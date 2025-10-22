#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2025 Apple Inc. All Rights Reserved.
#
# ruff: noqa
# pylint: skip-file

from typing import Optional, Tuple, Union, Literal, List, Dict, Any
from dataclasses import dataclass, asdict, field
import numpy as np
from functools import partial
import re

import torch
from torch import nn

import torch.nn.functional as F
from einops import rearrange
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.loaders.single_file_model import FromOriginalModelMixin
from diffusers.utils.accelerate_utils import apply_forward_hook
from diffusers.models.modeling_outputs import AutoencoderKLOutput
from diffusers.models.modeling_utils import ModelMixin
from diffusers.models.autoencoders.vae import DecoderOutput

from .linear import SparseLinear
from .basic import (
    SparseTensor,
    reconstruct_from_temporal_slices,
)
from .utils import (
    sparse_to_img_list,
)
from .finite_scaler_quantize import (
    levels_from_codebook_size,
    FSQ,
)
from .utils import (
    batch_tensor_to_sparse,
    DiagonalGaussianDistribution,
    Encoder,
    Decoder,
)

from .lookup_free_quantize import LFQ


@dataclass
class AutoencoderKLConfig:
    patch_size: Tuple[int, int, int] = (1, 8, 8)
    in_channels: int = 192
    out_channels: int = 192
    latent_channels: int = 8
    encoder_model_channels: int = 768
    encoder_num_blocks: int = 12
    encoder_num_heads: Optional[int] = None
    encoder_mlp_channels: float = 2048
    encoder_attn_mode: str = "swin"
    encoder_window_size: int = 8
    encoder_pe_mode: str = "rope"
    encoder_qk_rms_norm: bool = True
    encoder_use_bias: bool = True
    encoder_use_rms_norm: bool = False
    decoder_model_channels: int = 768
    decoder_num_blocks: int = 12
    decoder_num_heads: Optional[int] = None
    decoder_mlp_channels: float = 2048
    decoder_attn_mode: str = "swin"
    decoder_window_size: int = 8
    decoder_pe_mode: str = "rope"
    decoder_qk_rms_norm: bool = True
    decoder_use_bias: bool = True
    decoder_use_rms_norm: bool = False
    use_decoder: bool = True
    use_quantizer: bool = False
    quantizer_type: Literal["fsq", "lfq"] = "fsq"
    quantizer_codebook_size: int = 65536
    quantizer_num_codebooks: int = 1
    quantizer_feature_dim: int = 48
    quantizer_chunk_size: int = 1


class AutoencoderKL(ModelMixin, ConfigMixin, FromOriginalModelMixin):
    r"""
    A VAE model with KL loss for encoding images into latents and decoding latent representations into images.

    This model inherits from [`ModelMixin`]. Check the superclass documentation for it's generic methods implemented
    for all models (such as downloading or saving).

    Parameters:
        in_channels (int, *optional*, defaults to 3): Number of channels in the input image.
        out_channels (int,  *optional*, defaults to 3): Number of channels in the output.
        down_block_types (`Tuple[str]`, *optional*, defaults to `("DownEncoderBlock2D",)`):
            Tuple of downsample block types.
        up_block_types (`Tuple[str]`, *optional*, defaults to `("UpDecoderBlock2D",)`):
            Tuple of upsample block types.
        block_out_channels (`Tuple[int]`, *optional*, defaults to `(64,)`):
            Tuple of block output channels.
        act_fn (`str`, *optional*, defaults to `"silu"`): The activation function to use.
        latent_channels (`int`, *optional*, defaults to 4): Number of channels in the latent space.
        sample_size (`int`, *optional*, defaults to `32`): Sample input size.
        scaling_factor (`float`, *optional*, defaults to 0.18215):
            The component-wise standard deviation of the trained latent space computed using the first batch of the
            training set. This is used to scale the latent space to have unit variance when training the diffusion
            model. The latents are scaled with the formula `z = z * scaling_factor` before being passed to the
            diffusion model. When decoding, the latents are scaled back to the original scale with the formula: `z = 1
            / scaling_factor * z`. For more details, refer to sections 4.3.2 and D.1 of the [High-Resolution Image
            Synthesis with Latent Diffusion Models](https://arxiv.org/abs/2112.10752) paper.
        force_upcast (`bool`, *optional*, default to `True`):
            If enabled it will force the VAE to run in float32 for high image resolution pipelines, such as SD-XL. VAE
            can be fine-tuned / trained to a lower range without loosing too much precision in which case
            `force_upcast` can be set to `False` - see: https://huggingface.co/madebyollin/sdxl-vae-fp16-fix
        mid_block_add_attention (`bool`, *optional*, default to `True`):
            If enabled, the mid_block of the Encoder and Decoder will have attention blocks. If set to false, the
            mid_block will only have resnet blocks
    """

    _supports_gradient_checkpointing = True
    _no_split_modules = ["BasicTransformerBlock", "ResnetBlock2D"]

    # pylint: disable=unused-argument
    @register_to_config
    def __init__(
        self,
        patch_size: Tuple[int, int, int] = (1, 8, 8),
        in_channels: int = 192,
        out_channels: int = 192,
        latent_channels: int = 8,
        encoder_model_channels: int = 768,
        encoder_num_blocks: int = 12,
        encoder_num_heads: Optional[int] = None,
        encoder_mlp_channels: float = 2048,
        encoder_attn_mode: str = "swin",
        encoder_window_size: int = 8,
        encoder_pe_mode: str = "rope",
        encoder_qk_rms_norm: bool = True,
        encoder_use_bias: bool = False,
        encoder_use_rms_norm: bool = True,
        decoder_model_channels: int = 768,
        decoder_num_blocks: int = 12,
        decoder_num_heads: Optional[int] = None,
        decoder_mlp_channels: float = 2048,
        decoder_attn_mode: str = "swin",
        decoder_window_size: int = 8,
        decoder_pe_mode: str = "rope",
        decoder_qk_rms_norm: bool = True,
        decoder_use_bias: bool = False,
        decoder_use_rms_norm: bool = True,
        use_decoder: bool = True,
        use_quantizer: bool = False,
        quantizer_type: Literal["fsq", "lfq"] = "fsq",
        quantizer_codebook_size: int = 65536,
        quantizer_num_codebooks: int = 1,
        quantizer_feature_dim: int = 6,
        quantizer_chunk_size: int = 1,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.use_quantizer = use_quantizer
        self.quantizer_type = quantizer_type
        self.quantizer_codebook_size = quantizer_codebook_size
        self.quantizer_num_codebooks = quantizer_num_codebooks
        self.quantizer_feature_dim = quantizer_feature_dim
        self.quantizer_chunk_size = quantizer_chunk_size

        self.in_channels = in_channels
        self.num_sample_frames_batch_size = 16
        self.num_sample_frames_stride = 16  # this is only for inference
        self.kv_cache_size = 0  # this is only for inference. the sum of this and num_sample_frames_stride should be equal or less to num_sample_frames_batch_size

        # Add use_slicing attribute for compatibility with diffusers
        self.use_slicing = False
        # pass init params to Encoder
        self.encoder = Encoder(
            in_channels=in_channels,
            model_channels=encoder_model_channels,
            num_blocks=encoder_num_blocks,
            num_heads=encoder_num_heads,
            mlp_channels=encoder_mlp_channels,
            attn_mode=encoder_attn_mode,
            window_size=encoder_window_size,
            pe_mode=encoder_pe_mode,
            qk_rms_norm=encoder_qk_rms_norm,
            use_bias=encoder_use_bias,
            use_rms_norm=encoder_use_rms_norm,
            use_checkpoint=True,
        )

        self.proj = SparseLinear(encoder_model_channels, 2 * latent_channels)

        if use_quantizer:
            # init fsq quantizer here:
            if self.quantizer_type == "lfq":
                self.quantizer = LFQ(
                    dim=latent_channels,
                    codebook_size=self.quantizer_codebook_size,
                    num_codebooks=self.quantizer_num_codebooks,
                    sample_minimization_weight=1.0,
                    batch_maximization_weight=1.0,
                    token_factorization=False,
                    factorized_bits=[9, 9],
                )
            else:
                levels, _ = levels_from_codebook_size(self.quantizer_codebook_size)
                self.quantizer = FSQ(
                    levels=levels,
                    dim=self.quantizer_feature_dim // self.quantizer_chunk_size,
                    num_codebooks=self.quantizer_num_codebooks,
                )

        if use_decoder:
            # pass init params to Decoder
            self.decoder = Decoder(
                in_channels=latent_channels,
                out_channels=out_channels,
                model_channels=decoder_model_channels,
                num_blocks=decoder_num_blocks,
                num_heads=decoder_num_heads,
                mlp_channels=decoder_mlp_channels,
                attn_mode=decoder_attn_mode,
                window_size=decoder_window_size,
                qk_rms_norm=decoder_qk_rms_norm,
                use_bias=decoder_use_bias,
                use_rms_norm=decoder_use_rms_norm,
                pe_mode=decoder_pe_mode,
                use_checkpoint=True,
            )

        self.logit_scale = nn.Parameter(torch.ones(1))
        self.logit_bias = nn.Parameter(torch.ones(1))

        self.use_slicing = False

    def _set_gradient_checkpointing(self, module, value=False):
        if isinstance(module, (Encoder, Decoder)):
            module.gradient_checkpointing = value

    def enable_slicing(self):
        r"""
        Enable sliced VAE decoding. When this option is enabled, the VAE will split the input tensor in slices to
        compute decoding in several steps. This is useful to save some memory and allow larger batch sizes.
        """
        self.use_slicing = True

    def disable_slicing(self):
        r"""
        Disable sliced VAE decoding. If `enable_slicing` was previously enabled, this method will go back to computing
        decoding in one step.
        """
        self.use_slicing = False

    def _encode(self, x: SparseTensor, normalize: bool = False) -> SparseTensor:
        """
        Fixed version of _encode that preserves slice alignment during processing
        """
        frame_batch_size = self.num_sample_frames_batch_size // self.patch_size[0]

        # Split into temporal batches with adjusted coordinates, we are not encode with cache, thus we can ignore the offsets.
        temporal_slices = x.split_by_temporal_batches(frame_batch_size, adjust_temporal=True)
        processed_slices = []
        kv_cache = None

        for x_slice in temporal_slices:
            if x_slice.coords.shape[0] > 0:
                # Process non-empty slice
                enc_slice, kv_cache = self.encoder(x_slice, kv_cache)
                processed_slices.append(enc_slice)
            else:
                processed_slices.append(x_slice)

        # Reconstruct with proper alignment
        enc_full = reconstruct_from_temporal_slices(
            processed_slices, target_coords=x.coords, use_cached_offsets=True
        )
        image_feat = self.encoder.head(enc_full).feats

        if normalize:
            image_feat = F.normalize(image_feat, dim=-1)

        enc_proj = self.proj(enc_full)

        return enc_proj, image_feat, enc_full

    @apply_forward_hook
    def encode(
        self,
        x: Union[SparseTensor, torch.Tensor],
        normalize: bool = False,
        return_dict: bool = False,
    ) -> Union[AutoencoderKLOutput, Tuple[DiagonalGaussianDistribution]]:
        """
        Encode a batch of images into latents.
        """
        del return_dict
        if self.use_slicing and isinstance(x, torch.Tensor) and x.shape[0] > 1:
            raise ValueError("Legacy tensor slicing not implemented yet")

        if isinstance(x, torch.Tensor):
            x = batch_tensor_to_sparse(x, self.patch_size)

        x, image_feat, x_no_proj = self._encode(x, normalize=normalize)
        return x, image_feat, x_no_proj

    def _decode(
        self, z: torch.Tensor, return_dict: bool = True, training: bool = True
    ) -> Union[DecoderOutput, SparseTensor]:
        """
        Fixed version of _decode that preserves slice alignment during processing
        """

        kv_cache = None
        frame_batch_size = self.num_sample_frames_batch_size // self.patch_size[0]
        frame_batch_strides = self.num_sample_frames_stride // self.patch_size[0]
        kv_cache_size = self.kv_cache_size // self.patch_size[0]

        if training:
            kv_cache_size = 0
            frame_batch_strides = frame_batch_size
        else:
            frame_batch_size = frame_batch_strides
            kv_cache_size = kv_cache_size

        temporal_slices = z.split_by_temporal_batches(
            frame_batch_size,
            frame_batch_strides,
            adjust_temporal=True,
            offset=kv_cache_size,
        )

        # Process each slice
        processed_slices = []

        for z_slice in temporal_slices:
            if z_slice.coords.shape[0] > 0:
                # Process non-empty slice
                dec_slice, kv_cache = self.decoder(z_slice, kv_cache, kv_cache_size)
                processed_slices.append(dec_slice)
            else:
                processed_slices.append(z_slice)

        # Reconstruct with proper alignment
        dec = reconstruct_from_temporal_slices(
            processed_slices, target_coords=z.coords, use_cached_offsets=True
        )

        if not return_dict:
            return (dec,)

        return DecoderOutput(sample=dec)

    @apply_forward_hook
    def decode(
        self,
        z: Union[SparseTensor, torch.Tensor],
        return_dict: bool = True,
        return_batched_tensor=False,
        training: bool = True,
    ) -> Union[DecoderOutput, SparseTensor]:
        """
        Decode a batch of latent representations.
        """
        if self.use_slicing and isinstance(z, torch.Tensor) and z.shape[0] > 1:
            # Legacy tensor slicing (not implemented)
            raise ValueError("Legacy tensor slicing not implemented yet")
        else:
            decoded = self._decode(z, training=training).sample

        if return_batched_tensor and isinstance(decoded, SparseTensor):
            decoded = sparse_to_img_list(decoded, self.patch_size)
            # check if all shape is the same
            if len(set([x.shape for x in decoded])) > 1:
                print("WARNING: decoded shapes are not the same")
                print(f"Decoded shapes: {[x.shape for x in decoded]}")
            else:
                # stack on the batch size
                decoded = torch.stack(decoded, dim=0)
                decoded = rearrange(decoded, "b t c h w -> b t h w c")
                if decoded.shape[1] == 1:
                    decoded = decoded.squeeze(1)

        if not return_dict:
            return (decoded,)

        return DecoderOutput(sample=decoded)
