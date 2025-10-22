#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2025 Apple Inc. All Rights Reserved.
#
# ruff: noqa
# pylint: skip-file

from typing import Optional, Tuple, Union, Literal, List, Dict, Any
from dataclasses import dataclass, asdict, field
import numpy as np

import torch
from torch import nn
from einops import rearrange
from diffusers.utils.torch_utils import randn_tensor

from .attention.serialized_attn import (
    SerializeMode,
    SerializeModes,
)
from .transformer.blocks import (
    AbsolutePositionEmbedder,
    LearnedPositionEmbedder,
    SparseTransformerBlock,
    SparseMultiheadAttentionPoolingHead,
    LearnedPositionEmbedder4D,
)
from .linear import SparseLinear
from .basic import SparseTensor
from .norm import LayerNorm32, RMSNorm32
from .sparse_preprocessors import process_image

import re


def image_to_sparse(image, patch_size):
    if len(image.shape) == 3:  # Single image: (H, W, C)
        image = image.unsqueeze(0)  # Add batch dimension: (1, H, W, C)

    if image.shape[-1] != 3:  # Single image: (C, H, W)
        image = image.permute(0, 2, 3, 1)  # Convert to (B, C, H, W)

    T, H, W, D = image.shape
    Pt, Ph, Pw = patch_size  # Unpack Ph and Pw from patch_size

    if T < Pt:
        # Zero padding
        pad = (torch.zeros((Pt - T, H, W, D), dtype=image.dtype)).to(image)
        image = torch.cat([image, pad], dim=0)
        T = Pt

    # Ensure dimensions are divisible by size_factor Pt.
    T = T // Pt * Pt
    image = image[:T]

    target_H = H // Ph * Ph
    target_W = W // Pw * Pw

    # Crop if the image is larger than target size
    if H > target_H or W > target_W:
        # Calculate valid ranges for crop
        h_start_max = H - target_H
        w_start_max = W - target_W

        # Center crop for evaluation
        h_start = h_start_max // 2
        w_start = w_start_max // 2
        # Perform crop
        image = image[:, h_start : h_start + target_H, w_start : w_start + target_W, :]

    # Z, Pz make to 1 for images
    Z = 1
    Pz = 1

    # reshape to (T//Pt, H//Ph, W//Pw, Pt*Ph*Pw*C)
    feats = rearrange(
        image,
        "(nt pt) (nh ph) (nw pw) c -> nt nh nw (pt ph pw c)",
        nt=T // Pt,
        nh=H // Ph,
        nw=W // Pw,
        pt=Pt,
        ph=Ph,
        pw=Pw,
    )

    feats = rearrange(feats, "t h w c -> (t h w) c")

    coords = (
        torch.stack(
            torch.meshgrid(
                torch.arange(T // Pt),
                torch.arange(H // Ph),
                torch.arange(W // Pw),
                torch.arange(Z // Pz),
                indexing="ij",
            ),
            dim=-1,
        )
        .reshape(-1, 4)
        .int()
    )

    return feats, coords


def sparse_to_img_list(sparse_tensor, patch_size, var_patch_axis=0, task_types=["image"]):
    """Sparse tensor to image list."""
    # check if this is a 3d batch, i.e., coords with different z index
    # if yes, then skip the list conversion
    if re.search(r"3D", task_types[0]):
        return [sparse_tensor]

    Pt, Ph, Pw = patch_size

    # we need to select patch_size based on input.
    full_patch_size = [Pt, Ph, Pw, 3]
    variable_factor = np.prod(full_patch_size) // sparse_tensor.feats.shape[1]
    select_size = full_patch_size[var_patch_axis] // variable_factor
    if var_patch_axis == 0:
        Pt = select_size
    else:
        ValueError("varibale_axis value wrong")

    images = []
    for batch_idx in range(sparse_tensor.shape[0]):
        task_type = task_types[batch_idx]
        batch_sparse_tensor = sparse_tensor[batch_idx]
        T, H, W, Z = batch_sparse_tensor.coords.max(dim=0)[0][1:] + 1
        img = rearrange(
            batch_sparse_tensor.feats,
            "(t h w) (Pt Ph Pw c) -> (t Pt) c (h Ph) (w Pw)",
            t=T,
            h=H,
            w=W,
            Pt=Pt,
            Ph=Ph,
            Pw=Pw,
        )

        if "image" in task_type and img.shape[0] > 1:
            img = img[0:1]

        images.append(img)

    return images


def download_if_s3(path: str, retry_count: int = 3, timeout: int = 120) -> str:
    """Return the path as-is (S3 functionality removed)."""
    return path


def multiply_all_factors(config):
    config_values = list(config.values())

    if not config_values:
        return []

    first_factor = config_values[0]["factor"]
    result = [1] * len(first_factor)

    for value_dict in config_values:
        factor_list = value_dict["factor"]
        for i in range(len(result)):
            result[i] *= factor_list[i]
    return result


def average_with_scatter_add(features, batch_mapping):
    device = features.device
    feature_dim = features.shape[1]
    num_original_batches = batch_mapping.max().item() + 1
    # Initialize output tensor
    averaged_feats = torch.zeros(
        (num_original_batches, feature_dim), dtype=features.dtype, device=device
    )
    batch_mapping = batch_mapping.long()
    # Count how many points contribute to each original batch
    counts = torch.zeros(num_original_batches, dtype=torch.float32, device=device)
    counts.scatter_add_(0, batch_mapping, torch.ones_like(batch_mapping, dtype=torch.float32))

    # Sum features for each original batch
    averaged_feats.scatter_add_(0, batch_mapping.unsqueeze(1).expand(-1, feature_dim), features)

    # Divide by counts to get average
    counts = counts.clamp(min=1)
    averaged_feats = averaged_feats / counts.unsqueeze(1)

    return averaged_feats


def batch_tensor_to_sparse(batch_tensor, patch_size):
    # input shape: b,t,h,w,c
    if batch_tensor.dim() == 4:  # img input
        if batch_tensor.shape[1] == 3:
            batch_tensor = rearrange(batch_tensor, "b c h w -> b h w c")
        batch_tensor = batch_tensor.unsqueeze(1)
    elif batch_tensor.dim() == 5:  # video input
        if batch_tensor.shape[1] == 3:
            batch_tensor = rearrange(batch_tensor, "b c t h w -> b t h w c")
    if batch_tensor.min() < 0:
        # assume input to be [-1, 1], convert to [0, 255]
        batch_tensor = batch_tensor * 127.5 + 127.5
    all_feats, all_coords = [], []
    for batch_idx in range(batch_tensor.shape[0]):
        feats, coords = process_image(
            batch_tensor[batch_idx],  # this asks for [0, 255]
            patch_size=patch_size,
            min_resolution=(64, 64),
            max_resolution=(512, 512),
            size_factor=16,
            temporal_padding_to=patch_size[0],
            random_crop=False,
            random_time=0,
            patch_sample_ratio=1.0,
            square_image_length=None,
            random_resolution=False,
        )
        batch_indices = torch.full(
            (coords.shape[0], 1), batch_idx, dtype=torch.int32, device=coords.device
        )
        coords = torch.cat([batch_indices, coords], dim=1)
        all_feats.append(feats)
        all_coords.append(coords)
    feats = torch.cat(all_feats, dim=0)
    coords = torch.cat(all_coords, dim=0)
    return SparseTensor(feats, coords).to(batch_tensor.device)


def block_attn_config(self, num_blocks=None, cfg={}):
    """
    Return the attention configuration of the model.
    """
    if num_blocks is None:
        num_blocks = self.num_blocks

    for i in range(num_blocks):
        attn_mode = cfg[i]["attn_mode"] if i in cfg else self.attn_mode
        window_size = cfg[i]["window_size"] if i in cfg else self.window_size

        if attn_mode == "shift_window":
            yield (
                "serialized",
                window_size,
                0,
                (16 * (i % 2),) * 3,
                SerializeMode.Z_ORDER,
            )
        elif attn_mode == "shift_sequence":
            yield (
                "serialized",
                window_size,
                window_size // 2 * (i % 2),
                (0, 0, 0),
                SerializeMode.Z_ORDER,
            )
        elif attn_mode == "shift_order":
            yield "serialized", window_size, 0, (0, 0, 0), SerializeModes[i % 4]
        elif attn_mode == "full":
            yield "full", None, None, None, None
        elif attn_mode == "swin":
            yield (
                "windowed",
                window_size,
                None,
                window_size // 2 * (i % 2),
                None,
            )
        else:
            raise ValueError(
                f"Unknown attention mode: {self.attn_mode}. Supported modes are: full, shift_window, shift_sequence, shift_order, swin."
            )


class DiagonalGaussianDistribution(object):
    def __init__(self, parameters: Union[torch.Tensor, SparseTensor], deterministic: bool = False):
        if isinstance(parameters, SparseTensor):
            self.raw_parameters = parameters
            self.parameters = parameters.feats
        else:
            self.raw_parameters = None
            self.parameters = parameters

        self.original_dtype = self.parameters.dtype
        self.parameters = self.parameters.to(torch.float32)
        self.mean, self.logvar = torch.chunk(self.parameters, 2, dim=1)
        # self.logvar = torch.clamp(self.logvar, -30.0, 20.0)
        self.deterministic = deterministic
        self.std = torch.exp(0.5 * self.logvar)
        self.var = torch.exp(self.logvar)
        if self.deterministic:
            self.var = self.std = torch.zeros_like(
                self.mean, device=self.parameters.device, dtype=self.parameters.dtype
            )

    def sample(self, generator: Optional[torch.Generator] = None) -> torch.Tensor:
        # make sure sample is on the same device as the parameters and has same dtype
        sample = randn_tensor(
            self.mean.shape,
            generator=generator,
            device=self.parameters.device,
            dtype=self.parameters.dtype,
        )
        x = self.mean + self.std * sample
        return x.to(self.original_dtype)

    def kl(self, other: "DiagonalGaussianDistribution" = None) -> torch.Tensor:
        if self.deterministic:
            return torch.Tensor([0.0])
        else:
            if other is None:
                return 0.5 * torch.sum(
                    torch.pow(self.mean, 2) + self.var - 1.0 - self.logvar, dim=[1]
                )
            else:
                return 0.5 * torch.sum(
                    torch.pow(self.mean - other.mean, 2) / other.var
                    + self.var / other.var
                    - 1.0
                    - self.logvar
                    + other.logvar,
                    dim=[1, 2, 3],
                )

    def nll(self, sample: torch.Tensor, dims: Tuple[int, ...] = [1, 2, 3]) -> torch.Tensor:
        if self.deterministic:
            return torch.Tensor([0.0])
        logtwopi = np.log(2.0 * np.pi)
        return 0.5 * torch.sum(
            logtwopi + self.logvar + torch.pow(sample - self.mean, 2) / self.var,
            dim=dims,
        )

    def mode(self) -> torch.Tensor:
        return self.mean.to(self.original_dtype)


class SparseTransformerBase(nn.Module):
    """
    Sparse Transformer without output layers.
    Serve as the base class for encoder and decoder.
    """

    def __init__(
        self,
        in_channels: int,
        model_channels: int,
        num_blocks: int,
        num_heads: Optional[int] = None,
        num_head_channels: Optional[int] = 64,
        mlp_channels: float = 2048,
        attn_mode: Literal[
            "full", "shift_window", "shift_sequence", "shift_order", "swin"
        ] = "full",
        window_size: Optional[int] = None,
        pe_mode: Literal["ape", "rope", "learned", "learned4d", "joint"] = "rope",
        use_checkpoint: bool = False,
        qk_rms_norm: bool = False,
        use_bias: bool = False,
        use_rms_norm: bool = True,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.model_channels = model_channels
        self.num_blocks = num_blocks
        self.window_size = window_size
        self.num_heads = num_heads or model_channels // num_head_channels
        self.mlp_channels = mlp_channels
        self.attn_mode = attn_mode
        self.pe_mode = pe_mode
        self.use_checkpoint = use_checkpoint
        self.qk_rms_norm = qk_rms_norm
        self.use_bias = use_bias
        self.use_rms_norm = use_rms_norm

        if pe_mode == "ape":
            self.pos_embedder = AbsolutePositionEmbedder(model_channels)
        elif pe_mode == "learned4d":
            self.pos_embedder = LearnedPositionEmbedder4D(model_channels)
        elif pe_mode in ["joint", "learned"]:
            self.pos_embedder = LearnedPositionEmbedder(model_channels)

        self.input_layer = SparseLinear(in_channels, model_channels, bias=use_bias)
        cfg = {}

        self.blocks = nn.ModuleList(
            [
                SparseTransformerBlock(
                    cfg[i]["model_channels"] if i in cfg else model_channels,
                    num_heads=cfg[i]["num_heads"] if i in cfg else num_heads,
                    mlp_channels=cfg[i]["mlp_channels"] if i in cfg else mlp_channels,
                    attn_mode=attn_mode,
                    window_size=window_size,
                    shift_sequence=shift_sequence,
                    shift_window=shift_window,
                    serialize_mode=serialize_mode,
                    use_checkpoint=self.use_checkpoint,
                    use_rope=(pe_mode in ["rope", "joint"]),
                    qk_rms_norm=self.qk_rms_norm,
                    use_bias=self.use_bias,
                    use_rms_norm=self.use_rms_norm,
                )
                for i, (
                    attn_mode,
                    window_size,
                    shift_sequence,
                    shift_window,
                    serialize_mode,
                ) in enumerate(block_attn_config(self, num_blocks=num_blocks, cfg=cfg))
            ]
        )

    @property
    def device(self) -> torch.device:
        """
        Return the device of the model.
        """
        return next(self.parameters()).device

    def forward(
        self,
        x: SparseTensor,
        kv_cache: Optional[Dict[str, SparseTensor]] = None,
        kv_cache_size: Optional[int] = None,
    ) -> SparseTensor:
        hs = []

        # Initialize kv_cache structure if not provided
        if kv_cache is None:
            kv_cache = {}

        x = x.to(next(self.input_layer.parameters()).dtype)

        h = self.input_layer(x)

        if self.pe_mode == "ape":
            h = h + self.pos_embedder(x.coords[:, 1:]).to(h.dtype)
        elif self.pe_mode in ["learned", "learned4d", "joint"]:
            h = h + self.pos_embedder(x)

        h = h.to(next(self.blocks.parameters()).dtype)

        hs.append(h)
        for block_idx, block in enumerate(self.blocks):
            block_cache_key = f"block_{block_idx}"
            if block_cache_key not in kv_cache:
                kv_cache[block_cache_key] = {}

            block_cache = kv_cache[block_cache_key]

            h, updated_block_cache = block(h, kv_cache=block_cache, kv_cache_size=kv_cache_size)

            # Update the block's cache
            kv_cache[block_cache_key] = updated_block_cache

            hs.append(h)

        return h, hs, kv_cache


class Encoder(SparseTransformerBase):
    def __init__(
        self,
        in_channels,
        model_channels,
        num_blocks,
        num_heads=None,
        num_head_channels=64,
        mlp_channels=2048,
        attn_mode="full",
        window_size=None,
        pe_mode="ape",
        use_checkpoint=False,
        qk_rms_norm=False,
        use_bias=False,
        use_rms_norm=True,
        use_head=False,
    ):
        super().__init__(
            in_channels,
            model_channels,
            num_blocks,
            num_heads,
            num_head_channels,
            mlp_channels,
            attn_mode,
            window_size,
            pe_mode,
            use_checkpoint,
            qk_rms_norm,
            use_bias=use_bias,
            use_rms_norm=use_rms_norm,
        )
        self.use_head = use_head

        if use_rms_norm:
            self.post_layernorm = RMSNorm32(model_channels, eps=1e-6)
        else:
            self.post_layernorm = LayerNorm32(model_channels, elementwise_affine=True, eps=1e-6)

        self.head = SparseMultiheadAttentionPoolingHead(
            hidden_size=model_channels,
            num_attention_heads=num_heads,
            intermediate_size=mlp_channels,
            use_bias=use_bias,
            use_rms_norm=use_rms_norm,
            qk_rms_norm=qk_rms_norm,
        )

    def forward(
        self, x: SparseTensor, kv_cache: Optional[Dict[str, SparseTensor]] = None
    ) -> Union[SparseTensor, Tuple[SparseTensor, torch.Tensor]]:
        h, hs, kv_cache = super().forward(x, kv_cache=kv_cache)
        h = h.replace(self.post_layernorm(h.feats))

        if self.use_head:
            pooler_output = self.head(h)
            return h, pooler_output.feats, kv_cache
        else:
            return h, kv_cache


class Decoder(SparseTransformerBase):
    def __init__(
        self,
        in_channels,
        out_channels,
        model_channels,
        num_blocks,
        num_heads=None,
        num_head_channels=64,
        mlp_channels=2048,
        attn_mode="full",
        window_size=None,
        pe_mode="ape",
        use_checkpoint=False,
        qk_rms_norm=False,
        use_bias=False,
        use_rms_norm=True,
    ):
        super().__init__(
            in_channels,
            model_channels,
            num_blocks,
            num_heads,
            num_head_channels,
            mlp_channels,
            attn_mode,
            window_size,
            pe_mode,
            use_checkpoint,
            qk_rms_norm,
            use_bias=use_bias,
            use_rms_norm=use_rms_norm,
        )

        if use_rms_norm:
            self.out_norm = RMSNorm32(model_channels, eps=1e-6)
        else:
            self.out_norm = LayerNorm32(model_channels, elementwise_affine=False, eps=1e-6)

        # original output layer.
        self.out_layer = SparseLinear(
            model_channels,
            out_channels,
            bias=False,
        )

    def forward(
        self,
        x: SparseTensor,
        kv_cache: Optional[Dict[str, SparseTensor]] = None,
        kv_cache_size: Optional[int] = None,
    ):
        h, hs, kv_cache = super().forward(x, kv_cache=kv_cache, kv_cache_size=kv_cache_size)

        h = h.replace(self.out_norm(h.feats))
        h = self.out_layer(h)
        return h, kv_cache
