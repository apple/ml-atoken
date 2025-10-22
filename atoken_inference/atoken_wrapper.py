#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2025 Apple Inc. All Rights Reserved.
#
from dataclasses import asdict
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
import yaml
from safetensors.torch import load_file

# Relative imports for local atoken modules
from .model.autoencoder_kl import (
    AutoencoderKL,
    AutoencoderKLConfig,
    DiagonalGaussianDistribution,
)
from .model.basic import SparseTensor
from .model.decoder_gs import SLatGaussianDecoderTrain
from .model.utils import download_if_s3, image_to_sparse


class ATokenWrapper(nn.Module):
    def __init__(self, config_path, model_path, gs_config_path=None, gs_model_path=None):
        super().__init__()

        self.is_loaded = False
        self.config_path = config_path
        self.model_path = model_path

        self.gs_config_path = gs_config_path
        self.gs_model_path = gs_model_path

        self.pretrained_strict = True
        self.use_quantizer = False
        # Initialize model components
        self.model: Optional[AutoencoderKL] = None
        self.config: Optional[Dict[str, Any]] = None

        self._load_model()

        if gs_config_path is not None and gs_model_path is not None:
            self._load_gs_model()

    def _load_model(self):
        """Load the AToken model from config and checkpoint."""

        # Load configuration
        with open(self.config_path, "r", encoding="utf-8") as file:
            self.config = yaml.safe_load(file)

        # Extract model configuration
        if self.config is None:
            raise ValueError("Failed to load config file")

        model_config = self.config.get("model", {})
        model_cfg = model_config.get("model_cfg", {})
        arch_cfg = model_cfg.get("arch_cfg", {})

        self.arch_cfg = arch_cfg
        self.model_cfg = model_cfg

        # Create AutoencoderKLConfig from arch_cfg
        autoencoder_config = AutoencoderKLConfig(**arch_cfg)

        # Initialize the model directly
        self.model = AutoencoderKL(**asdict(autoencoder_config))

        if self.model_path is not None:
            pretrained_path = download_if_s3(self.model_path)
            if pretrained_path.endswith("safetensors"):
                pretrained_ckpt = load_file(pretrained_path)
            else:
                pretrained_ckpt = torch.load(pretrained_path, map_location="cpu")

            self.model.load_state_dict(pretrained_ckpt, strict=self.pretrained_strict)

        self.is_loaded = True

    def _load_gs_model(self):
        """Load the gs model from config and checkpoint."""

        # Load configuration
        with open(self.gs_config_path, "r", encoding="utf-8") as file:
            self.gs_config = yaml.safe_load(file)

        # Extract model configuration
        if self.gs_config is None:
            raise ValueError("Failed to load config file")

        model_config = self.gs_config.get("model", {})

        self.gs_model = SLatGaussianDecoderTrain(**model_config)

        if self.gs_model_path is not None:
            pretrained_path = download_if_s3(self.gs_model_path)
            if pretrained_path.endswith("safetensors"):
                pretrained_ckpt = load_file(pretrained_path)
            else:
                pretrained_ckpt = torch.load(pretrained_path, map_location="cpu")

            self.gs_model.load_state_dict(pretrained_ckpt, strict=self.pretrained_strict)

    def encode(self, x, normalize=False, return_dict=False):
        """
        Encode input to latent representation.

        Args:
            x: Input tensor or SparseTensor
            normalize: Whether to normalize the output
            return_dict: Whether to return as dictionary

        Returns:
            Encoded latent representation
        """
        if self.model is None:
            self._load_model()

        if self.model is None:
            raise RuntimeError("Failed to load AToken model")

        return self.model.encode(x, normalize=normalize, return_dict=return_dict)

    def decode(self, z, return_dict=True, training=False):
        """
        Decode latent representation to output.

        Args:
            z: Latent representation
            return_dict: Whether to return as dictionary
            training: Whether in training mode

        Returns:
            Decoded output
        """
        if self.model is None:
            self._load_model()

        if self.model is None:
            raise RuntimeError("Failed to load AToken model")

        return self.model.decode(z, return_dict=return_dict, training=training)

    @torch.no_grad()
    def inference(self, x, **kwargs):
        """
        Inference the video and get the rec.
        """
        task_types = kwargs.get("task_types", ["image"])

        dtype = next(self.model.parameters()).dtype
        device = next(self.model.parameters()).device

        feats = x.feats.to(device=device, dtype=dtype)
        coords = x.coords.to(device).int()
        x = SparseTensor(feats=feats, coords=coords)

        z, image_feat, x_no_proj = self.encode(x, normalize=True)

        pos = DiagonalGaussianDistribution(z)
        latent = z.replace(pos.sample())

        if self.arch_cfg.get("use_quantizer", False):
            z_feat = z.feats[:, : self.model.quantizer_feature_dim]
            chunk_dim = self.model.quantizer_feature_dim // self.model.quantizer_chunk_size
            bs = z_feat.shape[0]
            if self.model.quantizer_chunk_size > 1:
                z_feat = z_feat.reshape(bs * self.model.quantizer_chunk_size, chunk_dim)

            quantized, indices, commit_loss = self.model.quantizer(
                z_feat,
            )

            if self.model.quantizer_chunk_size > 1:
                quantized = quantized.reshape(bs, self.model.quantizer_feature_dim)

            latent = latent.replace(quantized)

        rec = self.decode(latent, training=False).sample

        return rec, image_feat, x_no_proj

    @torch.no_grad()
    def inference_3d(self, feats, coords, in_channel=3072):
        feats = torch.tensor(feats).float().cuda() * 2 - 1
        data_in_channels = feats.shape[1]

        feats = torch.cat(
            [
                feats,
                torch.zeros(
                    [
                        feats.shape[0],
                        in_channel - data_in_channels,
                    ]
                ).to(feats),
            ],
            dim=1,
        )
        coords = torch.tensor(coords).int().cuda()
        coords = torch.cat(
            [torch.zeros_like(coords[:, :1]), torch.ones_like(coords[:, :1]), coords],
            dim=-1,
        )

        dtype = next(self.model.parameters()).dtype
        device = next(self.model.parameters()).device

        feats = feats.to(device=device, dtype=dtype)
        coords = coords.to(device).int()

        x = SparseTensor(feats=feats, coords=coords)

        z, image_feat, x_no_proj = self.encode(x, normalize=True)

        pos = DiagonalGaussianDistribution(z)
        latent = z.replace(pos.sample())

        if self.arch_cfg.get("use_quantizer", False):
            z_feat = z.feats[:, : self.model.quantizer_feature_dim]
            chunk_dim = self.model.quantizer_feature_dim // self.model.quantizer_chunk_size
            bs = z_feat.shape[0]
            if self.model.quantizer_chunk_size > 1:
                z_feat = z_feat.reshape(bs * self.model.quantizer_chunk_size, chunk_dim)

            quantized, indices, commit_loss = self.model.quantizer(
                z_feat,
            )

            if self.model.quantizer_chunk_size > 1:
                quantized = quantized.reshape(bs, self.model.quantizer_feature_dim)

            latent = latent.replace(quantized)

        rec = self.decode(latent, training=False).sample

        rec_x_no_t = (
            SparseTensor(feats=rec.feats[:, :data_in_channels], coords=coords[:, [0, 2, 3, 4]])
            .to(device="cuda")
            .to(dtype=torch.bfloat16)
        )

        gaussians = self.gs_model.forward_decoder(rec_x_no_t)

        return gaussians, image_feat, x_no_proj

    def image_video_to_sparse_tensor(self, images, patch_size=None):
        """
        convert image and video to sparse format.
        """
        batch_coords = []
        batch_feats = []
        image_sizes = []

        if patch_size is None:
            patch_size = self.arch_cfg["patch_size"]

        for i, image in enumerate(images):
            feats, coords = image_to_sparse(image, patch_size)
            bcoords = (
                torch.zeros([coords.shape[0], 1]).to(device=coords.device, dtype=coords.dtype) + i
            )
            new_coords = torch.cat([bcoords, coords], dim=1)
            # get the image size from the coords
            image_sizes.append((coords[:, 1].max().item() + 1, coords[:, 2].max().item() + 1))
            batch_coords.append(new_coords)
            batch_feats.append(feats)

        coords = torch.cat(batch_coords, dim=0).to(images[0].device).int()
        feats = torch.cat(batch_feats, dim=0).to(images[0].device)
        x = SparseTensor(feats=feats, coords=coords)
        return x

    def video_to_padded_sparse_tensor(self, images, patch_size=[4, 16, 16]):
        """
        convert video into padded sparse tensor.
        """
        batch_coords = []
        batch_feats = []
        image_sizes = []

        target_dim = patch_size[0] * patch_size[1] * patch_size[2] * 3
        for i, image in enumerate(images):
            feats, coords = image_to_sparse(image, [1, patch_size[1], patch_size[2]])
            padded = torch.zeros([feats.shape[0], target_dim - feats.shape[1]]).to(
                device=feats.device, dtype=feats.dtype
            )

            feats = torch.cat([feats, padded], dim=-1)
            # paded to target batch size.
            bcoords = (
                torch.zeros([coords.shape[0], 1]).to(device=coords.device, dtype=coords.dtype) + i
            )
            new_coords = torch.cat([bcoords, coords], dim=1)
            # get the image size from the coords
            image_sizes.append((coords[:, 1].max().item() + 1, coords[:, 2].max().item() + 1))
            batch_coords.append(new_coords)
            batch_feats.append(feats)

        coords = torch.cat(batch_coords, dim=0).to(images[0].device).int()
        feats = torch.cat(batch_feats, dim=0).to(images[0].device)
        x = SparseTensor(feats=feats, coords=coords)
        return x

    @torch.no_grad()
    def forward(self, images):
        """
        image: List of images.
        """

        x = self.image_video_to_sparse_tensor(images)

        # first convert the images to SparseTensor
        z, image_feat, x_no_proj = self.encode(x, normalize=True)

        # convert the x_no_proj to list of dense tensors.
        image_features = []
        for i in range(len(images)):
            image_features.append(x_no_proj[i].feats.unsqueeze(0))

        if len(image_sizes) == 1:
            image_sizes = image_sizes[0]
            image_features = image_features[0]

        return image_features, image_sizes

    @property
    def hidden_size(self):
        return self.arch_cfg["encoder_model_channels"]
