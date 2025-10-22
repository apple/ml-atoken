#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2025 Apple Inc. All Rights Reserved.
#
# ruff: noqa
# pylint: skip-file
from typing import List, Optional, Literal
import json

import torch
from torch import nn
import torch.nn.functional as F
from torchmetrics.image import (
    MultiScaleStructuralSimilarityIndexMeasure,
    LearnedPerceptualImagePatchSimilarity,
    PeakSignalNoiseRatio,
)
from safetensors.torch import save_file

from .base_module import BaseModule
from .basic import SparseTensor
from .linear import SparseLinear
from .gs.random_utils import hammersley_sequence
from .gs.representations import Gaussian
from .gs.renderers.gaussian_render import GaussianRenderer
from .utils import SparseTransformerBase


class SLatGaussianDecoder(SparseTransformerBase):
    def __init__(
        self,
        resolution: int,
        model_channels: int,
        latent_channels: int,
        num_blocks: int,
        num_heads: Optional[int] = None,
        mlp_channels: int = 3072,
        num_head_channels: Optional[int] = 64,
        attn_mode: Literal[
            "full", "shift_window", "shift_sequence", "shift_order", "swin"
        ] = "swin",
        window_size: int = 8,
        pe_mode: Literal["ape", "rope"] = "ape",
        use_checkpoint: bool = False,
        qk_rms_norm: bool = False,
        representation_config: dict = None,
        **kwargs,
    ):
        del kwargs
        super().__init__(
            in_channels=latent_channels,
            model_channels=model_channels,
            num_blocks=num_blocks,
            num_heads=num_heads,
            mlp_channels=mlp_channels,
            num_head_channels=num_head_channels,
            attn_mode=attn_mode,
            window_size=window_size,
            pe_mode=pe_mode,
            use_checkpoint=use_checkpoint,
            qk_rms_norm=qk_rms_norm,
        )
        self.latent_channels = latent_channels
        self.num_head_channels = num_head_channels
        self.resolution = resolution
        self.rep_config = representation_config
        self._calc_layout()
        self.out_layer = SparseLinear(model_channels, self.out_channels, bias=True)
        self._build_perturbation()
        self.initialize_weights()

    def initialize_weights(self) -> None:
        # Zero-out output layers:
        nn.init.constant_(self.out_layer.weight, 0)
        nn.init.constant_(self.out_layer.bias, 0)

    def _build_perturbation(self) -> None:
        perturbation = [
            hammersley_sequence(3, i, self.rep_config["num_gaussians"])
            for i in range(self.rep_config["num_gaussians"])
        ]
        perturbation = torch.tensor(perturbation).float() * 2 - 1
        perturbation = perturbation / self.rep_config["voxel_size"]
        perturbation = torch.atanh(perturbation).to(self.device)
        self.register_buffer("offset_perturbation", perturbation)

    def _calc_layout(self) -> None:
        self.layout = {
            "_xyz": {
                "shape": (self.rep_config["num_gaussians"], 3),
                "size": self.rep_config["num_gaussians"] * 3,
            },
            "_features_dc": {
                "shape": (self.rep_config["num_gaussians"], 1, 3),
                "size": self.rep_config["num_gaussians"] * 3,
            },
            "_scaling": {
                "shape": (self.rep_config["num_gaussians"], 3),
                "size": self.rep_config["num_gaussians"] * 3,
            },
            "_rotation": {
                "shape": (self.rep_config["num_gaussians"], 4),
                "size": self.rep_config["num_gaussians"] * 4,
            },
            "_opacity": {
                "shape": (self.rep_config["num_gaussians"], 1),
                "size": self.rep_config["num_gaussians"],
            },
        }
        start = 0
        for _, v in self.layout.items():
            v["range"] = (start, start + v["size"])
            start += v["size"]
        self.out_channels = start

    def to_representation(self, x: SparseTensor) -> List[Gaussian]:
        """
        Convert a batch of network outputs to 3D representations.

        Args:
            x: The [N x * x C] sparse tensor output by the network.

        Returns:
            list of representations
        """
        x = x.to(torch.float32)  # gaussian parameter should be torch.float32 type
        ret = []
        for i in range(x.shape[0]):
            representation = Gaussian(
                sh_degree=0,
                aabb=[-0.5, -0.5, -0.5, 1.0, 1.0, 1.0],
                mininum_kernel_size=self.rep_config["3d_filter_kernel_size"],
                scaling_bias=self.rep_config["scaling_bias"],
                opacity_bias=self.rep_config["opacity_bias"],
                scaling_activation=self.rep_config["scaling_activation"],
                device=x.device,
            )
            xyz = (x.coords[x.layout[i]][:, 1:].float() + 0.5) / self.resolution
            for k, v in self.layout.items():
                if k == "_xyz":
                    offset = x.feats[x.layout[i]][:, v["range"][0] : v["range"][1]].reshape(
                        -1, *v["shape"]
                    )
                    offset = offset * self.rep_config["lr"][k]
                    if self.rep_config["perturb_offset"]:
                        offset = offset + self.offset_perturbation
                    offset = (
                        torch.tanh(offset) / self.resolution * 0.5 * self.rep_config["voxel_size"]
                    )
                    _xyz = xyz.unsqueeze(1) + offset
                    setattr(representation, k, _xyz.flatten(0, 1))
                else:
                    feats = (
                        x.feats[x.layout[i]][:, v["range"][0] : v["range"][1]]
                        .reshape(-1, *v["shape"])
                        .flatten(0, 1)
                    )
                    feats = feats * self.rep_config["lr"][k]
                    setattr(representation, k, feats)
            ret.append(representation)
        return ret

    def forward(self, x: SparseTensor) -> List[Gaussian]:
        h = super().forward(x)[0]
        h = h.type(x.dtype)
        h = h.replace(F.layer_norm(h.feats, h.feats.shape[-1:]))
        h = self.out_layer(h)
        return self.to_representation(h)

    def save_pretrained(self, path: str) -> None:
        """Save the model to the given path."""
        cfg_json = {
            "name": "atoken.models.pretrained.decoder_gs.SLatGaussianDecoder",
            "args": {
                "resolution": self.resolution,
                "model_channels": self.model_channels,
                "latent_channels": self.latent_channels,
                "num_blocks": self.num_blocks,
                "num_heads": self.num_heads,
                "mlp_channels": self.mlp_channels,
                "num_head_channels": self.num_head_channels,
                "attn_mode": self.attn_mode,
                "window_size": self.window_size,
                "pe_mode": self.pe_mode,
                "use_checkpoint": self.use_checkpoint,
                "qk_rms_norm": self.qk_rms_norm,
                "representation_config": self.rep_config,
            },
        }
        with open(f"{path}/decoder_gs.json", "w", encoding="utf-8") as f:
            json.dump(cfg_json, f)
        save_file(self.state_dict(), f"{path}/decoder_gs.safetensors")


class SLatGaussianDecoderTrain(SLatGaussianDecoder, BaseModule):
    def __init__(self, *args, **kwargs):
        self.loss_coefficients = kwargs.pop(
            "loss_coefficients",
            {
                "loss_rgb_l1": 1.0,
                "loss_lpips": 0.2,
                "loss_ssim": 0.2,
                "loss_reg": 0.0,
            },
        )
        super().__init__(*args, **kwargs)

        self.rendering_options = {
            "resolution": 512,
            "near": 0.8,
            "far": 1.6,
            "ssaa": 1,
            "bg_color": (0.0, 0.0, 0.0),
        }
        gs_renderer = GaussianRenderer(rendering_options=self.rendering_options)
        gs_renderer.pipe.kernel_size = 0.1
        gs_renderer.pipe.use_mip_gaussian = True
        self.renderer = gs_renderer

        self.lpips_loss = LearnedPerceptualImagePatchSimilarity(net_type="vgg", normalize=True)
        for param in self.lpips_loss.parameters():
            param.requires_grad = False
        self.ssim_loss = MultiScaleStructuralSimilarityIndexMeasure(data_range=1.0)
        self.get_psnr = PeakSignalNoiseRatio(data_range=1.0)

    # def state_dict(self):
    #     # remove the params that don't need gradient
    #     return {k: v for k, v in super().state_dict().items() if v.requires_grad}

    def get_loss(self, images, intrinsics, w2c, coords, feats, trainer, **kwargs):
        del kwargs
        device = trainer.accelerator.device
        images = images.to(device)
        intrinsics = intrinsics.to(device)
        w2c = w2c.to(device)
        if coords.shape[1] == 5:
            coords = coords[:, [0, 2, 3, 4]]
        coords = coords.to(device)
        feats = feats.to(device)

        input_x = SparseTensor(coords=coords, feats=feats)
        gaussian = SLatGaussianDecoder.forward(self, input_x.to(trainer.config.weight_dtype))

        all_scale, all_alpha = [], []
        for gs in gaussian:
            all_scale.append(gs.get_scaling)
            all_alpha.append(gs.get_opacity)
        loss_reg = torch.cat(all_scale).mean() + (1 - torch.cat(all_alpha).mean())

        pred_images = []
        batch_size, num_camera = images.shape[:2]
        for i in range(batch_size):
            pred_images_batch = []
            for j in range(num_camera):
                rendering = self.renderer.render(
                    gausssian=gaussian[i],
                    intrinsics=intrinsics[i, j],
                    extrinsics=w2c[i, j],
                )
                pred_images_batch.append(rendering["color"])
            pred_images.append(torch.stack(pred_images_batch, dim=0))
        pred_images = torch.stack(pred_images, dim=0).flatten(0, 1)
        images = images.permute(0, 1, 4, 2, 3).flatten(0, 1)

        loss_rgb_l1 = torch.mean(torch.abs(pred_images - images))
        loss_lpips = self.lpips_loss(
            pred_images.clamp(0, 1).to(trainer.config.weight_dtype),
            images.to(trainer.config.weight_dtype),
        )
        # SSIM loss requires the input to be interpolated
        new_size = int(pred_images.shape[-1] // 8 * 8)
        pred_images = torch.nn.functional.interpolate(
            pred_images, size=(new_size, new_size), mode="bilinear", align_corners=False
        )
        images = torch.nn.functional.interpolate(
            images, size=(new_size, new_size), mode="bilinear", align_corners=False
        )
        loss_ssim = 1 - self.ssim_loss(pred_images, images)
        # just average them all, since the sparse tensor is not evenly distributed

        loss = (
            self.loss_coefficients["loss_rgb_l1"] * loss_rgb_l1
            + self.loss_coefficients["loss_lpips"] * loss_lpips
            + self.loss_coefficients["loss_ssim"] * loss_ssim
            + self.loss_coefficients["loss_reg"] * loss_reg
        )

        trainer.accelerator.log(
            {
                "train/loss_rgb_l1(no_weight)": loss_rgb_l1,
                "train/loss_lpips(no_weight)": loss_lpips,
                "train/loss_ssim(no_weight)": loss_ssim,
                "train/loss_reg(no_weight)": loss_reg,
            },
            step=trainer.global_step,
        )
        return loss, None

    def inference(self, intrinsics, w2c, coords, feats, weight_dtype):
        """Inference the model on the given data."""
        input_x = SparseTensor(coords=coords, feats=feats)
        gaussian = SLatGaussianDecoder.forward(self, input_x.to(weight_dtype))

        pred_images = []
        batch_size, num_camera = w2c.shape[:2]
        for i in range(batch_size):
            pred_images_batch = []
            for j in range(num_camera):
                rendering = self.renderer.render(
                    gausssian=gaussian[i],
                    intrinsics=intrinsics[i, j],
                    extrinsics=w2c[i, j],
                )
                pred_images_batch.append(rendering["color"])
            pred_images.append(torch.stack(pred_images_batch, dim=0))
        pred_images = torch.stack(pred_images, dim=0)
        return pred_images

    def get_psnr_lpips_ssim(self, pred, gt, weight_dtype):
        """Calculate PSNR, LPIPS, and SSIM between the prediction and the ground truth."""
        psnr = self.get_psnr(pred, gt)
        # lpips = self.lpips_loss(
        #     pred.clamp(0, 1).to(weight_dtype),
        #     gt.to(weight_dtype),
        # )
        # TODO: there is a bug of LPIPS
        # RuntimeError: CUDA error: an illegal memory access was encountered
        # CUDA kernel errors might be asynchronously reported at some other API call
        # , so the stacktrace below might be incorrect.
        lpips = torch.tensor(0, device=gt.device, dtype=weight_dtype)
        new_size = int(pred.shape[-1] // 8 * 8)
        pred = torch.nn.functional.interpolate(
            pred, size=(new_size, new_size), mode="bilinear", align_corners=False
        )
        gt = torch.nn.functional.interpolate(
            gt, size=(new_size, new_size), mode="bilinear", align_corners=False
        )
        ssim = self.ssim_loss(pred, gt)
        return psnr, lpips, ssim

    @torch.no_grad()
    def evaluate(self, images, intrinsics, w2c, coords, feats, trainer):
        """Evaluate the model on the given data,
        model.evaluate is designed to take the same params as training
        """
        device = trainer.accelerator.device
        images = images.to(device)
        intrinsics = intrinsics.to(device)
        w2c = w2c.to(device)
        if coords.shape[1] == 5:
            coords = coords[:, [0, 2, 3, 4]]
        coords = coords.to(device)
        feats = feats.to(device)

        pred_images = self.inference(intrinsics, w2c, coords, feats, trainer.config.weight_dtype)
        pred_images = pred_images.flatten(0, 1)
        images = images.permute(0, 1, 4, 2, 3).flatten(0, 1)

        psnr, lpips, ssim = self.get_psnr_lpips_ssim(
            pred_images, images, trainer.config.weight_dtype
        )
        return {"psnr": psnr, "lpips": lpips, "ssim": ssim}

    @torch.no_grad()
    def validation(self, images, intrinsics, w2c, coords, feats, trainer) -> torch.Tensor:
        """Validate on data.
        This method is for logging some visualization results.
        We separate the evaluation and validation since eval may take a larger dataset.
        """
        device = trainer.accelerator.device
        images = images.to(device)
        intrinsics = intrinsics.to(device)
        w2c = w2c.to(device)
        if coords.shape[1] == 5:
            coords = coords[:, [0, 2, 3, 4]]
        coords = coords.to(device)
        feats = feats.to(device)

        pred_images = self.inference(intrinsics, w2c, coords, feats, trainer.config.weight_dtype)
        pred_images = pred_images.flatten(0, 1)
        images = images.permute(0, 1, 4, 2, 3).flatten(0, 1)

        psnr, lpips, ssim = self.get_psnr_lpips_ssim(
            pred_images, images, trainer.config.weight_dtype
        )
        img_gt = images[:4].split(1, dim=0)
        img_pred = pred_images[:4].split(1, dim=0)
        img_gt = torch.cat(img_gt, dim=2)
        img_pred = torch.cat(img_pred, dim=2)
        img = torch.cat([img_gt, img_pred], dim=3)
        return {
            "image": img,
            "psnr": psnr[None],
            "lpips": lpips[None],
            "ssim": ssim[None],
        }

    @staticmethod
    def log_validation(val_out_list, global_step, trainer):
        """Log validation output."""
        val_out = {k: torch.cat([v[k] for v in val_out_list], dim=0) for k in val_out_list[0]}
        val_image = val_out["image"]
        val_image = trainer.accelerator.gather(val_image)
        val_image = val_image[: trainer.config.validation_log_examples]
        if trainer.accelerator.is_main_process:
            for tracker in trainer.accelerator.trackers:
                if tracker.name == "tensorboard":
                    tracker.writer.add_images(
                        "val/images",
                        val_image.cpu(),
                        global_step,
                        dataformats="NCHW",
                    )
                    tracker.writer.add_scalar(
                        "val/psnr",
                        val_out["psnr"].mean().item(),
                        global_step,
                    )
                    tracker.writer.add_scalar(
                        "val/lpips",
                        val_out["lpips"].mean().item(),
                        global_step,
                    )
                    tracker.writer.add_scalar(
                        "val/ssim",
                        val_out["ssim"].mean().item(),
                        global_step,
                    )
                else:
                    trainer.logger.warning("image logging not implemented for %s", tracker.name)

    def forward(self, *args, **kwargs):
        return BaseModule.forward(self, *args, **kwargs)

    def forward_decoder(self, *args, **kwargs):
        return SLatGaussianDecoder.forward(self, *args, **kwargs)
