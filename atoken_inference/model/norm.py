#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2025 Apple Inc. All Rights Reserved.
#
# ruff: noqa
# pylint: skip-file
import torch
import torch.nn as nn
from . import SparseTensor
from . import DEBUG

__all__ = [
    "SparseGroupNorm",
    "SparseLayerNorm",
    "SparseGroupNorm32",
    "SparseLayerNorm32",
    "SparseRMSNorm32",
]


class SparseGroupNorm(nn.GroupNorm):
    def __init__(self, num_groups, num_channels, eps=1e-5, affine=True):
        super(SparseGroupNorm, self).__init__(num_groups, num_channels, eps, affine)

    def forward(self, input: SparseTensor) -> SparseTensor:
        nfeats = torch.zeros_like(input.feats)
        for k in range(input.shape[0]):
            if DEBUG:
                assert (
                    input.coords[input.layout[k], 0] == k
                ).all(), "SparseGroupNorm: batch index mismatch"
            bfeats = input.feats[input.layout[k]]
            bfeats = bfeats.permute(1, 0).reshape(1, input.shape[1], -1)
            # bfeats = super().forward(bfeats)
            bfeats = nn.functional.group_norm(
                bfeats,
                self.num_groups,
                self.weight.to(bfeats.dtype) if self.weight is not None else None,
                self.bias.to(bfeats.dtype) if self.bias is not None else None,
                self.eps,
            )
            bfeats = bfeats.reshape(input.shape[1], -1).permute(1, 0)
            nfeats[input.layout[k]] = bfeats
        return input.replace(nfeats)


class SparseLayerNorm(nn.LayerNorm):
    def __init__(self, normalized_shape, eps=1e-5, elementwise_affine=True):
        super(SparseLayerNorm, self).__init__(normalized_shape, eps, elementwise_affine)

    def forward(self, input: SparseTensor) -> SparseTensor:
        nfeats = torch.zeros_like(input.feats)
        for k in range(input.shape[0]):
            bfeats = input.feats[input.layout[k]]
            bfeats = bfeats.permute(1, 0).reshape(1, input.shape[1], -1)
            # bfeats = super().forward(bfeats)
            bfeats = nn.functional.layer_norm(
                bfeats,
                self.normalized_shape,
                self.weight.to(bfeats.dtype) if self.weight is not None else None,
                self.bias.to(bfeats.dtype) if self.bias is not None else None,
                self.eps,
            )
            bfeats = bfeats.reshape(input.shape[1], -1).permute(1, 0)
            nfeats[input.layout[k]] = bfeats
        return input.replace(nfeats)


class SparseGroupNorm32(SparseGroupNorm):
    """
    A GroupNorm layer that converts to float32 before the forward pass.
    """

    def forward(self, x: SparseTensor) -> SparseTensor:
        return super().forward(x.float()).type(x.dtype)


class SparseLayerNorm32(SparseLayerNorm):
    """
    A LayerNorm layer that converts to float32 before the forward pass.
    """

    def forward(self, x: SparseTensor) -> SparseTensor:
        return super().forward(x.float()).type(x.dtype)


class LayerNorm32(nn.LayerNorm):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # return super().forward(x.float()).type(x.dtype)
        output = nn.functional.layer_norm(
            x.float(),
            self.normalized_shape,
            self.weight.float() if self.weight is not None else None,
            self.bias.float() if self.bias is not None else None,
            self.eps,
        )
        return output.to(x.dtype)


class GroupNorm32(nn.GroupNorm):
    """
    A GroupNorm layer that converts to float32 before the forward pass.
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return super().forward(x.float()).type(x.dtype)


class ChannelLayerNorm32(LayerNorm32):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        DIM = x.dim()
        x = x.permute(0, *range(2, DIM), 1).contiguous()
        x = super().forward(x)
        x = x.permute(0, DIM - 1, *range(1, DIM - 1)).contiguous()
        return x


class RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """ """
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(hidden_size))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight

    def extra_repr(self):
        return f"{tuple(self.weight.shape)}, eps={self.eps}"


class RMSNorm32(RMSNorm):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = self._norm(x.float()).type_as(x)
        output = output * self.weight.float()
        return output.to(x.dtype)


class SparseRMSNorm32(RMSNorm):
    def forward(self, input: SparseTensor) -> SparseTensor:
        nfeats = torch.zeros_like(input.feats)
        for k in range(input.shape[0]):
            bfeats = input.feats[input.layout[k]][None, :, :]
            bfeats = self._norm(bfeats.float())
            bfeats = bfeats * self.weight.float()
            nfeats[input.layout[k]] = bfeats.to(nfeats.dtype)

        return input.replace(nfeats)
