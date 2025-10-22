#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2025 Apple Inc. All Rights Reserved.
#
# ruff: noqa
# pylint: skip-file

"""
Finite Scalar Quantization: VQ-VAE Made Simple - https://arxiv.org/abs/2309.15505
Code adapted from Jax version in Appendix A.1
"""

from typing import List, Tuple, Optional
import math

import torch
import torch.nn as nn
from torch.nn import Module
from torch import Tensor, int32
from torch.amp import autocast

from einops import rearrange, pack, unpack


def levels_from_codebook_size(codebook_size: int):
    """Levels for FSQ for each codebook size (suggested by the authors)"""
    if codebook_size == 2**4:  # 16
        levels = [5, 3]
    elif codebook_size == 2**6:  # 64
        levels = [8, 8]
    elif codebook_size == 2**8:  # 256
        levels = [8, 6, 5]
    elif codebook_size == 2**9:  # 512
        levels = [8, 8, 8]
    elif codebook_size == 2**10:  # 1024
        levels = [8, 5, 5, 5]
    elif codebook_size == 2**11:  # 2048
        levels = [8, 8, 6, 5]
    elif codebook_size == 2**12:  # 4096
        levels = [4, 4, 4, 4, 4, 4]
    elif codebook_size == 2**13:  # 8192
        levels = [8, 8, 8, 4, 4]
    elif codebook_size == 2**14:  # 16384 'c5-b14',
        levels = [8, 8, 8, 6, 5]
    elif codebook_size == 2**16:  # 65536 'c6-b16',
        levels = [8, 8, 8, 5, 5, 5]
    elif codebook_size == (2**18 - 2):  # 262142 'c4-b18',
        levels = [23, 23, 23, 23]  # [23, 23, 22, 22]
    elif codebook_size == (2**18 - 1):  # 262143 'c5-b18',
        levels = [13, 12, 12, 12, 12]
    elif codebook_size == 2**18:  # 262144 'c7-b18',
        levels = [8, 8, 6, 5, 5, 5, 5]
    elif codebook_size == (2**18 + 1):  # 262145 'c6-b18'
        levels = [8, 8, 8, 8, 8, 8]
    elif codebook_size == (5**8 - 1):  # 390624 'c4-b18'
        levels = [25, 25, 25, 25]
    elif codebook_size == (5**8):  # 390625 'c8-b18'
        levels = [5, 5, 5, 5, 5, 5, 5, 5]
    elif codebook_size < 0:  # temporarily use negative value for log2(codebook_size) like LFQ
        channel_size = int(math.log2(-codebook_size))
        levels = [2] * channel_size
    elif codebook_size == 0:  # continuous c16
        channel_size = 16
        levels = [0] * channel_size  # placeholder for legacy
    else:
        raise NotImplementedError

    updated_codebook_size = 1
    for level in levels:
        updated_codebook_size *= level

    return levels, updated_codebook_size


# helper functions


def exists(v):
    return v is not None


def default(*args):
    for arg in args:
        if exists(arg):
            return arg
    return None


def pack_one(t, pattern):
    return pack([t], pattern)


def unpack_one(t, ps, pattern):
    return unpack(t, ps, pattern)[0]


# tensor helpers


def round_ste(z: Tensor) -> Tensor:
    """Round with straight through gradients."""
    zhat = z.round()
    return z + (zhat - z).detach()


# main class


class FSQ(Module):
    def __init__(
        self,
        levels: List[int],
        dim: Optional[int] = None,
        num_codebooks=1,
        keep_num_codebooks_dim: Optional[bool] = None,
        scale: Optional[float] = None,
        allowed_dtypes: Tuple[torch.dtype, ...] = (torch.float32, torch.float64),
    ):
        super().__init__()
        _levels = torch.tensor(levels, dtype=int32)
        self.register_buffer("_levels", _levels, persistent=False)

        _basis = torch.cumprod(torch.tensor([1] + levels[:-1]), dim=0, dtype=int32)
        self.register_buffer("_basis", _basis, persistent=False)

        self.scale = scale

        codebook_dim = len(levels)
        self.codebook_dim = codebook_dim

        effective_codebook_dim = codebook_dim * num_codebooks
        self.num_codebooks = num_codebooks
        self.effective_codebook_dim = effective_codebook_dim

        keep_num_codebooks_dim = default(keep_num_codebooks_dim, num_codebooks > 1)
        assert not (num_codebooks > 1 and not keep_num_codebooks_dim)
        self.keep_num_codebooks_dim = keep_num_codebooks_dim

        self.dim = default(dim, len(_levels) * num_codebooks)

        has_projections = self.dim != effective_codebook_dim

        self.project_in = (
            nn.Linear(self.dim, effective_codebook_dim) if has_projections else nn.Identity()
        )
        self.project_out = (
            nn.Linear(effective_codebook_dim, self.dim) if has_projections else nn.Identity()
        )
        self.has_projections = has_projections

        self.codebook_size = self._levels.prod().item()

        implicit_codebook = self.indices_to_codes(
            torch.arange(self.codebook_size), project_out=False
        )
        self.register_buffer("implicit_codebook", implicit_codebook, persistent=False)

        self.allowed_dtypes = allowed_dtypes

    def bound(self, z: Tensor, eps: float = 1e-3) -> Tensor:
        """Bound `z`, an array of shape (..., d)."""
        half_l = (self._levels - 1) * (1 + eps) / 2
        offset = torch.where(self._levels % 2 == 0, 0.5, 0.0)
        shift = (offset / half_l).atanh()
        return (z + shift).tanh() * half_l - offset

    def quantize(self, z: Tensor) -> Tensor:
        """Quantizes z, returns quantized zhat, same shape as z."""
        quantized = round_ste(self.bound(z))
        half_width = self._levels // 2  # Renormalize to [-1, 1].
        return quantized / half_width

    def _scale_and_shift(self, zhat_normalized: Tensor) -> Tensor:
        half_width = self._levels // 2
        return (zhat_normalized * half_width) + half_width

    def _scale_and_shift_inverse(self, zhat: Tensor) -> Tensor:
        half_width = self._levels // 2
        return (zhat - half_width) / half_width

    def codes_to_indices(self, zhat: Tensor) -> Tensor:
        """Converts a `code` to an index in the codebook."""
        assert zhat.shape[-1] == self.codebook_dim
        zhat = self._scale_and_shift(zhat)
        return (zhat * self._basis).sum(dim=-1).to(int32)

    def indices_to_codes(self, indices: Tensor, project_out=True) -> Tensor:
        """Inverse of `codes_to_indices`."""

        is_img_or_video = indices.ndim >= (3 + int(self.keep_num_codebooks_dim))

        indices = rearrange(indices, "... -> ... 1")
        codes_non_centered = (indices // self._basis) % self._levels
        codes = self._scale_and_shift_inverse(codes_non_centered)

        if self.keep_num_codebooks_dim:
            codes = rearrange(codes, "... c d -> ... (c d)")

        if project_out:
            codes = self.project_out(codes)

        if is_img_or_video:
            codes = rearrange(codes, "b ... d -> b d ...")

        return codes

    @autocast("cuda", enabled=False)
    def forward(self, z: Tensor) -> Tensor:
        """
        einstein notation
        b - batch
        n - sequence (or flattened spatial dimensions)
        d - feature dimension
        c - number of codebook dim
        """

        orig_dtype = z.dtype

        # make sure allowed dtype
        if z.dtype not in self.allowed_dtypes:
            z = z.float()

        assert (
            z.shape[-1] == self.dim
        ), f"expected dimension of {self.dim} but found dimension of {z.shape[-1]}"

        z = self.project_in.to(z.dtype)(z)

        z = rearrange(z, "b (c d) -> b c d", c=self.num_codebooks)

        codes = self.quantize(z)
        indices = self.codes_to_indices(codes)

        codes = rearrange(codes, "b c d -> b (c d)")

        out = self.project_out.to(z.dtype)(codes)

        # reconstitute image or video dimensions

        if not self.keep_num_codebooks_dim:
            indices = rearrange(indices, "... 1 -> ...")

        # cast back to original dtype

        if out.dtype != orig_dtype:
            out = out.type(orig_dtype)

        # placeholder for compatibility

        commit_loss = torch.tensor(0.0, device=z.device)

        # return quantized output and indices

        return out, indices, commit_loss
