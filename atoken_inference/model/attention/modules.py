#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2025 Apple Inc. All Rights Reserved.
#
# ruff: noqa
# pylint: skip-file
from typing import *
import einops

import torch
import torch.nn as nn
import torch.nn.functional as F

from .. import SparseTensor
from .full_attn import sparse_scaled_dot_product_attention
from .serialized_attn import (
    SerializeMode,
    sparse_serialized_scaled_dot_product_self_attention,
)
from .windowed_attn import sparse_windowed_scaled_dot_product_self_attention


class RotaryPositionEmbedder(nn.Module):
    def __init__(self, head_dim: int, pos_cls_token: int = 0):
        super().__init__()
        self.head_dim = head_dim
        self.dim_rope = head_dim // 8

        # Precompute frequency bases instead of full tensor
        self.freqs = 1.0 / (10000.0 ** (torch.arange(0, self.dim_rope) / head_dim))

    def compute_freqs_cis(self, positions):
        """Compute RoPE frequencies for given positions on-the-fly"""
        # positions: [..., 4] tensor containing t,h,w,z positions

        # Calculate frequencies for each dimension
        t_freq = torch.outer(positions[..., 0].float(), self.freqs)
        h_freq = torch.outer(positions[..., 1].float(), self.freqs)
        w_freq = torch.outer(positions[..., 2].float(), self.freqs)
        z_freq = torch.outer(positions[..., 3].float(), self.freqs)

        # Convert to complex numbers
        freqs_cis_t = torch.polar(torch.ones_like(t_freq), t_freq)
        freqs_cis_h = torch.polar(torch.ones_like(h_freq), h_freq)
        freqs_cis_w = torch.polar(torch.ones_like(w_freq), w_freq)
        freqs_cis_z = torch.polar(torch.ones_like(z_freq), z_freq)

        # Concatenate all dimensions
        freqs_cis = torch.cat([freqs_cis_t, freqs_cis_h, freqs_cis_w, freqs_cis_z], dim=-1)

        return freqs_cis

    @staticmethod
    def apply_rotary_emb(
        xq: torch.Tensor,
        xk: torch.Tensor,
        freqs_cis: torch.Tensor,
        xk_freqs_cis: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        reshape_xq = xq.to(torch.float32).reshape(*xq.shape[:-1], -1, 2)
        reshape_xk = xk.to(torch.float32).reshape(*xk.shape[:-1], -1, 2)

        xq_ = torch.complex(reshape_xq[..., 0], reshape_xq[..., 1])
        xk_ = torch.complex(reshape_xk[..., 0], reshape_xk[..., 1])

        # add head dim
        freqs_cis = freqs_cis.unsqueeze(-2)
        xk_freqs_cis = xk_freqs_cis.unsqueeze(-2)

        xq_out = xq_ * freqs_cis
        xq_out = torch.stack((xq_out.real, xq_out.imag), dim=-1).reshape(*xq_out.shape[:-1], -1)

        xk_out = xk_ * xk_freqs_cis
        xk_out = torch.stack((xk_out.real, xk_out.imag), dim=-1).reshape(*xk_out.shape[:-1], -1)

        return xq_out.to(xq.dtype), xk_out.to(xk.dtype)

    def forward(self, q, k, indices, k_indices=None):
        # Compute frequencies on-the-fly
        self.freqs = self.freqs.to(q.device)

        q_freqs = self.compute_freqs_cis(indices)

        if k_indices is None:
            k_freqs = q_freqs
        else:
            k_freqs = self.compute_freqs_cis(k_indices)

        q_embed, k_embed = self.apply_rotary_emb(q, k, freqs_cis=q_freqs, xk_freqs_cis=k_freqs)

        return q_embed, k_embed


class SparseMultiHeadRMSNorm(nn.Module):
    def __init__(self, dim: int, heads: int):
        super().__init__()
        self.scale = dim**0.5
        self.gamma = nn.Parameter(torch.ones(heads, dim))

    def forward(self, x: Union[SparseTensor, torch.Tensor]) -> Union[SparseTensor, torch.Tensor]:
        x_type = x.dtype
        x = x.float()
        if isinstance(x, SparseTensor):
            x = x.replace(F.normalize(x.feats, dim=-1))
        else:
            x = F.normalize(x, dim=-1)
        return (x * self.gamma * self.scale).to(x_type)


class SparseMultiHeadAttention(nn.Module):
    def __init__(
        self,
        channels: int,
        num_heads: int,
        ctx_channels: Optional[int] = None,
        type: Literal["self", "cross"] = "self",
        attn_mode: Literal["full", "serialized", "windowed"] = "full",
        window_size: Optional[int] = None,
        shift_sequence: Optional[int] = None,
        shift_window: Optional[Tuple[int, int, int]] = None,
        serialize_mode: Optional[SerializeMode] = None,
        use_bias: bool = False,
        use_rope: bool = False,
        qk_rms_norm: bool = False,
        out_channels: Optional[int] = None,
    ):
        super().__init__()

        if out_channels is None:
            out_channels = channels
        assert channels % num_heads == 0
        assert out_channels % num_heads == 0
        assert type in ["self", "cross"], f"Invalid attention type: {type}"
        assert attn_mode in [
            "full",
            "serialized",
            "windowed",
        ], f"Invalid attention mode: {attn_mode}"
        assert type == "self" or attn_mode == "full", "Cross-attention only supports full attention"
        self.channels = channels
        self.ctx_channels = ctx_channels if ctx_channels is not None else channels
        self.num_heads = num_heads
        self._type = type
        self.attn_mode = attn_mode
        self.window_size = window_size
        self.shift_sequence = shift_sequence
        self.shift_window = shift_window
        self.serialize_mode = serialize_mode
        self.use_rope = use_rope
        self.qk_rms_norm = qk_rms_norm

        head_dim = channels // num_heads
        if self._type == "self":
            self.to_qkv = nn.Linear(channels, channels * 3, bias=use_bias)
        else:
            self.to_q = nn.Linear(channels, channels, bias=use_bias)
            self.to_kv = nn.Linear(self.ctx_channels, channels * 2, bias=use_bias)

        if self.qk_rms_norm:
            self.q_rms_norm = SparseMultiHeadRMSNorm(head_dim, num_heads)
            self.k_rms_norm = SparseMultiHeadRMSNorm(head_dim, num_heads)

        self.to_out = nn.Linear(channels, out_channels, bias=use_bias)

        if use_rope:
            self.rope = RotaryPositionEmbedder(head_dim)

    @staticmethod
    def _linear(
        module: nn.Linear, x: Union[SparseTensor, torch.Tensor]
    ) -> Union[SparseTensor, torch.Tensor]:
        if isinstance(x, SparseTensor):
            return x.replace(module(x.feats))
        else:
            return module(x)

    @staticmethod
    def _reshape_chs(
        x: Union[SparseTensor, torch.Tensor], shape: Tuple[int, ...]
    ) -> Union[SparseTensor, torch.Tensor]:
        if isinstance(x, SparseTensor):
            return x.reshape(*shape)
        else:
            return x.reshape(*x.shape[:2], *shape)

    def _fused_pre(
        self, x: Union[SparseTensor, torch.Tensor], num_fused: int
    ) -> Union[SparseTensor, torch.Tensor]:
        if isinstance(x, SparseTensor):
            x_feats = x.feats.unsqueeze(0)
        else:
            x_feats = x
        x_feats = x_feats.reshape(*x_feats.shape[:2], num_fused, self.num_heads, -1)
        return x.replace(x_feats.squeeze(0)) if isinstance(x, SparseTensor) else x_feats

    def _qkv_rope(self, qkv: SparseTensor) -> SparseTensor:
        q, k, v = qkv.feats.unbind(dim=1)
        q, k = self.rope(q, k, qkv.coords[:, 1:])  # [T, H, C]
        qkv = qkv.replace(torch.stack([q, k, v], dim=1))
        return qkv

    def _q_kv_rope(self, q: SparseTensor, k: SparseTensor) -> SparseTensor:
        q_feats, k_feats = self.rope(
            q.feats, k.feats, q.coords[:, 1:], k.coords[:, 1:]
        )  # [T, H, C]
        return q.replace(q_feats), k.replace(k_feats)

    def _prepare_kv_cache(
        self,
        k: SparseTensor,
        v: SparseTensor,
        cached_k: Optional[SparseTensor] = None,
        cached_v: Optional[SparseTensor] = None,
    ) -> Tuple[SparseTensor, SparseTensor]:
        """
        Prepare K and V tensors by concatenating with cached values.

        Args:
            k: Current key tensor
            v: Current value tensor
            cached_k: Cached key tensor from previous timesteps
            cached_v: Cached value tensor from previous timesteps

        Returns:
            Tuple of (combined_k, combined_v)
        """
        if cached_k is not None and cached_v is not None:
            # Concatenate cached KV with current KV
            # Note: sparse_cat concatenates along batch dimension (dim=0) by default
            # but we want to concatenate along the sequence dimension

            combined_k = self._concat_temporal_sparse(cached_k, k)
            combined_v = self._concat_temporal_sparse(cached_v, v)
        else:
            combined_k = k
            combined_v = v

        return combined_k, combined_v

    def _concat_temporal_sparse(self, cached: SparseTensor, current: SparseTensor) -> SparseTensor:
        """
        Concatenate cached and current sparse tensors along temporal dimension.
        Injects cached data at the beginning of each batch to maintain proper temporal ordering.
        """
        return current.concat_temporal_at_batch_start(cached)

    def _update_kv_cache(
        self,
        k: SparseTensor,
        v: SparseTensor,
        kv_cache: Optional[Dict[str, SparseTensor]] = None,
        kv_cache_size: Optional[int] = None,
    ) -> Dict[str, SparseTensor]:
        """
        Update the KV cache with the last k timesteps from current K and V.

        Args:
            k: Current key tensor
            v: Current value tensor
            kv_cache: Existing cache dictionary

        Returns:
            Updated cache dictionary
        """
        if kv_cache is None:
            kv_cache = {}

        if kv_cache_size is not None and kv_cache_size > 0:
            # Get last k timesteps without gradients for caching
            cached_k = k.get_last_k_timesteps(kv_cache_size).detach()
            cached_v = v.get_last_k_timesteps(kv_cache_size).detach()

            # Store in cache
            kv_cache["cached_k"] = cached_k
            kv_cache["cached_v"] = cached_v
        else:
            # Clear cache if temporal size is 0 or None
            kv_cache.pop("cached_k", None)
            kv_cache.pop("cached_v", None)

        return kv_cache

    def forward(
        self,
        x: Union[SparseTensor, torch.Tensor],
        context: Optional[Union[SparseTensor, torch.Tensor]] = None,
        kv_cache: Optional[Dict[str, SparseTensor]] = None,
        kv_cache_size: Optional[int] = None,
    ) -> Union[SparseTensor, torch.Tensor]:
        if kv_cache is None:
            kv_cache = {}

        if self._type == "self":
            qkv = self._linear(self.to_qkv, x)
            qkv = self._fused_pre(qkv, num_fused=3)

            q, k, v = qkv.unbind(dim=1)
            if self.qk_rms_norm:
                q = self.q_rms_norm(q)
                k = self.k_rms_norm(k)

            # Handle KV cache AFTER RMS norm, BEFORE RoPE
            if isinstance(k, SparseTensor) and kv_cache_size is not None:
                cached_k = kv_cache.get("cached_k", None)
                cached_v = kv_cache.get("cached_v", None)

                # Prepare KV cache (concatenate cached with current)
                k_with_cache, v_with_cache = self._prepare_kv_cache(k, v, cached_k, cached_v)
            else:
                k_with_cache, v_with_cache = k, v

            # Update cache with current K, V, this should happen before rope to store kv without position information.
            kv_cache = self._update_kv_cache(k_with_cache, v_with_cache, kv_cache, kv_cache_size)

            if self.use_rope:
                q, k_with_cache = self._q_kv_rope(q, k_with_cache)

            # Perform attention
            if self.attn_mode == "full":
                # Use 3-argument API for cleaner code when we have separate q, k, v
                h = sparse_scaled_dot_product_attention(q, k_with_cache, v_with_cache)
            elif self.attn_mode == "serialized":
                # For serialized attention, reconstruct the full qkv tensor (required by this function)
                qkv_with_cache = q.replace(
                    torch.stack([q.feats, k_with_cache.feats, v_with_cache.feats], dim=1)
                )
                h = sparse_serialized_scaled_dot_product_self_attention(
                    qkv_with_cache,
                    self.window_size,
                    serialize_mode=self.serialize_mode,
                    shift_sequence=self.shift_sequence,
                    shift_window=self.shift_window,
                )
            elif self.attn_mode == "windowed":
                # For windowed attention, reconstruct the full qkv tensor (required by this function)
                qkv_with_cache = q.replace(
                    torch.stack([q.feats, k_with_cache.feats, v_with_cache.feats], dim=1)
                )
                h = sparse_windowed_scaled_dot_product_self_attention(
                    qkv_with_cache, self.window_size, shift_window=self.shift_window
                )

        else:
            q = self._linear(self.to_q, x)
            q = self._reshape_chs(q, (self.num_heads, -1))
            kv = self._linear(self.to_kv, context)
            kv = self._fused_pre(kv, num_fused=2)

            if self.qk_rms_norm:
                q = self.q_rms_norm(q)
                k, v = kv.unbind(dim=1)
                k = self.k_rms_norm(k)
                kv = kv.replace(torch.stack([k.feats, v.feats], dim=1))

            if self.use_rope:
                k, v = kv.unbind(dim=1)
                q, k = self._q_kv_rope(q, k)
                kv = kv.replace(torch.stack([k.feats, v.feats], dim=1))

            h = sparse_scaled_dot_product_attention(q, kv)
        h = self._reshape_chs(h, (-1,))
        h = self._linear(self.to_out, h)
        return h, kv_cache
