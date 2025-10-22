#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2025 Apple Inc. All Rights Reserved.
#
# ruff: noqa
# pylint: skip-file
from typing import *
import torch
from ..basic import SparseTensor
from .. import DEBUG, ATTN

if ATTN == "xformers":
    import xformers.ops as xops
    from xformers.ops.fmha.attn_bias import BlockDiagonalCausalMask
elif ATTN == "flash_attn":
    import flash_attn
else:
    raise ValueError(f"Unknown attention module: {ATTN}")


__all__ = [
    "sparse_scaled_dot_product_attention",
]


@overload
def sparse_scaled_dot_product_attention(
    q: torch.Tensor, k: SparseTensor, v: SparseTensor, temporal_causal: bool = False
) -> torch.Tensor:
    """
    Apply scaled dot product attention to a sparse tensor.

    Args:
        q (torch.Tensor): A [N, L, H, Ci] dense tensor containing Qs.
        k (SparseTensor): A [N, *, H, Ci] sparse tensor containing Ks.
        v (SparseTensor): A [N, *, H, Co] sparse tensor containing Vs.
        temporal_causal (bool): If True, apply temporal causality where coordinate
                               dimension 1 (time) is causal, other dims bidirectional.
    """
    ...


def _create_temporal_causal_mask_xformers(coords, device):
    """
    Create BlockDiagonalCausalMask for temporal causality + spatial bidirectionality

    Args:
        coords: SparseTensor.coords with shape [N, coord_dims] where coord_dims >= 2
                coords[:, 0] = batch_idx, coords[:, 1] = time_idx
    """
    # Group coordinates by (batch, timestep)
    batch_time_groups = {}
    for i, coord in enumerate(coords):
        batch_idx = coord[0].item()
        time_idx = coord[1].item()
        key = (batch_idx, time_idx)
        if key not in batch_time_groups:
            batch_time_groups[key] = []
        batch_time_groups[key].append(i)

    # Create sequence lengths for BlockDiagonalCausalMask
    # Each (batch, timestep) becomes a block
    seqlens = []
    reorder_indices = []

    # Process in sorted order to ensure proper temporal sequencing
    for batch_idx, time_idx in sorted(batch_time_groups.keys()):
        indices = batch_time_groups[(batch_idx, time_idx)]
        seqlens.append(len(indices))
        reorder_indices.extend(indices)

    # Create mask - this handles causal constraints between blocks
    # and full attention within blocks
    mask = BlockDiagonalCausalMask.from_seqlens(q_seqlen=seqlens, kv_seqlen=seqlens, device=device)

    return mask, torch.tensor(reorder_indices, device=device, dtype=torch.long)


def _create_temporal_causal_seqlens_flash(coords):
    """
    Create cumulative sequence lengths for flash_attn with temporal causality

    The key insight is to arrange sequences so that within each batch,
    timesteps are consecutive, enabling causal=True to work correctly.
    """
    # Group by batch first
    batch_groups = {}
    for i, coord in enumerate(coords):
        batch_idx = coord[0].item()
        time_idx = coord[1].item()

        if batch_idx not in batch_groups:
            batch_groups[batch_idx] = {}
        if time_idx not in batch_groups[batch_idx]:
            batch_groups[batch_idx][time_idx] = []
        batch_groups[batch_idx][time_idx].append(i)

    # Create reordering: batch0_t0, batch0_t1, ..., batch1_t0, batch1_t1, ...
    reorder_indices = []
    cu_seqlens = [0]

    for batch_idx in sorted(batch_groups.keys()):
        batch_length = 0
        for time_idx in sorted(batch_groups[batch_idx].keys()):
            indices = batch_groups[batch_idx][time_idx]
            reorder_indices.extend(indices)
            batch_length += len(indices)

        cu_seqlens.append(cu_seqlens[-1] + batch_length)

    return torch.tensor(reorder_indices, dtype=torch.long), torch.tensor(
        cu_seqlens, dtype=torch.int32
    )


def sparse_scaled_dot_product_attention(*args, temporal_causal=False, **kwargs):
    arg_names_dict = {1: ["qkv"], 2: ["q", "kv"], 3: ["q", "k", "v"]}
    num_all_args = len(args) + len(kwargs)
    assert (
        num_all_args in arg_names_dict
    ), f"Invalid number of arguments, got {num_all_args}, expected 1, 2, or 3"
    for key in arg_names_dict[num_all_args][len(args) :]:
        assert key in kwargs, f"Missing argument {key}"

    if num_all_args == 1:
        qkv = args[0] if len(args) > 0 else kwargs["qkv"]
        assert isinstance(qkv, SparseTensor), f"qkv must be a SparseTensor, got {type(qkv)}"
        assert (
            len(qkv.shape) == 4 and qkv.shape[1] == 3
        ), f"Invalid shape for qkv, got {qkv.shape}, expected [N, *, 3, H, C]"
        device = qkv.device

        s = qkv
        q_seqlen = [qkv.layout[i].stop - qkv.layout[i].start for i in range(qkv.shape[0])]
        kv_seqlen = q_seqlen
        qkv = qkv.feats  # [T, 3, H, C]

    elif num_all_args == 2:
        q = args[0] if len(args) > 0 else kwargs["q"]
        kv = args[1] if len(args) > 1 else kwargs["kv"]
        assert (
            isinstance(q, SparseTensor)
            and isinstance(kv, (SparseTensor, torch.Tensor))
            or isinstance(q, torch.Tensor)
            and isinstance(kv, SparseTensor)
        ), f"Invalid types, got {type(q)} and {type(kv)}"
        assert q.shape[0] == kv.shape[0], f"Batch size mismatch, got {q.shape[0]} and {kv.shape[0]}"
        device = q.device

        if isinstance(q, SparseTensor):
            assert len(q.shape) == 3, f"Invalid shape for q, got {q.shape}, expected [N, *, H, C]"
            s = q
            q_seqlen = [q.layout[i].stop - q.layout[i].start for i in range(q.shape[0])]
            q = q.feats  # [T_Q, H, C]
        else:
            assert len(q.shape) == 4, f"Invalid shape for q, got {q.shape}, expected [N, L, H, C]"
            s = None
            N, L, H, C = q.shape
            q_seqlen = [L] * N
            q = q.reshape(N * L, H, C)  # [T_Q, H, C]

        if isinstance(kv, SparseTensor):
            assert (
                len(kv.shape) == 4 and kv.shape[1] == 2
            ), f"Invalid shape for kv, got {kv.shape}, expected [N, *, 2, H, C]"
            kv_seqlen = [kv.layout[i].stop - kv.layout[i].start for i in range(kv.shape[0])]
            kv = kv.feats  # [T_KV, 2, H, C]
        else:
            assert (
                len(kv.shape) == 5
            ), f"Invalid shape for kv, got {kv.shape}, expected [N, L, 2, H, C]"
            N, L, _, H, C = kv.shape
            kv_seqlen = [L] * N
            kv = kv.reshape(N * L, 2, H, C)  # [T_KV, 2, H, C]

    elif num_all_args == 3:
        q = args[0] if len(args) > 0 else kwargs["q"]
        k = args[1] if len(args) > 1 else kwargs["k"]
        v = args[2] if len(args) > 2 else kwargs["v"]
        assert (
            isinstance(q, SparseTensor)
            and isinstance(k, (SparseTensor, torch.Tensor))
            and type(k) == type(v)
            or isinstance(q, torch.Tensor)
            and isinstance(k, SparseTensor)
            and isinstance(v, SparseTensor)
        ), f"Invalid types, got {type(q)}, {type(k)}, and {type(v)}"

        assert (
            q.shape[0] == k.shape[0] == v.shape[0]
        ), f"Batch size mismatch, got {q.shape[0]}, {k.shape[0]}, and {v.shape[0]}"
        device = q.device

        if isinstance(q, SparseTensor):
            assert len(q.shape) == 3, f"Invalid shape for q, got {q.shape}, expected [N, *, H, Ci]"
            s = q
            q_seqlen = [q.layout[i].stop - q.layout[i].start for i in range(q.shape[0])]
            q = q.feats  # [T_Q, H, Ci]
        else:
            assert len(q.shape) == 4, f"Invalid shape for q, got {q.shape}, expected [N, L, H, Ci]"
            s = None
            N, L, H, CI = q.shape
            q_seqlen = [L] * N
            q = q.reshape(N * L, H, CI)  # [T_Q, H, Ci]

        if isinstance(k, SparseTensor):
            assert len(k.shape) == 3, f"Invalid shape for k, got {k.shape}, expected [N, *, H, Ci]"
            assert len(v.shape) == 3, f"Invalid shape for v, got {v.shape}, expected [N, *, H, Co]"
            kv_seqlen = [k.layout[i].stop - k.layout[i].start for i in range(k.shape[0])]
            k = k.feats  # [T_KV, H, Ci]
            v = v.feats  # [T_KV, H, Co]
        else:
            assert len(k.shape) == 4, f"Invalid shape for k, got {k.shape}, expected [N, L, H, Ci]"
            assert len(v.shape) == 4, f"Invalid shape for v, got {v.shape}, expected [N, L, H, Co]"
            N, L, H, CI, CO = *k.shape, v.shape[-1]
            kv_seqlen = [L] * N
            k = k.reshape(N * L, H, CI)  # [T_KV, H, Ci]
            v = v.reshape(N * L, H, CO)  # [T_KV, H, Co]

    if DEBUG:
        if s is not None:
            for i in range(s.shape[0]):
                assert (
                    s.coords[s.layout[i]] == i
                ).all(), "SparseScaledDotProductSelfAttention: batch index mismatch"
        if num_all_args in [2, 3]:
            assert q.shape[:2] == [
                1,
                sum(q_seqlen),
            ], "SparseScaledDotProductSelfAttention: q shape mismatch"
        if num_all_args == 3:
            assert k.shape[:2] == [
                1,
                sum(kv_seqlen),
            ], "SparseScaledDotProductSelfAttention: k shape mismatch"
            assert v.shape[:2] == [
                1,
                sum(kv_seqlen),
            ], "SparseScaledDotProductSelfAttention: v shape mismatch"

    # Handle temporal causality
    if temporal_causal and s is not None:
        if ATTN == "xformers":
            # Create temporal causal mask and reordering
            mask, reorder_indices = _create_temporal_causal_mask_xformers(s.coords, device)

            # Reorder tensors according to temporal structure
            if num_all_args == 1:
                qkv = qkv[reorder_indices]
                q, k, v = qkv.unbind(dim=1)
            elif num_all_args == 2:
                q = q[reorder_indices]
                kv = kv[reorder_indices]
                k, v = kv.unbind(dim=1)
            elif num_all_args == 3:
                q = q[reorder_indices]
                k = k[reorder_indices]
                v = v[reorder_indices]

            q = q.unsqueeze(0)
            k = k.unsqueeze(0)
            v = v.unsqueeze(0)

            # Apply attention with temporal causal mask
            out = xops.memory_efficient_attention(q, k, v, mask)[0]

            # Reorder output back to original coordinate order
            reverse_indices = torch.argsort(reorder_indices)
            out = out[reverse_indices]

        elif ATTN == "flash_attn":
            # Create temporal causal sequence arrangement
            reorder_indices, cu_seqlens = _create_temporal_causal_seqlens_flash(s.coords)
            reorder_indices = reorder_indices.to(device)
            cu_seqlens = cu_seqlens.to(device)

            # Reorder tensors
            if num_all_args == 1:
                qkv = qkv[reorder_indices]
                max_seqlen = max(
                    cu_seqlens[i + 1] - cu_seqlens[i] for i in range(len(cu_seqlens) - 1)
                )
                out = flash_attn.flash_attn_varlen_qkvpacked_func(
                    qkv, cu_seqlens, max_seqlen, causal=True, softcap=30.0
                )
            elif num_all_args == 2:
                q = q[reorder_indices]
                kv = kv[reorder_indices]
                max_seqlen = max(
                    cu_seqlens[i + 1] - cu_seqlens[i] for i in range(len(cu_seqlens) - 1)
                )
                out = flash_attn.flash_attn_varlen_kvpacked_func(
                    q,
                    kv,
                    cu_seqlens,
                    cu_seqlens,
                    max_seqlen,
                    max_seqlen,
                    causal=True,
                    softcap=30.0,
                )
            elif num_all_args == 3:
                q = q[reorder_indices]
                k = k[reorder_indices]
                v = v[reorder_indices]
                max_seqlen = max(
                    cu_seqlens[i + 1] - cu_seqlens[i] for i in range(len(cu_seqlens) - 1)
                )
                out = flash_attn.flash_attn_varlen_func(
                    q,
                    k,
                    v,
                    cu_seqlens,
                    cu_seqlens,
                    max_seqlen,
                    max_seqlen,
                    causal=True,
                    softcap=30.0,
                )

            # Reorder output back to original coordinate order
            reverse_indices = torch.argsort(reorder_indices)
            out = out[reverse_indices]

    else:
        # Original non-temporal-causal path
        if ATTN == "xformers":
            if num_all_args == 1:
                q, k, v = qkv.unbind(dim=1)
            elif num_all_args == 2:
                k, v = kv.unbind(dim=1)
            q = q.unsqueeze(0)
            k = k.unsqueeze(0)
            v = v.unsqueeze(0)
            mask = xops.fmha.BlockDiagonalMask.from_seqlens(q_seqlen, kv_seqlen)
            out = xops.memory_efficient_attention(q, k, v, mask)[0]

        elif ATTN == "flash_attn":
            cu_seqlens_q = (
                torch.cat([torch.tensor([0]), torch.cumsum(torch.tensor(q_seqlen), dim=0)])
                .int()
                .to(device)
            )
            if num_all_args in [2, 3]:
                cu_seqlens_kv = (
                    torch.cat(
                        [
                            torch.tensor([0]),
                            torch.cumsum(torch.tensor(kv_seqlen), dim=0),
                        ]
                    )
                    .int()
                    .to(device)
                )
            if num_all_args == 1:
                out = flash_attn.flash_attn_varlen_qkvpacked_func(
                    qkv, cu_seqlens_q, max(q_seqlen), softcap=30.0
                )
            elif num_all_args == 2:
                out = flash_attn.flash_attn_varlen_kvpacked_func(
                    q,
                    kv,
                    cu_seqlens_q,
                    cu_seqlens_kv,
                    max(q_seqlen),
                    max(kv_seqlen),
                    softcap=30.0,
                )
            elif num_all_args == 3:
                out = flash_attn.flash_attn_varlen_func(
                    q,
                    k,
                    v,
                    cu_seqlens_q,
                    cu_seqlens_kv,
                    max(q_seqlen),
                    max(kv_seqlen),
                    softcap=30.0,
                )
        else:
            raise ValueError(f"Unknown attention module: {ATTN}")

    if s is not None:
        return s.replace(out)
    else:
        return out.reshape(N, L, H, -1)


# Example usage showing how to use temporal causality
def example_usage():
    """
    Example showing how to use temporal causal + spatial bidirectional attention
    """
    import torch

    # Create example sparse tensor with temporal coordinates
    # Coordinates: [batch_idx, time_idx, x, y, z]
    coords = torch.tensor(
        [
            [0, 0, 0, 0, 0],  # batch 0, time 0, spatial (0,0,0)
            [0, 0, 0, 1, 0],  # batch 0, time 0, spatial (0,1,0)
            [0, 0, 1, 0, 0],  # batch 0, time 0, spatial (1,0,0)
            [0, 1, 0, 0, 0],  # batch 0, time 1, spatial (0,0,0)
            [0, 1, 0, 1, 0],  # batch 0, time 1, spatial (0,1,0)
            [0, 2, 0, 0, 0],  # batch 0, time 2, spatial (0,0,0)
        ],
        dtype=torch.int32,
    )

    # Create features for each coordinate
    N, H, C = coords.shape[0], 8, 64
    features = torch.randn(N, H, C)

    # Create sparse tensor (this depends on your SparseTensor implementation)
    # Assuming it has coords and feats attributes
    k_sparse = SparseTensor(feats=features, coords=coords)
    v_sparse = SparseTensor(feats=features, coords=coords)

    # Apply temporal causal + spatial bidirectional attention
    output = sparse_scaled_dot_product_attention(
        features,  # q as dense tensor
        k_sparse,  # k as sparse tensor
        v_sparse,  # v as sparse tensor
        temporal_causal=True,  # Enable temporal causality
    )

    print("Attention with temporal causality applied!")
    print(f"Output shape: {output.shape}")

    return output


example_usage()
