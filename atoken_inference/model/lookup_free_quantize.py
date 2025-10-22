#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2025 Apple Inc. All Rights Reserved.
#
# ruff: noqa
# pylint: skip-file

"""
Lookup Free Quantization
Proposed in https://arxiv.org/abs/2310.05737

In the simplest setup, each dimension is quantized into {-1, 1}.
An entropy penalty is used to encourage utilization.

Refer to
https://github.com/lucidrains/vector-quantize-pytorch/blob/master/vector_quantize_pytorch/lookup_free_quantization.py
https://github.com/theAdamColton/ijepa-enhanced/blob/7edef5f7288ae8f537f0db8a10044a2a487f70c9/ijepa_enhanced/lfq.py
"""

from math import log2, ceil
from collections import namedtuple

import torch
from torch import einsum
import torch.nn.functional as F
from torch.nn import Module

from einops import rearrange, reduce, pack, unpack

from .basic import SparseTensor
# constants

LossBreakdown = namedtuple(
    "LossBreakdown",
    ["per_sample_entropy", "codebook_entropy", "commitment", "avg_probs"],
)

# helper functions


def exists(v):
    return v is not None


def default(*args):
    for arg in args:
        if exists(arg):
            return arg() if callable(arg) else arg
    return None


def pack_one(t, pattern):
    return pack([t], pattern)


def unpack_one(t, ps, pattern):
    return unpack(t, ps, pattern)[0]


# entropy

# def log(t, eps = 1e-5):
#     return t.clamp(min = eps).log()


def entropy(prob):
    return (-prob * torch.log(prob + 1e-5)).sum(dim=-1)


# class
def mult_along_first_dims(x, y):
    """
    returns x * y elementwise along the leading dimensions of y
    """
    ndim_to_expand = x.ndim - y.ndim
    for _ in range(ndim_to_expand):
        y = y.unsqueeze(-1)
    return x * y


def masked_mean(x, m):
    """
    takes the mean of the elements of x that are not masked
    the mean is taken along the shared leading dims of m
    equivalent to: x[m].mean(tuple(range(m.ndim)))

    The benefit of using masked_mean rather than using
    tensor indexing is that masked_mean is much faster
    for torch-compile on batches.

    The drawback is larger floating point errors
    """
    x = mult_along_first_dims(x, m)
    x = x / m.sum()
    return x.sum(tuple(range(m.ndim)))


def entropy_loss(
    logits,
    mask=None,
    temperature=0.01,
    sample_minimization_weight=1.0,
    batch_maximization_weight=1.0,
    eps=1e-5,
):
    """
    Entropy loss of unnormalized logits
    logits: Affinities are over the last dimension
    https://github.com/google-research/magvit/blob/05e8cfd6559c47955793d70602d62a2f9b0bdef5/videogvt/train_lib/losses.py#L279
    LANGUAGE MODEL BEATS DIFFUSION — TOKENIZER IS KEY TO VISUAL GENERATION (2024)

    The loss encourages:
    - LOW per-sample entropy (focused code usage per sample)
    - HIGH batch-level entropy (diverse code usage across batch)

    Formula: loss = sample_entropy - batch_entropy

    NEGATIVE loss is GOOD: means batch entropy > sample entropy
    This indicates good codebook utilization with focused per-sample usage.
    """
    probs = F.softmax(logits / temperature, -1)
    log_probs = F.log_softmax(logits / temperature + eps, -1)

    if mask is not None:
        # avg_probs = probs[mask].mean(tuple(range(probs.ndim - 1)))
        # avg_probs = einx.mean("... D -> D", probs[mask])
        avg_probs = masked_mean(probs, mask)
        # avg_probs = einx.mean("... D -> D", avg_probs)
    else:
        avg_probs = reduce(probs, "... D -> D", "mean")

    avg_entropy = -torch.sum(avg_probs * torch.log(avg_probs + eps))

    sample_entropy = -torch.sum(probs * log_probs, -1)
    if mask is not None:
        # sample_entropy = sample_entropy[mask].mean()
        sample_entropy = masked_mean(sample_entropy, mask).mean()
    else:
        sample_entropy = torch.mean(sample_entropy)

    # Key insight: loss = sample_entropy - batch_entropy
    # Negative loss is DESIRABLE (batch entropy > sample entropy)
    loss = (sample_minimization_weight * sample_entropy) - (batch_maximization_weight * avg_entropy)

    return sample_entropy, avg_entropy, loss


class LFQ(Module):
    def __init__(
        self,
        *,
        dim=None,
        codebook_size=None,
        num_codebooks=1,
        sample_minimization_weight=1.0,
        batch_maximization_weight=1.0,
        token_factorization=False,
        factorized_bits=[9, 9],
    ):
        super().__init__()

        # some assert validations

        assert exists(dim) or exists(
            codebook_size
        ), "either dim or codebook_size must be specified for LFQ"
        assert (
            not exists(codebook_size) or log2(codebook_size).is_integer()
        ), f"your codebook size must be a power of 2 for lookup free quantization (suggested {2 ** ceil(log2(codebook_size))})"

        self.codebook_size = default(codebook_size, lambda: 2**dim)
        self.codebook_dim = int(log2(codebook_size))

        codebook_dims = self.codebook_dim * num_codebooks
        dim = default(dim, codebook_dims)

        has_projections = dim != codebook_dims
        self.has_projections = has_projections

        self.dim = dim
        self.codebook_dim = self.codebook_dim
        self.num_codebooks = num_codebooks

        # for entropy loss
        self.sample_minimization_weight = sample_minimization_weight
        self.batch_maximization_weight = batch_maximization_weight

        # for no auxiliary loss, during inference
        self.token_factorization = token_factorization
        if not self.token_factorization:  # for first stage model
            self.register_buffer("mask", 2 ** torch.arange(self.codebook_dim), persistent=False)
        else:
            self.factorized_bits = factorized_bits
            self.register_buffer(
                "pre_mask", 2 ** torch.arange(factorized_bits[0]), persistent=False
            )
            self.register_buffer(
                "post_mask", 2 ** torch.arange(factorized_bits[1]), persistent=False
            )

        self.register_buffer("zero", torch.tensor(0.0), persistent=False)

        # codes
        all_codes = torch.arange(codebook_size)
        bits = self.indices_to_bits(all_codes)
        codebook = bits * 2.0 - 1.0

        self.register_buffer("codebook", codebook, persistent=False)

    @property
    def dtype(self):
        return self.codebook.dtype

    def indices_to_bits(self, x):
        """
        x: long tensor of indices

        returns big endian bits
        """
        mask = 2 ** torch.arange(self.codebook_dim, device=x.device, dtype=torch.long)
        # x is now big endian bits, the last dimension being the bits
        x = (x.unsqueeze(-1) & mask) != 0
        return x

    def get_codebook_entry(self, x, bhwc, order):  # 0610
        if self.token_factorization:
            if order == "pre":
                mask = 2 ** torch.arange(self.factorized_bits[0], device=x.device, dtype=torch.long)
            else:
                mask = 2 ** torch.arange(self.factorized_bits[1], device=x.device, dtype=torch.long)
        else:
            mask = 2 ** torch.arange(self.codebook_dim, device=x.device, dtype=torch.long)

        x = (x.unsqueeze(-1) & mask) != 0
        x = x * 2.0 - 1.0  # back to the float
        ## scale back to the
        b, h, w, c = bhwc
        x = rearrange(x, "b (h w) c -> b h w c", h=h, w=w, c=c)
        x = rearrange(x, "b h w c -> b c h w")
        return x

    def bits_to_indices(self, bits):
        """
        bits: bool tensor of big endian bits, where the last dimension is the bit dimension

        returns indices, which are long integers from 0 to self.codebook_size
        """
        assert bits.shape[-1] == self.codebook_dim
        indices = 2 ** torch.arange(
            0,
            self.codebook_dim,
            1,
            dtype=torch.long,
            device=bits.device,
        )
        return (bits * indices).sum(-1)

    def decode(self, x):
        """
        x: ... NH
            where NH is number of codebook heads
            A longtensor of codebook indices, containing values from
            0 to self.codebook_size
        """
        x = self.indices_to_bits(x)
        # to some sort of float
        x = x.to(self.dtype)
        # -1 or 1
        x = x * 2 - 1
        x = rearrange(x, "... NC Z-> ... (NC Z)")
        return x

    def forward(
        self,
        x: SparseTensor,  # Custom SparseTensor class
        inv_temperature=100.0,
        return_loss_breakdown=False,
        mask=None,
        return_loss=True,
        fp32_loss_computation=False,  # New parameter for fp32 loss computation
    ):
        """
        Forward pass for custom SparseTensor

        Args:
            x: Custom SparseTensor input with .feats, .coords, .shape, .layout
            inv_temperature: Controls the sharpness of the softmax in entropy loss
            return_loss_breakdown: Whether to return detailed loss components
            mask: Optional mask for selective processing (applied to features)
            return_loss: Whether to compute training losses
            fp32_loss_computation: Whether to compute losses in fp32 for numerical stability
        """

        # Extract features and coordinates from the custom sparse tensor

        # ===== FEATURE PROCESSING =====
        N, feature_dim = x.shape

        # Ensure feature_dim is compatible with our codebook structure
        expected_dim = self.num_codebooks * self.codebook_dim
        if feature_dim != expected_dim:
            raise ValueError(
                f"Feature dimension {feature_dim} doesn't match expected {expected_dim} "
                f"(num_codebooks={self.num_codebooks} * codebook_dim={self.codebook_dim})"
            )

        # Reshape features for codebook processing: [N, num_codebooks, codebook_dim]
        features_reshaped = x.view(N, self.num_codebooks, self.codebook_dim)

        # ===== QUANTIZATION STEP =====
        codebook_value = torch.tensor(1.0, device=x.device, dtype=x.dtype)
        quantized_values = torch.where(features_reshaped > 0, codebook_value, -codebook_value)

        # ===== INDEX CALCULATION =====
        if self.token_factorization:
            # Split for factorized tokens
            pre_bits = quantized_values[
                ..., : self.factorized_bits[0]
            ]  # [N, num_codebooks, factorized_bits[0]]
            post_bits = quantized_values[
                ..., self.factorized_bits[0] :
            ]  # [N, num_codebooks, factorized_bits[1]]

            # Calculate indices by applying bit masks
            indices_pre = ((pre_bits > 0).int() * self.pre_mask.int()).sum(-1)  # [N, num_codebooks]
            indices_post = ((post_bits > 0).int() * self.post_mask.int()).sum(
                -1
            )  # [N, num_codebooks]

            # Flatten indices for output to match original format
            indices_pre_flat = indices_pre.flatten()  # [N * num_codebooks]
            indices_post_flat = indices_post.flatten()  # [N * num_codebooks]
            sparse_indices_quantized = (indices_pre_flat, indices_post_flat)
        else:
            # Standard index calculation
            indices = ((quantized_values > 0).int() * self.mask.int()).sum(-1)  # [N, num_codebooks]
            sparse_indices_quantized = indices.flatten()  # [N * num_codebooks]

        # ===== ENTROPY LOSS (Training Only) =====
        if self.training and return_loss:
            # Convert to fp32 for loss calculations if requested (numerical stability)
            if fp32_loss_computation:
                features_flat_fp32 = features_reshaped.view(-1, self.codebook_dim).float()
                codebook_fp32 = self.codebook.float()
            else:
                features_flat_fp32 = features_reshaped.view(-1, self.codebook_dim)
                codebook_fp32 = self.codebook

            # Compute similarities to all codebook entries
            # self.codebook should have shape [codebook_size, codebook_dim]
            logits = 2 * torch.mm(
                features_flat_fp32, codebook_fp32.T
            )  # [N*num_codebooks, codebook_size]

            # Apply mask if provided
            if mask is not None:
                if mask.shape[0] != N:
                    raise ValueError(
                        f"Mask shape {mask.shape} doesn't match number of features {N}"
                    )
                # Expand mask to cover all codebooks: [N] -> [N*num_codebooks]
                mask_expanded = mask.unsqueeze(1).repeat(1, self.num_codebooks).view(-1)
            else:
                mask_expanded = None

            # Compute entropy loss with temperature
            temperature = 1.0 / inv_temperature if inv_temperature > 0 else 0.01
            per_sample_entropy, codebook_entropy, entropy_aux_loss = entropy_loss(
                logits=logits,
                mask=mask_expanded,
                temperature=temperature,
                sample_minimization_weight=self.sample_minimization_weight,
                batch_maximization_weight=self.batch_maximization_weight,
            )

        else:
            # Set losses to appropriate dtype zeros for consistency
            dtype = torch.float32 if fp32_loss_computation else x.dtype
            per_sample_entropy = torch.tensor(0.0, dtype=dtype, device=x.device)
            codebook_entropy = torch.tensor(0.0, dtype=dtype, device=x.device)
            entropy_aux_loss = torch.tensor(0.0, dtype=dtype, device=x.device)

        # ===== COMMITMENT LOSS =====
        if self.training:
            # Convert to fp32 for commitment loss calculation if requested (numerical stability)
            if fp32_loss_computation:
                features_fp32 = features_reshaped.float()
                quantized_fp32 = quantized_values.float()
            else:
                features_fp32 = features_reshaped
                quantized_fp32 = quantized_values

            # Commitment loss on features
            commit_loss = F.mse_loss(
                features_fp32, quantized_fp32.detach(), reduction="none"
            )  # [N, num_codebooks, codebook_dim]

            # Apply mask if provided
            if mask is not None:
                # Expand mask to cover all codebooks and dimensions: [N] -> [N, num_codebooks, codebook_dim]
                mask_expanded = mask.view(N, 1, 1).expand_as(commit_loss)
                commit_loss = commit_loss[mask_expanded].mean()
            else:
                commit_loss = commit_loss.mean()
        else:
            dtype = torch.float32 if fp32_loss_computation else x.dtype
            commit_loss = torch.tensor(0.0, dtype=dtype, device=x.device)

        # ===== STRAIGHT-THROUGH ESTIMATOR =====
        # Apply straight-through gradients
        quantized_values_ste = features_reshaped + (quantized_values - features_reshaped).detach()

        # ===== OUTPUT CONSTRUCTION =====
        # Reshape back to original feature format: [N, feature_dim]
        # Keep quantized output in original dtype for model consistency
        quantized_feats = quantized_values_ste.view(N, feature_dim)

        # Ensure losses are in the requested dtype (fp32 for numerical stability by default)
        if self.training and return_loss and fp32_loss_computation:
            # Convert any non-fp32 losses to fp32
            entropy_aux_loss = (
                entropy_aux_loss.float()
                if entropy_aux_loss.dtype != torch.float32
                else entropy_aux_loss
            )
            per_sample_entropy = (
                per_sample_entropy.float()
                if per_sample_entropy.dtype != torch.float32
                else per_sample_entropy
            )
            codebook_entropy = (
                codebook_entropy.float()
                if codebook_entropy.dtype != torch.float32
                else codebook_entropy
            )
            commit_loss = commit_loss.float() if commit_loss.dtype != torch.float32 else commit_loss

        # Return values
        ret = (quantized_feats, entropy_aux_loss, sparse_indices_quantized)

        if not return_loss_breakdown:
            return ret

        # Set appropriate dtype for placeholder
        placeholder_dtype = torch.float32 if fp32_loss_computation else x.dtype
        return ret, LossBreakdown(
            per_sample_entropy,
            codebook_entropy,
            commit_loss,
            torch.tensor(0.0, dtype=placeholder_dtype, device=x.device),  # avg_probs placeholder
        )


if __name__ == "__main__":
    quantizer = LFQ(
        codebook_size=2**18,  # codebook size, must be a power of 2
        dim=18,  # this is the input feature dimension, defaults to log2(codebook_size) if not defined
        sample_minimization_weight=1.0,  # within entropy loss, how much weight to give to diversity of codes, taken from https://arxiv.org/abs/1911.05894
        batch_maximization_weight=1.0,
    )

    image_feats = torch.randn(2, 18, 16, 16)  # 16 is dim, must be power of 2 of codebook_size

    quantized, indices, entropy_aux_loss = quantizer(
        image_feats, inv_temperature=100.0
    )  # you may want to experiment with temperature

    assert image_feats.shape == quantized.shape
    assert (quantized == quantizer.indices_to_codes(indices)).all()
