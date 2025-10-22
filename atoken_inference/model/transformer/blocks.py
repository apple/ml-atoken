#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2025 Apple Inc. All Rights Reserved.
#
# ruff: noqa
# pylint: skip-file
from typing import *
import torch
import torch.nn as nn
import torch.nn.functional as F
from ..basic import SparseTensor
from ..linear import SparseLinear
from ..nonlinearity import SparseGELU
from ..attention import SparseMultiHeadAttention, SerializeMode
from ..norm import LayerNorm32, RMSNorm32


class SparseMultiheadAttentionPoolingHead(nn.Module):
    """
    Sparse Multihead Attention Pooling Head using cross-attention.
    """

    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int,
        intermediate_size: Optional[int] = None,
        layer_norm_eps: float = 1e-6,
        attn_mode: str = "full",
        window_size: Optional[int] = None,
        use_checkpoint: bool = False,
        use_bias: bool = True,
        use_rms_norm: bool = False,
        qk_rms_norm: bool = False,
    ):
        super().__init__()

        self.hidden_size = hidden_size
        self.num_heads = num_attention_heads
        self.use_checkpoint = use_checkpoint

        if intermediate_size is None:
            intermediate_size = hidden_size * 4

        # Learnable probe parameter
        self.probe = nn.Parameter(torch.randn(hidden_size))

        # Sparse cross-attention module
        self.attention = SparseMultiHeadAttention(
            hidden_size,
            num_heads=num_attention_heads,
            ctx_channels=hidden_size,  # context (K,V) has same dim as query
            type="cross",  # Use cross-attention
            attn_mode=attn_mode,
            window_size=window_size,
            use_bias=use_bias,
            qk_rms_norm=qk_rms_norm,
        )

        # Layer normalization
        if use_rms_norm:
            self.layernorm = RMSNorm32(hidden_size, eps=layer_norm_eps)
        else:
            self.layernorm = LayerNorm32(hidden_size, eps=layer_norm_eps)

        # MLP
        self.mlp = SparseFeedForwardNet(
            hidden_size,
            mlp_channels=intermediate_size,
            use_bias=use_bias,
        )

    def _forward(self, hidden_state: SparseTensor) -> torch.Tensor:
        """
        Forward pass using cross-attention between probe tokens and input features.

        Args:
            hidden_state: SparseTensor with input features (used as context for K,V)

        Returns:
            torch.Tensor: Pooled features of shape (batch_size, hidden_size)
        """
        batch_size = hidden_state.shape[0]
        device = hidden_state.device

        # Create probe tokens with special coordinates [-1, -1, -1, -1]
        probe_coords = torch.full((batch_size, 4), -1, dtype=torch.int32, device=device)
        probe_coords[:, 0] = torch.arange(batch_size, device=device)

        # Repeat probe for each batch
        probe_feats = self.probe.unsqueeze(0).repeat(batch_size, 1)
        probe_tokens = SparseTensor(feats=probe_feats, coords=probe_coords)

        hidden_state, _ = self.attention(probe_tokens, context=hidden_state)

        residual = hidden_state

        hidden_state = hidden_state.replace(self.layernorm(hidden_state.feats))
        hidden_state = residual + self.mlp(hidden_state)

        return hidden_state

    def forward(self, hidden_state: SparseTensor) -> torch.Tensor:
        """
        Forward pass with optional checkpointing.

        Args:
            hidden_state: SparseTensor with input features

        Returns:
            torch.Tensor: Pooled features of shape (batch_size, hidden_size)
        """
        if self.use_checkpoint:
            return torch.utils.checkpoint.checkpoint(
                self._forward, hidden_state, use_reentrant=False
            )
        else:
            return self._forward(hidden_state)


class LearnedPositionEmbedder4D(nn.Module):
    def __init__(self, hidden_size, num_patches_2d=256, max_t=32, max_z=16):
        super().__init__()
        self.embed_dim = hidden_size

        # 2D spatial embedding (x, y only)
        self.num_patches_2d = num_patches_2d
        self.spatial_embedding_size_2d = int(round(self.num_patches_2d**0.5))  # Square root for 2D

        # Validate that num_patches_2d is a perfect square (or close to it)
        expected_patches = self.spatial_embedding_size_2d**2
        if abs(expected_patches - num_patches_2d) > 1:
            import warnings

            warnings.warn(
                f"num_patches_2d={num_patches_2d} is not a perfect square. "
                f"Using spatial_embedding_size_2d={self.spatial_embedding_size_2d} "
                f"(square={expected_patches})"
            )

        self.position_embedding = nn.Embedding(self.num_patches_2d, self.embed_dim)

        # Separate 1D embeddings for temporal and depth dimensions
        self.position_embedding_t = nn.Embedding(max_t, self.embed_dim)
        self.position_embedding_z = nn.Embedding(max_z, self.embed_dim)

        # Initialize temporal and z embeddings to zero (spatial embeddings use default initialization)
        self.position_embedding_t.weight.data.zero_()
        self.position_embedding_z.weight.data.zero_()

    def infer_shapes(self, sparse_patches: "SparseTensor") -> torch.LongTensor:
        """
        Infer 4D spatial shapes from sparse tensor using layout property.

        Args:
            sparse_patches (SparseTensor): Sparse tensor with 4D coordinates
                                         [batch_idx, t, x, y, z]

        Returns:
            torch.LongTensor: Spatial shapes of shape (batch_size, 4) with [max_t, max_x, max_y, max_z]
        """
        batch_size = sparse_patches.shape[0]
        device = sparse_patches.device

        spatial_shapes = torch.zeros((batch_size, 4), dtype=torch.long, device=device)

        # Use layout property for efficient batch processing
        for batch_idx, batch_slice in enumerate(sparse_patches.layout):
            batch_coords = sparse_patches.coords[batch_slice]

            # Get max coordinates for this batch (add 1 since coordinates are 0-indexed)
            max_t = batch_coords[:, 1].max().item() + 1 if batch_coords.shape[1] > 1 else 1
            max_x = batch_coords[:, 2].max().item() + 1
            max_y = batch_coords[:, 3].max().item() + 1
            max_z = batch_coords[:, 4].max().item() + 1 if batch_coords.shape[1] > 4 else 1

            spatial_shapes[batch_idx] = torch.tensor([max_t, max_x, max_y, max_z], device=device)

        return spatial_shapes

    def resize_spatial_embeddings_2d(
        self,
        spatial_embeddings: torch.Tensor,
        sparse_patches: "SparseTensor",
    ) -> torch.Tensor:
        """
        Resize 2D spatial (x,y) positional embeddings for sparse tensor format.
        Always resizes embeddings to match actual coordinate ranges - no clamping.

        Args:
            spatial_embeddings (torch.Tensor): Base 2D spatial embeddings
                                              of shape (height, width, embed_dim)
            sparse_patches (SparseTensor): Sparse tensor with coordinates

        Returns:
            torch.Tensor: Resized spatial embeddings of shape (num_patches, embed_dim)
        """
        # Infer spatial shapes from coordinates
        spatial_shapes = self.infer_shapes(sparse_patches)

        num_patches = sparse_patches.coords.shape[0]
        embed_dim = spatial_embeddings.shape[-1]
        source_dtype = spatial_embeddings.dtype
        device = sparse_patches.device

        # Initialize output tensor
        resized_embeddings = torch.zeros(
            (num_patches, embed_dim),
            device=device,
            dtype=source_dtype,
        )

        # (height, width, embed_dim) -> (1, embed_dim, height, width) for 2D interpolation
        spatial_embeddings = spatial_embeddings.permute(2, 0, 1).unsqueeze(0)

        # Upcast to float32 on CPU because antialias is not supported for bfloat16/float16 on CPU
        if spatial_embeddings.device.type == "cpu":
            spatial_embeddings = spatial_embeddings.to(torch.float32)

        # Process each batch using layout property
        for batch_idx, batch_slice in enumerate(sparse_patches.layout):
            batch_coords = sparse_patches.coords[batch_slice]

            # Get 2D spatial shape for this batch (x, y dimensions only)
            _, height, width, _ = spatial_shapes[batch_idx]

            # Always resize 2D spatial embeddings to match actual coordinate ranges
            batch_spatial_emb = F.interpolate(
                spatial_embeddings,
                size=(height, width),  # Resize to actual coordinate ranges
                mode="bilinear",
                align_corners=False,
                antialias=True,
            )

            # (1, embed_dim, height, width) -> (height, width, embed_dim)
            batch_spatial_emb = batch_spatial_emb.squeeze(0).permute(1, 2, 0)

            # Extract x,y coordinates for this batch - no clamping needed since we resized
            patch_x = batch_coords[:, 2].long()
            patch_y = batch_coords[:, 3].long()

            # Index directly into resized 2D spatial embeddings (no clamping needed)
            batch_embeddings = batch_spatial_emb[patch_x, patch_y]

            # Cast back to original dtype
            batch_embeddings = batch_embeddings.to(source_dtype)

            # Assign to output tensor using batch slice
            resized_embeddings[batch_slice] = batch_embeddings

        return resized_embeddings

    def get_1d_embeddings(
        self,
        sparse_patches: "SparseTensor",
        coord_index: int,
        shape_index: int,
        embedding_layer: nn.Embedding,
        dim_name: str = "1D",
    ) -> torch.Tensor:
        """
        Generic method for 1D positional embeddings (temporal or depth) with interpolation support.

        Args:
            sparse_patches (SparseTensor): Sparse tensor with coordinates
            coord_index (int): Index in coordinates for this dimension (1 for t, 4 for z)
            shape_index (int): Index in spatial_shapes for this dimension (0 for t, 3 for z)
            embedding_layer (nn.Embedding): The embedding layer to use
            dim_name (str): Name for debugging/logging

        Returns:
            torch.Tensor: 1D embeddings of shape (num_patches, embed_dim)
        """
        coords = sparse_patches.coords
        device = sparse_patches.device
        num_patches = coords.shape[0]

        # Initialize output tensor
        embeddings = torch.zeros(
            (num_patches, self.embed_dim),
            device=device,
            dtype=embedding_layer.weight.dtype,
        )

        # Check if this dimension exists in coordinates
        if coords.shape[1] > coord_index:
            # Infer shapes to check if interpolation is needed
            shapes = self.infer_shapes(sparse_patches)

            # Process each batch separately for interpolation
            for batch_idx, batch_slice in enumerate(sparse_patches.layout):
                batch_coords = sparse_patches.coords[batch_slice]
                dim_coords = batch_coords[:, coord_index].long()

                max_dim_in_batch = shapes[batch_idx, shape_index].item()

                if max_dim_in_batch <= embedding_layer.num_embeddings:
                    # No interpolation needed, use direct embedding lookup
                    embeddings[batch_slice] = embedding_layer(dim_coords)
                else:
                    # Interpolation needed - resize embeddings
                    # Get base embeddings: (max_dim, embed_dim)
                    base_emb = embedding_layer.weight

                    # Reshape for interpolation: (max_dim, embed_dim) -> (1, embed_dim, max_dim)
                    base_emb = base_emb.transpose(0, 1).unsqueeze(0)

                    # Upcast to float32 on CPU for interpolation
                    if base_emb.device.type == "cpu":
                        base_emb = base_emb.to(torch.float32)

                    # Interpolate to the required length
                    resized_emb = F.interpolate(
                        base_emb,
                        size=max_dim_in_batch,
                        mode="linear",
                        align_corners=False,
                    )

                    # Reshape back: (1, embed_dim, max_dim) -> (max_dim, embed_dim)
                    resized_emb = resized_emb.squeeze(0).transpose(0, 1)

                    # Cast back to original dtype
                    resized_emb = resized_emb.to(embedding_layer.weight.dtype)

                    # Index into resized embeddings
                    embeddings[batch_slice] = resized_emb[dim_coords]

        return embeddings

    def get_temporal_embeddings(
        self,
        sparse_patches: "SparseTensor",
    ) -> torch.Tensor:
        """
        Get temporal positional embeddings for sparse tensor with interpolation support.

        Args:
            sparse_patches (SparseTensor): Sparse tensor with coordinates

        Returns:
            torch.Tensor: Temporal embeddings of shape (num_patches, embed_dim)
        """
        return self.get_1d_embeddings(
            sparse_patches,
            coord_index=1,  # t coordinate
            shape_index=0,  # temporal shape index
            embedding_layer=self.position_embedding_t,
            dim_name="temporal",
        )

    def get_depth_embeddings(
        self,
        sparse_patches: "SparseTensor",
    ) -> torch.Tensor:
        """
        Get depth (z) positional embeddings for sparse tensor with interpolation support.
        Treats z dimension similar to temporal dimension.

        Args:
            sparse_patches (SparseTensor): Sparse tensor with coordinates

        Returns:
            torch.Tensor: Depth embeddings of shape (num_patches, embed_dim)
        """
        return self.get_1d_embeddings(
            sparse_patches,
            coord_index=4,  # z coordinate
            shape_index=3,  # depth shape index
            embedding_layer=self.position_embedding_z,
            dim_name="depth",
        )

    def forward(self, sparse_patches: "SparseTensor") -> "SparseTensor":
        """
        Forward pass for 4D sparse vision embeddings with modular design:
        - 2D Spatial (x,y): True spatial relationships
        - 1D Temporal (t): Time progression
        - 1D Depth (z): Depth/layer progression

        Args:
            sparse_patches (SparseTensor): Sparse tensor with patch features
                - coords: (num_patches, 5) with [batch_idx, t, x, y, z]
                - feats: (num_patches, embed_dim) - already embedded patch features

        Returns:
            SparseTensor: Patches with all positional embeddings added
        """
        # Get base 2D spatial (x,y) positional embeddings reshaped to 2D spatial grid
        spatial_embeddings_2d = self.position_embedding.weight.reshape(
            self.spatial_embedding_size_2d, self.spatial_embedding_size_2d, -1
        )

        # Resize 2D spatial positional embeddings for sparse format
        resized_spatial_embeddings = self.resize_spatial_embeddings_2d(
            spatial_embeddings_2d, sparse_patches
        )

        # Get temporal embeddings (1D, separate)
        temporal_embeddings = self.get_temporal_embeddings(sparse_patches)

        # Get depth embeddings (1D, separate, similar to temporal)
        depth_embeddings = self.get_depth_embeddings(sparse_patches)

        # Combine all embeddings additively
        total_positional_embeddings = (
            resized_spatial_embeddings + temporal_embeddings + depth_embeddings
        )

        # Return new sparse tensor with embedded features using replace method
        return sparse_patches.replace(total_positional_embeddings)


class LearnedPositionEmbedder(nn.Module):
    def __init__(self, hidden_size, num_patches=256, max_t=32, max_z=16):
        super().__init__()
        self.embed_dim = hidden_size

        self.num_patches = num_patches
        self.position_embedding_size = int(self.num_patches**0.5)
        self.position_embedding = nn.Embedding(self.num_patches, self.embed_dim)

    def infer_spatial_shapes(self, sparse_patches: "SparseTensor") -> torch.LongTensor:
        """
        Infer spatial shapes from sparse tensor using its layout property.

        Args:
            sparse_patches (SparseTensor): Sparse tensor with coordinates

        Returns:
            torch.LongTensor: Spatial shapes of shape (batch_size, 2) with [height, width]
        """
        batch_size = sparse_patches.shape[0]
        device = sparse_patches.device

        spatial_shapes = torch.zeros((batch_size, 2), dtype=torch.long, device=device)

        # Use the layout property to efficiently process each batch
        for batch_idx, batch_slice in enumerate(sparse_patches.layout):
            batch_coords = sparse_patches.coords[batch_slice]

            # Handle empty batches
            if batch_coords.shape[0] == 0:
                # Set default spatial shape for empty batches (or you could skip them)
                spatial_shapes[batch_idx] = torch.tensor([1, 1], device=device)
                continue

            # Get max coordinates for this batch (add 1 since coordinates are 0-indexed)
            max_x = batch_coords[:, 2].max().item() + 1
            max_y = batch_coords[:, 3].max().item() + 1

            spatial_shapes[batch_idx] = torch.tensor([max_x, max_y], device=device)

        return spatial_shapes

    def resize_positional_embeddings_sparse(
        self,
        positional_embeddings: torch.Tensor,
        sparse_patches: "SparseTensor",
    ) -> torch.Tensor:
        """
        Resize positional embeddings for sparse tensor format using layout property.

        Args:
            positional_embeddings (torch.Tensor): Base positional embeddings
                                                 of shape (height, width, embed_dim)
            sparse_patches (SparseTensor): Sparse tensor with coordinates

        Returns:
            torch.Tensor: Resized positional embeddings of shape (num_patches, embed_dim)
        """
        # Infer spatial shapes from coordinates
        spatial_shapes = self.infer_spatial_shapes(sparse_patches)

        num_patches = sparse_patches.coords.shape[0]
        embed_dim = positional_embeddings.shape[-1]
        source_dtype = positional_embeddings.dtype
        device = sparse_patches.device

        # Handle empty tensor case
        if num_patches == 0:
            return torch.empty((0, embed_dim), device=device, dtype=source_dtype)

        # Initialize output tensor
        resized_embeddings = torch.zeros(
            (num_patches, embed_dim),
            device=device,
            dtype=source_dtype,
        )

        # (height, width, embed_dim) -> (1, embed_dim, height, width) for interpolation
        positional_embeddings = positional_embeddings.permute(2, 0, 1).unsqueeze(0)

        # Upcast to float32 on CPU because antialias is not supported for bfloat16/float16 on CPU
        if positional_embeddings.device.type == "cpu":
            positional_embeddings = positional_embeddings.to(torch.float32)

        # Process each batch using the layout property
        for batch_idx, batch_slice in enumerate(sparse_patches.layout):
            batch_coords = sparse_patches.coords[batch_slice]

            # Skip empty batches
            if batch_coords.shape[0] == 0:
                continue

            # Get spatial shape for this batch
            height, width = spatial_shapes[batch_idx]

            # Resize positional embeddings for this batch
            batch_pos_emb = F.interpolate(
                positional_embeddings,
                size=(height, width),
                mode="bilinear",
                align_corners=False,
                antialias=True,
            )

            # (1, embed_dim, height, width) -> (height, width, embed_dim)
            batch_pos_emb = batch_pos_emb.squeeze(0).permute(1, 2, 0)

            # Extract coordinates for this batch (removing batch dimension and unused dimension)
            patch_x = batch_coords[:, 2].long()
            patch_y = batch_coords[:, 3].long()

            # Clamp coordinates to valid range
            patch_x = torch.clamp(patch_x, 0, height - 1)
            patch_y = torch.clamp(patch_y, 0, width - 1)

            # Index into resized positional embeddings
            batch_embeddings = batch_pos_emb[patch_x, patch_y]

            # Cast back to original dtype
            batch_embeddings = batch_embeddings.to(source_dtype)

            # Assign to output tensor using the batch slice
            resized_embeddings[batch_slice] = batch_embeddings

        return resized_embeddings

    def forward(self, sparse_patches: "SparseTensor") -> "SparseTensor":
        """
        Forward pass for sparse vision embeddings.

        Args:
            sparse_patches (SparseTensor): Sparse tensor with patch features
                - coords: (num_patches, 4) with [batch_idx, 0, patch_x, patch_y]
                - feats: (num_patches, embed_dim) - already embedded patch features

        Returns:
            SparseTensor: Patches with positional embeddings added
        """
        # Handle empty input
        if sparse_patches.coords.shape[0] == 0:
            return sparse_patches

        # Get base positional embeddings reshaped to spatial grid
        positional_embeddings = self.position_embedding.weight.reshape(
            self.position_embedding_size, self.position_embedding_size, -1
        )

        # Resize positional embeddings for sparse format (spatial shapes inferred from coords)
        resized_positional_embeddings = self.resize_positional_embeddings_sparse(
            positional_embeddings, sparse_patches
        )

        # Return new sparse tensor with embedded features using replace method
        return sparse_patches.replace(resized_positional_embeddings)


class AbsolutePositionEmbedder(nn.Module):
    """
    Embeds spatial positions into vector representations.
    """

    def __init__(self, channels: int, in_channels: int = 3):
        super().__init__()
        self.channels = channels
        self.in_channels = in_channels
        self.freq_dim = channels // in_channels // 2
        self.freqs = torch.arange(self.freq_dim, dtype=torch.float32) / self.freq_dim
        self.freqs = 1.0 / (10000**self.freqs)

    def _sin_cos_embedding(self, x: torch.Tensor) -> torch.Tensor:
        """
        Create sinusoidal position embeddings.

        Args:
            x: a 1-D Tensor of N indices

        Returns:
            an (N, D) Tensor of positional embeddings.
        """
        self.freqs = self.freqs.to(x.device)
        out = torch.outer(x.to(self.freqs.dtype), self.freqs)
        out = torch.cat([torch.sin(out), torch.cos(out)], dim=-1)
        return out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): (N, D) tensor of spatial positions
        """
        N, D = x.shape
        assert D == self.in_channels, "Input dimension must match number of input channels"
        embed = self._sin_cos_embedding(x.reshape(-1))
        embed = embed.reshape(N, -1)
        if embed.shape[1] < self.channels:
            embed = torch.cat(
                [
                    embed,
                    torch.zeros(N, self.channels - embed.shape[1], device=embed.device),
                ],
                dim=-1,
            )
        return embed


class SparseFeedForwardNet(nn.Module):
    def __init__(self, channels: int, mlp_channels: float = 2048, use_bias=False):
        super().__init__()
        self.mlp = nn.Sequential(
            SparseLinear(channels, mlp_channels, bias=use_bias),
            SparseGELU(approximate="tanh"),
            SparseLinear(mlp_channels, channels, bias=use_bias),
        )

    def forward(self, x: SparseTensor) -> SparseTensor:
        return self.mlp(x)


class SparseTransformerBlock(nn.Module):
    """
    Sparse Transformer block (MSA + FFN).
    """

    def __init__(
        self,
        channels: int,
        num_heads: int,
        mlp_channels: float = 2048,
        attn_mode: Literal[
            "full", "shift_window", "shift_sequence", "shift_order", "swin"
        ] = "full",
        window_size: Optional[int] = None,
        shift_sequence: Optional[int] = None,
        shift_window: Optional[Tuple[int, int, int]] = None,
        serialize_mode: Optional[SerializeMode] = None,
        use_checkpoint: bool = False,
        use_rope: bool = False,
        qk_rms_norm: bool = False,
        use_bias: bool = False,
        use_rms_norm: bool = True,
        ln_affine: bool = True,
    ):
        super().__init__()
        self.use_checkpoint = use_checkpoint

        if use_rms_norm:
            self.norm1 = RMSNorm32(channels, eps=1e-6)
            self.norm2 = RMSNorm32(channels, eps=1e-6)
        else:
            self.norm1 = LayerNorm32(channels, elementwise_affine=ln_affine, eps=1e-6)
            self.norm2 = LayerNorm32(channels, elementwise_affine=ln_affine, eps=1e-6)

        self.attn = SparseMultiHeadAttention(
            channels,
            num_heads=num_heads,
            attn_mode=attn_mode,
            window_size=window_size,
            shift_sequence=shift_sequence,
            shift_window=shift_window,
            serialize_mode=serialize_mode,
            use_bias=use_bias,
            use_rope=use_rope,
            qk_rms_norm=qk_rms_norm,
        )
        self.mlp = SparseFeedForwardNet(
            channels,
            mlp_channels=mlp_channels,
            use_bias=use_bias,
        )

    def _forward(
        self,
        x: SparseTensor,
        kv_cache: Optional[Dict[str, SparseTensor]] = None,
        kv_cache_size: Optional[int] = None,
    ) -> SparseTensor:
        if kv_cache is None:
            kv_cache = {}

        h = x.replace(self.norm1(x.feats))
        h, kv_cache = self.attn(h, kv_cache=kv_cache, kv_cache_size=kv_cache_size)
        x = x + h
        h = x.replace(self.norm2(x.feats))
        h = self.mlp(h)
        x = x + h

        return x, kv_cache

    def forward(
        self,
        x: SparseTensor,
        kv_cache: Optional[Dict[str, SparseTensor]] = None,
        kv_cache_size: Optional[int] = None,
    ) -> SparseTensor:
        if self.use_checkpoint:
            return torch.utils.checkpoint.checkpoint(
                self._forward, x, kv_cache, kv_cache_size, use_reentrant=False
            )
        else:
            return self._forward(x, kv_cache)
