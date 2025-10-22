#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2025 Apple Inc. All Rights Reserved.
#
# ruff: noqa
# pylint: skip-file
from typing import *
import torch
from . import DEBUG
import numpy as np
import math

__all__ = [
    "SparseTensor",
    "sparse_batch_broadcast",
    "sparse_batch_op",
    "sparse_cat",
    "sparse_unbind",
    "reconstruct_from_temporal_slices",
]


class SparseTensor:
    """
    Sparse tensor implementation using only PyTorch tensors (no backend dependencies).

    Parameters:
    - feats (torch.Tensor): Features of the sparse tensor.
    - coords (torch.Tensor): Coordinates of the sparse tensor.
    - shape (torch.Size): Shape of the sparse tensor.
    - layout (List[slice]): Layout of the sparse tensor for each batch

    NOTE:
    - Data corresponding to a same batch should be contiguous.
    - Coords should be in [0, 1023]
    """

    def __init__(
        self,
        feats: torch.Tensor,
        coords: torch.Tensor,
        shape: Optional[torch.Size] = None,
        layout: Optional[List[slice]] = None,
        **kwargs,
    ):
        self._feats = feats
        self._coords = coords

        if shape is None:
            shape = self.__cal_shape(feats, coords)
        if layout is None:
            layout = self.__cal_layout(coords, shape[0])

        self._shape = shape
        self._layout = layout
        self._scale = kwargs.get("scale", (1, 1, 1, 1))
        self._spatial_cache = kwargs.get("spatial_cache", {})

        if DEBUG:
            try:
                assert (
                    self.feats.shape[0] == self.coords.shape[0]
                ), f"Invalid feats shape: {self.feats.shape}, coords shape: {self.coords.shape}"
                assert self.shape == self.__cal_shape(
                    self.feats, self.coords
                ), f"Invalid shape: {self.shape}"
                assert self.layout == self.__cal_layout(
                    self.coords, self.shape[0]
                ), f"Invalid layout: {self.layout}"
                for i in range(self.shape[0]):
                    batch_slice = self.layout[i]
                    if batch_slice.start < batch_slice.stop:
                        assert torch.all(
                            self.coords[batch_slice, 0] == i
                        ), f"The data of batch {i} is not contiguous"
            except Exception as e:
                print("Debugging information:")
                print(f"- Shape: {self.shape}")
                print(f"- Layout: {self.layout}")
                print(f"- Scale: {self._scale}")
                print(f"- Coords shape: {self.coords.shape}")
                print(f"- Feats shape: {self.feats.shape}")
                raise e

    def __cal_shape(self, feats, coords):
        shape = []

        # Handle empty tensor case
        if coords.shape[0] == 0:
            shape.append(0)  # batch size 0
            shape.extend([*feats.shape[1:]])  # feature dimensions
            return torch.Size(shape)

        shape.append(coords[:, 0].max().item() + 1)
        shape.extend([*feats.shape[1:]])
        return torch.Size(shape)

    def __cal_layout(self, coords, batch_size):
        # Handle empty tensor case
        if coords.shape[0] == 0:
            return [slice(0, 0) for _ in range(batch_size)]

        seq_len = torch.bincount(coords[:, 0], minlength=batch_size)
        offset = torch.cumsum(seq_len, dim=0)
        layout = [
            slice((offset[i] - seq_len[i]).item(), offset[i].item()) for i in range(batch_size)
        ]
        return layout

    @property
    def shape(self) -> torch.Size:
        return self._shape

    def dim(self) -> int:
        return len(self.shape)

    @property
    def layout(self) -> List[slice]:
        return self._layout

    @property
    def feats(self) -> torch.Tensor:
        return self._feats

    @feats.setter
    def feats(self, value: torch.Tensor):
        self._feats = value

    @property
    def coords(self) -> torch.Tensor:
        return self._coords

    @coords.setter
    def coords(self, value: torch.Tensor):
        self._coords = value

    @property
    def dtype(self):
        return self.feats.dtype

    @property
    def device(self):
        return self.feats.device

    @overload
    def to(self, dtype: torch.dtype) -> "SparseTensor": ...

    @overload
    def to(
        self,
        device: Optional[Union[str, torch.device]] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> "SparseTensor": ...

    def to(self, *args, **kwargs) -> "SparseTensor":
        device = None
        dtype = None
        if len(args) == 2:
            device, dtype = args
        elif len(args) == 1:
            if isinstance(args[0], torch.dtype):
                dtype = args[0]
            else:
                device = args[0]
        if "dtype" in kwargs:
            assert dtype is None, "to() received multiple values for argument 'dtype'"
            dtype = kwargs["dtype"]
        if "device" in kwargs:
            assert device is None, "to() received multiple values for argument 'device'"
            device = kwargs["device"]

        new_feats = self.feats.to(device=device, dtype=dtype)
        new_coords = self.coords.to(device=device)
        return self.replace(new_feats, new_coords)

    def type(self, dtype):
        new_feats = self.feats.type(dtype)
        return self.replace(new_feats)

    def cpu(self) -> "SparseTensor":
        new_feats = self.feats.cpu()
        new_coords = self.coords.cpu()
        return self.replace(new_feats, new_coords)

    def cuda(self) -> "SparseTensor":
        new_feats = self.feats.cuda()
        new_coords = self.coords.cuda()
        return self.replace(new_feats, new_coords)

    def half(self) -> "SparseTensor":
        new_feats = self.feats.half()
        return self.replace(new_feats)

    def float(self) -> "SparseTensor":
        new_feats = self.feats.float()
        return self.replace(new_feats)

    def detach(self) -> "SparseTensor":
        new_coords = self.coords.detach()
        new_feats = self.feats.detach()
        return self.replace(new_feats, new_coords)

    def dense(self) -> torch.Tensor:
        """
        Convert sparse tensor to dense representation.
        Note: This is a basic implementation and may need customization based on use case.
        """
        batch_size = self.shape[0]
        feat_dims = self.shape[1:]

        # Get maximum coordinates for spatial dimensions
        if self.coords.shape[0] == 0:
            spatial_shape = [1, 1, 1, 1]  # Default minimal shape
        else:
            max_coords = self.coords.max(0)[0]
            spatial_shape = [
                max_coords[i].item() + 1 for i in range(1, min(5, self.coords.shape[1]))
            ]
            while len(spatial_shape) < 4:
                spatial_shape.append(1)

        # Create dense tensor
        dense_shape = [batch_size] + spatial_shape + list(feat_dims)
        dense = torch.zeros(dense_shape, dtype=self.dtype, device=self.device)

        # Fill in values
        for i in range(self.coords.shape[0]):
            coord = self.coords[i]
            batch_idx = coord[0].long()
            spatial_idx = tuple(coord[j].long() for j in range(1, min(5, len(coord))))
            dense[(batch_idx,) + spatial_idx] = self.feats[i]

        return dense

    def reshape(self, *shape) -> "SparseTensor":
        new_feats = self.feats.reshape(self.feats.shape[0], *shape)
        return self.replace(new_feats)

    def unbind(self, dim: int) -> List["SparseTensor"]:
        return sparse_unbind(self, dim)

    def replace(self, feats: torch.Tensor, coords: Optional[torch.Tensor] = None) -> "SparseTensor":
        new_shape = [self.shape[0]]
        new_shape.extend(feats.shape[1:])

        new_tensor = SparseTensor(
            feats=feats,
            coords=self.coords if coords is None else coords,
            shape=torch.Size(new_shape),
            layout=self.layout,
            scale=self._scale,
            spatial_cache=self._spatial_cache,
        )
        return new_tensor

    @staticmethod
    def full(aabb, dim, value, dtype=torch.float32, device=None) -> "SparseTensor":
        N, C = dim
        x = torch.arange(aabb[0], aabb[3] + 1)
        y = torch.arange(aabb[1], aabb[4] + 1)
        z = torch.arange(aabb[2], aabb[5] + 1)
        coords = torch.stack(torch.meshgrid(x, y, z, indexing="ij"), dim=-1).reshape(-1, 3)
        coords = torch.cat(
            [
                torch.arange(N).view(-1, 1).repeat(1, coords.shape[0]).view(-1, 1),
                coords.repeat(N, 1),
            ],
            dim=1,
        ).to(dtype=torch.int32, device=device)
        feats = torch.full((coords.shape[0], C), value, dtype=dtype, device=device)
        return SparseTensor(feats=feats, coords=coords)

    def __merge_sparse_cache(self, other: "SparseTensor") -> dict:
        new_cache = {}
        for k in set(list(self._spatial_cache.keys()) + list(other._spatial_cache.keys())):
            if k in self._spatial_cache:
                new_cache[k] = self._spatial_cache[k]
            if k in other._spatial_cache:
                if k not in new_cache:
                    new_cache[k] = other._spatial_cache[k]
                else:
                    new_cache[k].update(other._spatial_cache[k])
        return new_cache

    def __neg__(self) -> "SparseTensor":
        return self.replace(-self.feats)

    def __elemwise__(
        self, other: Union[torch.Tensor, "SparseTensor"], op: callable
    ) -> "SparseTensor":
        if isinstance(other, torch.Tensor):
            try:
                other = torch.broadcast_to(other, self.shape)
                other = sparse_batch_broadcast(self, other)
            except:
                pass
        if isinstance(other, SparseTensor):
            other = other.feats
        new_feats = op(self.feats, other)
        new_tensor = self.replace(new_feats)
        if isinstance(other, SparseTensor):
            new_tensor._spatial_cache = self.__merge_sparse_cache(other)
        return new_tensor

    def __add__(self, other: Union[torch.Tensor, "SparseTensor", float]) -> "SparseTensor":
        return self.__elemwise__(other, torch.add)

    def __radd__(self, other: Union[torch.Tensor, "SparseTensor", float]) -> "SparseTensor":
        return self.__elemwise__(other, torch.add)

    def __sub__(self, other: Union[torch.Tensor, "SparseTensor", float]) -> "SparseTensor":
        return self.__elemwise__(other, torch.sub)

    def __rsub__(self, other: Union[torch.Tensor, "SparseTensor", float]) -> "SparseTensor":
        return self.__elemwise__(other, lambda x, y: torch.sub(y, x))

    def __mul__(self, other: Union[torch.Tensor, "SparseTensor", float]) -> "SparseTensor":
        return self.__elemwise__(other, torch.mul)

    def __rmul__(self, other: Union[torch.Tensor, "SparseTensor", float]) -> "SparseTensor":
        return self.__elemwise__(other, torch.mul)

    def __truediv__(self, other: Union[torch.Tensor, "SparseTensor", float]) -> "SparseTensor":
        return self.__elemwise__(other, torch.div)

    def __rtruediv__(self, other: Union[torch.Tensor, "SparseTensor", float]) -> "SparseTensor":
        return self.__elemwise__(other, lambda x, y: torch.div(y, x))

    def __getitem__(self, idx):
        if isinstance(idx, int):
            idx = [idx]
        elif isinstance(idx, slice):
            idx = range(*idx.indices(self.shape[0]))
        elif isinstance(idx, torch.Tensor):
            if idx.dtype == torch.bool:
                assert idx.shape == (self.shape[0],), f"Invalid index shape: {idx.shape}"
                idx = idx.nonzero().squeeze(1)
            elif idx.dtype in [torch.int32, torch.int64]:
                assert len(idx.shape) == 1, f"Invalid index shape: {idx.shape}"
            else:
                raise ValueError(f"Unknown index type: {idx.dtype}")
        else:
            raise ValueError(f"Unknown index type: {type(idx)}")

        coords = []
        feats = []
        for new_idx, old_idx in enumerate(idx):
            # Check if old_idx is within bounds
            if old_idx >= len(self.layout):
                # Return empty tensors for out-of-bounds batch indices
                empty_coords = torch.empty(
                    (0, self.coords.shape[1]),
                    dtype=self.coords.dtype,
                    device=self.coords.device,
                )
                empty_feats = torch.empty(
                    (0,) + self.feats.shape[1:],
                    dtype=self.feats.dtype,
                    device=self.feats.device,
                )
                coords.append(empty_coords)
                feats.append(empty_feats)
            else:
                batch_coords = self.coords[self.layout[old_idx]]
                batch_feats = self.feats[self.layout[old_idx]]

                # Handle empty batches (can happen due to temporal slicing)
                if batch_coords.shape[0] > 0:
                    batch_coords = batch_coords.clone()
                    batch_coords[:, 0] = new_idx
                    coords.append(batch_coords)
                    feats.append(batch_feats)
                else:
                    # Create empty tensors for empty batches
                    empty_coords = torch.empty(
                        (0, self.coords.shape[1]),
                        dtype=self.coords.dtype,
                        device=self.coords.device,
                    )
                    empty_feats = torch.empty(
                        (0,) + self.feats.shape[1:],
                        dtype=self.feats.dtype,
                        device=self.feats.device,
                    )
                    coords.append(empty_coords)
                    feats.append(empty_feats)

        # Concatenate all coordinates and features (including empty ones)
        if len(coords) > 0:
            coords = torch.cat(coords, dim=0).contiguous()
            feats = torch.cat(feats, dim=0).contiguous()
        else:
            # Handle case where idx is empty
            coords = torch.empty(
                (0, self.coords.shape[1]),
                dtype=self.coords.dtype,
                device=self.coords.device,
            )
            feats = torch.empty(
                (0,) + self.feats.shape[1:],
                dtype=self.feats.dtype,
                device=self.feats.device,
            )

        return SparseTensor(feats=feats, coords=coords)

    def register_spatial_cache(self, key, value) -> None:
        """
        Register a spatial cache.
        The spatial cache can be any thing you want to cache.
        The registery and retrieval of the cache is based on current scale.
        """
        scale_key = str(self._scale)
        if scale_key not in self._spatial_cache:
            self._spatial_cache[scale_key] = {}
        self._spatial_cache[scale_key][key] = value

    def get_spatial_cache(self, key=None):
        """
        Get a spatial cache.
        """
        scale_key = str(self._scale)
        cur_scale_cache = self._spatial_cache.get(scale_key, {})
        if key is None:
            return cur_scale_cache
        return cur_scale_cache.get(key, None)

    def to_dense_padded(self) -> tuple:
        """
        Convert sparse tensor to a tuple of dense representations:
        - Dense features tensor with shape [batch, seq_len, feat_dim]
        - Dense coordinates tensor with shape [batch, seq_len, coord_dim]
        - Boolean mask with shape [batch, seq_len] indicating valid (non-padded) positions

        Returns:
            tuple: (dense_features, dense_coords, mask)
                - dense_features: Tensor with shape [batch, seq_len, feat_dim]
                - dense_coords: Tensor with shape [batch, seq_len, coord_dim]
                - mask: Boolean mask with shape [batch, seq_len] (True for valid positions)
        """
        batch_size = len(self.layout)

        # Calculate max sequence length across all batches
        max_seq_len = max(layout_slice.stop - layout_slice.start for layout_slice in self.layout)

        # Get feature and coordinate dimensions
        feat_dims = self.feats.shape[1:]
        coord_dims = self.coords.shape[1:]

        # Create output tensors
        feat_shape = (batch_size, max_seq_len) + feat_dims
        coord_shape = (batch_size, max_seq_len) + coord_dims

        dense_feats = torch.zeros(feat_shape, dtype=self.dtype, device=self.device)
        dense_coords = torch.full(
            coord_shape, -1, dtype=self.coords.dtype, device=self.coords.device
        )
        mask = torch.zeros((batch_size, max_seq_len), dtype=torch.bool, device=self.device)

        # Fill in the tensors with data from each batch
        for batch_idx in range(batch_size):
            batch_slice = self.layout[batch_idx]
            seq_len = batch_slice.stop - batch_slice.start

            # Copy features and coordinates from the sparse tensor
            dense_feats[batch_idx, :seq_len] = self.feats[batch_slice]
            dense_coords[batch_idx, :seq_len] = self.coords[batch_slice]

            # Mark valid positions in the mask
            mask[batch_idx, :seq_len] = True

        return dense_feats, dense_coords, mask

    def get_batch_mean(self) -> "SparseTensor":
        """
        Compute the mean of features for each batch in the sparse tensor.

        Returns:
            SparseTensor: A sparse tensor with a single point per batch containing
                        the mean feature values for that batch.
        """
        batch_size = len(self.layout)

        # Get the feature dimensions
        feature_dims = self.feats.shape[1:]

        # Create output tensor with shape (batch_size, feature_dim)
        batch_means = torch.zeros(
            (batch_size,) + feature_dims, dtype=self.dtype, device=self.device
        )

        # Create the coordinates for the mean features
        # Using [-1, -1, -1, -1] for spatial coordinates to indicate special tokens
        coords = torch.full(
            (batch_size, self.coords.shape[1]),
            -1,
            dtype=self.coords.dtype,
            device=self.coords.device,
        )

        # Set batch indices
        coords[:, 0] = torch.arange(batch_size, device=self.coords.device)

        # Compute mean for each batch
        for batch_idx in range(batch_size):
            # Get the batch slice
            batch_slice = self.layout[batch_idx]

            # Compute mean of features in this batch
            batch_features = self.feats[batch_slice]
            if len(batch_features) > 0:  # Check to avoid division by zero
                batch_means[batch_idx] = torch.mean(batch_features, dim=0)

        return SparseTensor(feats=batch_means, coords=coords)

    def expand_by_factors(self, factors, channel_duplicate=1):
        """
        Convert sparse representation to original locations, handling all dimensions.

        Args:
            factors: Tuple specifying patch dimensions (T, X, Y, Z) or (X, Y) or (T, X, Y)
            channel_duplicate: Factor to duplicate the channels after expansion (default: 1)
        """
        if len(factors) == 2:
            X, Y = factors
            T, Z = 1, 1
        elif len(factors) == 3:
            T, X, Y = factors
            Z = 1
        elif len(factors) == 4:
            T, X, Y, Z = factors
        else:
            raise ValueError(f"Invalid patch size: {factors}")

        patch_product = T * X * Y * Z

        N, C = self.feats.shape
        channels = C // patch_product

        assert (
            C == patch_product * channels
        ), f"Feature dimension {C} doesn't match patch_product ({patch_product}) * channels ({channels})"

        feats_reshaped = self.feats.reshape(N, T, X, Y, Z, channels)

        # Create mesh grids for offsets within each patch dimension
        t_offsets, x_offsets, y_offsets, z_offsets = torch.meshgrid(
            torch.arange(T, device=self.device),
            torch.arange(X, device=self.device),
            torch.arange(Y, device=self.device),
            torch.arange(Z, device=self.device),
            indexing="ij",
        )

        # Flatten the offsets
        t_offsets = t_offsets.reshape(-1)  # [patch_product]
        x_offsets = x_offsets.reshape(-1)  # [patch_product]
        y_offsets = y_offsets.reshape(-1)  # [patch_product]
        z_offsets = z_offsets.reshape(-1)  # [patch_product]

        # Repeat each coordinate patch_product times
        # [N, 5] -> [N*patch_product, 5]
        expanded_coords = self.coords.repeat_interleave(patch_product, dim=0)

        tiled_t_offsets = t_offsets.repeat(N)
        tiled_x_offsets = x_offsets.repeat(N)
        tiled_y_offsets = y_offsets.repeat(N)
        tiled_z_offsets = z_offsets.repeat(N)

        # Update T, X, Y, and Z coordinates (indices 1, 2, 3, 4)
        # Only apply patch expansion to dimensions with patch size > 1
        if T > 1:
            expanded_coords[:, 1] = expanded_coords[:, 1] * T + tiled_t_offsets

        expanded_coords[:, 2] = expanded_coords[:, 2] * X + tiled_x_offsets
        expanded_coords[:, 3] = expanded_coords[:, 3] * Y + tiled_y_offsets

        if Z > 1:
            expanded_coords[:, 4] = expanded_coords[:, 4] * Z + tiled_z_offsets

        # Reshape features to match the expanded coordinates
        # [N, t_patch_size, patch_size, patch_size, z_patch_size, channels] -> [N*patch_product, channels]
        expanded_feats = feats_reshaped.reshape(-1, channels)

        # Apply channel duplication if requested
        if channel_duplicate > 1:
            # Duplicate each channel by repeating and then reshaping
            # [N*patch_product, channels] -> [N*patch_product, channels*channel_duplicate]
            expanded_feats = (
                expanded_feats.unsqueeze(-1)
                .repeat(1, 1, channel_duplicate)
                .reshape(N * patch_product, channels * channel_duplicate)
            )

        return SparseTensor(feats=expanded_feats, coords=expanded_coords)

    def shrink_by_factors(self, factors, channel_average=1):
        """
        Convert sparse representation to a coarser resolution by merging points
        into larger patches and concatenating their features.

        This method is the inverse of expand_by_patch_size, converting from smaller patches
        (e.g., 8x8) to larger patches (e.g., 16x16) by grouping points and concatenating
        their features.

        Args:
            factors: Tuple specifying the target patch dimensions.
                     Can be 2D (X, Y), 3D (T, X, Y), or 4D (T, X, Y, Z).

        Returns:
            SparseTensor: A new sparse tensor with merged patches and concatenated features.
        """
        # Parse target patch size
        if len(factors) == 2:
            X, Y = factors
            T, Z = 1, 1
        elif len(factors) == 3:
            T, X, Y = factors
            Z = 1
        elif len(factors) == 4:
            T, X, Y, Z = factors
        else:
            raise ValueError(f"Invalid target patch size: {factors}")

        # Calculate patch product (total number of points in a patch)
        patch_product = T * X * Y * Z

        # Get feature dimensions
        N, C = self.feats.shape

        # New feature dimension after concatenation
        new_C = C * patch_product

        # Apply channel averaging if specified
        if channel_average > 1:
            # Ensure new_C is divisible by channel_average
            assert (
                new_C % channel_average == 0
            ), f"Feature dimension {new_C} must be divisible by channel_average {channel_average}"
            final_C = new_C // channel_average
        else:
            final_C = new_C

        # Calculate new coordinates by integer division
        new_coords = self.coords.clone()

        # Apply patch shrinking to each dimension with patch size > 1
        if T > 1:
            new_coords[:, 1] = new_coords[:, 1] // T

        new_coords[:, 2] = new_coords[:, 2] // X
        new_coords[:, 3] = new_coords[:, 3] // Y

        if Z > 1:
            new_coords[:, 4] = new_coords[:, 4] // Z

        # Calculate position within the patch (local offsets)
        local_t = self.coords[:, 1] % T if T > 1 else torch.zeros_like(self.coords[:, 1])
        local_x = self.coords[:, 2] % X
        local_y = self.coords[:, 3] % Y
        local_z = self.coords[:, 4] % Z if Z > 1 else torch.zeros_like(self.coords[:, 4])

        # Calculate flat offset within the patch
        flat_offset = (local_t * X * Y * Z) + (local_x * Y * Z) + (local_y * Z) + local_z

        # Get a unique ID for each coordinate
        coord_str = torch.cat([new_coords[:, 0:1], new_coords[:, 1:5]], dim=1)

        # Use torch.unique to find unique coordinates and their indices
        unique_coords, inverse_indices, counts = torch.unique(
            coord_str, dim=0, return_inverse=True, return_counts=True
        )

        # Create output tensors
        batch_N = unique_coords.shape[0]

        # If channel averaging, create intermediate tensor
        if channel_average > 1:
            merged_feats_intermediate = torch.zeros(
                (batch_N, new_C), dtype=self.dtype, device=self.device
            )
        else:
            merged_feats = torch.zeros((batch_N, new_C), dtype=self.dtype, device=self.device)

        merged_coords = unique_coords.to(dtype=self.coords.dtype)

        # FULLY VECTORIZED FEATURE PLACEMENT:
        # Expand dimensions for efficient indexing
        batch_idx = inverse_indices.unsqueeze(1)  # [N, 1]

        # For each feature dimension, we need to calculate where it goes in the output
        feature_indices = flat_offset.unsqueeze(1) * C + torch.arange(
            C, device=self.device
        ).unsqueeze(0)  # [N, C]

        # Reshape input features for efficient indexing
        input_feats_flat = self.feats.reshape(-1)  # [N*C]

        # Create indices for scatter operation
        batch_indices_flat = batch_idx.repeat(1, C).reshape(-1)  # [N*C]
        feature_indices_flat = feature_indices.reshape(-1)  # [N*C]

        # Use advanced indexing to place all features at once
        # This creates a sparse tensor-like operation but with PyTorch tensors
        if channel_average > 1:
            merged_feats_intermediate[batch_indices_flat, feature_indices_flat] = input_feats_flat

            # Apply channel averaging
            # Reshape to group channels for averaging
            merged_feats_reshaped = merged_feats_intermediate.reshape(
                batch_N, final_C, channel_average
            )
            # Average across channel groups
            merged_feats = torch.mean(merged_feats_reshaped, dim=2)
        else:
            merged_feats[batch_indices_flat, feature_indices_flat] = input_feats_flat

        # Calculate new layout based on batch indices
        batch_indices = merged_coords[:, 0].cpu().numpy()
        batch_size = int(torch.max(merged_coords[:, 0]).item()) + 1 if batch_N > 0 else 0

        layout_ends = np.zeros(batch_size + 1, dtype=np.int64)
        np.add.at(layout_ends[1:], batch_indices, 1)
        layout_ends = np.cumsum(layout_ends)

        new_layout = [slice(layout_ends[i], layout_ends[i + 1]) for i in range(batch_size)]

        # Create new shape
        new_shape = torch.Size([self.shape[0], final_C])

        # Preserve metadata from original tensor
        result = SparseTensor(
            feats=merged_feats, coords=merged_coords, shape=new_shape, layout=new_layout
        )

        # Copy scale and spatial cache from original tensor
        result._scale = self._scale
        result._spatial_cache = self._spatial_cache

        return result

    def slice_by_mask(self, mask: torch.Tensor, temporal_offset: int = None) -> "SparseTensor":
        """
        Slice the sparse tensor using a boolean mask.

        Args:
            mask (torch.Tensor): Boolean mask of shape [N] where N is the number of points
            temporal_offset (int, optional): If provided, subtract this offset from temporal coordinates

        Returns:
            SparseTensor: A new sparse tensor containing only the masked points
        """
        assert (
            mask.shape[0] == self.coords.shape[0]
        ), f"Mask shape {mask.shape[0]} doesn't match coords shape {self.coords.shape[0]}"

        if not mask.any():
            # Return empty sparse tensor
            empty_coords = torch.empty(
                (0, self.coords.shape[1]),
                dtype=self.coords.dtype,
                device=self.coords.device,
            )
            empty_feats = torch.empty(
                (0,) + self.feats.shape[1:],
                dtype=self.feats.dtype,
                device=self.feats.device,
            )
            return SparseTensor(feats=empty_feats, coords=empty_coords)

        # Extract masked coordinates and features
        new_coords = self.coords[mask].clone()
        new_feats = self.feats[mask]

        # Apply temporal offset if provided (assuming temporal coordinate is at index 1)
        if temporal_offset is not None:
            new_coords[:, 1] -= temporal_offset

        return SparseTensor(feats=new_feats, coords=new_coords)

    def slice_temporal_range(
        self,
        start_frame: int,
        end_frame: int,
        adjust_temporal: bool = True,
        offset: int = 0,
    ) -> "SparseTensor":
        """
        Slice the sparse tensor to keep only points within a temporal range.

        Args:
            start_frame (int): Starting frame index (inclusive)
            end_frame (int): Ending frame index (exclusive)
            adjust_temporal (bool): If True, adjust temporal coordinates to start from 0

        Returns:
            SparseTensor: A new sparse tensor containing only points in the temporal range
        """
        # Handle invalid range
        if start_frame >= end_frame:
            empty_coords = torch.empty(
                (0, self.coords.shape[1]),
                dtype=self.coords.dtype,
                device=self.coords.device,
            )
            empty_feats = torch.empty(
                (0,) + self.feats.shape[1:],
                dtype=self.feats.dtype,
                device=self.feats.device,
            )
            result = SparseTensor(feats=empty_feats, coords=empty_coords)
            if adjust_temporal:
                result.register_spatial_cache("temporal_offset", start_frame - offset)
            return result

        mask = (self.coords[:, 1] >= start_frame) & (self.coords[:, 1] < end_frame)
        temporal_offset = start_frame - offset if adjust_temporal else None

        result = self.slice_by_mask(mask, temporal_offset=temporal_offset)

        # Store the original temporal offset for reconstruction
        if adjust_temporal:
            result.register_spatial_cache("temporal_offset", start_frame - offset)

        return result

    def get_temporal_range(self) -> tuple:
        """
        Get the temporal range of the sparse tensor.

        Returns:
            tuple: (min_frame, max_frame) or (None, None) if empty
        """
        if self.coords.shape[0] == 0:
            return None, None

        temporal_coords = self.coords[:, 1]
        return temporal_coords.min().item(), temporal_coords.max().item()

    def split_by_temporal_batches(
        self,
        frame_batch_size: int,
        frame_batch_strides: int = None,
        adjust_temporal: bool = True,
        offset: int = 0,
    ) -> List["SparseTensor"]:
        """
        Split the sparse tensor into temporal batches with sliding window support.

        Args:
            frame_batch_size (int): Number of frames per batch (window size)
            frame_batch_strides (int): Stride between consecutive windows (step size)
            adjust_temporal (bool): Whether to adjust temporal coordinates to start from 0 for each slice
            offset (int): Offset to apply when adjusting temporal coordinates

        Returns:
            List[SparseTensor]: List of temporal slices with sliding window overlap

        Example:
            # frame_batch_size=16, frame_batch_strides=12
            # Slice 0: frames [0, 16)   -> temporal range 0-15
            # Slice 1: frames [12, 28)  -> temporal range 12-27
            # Slice 2: frames [24, 40)  -> temporal range 24-39
            # etc.
        """
        min_frame, max_frame = self.get_temporal_range()

        # Handle empty tensor
        if min_frame is None or max_frame is None:
            return []

        # Validate parameters
        if frame_batch_size <= 0:
            raise ValueError("frame_batch_size must be positive")

        if frame_batch_strides is None:
            frame_batch_strides = frame_batch_size

        if frame_batch_strides <= 0:
            raise ValueError("frame_batch_strides must be positive")

        slices = []
        current_start = min_frame

        # Continue creating slices while there's still data to process
        while current_start <= max_frame:
            # Calculate end frame for current slice
            current_end = current_start + frame_batch_size

            # Create slice for current window
            # Note: slice_temporal_range uses [start, end) so current_end is exclusive
            if current_start == min_frame:
                # No offset for the first slice
                update_offset = 0
            else:
                update_offset = offset

            slice_tensor = self.slice_temporal_range(
                current_start,
                current_end,
                adjust_temporal=adjust_temporal,
                offset=update_offset,
            )

            slices.append(slice_tensor)

            # Move to next window start position
            current_start += frame_batch_strides

            # Stop if we've moved beyond the available data
            # This prevents creating empty slices at the end
            if current_start > max_frame:
                break

        return slices

    def get_last_k_timesteps(self, k: int) -> "SparseTensor":
        """
        Return a sparse tensor containing only the last k consecutive time steps for each batch.
        Uses global max time step across all batches as reference.
        Preserves batch structure including empty batches for compatibility with concat operations.

        Args:
            k (int): Number of last time steps to keep

        Returns:
            SparseTensor: Sparse tensor with only the last k time steps per batch.
                        Batches without the required time steps will be empty but preserved.
        """
        if k <= 0 or self.coords.shape[0] == 0:
            # Return empty sparse tensor with same batch structure
            empty_coords = torch.empty(
                (0, self.coords.shape[1]),
                dtype=self.coords.dtype,
                device=self.coords.device,
            )
            empty_feats = torch.empty(
                (0,) + self.feats.shape[1:],
                dtype=self.feats.dtype,
                device=self.feats.device,
            )
            # Preserve the batch size in shape even if empty
            empty_shape = torch.Size([self.shape[0]] + list(self.feats.shape[1:]))
            # Create empty layout for all batches
            empty_layout = [slice(0, 0) for _ in range(self.shape[0])]
            return SparseTensor(
                feats=empty_feats,
                coords=empty_coords,
                shape=empty_shape,
                layout=empty_layout,
            )

        # Get global max time step
        global_max_time = self.coords[:, 1].max().item()

        # Calculate time threshold for last k steps
        time_threshold = global_max_time - k + 1

        # Create mask for points in the last k time steps
        valid_mask = self.coords[:, 1] >= time_threshold

        # Extract selected coordinates and features and coords minus the time threshold on the temporal dimension
        time_offset = torch.zeros(
            (1, self.coords.shape[1]),
            dtype=self.coords.dtype,
            device=self.coords.device,
        )
        time_offset[:, 1] = time_threshold

        selected_coords = self.coords[valid_mask] - time_offset
        selected_feats = self.feats[valid_mask]

        # Preserve the original batch size in the shape
        new_shape = torch.Size(
            [self.shape[0]] + list(selected_feats.shape[1:])
            if selected_feats.shape[0] > 0
            else [self.shape[0]] + list(self.feats.shape[1:])
        )

        return SparseTensor(feats=selected_feats, coords=selected_coords, shape=new_shape)

    def concat_temporal_at_batch_start(self, other: "SparseTensor") -> "SparseTensor":
        """
        Concatenate another sparse tensor at the beginning of each batch in this tensor.
        This is the robust implementation that handles all edge cases correctly.

        Args:
            other (SparseTensor): The sparse tensor to inject at the beginning of each batch

        Returns:
            SparseTensor: New sparse tensor with 'other' injected at the start of each batch
        """
        # Handle empty tensor cases
        if other.coords.shape[0] == 0:
            return self
        if self.coords.shape[0] == 0:
            return other

        # Check feature dimensions compatibility
        assert (
            self.feats.shape[1:] == other.feats.shape[1:]
        ), f"Feature shape mismatch: {self.feats.shape[1:]} vs {other.feats.shape[1:]}"

        current_batch_size = self.shape[0]

        # Filter other tensor to only include batches that exist in current tensor
        other_batch_indices = other.coords[:, 0]
        valid_mask = other_batch_indices < current_batch_size

        if not valid_mask.any():
            return self

        filtered_other_coords = other.coords[valid_mask]
        filtered_other_feats = other.feats[valid_mask]

        # Concatenate and sort to ensure proper batch and temporal ordering
        combined_coords = torch.cat([filtered_other_coords, self.coords], dim=0)
        combined_feats = torch.cat([filtered_other_feats, self.feats], dim=0)

        # Sort by (batch_idx, temporal_idx) - this handles empty batches correctly
        # Cached data will appear first within each batch due to earlier timestamps
        sort_key = (combined_coords[:, 0].long() << 20) + combined_coords[:, 1].long()
        sorted_indices = torch.argsort(sort_key)

        return SparseTensor(
            feats=combined_feats[sorted_indices], coords=combined_coords[sorted_indices]
        )

    def temporal_to_batch_transform(self, current_patch_size, target_feat_size=None):
        """
        Transform the temporal dimension into batch dimension by reshaping patches and features.
        """
        if len(current_patch_size) != 3:
            raise ValueError("Patch size must be a 3D tuple (T, X, Y)")

        T, X, Y = current_patch_size
        N, C = self.feats.shape

        # Validate feature dimension is divisible by temporal dimension
        if C % T != 0:
            raise ValueError(f"Feature dimension {C} must be divisible by temporal dimension {T}")

        new_feature_dim = C // T

        # Reshape features: [N, C] -> [N, T, C//T] -> [N*T, C//T]
        feats_reshaped = self.feats.reshape(N, T, new_feature_dim)
        new_feats = feats_reshaped.reshape(N * T, new_feature_dim)

        # Apply padding if target_feat_size is specified
        if target_feat_size is not None:
            if target_feat_size < new_feature_dim:
                raise ValueError(
                    f"Target feature size {target_feat_size} must be >= current feature size {new_feature_dim}"
                )

            if target_feat_size > new_feature_dim:
                # Pad features with zeros
                padding_size = target_feat_size - new_feature_dim
                padding = torch.zeros(
                    (new_feats.shape[0], padding_size),
                    dtype=new_feats.dtype,
                    device=new_feats.device,
                )
                new_feats = torch.cat([new_feats, padding], dim=1)
                new_feature_dim = target_feat_size

        # Repeat coordinates T times
        repeated_coords = self.coords.repeat_interleave(T, dim=0)  # [N*T, 5]

        # Create temporal indices: [0, 1, 2, 3, 0, 1, 2, 3, ...] for each original point
        temporal_indices = torch.arange(T, device=self.coords.device).repeat(N)  # [N*T]

        # Step 1: Expand temporal dimension by factor T and add temporal indices
        # Original time * T + temporal_index gives new expanded time coordinates
        repeated_coords[:, 1] = repeated_coords[:, 1] * T + temporal_indices

        # Step 2: Convert time dimension to batch dimension
        # New batch index = original_batch * T + time_index
        new_batch_indices = repeated_coords[:, 0] * T + repeated_coords[:, 1]
        repeated_coords[:, 0] = new_batch_indices

        # Set temporal coordinates to 0 (since temporal info is now in batch dimension)
        repeated_coords[:, 1] = 0

        # Sort by batch index to maintain proper ordering
        sorted_indices = torch.argsort(repeated_coords[:, 0])

        final_coords = repeated_coords[sorted_indices]
        final_feats = new_feats[sorted_indices]

        transformed_tensor = SparseTensor(feats=final_feats, coords=final_coords)

        # Create batch mapping: map each point to its new batch index
        changes = torch.cat(
            [
                torch.tensor([True], device=final_coords.device),
                final_coords[1:, 0] != final_coords[:-1, 0],
            ]
        )
        batch_mapping = self.coords.repeat_interleave(T, dim=0)[:, 0][changes]

        return transformed_tensor, batch_mapping


def sparse_batch_broadcast(input: SparseTensor, other: torch.Tensor) -> torch.Tensor:
    """
    Broadcast a 1D tensor to a sparse tensor along the batch dimension then perform an operation.

    Args:
        input (torch.Tensor): 1D tensor to broadcast.
        target (SparseTensor): Sparse tensor to broadcast to.
        op (callable): Operation to perform after broadcasting. Defaults to torch.add.
    """
    coords, feats = input.coords, input.feats
    broadcasted = torch.zeros_like(feats)
    for k in range(input.shape[0]):
        broadcasted[input.layout[k]] = other[k]
    return broadcasted


def sparse_batch_op(
    input: SparseTensor, other: torch.Tensor, op: callable = torch.add
) -> SparseTensor:
    """
    Broadcast a 1D tensor to a sparse tensor along the batch dimension then perform an operation.

    Args:
        input (torch.Tensor): 1D tensor to broadcast.
        target (SparseTensor): Sparse tensor to broadcast to.
        op (callable): Operation to perform after broadcasting. Defaults to torch.add.
    """
    return input.replace(op(input.feats, sparse_batch_broadcast(input, other)))


def sparse_cat(inputs: List[SparseTensor], dim: int = 0) -> SparseTensor:
    """
    Concatenate a list of sparse tensors.

    Args:
        inputs (List[SparseTensor]): List of sparse tensors to concatenate.
    """
    if dim == 0:
        start = 0
        coords = []
        for input in inputs:
            coords.append(input.coords.clone())
            coords[-1][:, 0] += start
            start += input.shape[0]
        coords = torch.cat(coords, dim=0)
        feats = torch.cat([input.feats for input in inputs], dim=0)
        output = SparseTensor(
            coords=coords,
            feats=feats,
        )
    else:
        feats = torch.cat([input.feats for input in inputs], dim=dim)
        output = inputs[0].replace(feats)

    return output


def sparse_unbind(input: SparseTensor, dim: int) -> List[SparseTensor]:
    """
    Unbind a sparse tensor along a dimension.

    Args:
        input (SparseTensor): Sparse tensor to unbind.
        dim (int): Dimension to unbind.
    """
    if dim == 0:
        return [input[i] for i in range(input.shape[0])]
    else:
        feats = input.feats.unbind(dim)
        return [input.replace(f) for f in feats]


# Standalone utility function (add to the module)
def reconstruct_from_temporal_slices(
    slices: List[SparseTensor],
    target_coords: Optional[torch.Tensor] = None,
    use_cached_offsets: bool = True,
) -> SparseTensor:
    """
    Reconstruct a sparse tensor from temporal slices.

    Args:
        slices (List[SparseTensor]): List of temporal slices
        target_coords (Optional[torch.Tensor]): Target coordinates to match exactly.
                                               If provided, the output will have these exact coordinates.
        use_cached_offsets (bool): Whether to use cached temporal offsets for reconstruction

    Returns:
        SparseTensor: Reconstructed sparse tensor with original temporal coordinates
    """
    if len(slices) == 0:
        raise ValueError("Cannot reconstruct from empty slice list")

    # Filter out any empty slices
    non_empty_slices = [s for s in slices if s.coords.shape[0] > 0]

    if len(non_empty_slices) == 0:
        # All slices are empty
        if target_coords is not None:
            # Return empty tensor with target coordinate structure
            empty_feats = torch.empty(
                (0,) + slices[0].feats.shape[1:],
                dtype=slices[0].feats.dtype,
                device=slices[0].feats.device,
            )
            return SparseTensor(
                feats=empty_feats, coords=target_coords[:0]
            )  # Empty with same structure
        else:
            # Return empty tensor based on the first slice structure
            first_slice = slices[0]
            empty_coords = torch.empty(
                (0, first_slice.coords.shape[1]),
                dtype=first_slice.coords.dtype,
                device=first_slice.coords.device,
            )
            empty_feats = torch.empty(
                (0,) + first_slice.feats.shape[1:],
                dtype=first_slice.feats.dtype,
                device=first_slice.feats.device,
            )
            return SparseTensor(feats=empty_feats, coords=empty_coords)

    # Process each slice to restore temporal coordinates
    restored_slices = []
    for slice_tensor in non_empty_slices:
        if use_cached_offsets:
            offset = slice_tensor.get_spatial_cache("temporal_offset")
            if offset is not None:
                new_coords = slice_tensor.coords.clone()
                new_coords[:, 1] += offset
                restored_slice = slice_tensor.replace(slice_tensor.feats, new_coords)
            else:
                restored_slice = slice_tensor
        else:
            restored_slice = slice_tensor

        restored_slices.append(restored_slice)

    # Collect all coordinates and features
    all_coords = []
    all_feats = []

    for restored_slice in restored_slices:
        all_coords.append(restored_slice.coords)
        all_feats.append(restored_slice.feats)

    # Concatenate everything
    combined_coords = torch.cat(all_coords, dim=0)
    combined_feats = torch.cat(all_feats, dim=0)

    if target_coords is not None:
        # Ensure exact match with target coordinates
        return _match_target_coordinates(combined_coords, combined_feats, target_coords)
    else:
        # Sort by (batch_idx, temporal_idx) to restore original order
        sort_key = (combined_coords[:, 0].long() << 20) + combined_coords[:, 1].long()
        sorted_indices = torch.argsort(sort_key)

        final_coords = combined_coords[sorted_indices]
        final_feats = combined_feats[sorted_indices]

        return SparseTensor(feats=final_feats, coords=final_coords)


def _match_target_coordinates(
    available_coords: torch.Tensor,
    available_feats: torch.Tensor,
    target_coords: torch.Tensor,
) -> SparseTensor:
    """
    Match available coordinates and features to target coordinates exactly.
    Assumes coordinates are the same but potentially shuffled.

    Args:
        available_coords: Coordinates from reconstructed slices
        available_feats: Features from reconstructed slices
        target_coords: Target coordinates to match

    Returns:
        SparseTensor with exact target coordinates
    """
    if len(available_coords) == 0:
        # No available data, return zeros for all target coordinates
        device = target_coords.device
        feat_dim = available_feats.shape[1:] if len(available_feats) > 0 else (1,)
        zero_feats = torch.zeros(
            (len(target_coords),) + feat_dim,
            dtype=available_feats.dtype if len(available_feats) > 0 else torch.float32,
            device=device,
        )
        return SparseTensor(feats=zero_feats, coords=target_coords)

    # Compute hash for available coordinates (vectorized)
    # Use a simple hash combining all coordinate dimensions
    available_hash = _compute_coord_hash(available_coords)
    target_hash = _compute_coord_hash(target_coords)

    # Find the mapping from target indices to available indices
    # This is much faster than nested loops
    available_hash_to_idx = {hash_val.item(): idx for idx, hash_val in enumerate(available_hash)}
    reorder_indices = torch.tensor(
        [available_hash_to_idx[hash_val.item()] for hash_val in target_hash],
        dtype=torch.long,
        device=available_coords.device,
    )

    # Reorder features to match target coordinate order
    reordered_feats = available_feats[reorder_indices]

    return SparseTensor(feats=reordered_feats, coords=target_coords)


def _compute_coord_hash(coords: torch.Tensor) -> torch.Tensor:
    """
    Compute hash values for coordinates efficiently using bit shifting.
    Optimized for 5D coordinates: batch, time, x, y, z

    Args:
        coords: Coordinate tensor of shape (N, 5) - [batch, time, x, y, z]

    Returns:
        Hash tensor of shape (N,)
    """
    # Use bit shifting - each dimension gets 12 bits (0-4095 range per dim)
    # This gives us plenty of room for typical sparse tensor coordinates
    # Total: 60 bits used out of 64-bit long integer

    hash_vals = coords[:, 0].long()  # batch
    hash_vals = (hash_vals << 12) + coords[:, 1].long()  # time
    hash_vals = (hash_vals << 12) + coords[:, 2].long()  # x
    hash_vals = (hash_vals << 12) + coords[:, 3].long()  # y
    hash_vals = (hash_vals << 12) + coords[:, 4].long()  # z

    return hash_vals
