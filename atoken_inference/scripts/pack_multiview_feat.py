#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2025 Apple Inc. All Rights Reserved.
#

"""
Multi-view Feature Volume Packing for Single Objects

This script demonstrates how to pack a single 3D object into a feature volume
representation using multiple rendered views (images + depth maps).

The process:
1. Load multi-view images and depth maps of an object
2. Extract features from images (RGB patches)
3. Project features into 3D space using camera parameters
4. Create a voxel grid and assign features to occupied voxels
5. Save the resulting feature volume (coords + features)

Example usage:
    # Prepare your data
    all_images = [Image.open(f"{i:03d}.png") for i in range(16)]
    all_depth = np.stack([np.load(f"{i:03d}_depth.npy") for i in range(16)])
    intr, w2c = convert_cam_poses("transforms.json")

    # Compute feature volume
    feats, coords = compute_feat_coords_pair(
        all_images=all_images,
        all_depth=all_depth,
        all_w2c=w2c,
        all_intrinsics=intr,
        vol_resolution=64,
        bounding_radius=0.5,
    )

    # Save results
    np.save("object.feats.npy", feats.cpu().numpy().astype(np.float16))
    np.save("object.coords.npy", coords.int().cpu().numpy())

For lib utils3d, run:
    pip install git+https://github.com/EasternJournalist/utils3d.git@9a4eb15e4021b67b12c460c7057d642626897ec8
"""

from typing import List, Union, Tuple, Callable
import json

import numpy as np
import cv2
import torch
from einops import rearrange
from PIL import Image
import utils3d


# Image processing constants
IMAGE_SIZE = 256
PATCH_SIZE = 16


class MvFeatExtractor:
    """Multi-view feature extractor using RGB patches.

    This simplified version extracts features by rearranging RGB images into patches.
    For more advanced features, you could replace this with a vision model like DINOv2.

    Args:
        processing_batch_size: Number of images to process at once.
    """

    def __init__(self, processing_batch_size: int = 16):
        self.processing_batch_size = processing_batch_size

    def __call__(self, image_list: List[Image.Image]) -> torch.Tensor:
        """Extract features from a list of images.

        Args:
            image_list: List of PIL images.

        Returns:
            Feature tensor of shape [N, C, H, W] where:
                N = number of images
                C = feature channels (PATCH_SIZE * PATCH_SIZE * 3 for RGB)
                H, W = spatial dimensions (IMAGE_SIZE / PATCH_SIZE)
        """
        feature_list = []
        start = 0
        while start < len(image_list):
            end = min(start + self.processing_batch_size, len(image_list))
            feature_list.append(self.process_batch(image_list[start:end]))
            start = end
        return torch.cat(feature_list, dim=0)

    @staticmethod
    def process_pil_image(image: Image.Image, bg_color: float = 0) -> torch.Tensor:
        """Process a single PIL image: resize, add background, normalize.

        Args:
            image: Input PIL image.
            bg_color: Background color (0-1 range).

        Returns:
            Processed image tensor [3, IMAGE_SIZE, IMAGE_SIZE].
        """
        image = image.resize((IMAGE_SIZE, IMAGE_SIZE), Image.LANCZOS)
        bg_color = int(bg_color * 255)
        bg_img = Image.new(
            "RGB", (IMAGE_SIZE, IMAGE_SIZE), (bg_color, bg_color, bg_color)
        )
        bg_img.paste(image, (0, 0))
        image = np.array(bg_img.convert("RGB")).astype(np.float32) / 255
        image = torch.from_numpy(image).permute(2, 0, 1).float()
        return image

    @torch.inference_mode()
    def process_batch(self, image_list: List[Image.Image]) -> torch.Tensor:
        """Process a batch of images and extract patch-based features.

        Args:
            image_list: List of PIL images.

        Returns:
            Feature tensor [N, PATCH_SIZE*PATCH_SIZE*3, H, W].
        """
        image = [self.process_pil_image(i) for i in image_list]
        image = torch.stack(image)  # [N, 3, IMAGE_SIZE, IMAGE_SIZE]
        # Rearrange into patches: split image into PATCH_SIZE x PATCH_SIZE blocks
        return rearrange(
            image, "N C (h p1) (w p2) -> N (p1 p2 C) h w", p1=PATCH_SIZE, p2=PATCH_SIZE
        )


def proj_world_to_view_uv(
    world_coord: torch.Tensor,
    intrinsics: torch.Tensor,
    w2c: torch.Tensor,
    return_depth: bool = False,
) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    """Project 3D world coordinates to 2D view UV coordinates.

    Args:
        world_coord: World coordinates [bs, n_points, 3].
        intrinsics: Camera intrinsics [bs, 3, 3].
        w2c: World-to-camera matrices [bs, 4, 4].
        return_depth: If True, also return projected depths.

    Returns:
        uv_coords: View UV coordinates [bs, n_points, 2].
        depths (optional): Projected depths [bs, n_points, 1].
    """
    bs, n_points, _ = world_coord.shape
    # Convert to homogeneous coordinates
    homo_coords = torch.cat(
        [world_coord, torch.ones(bs, n_points, 1, device=world_coord.device)], dim=-1
    )
    homo_coords = homo_coords.transpose(1, 2)  # [bs, 4, n_points]

    # Transform to camera coordinates
    cam_coords = torch.bmm(w2c, homo_coords)[:, :3]  # [bs, 3, n_points]

    # Project to image plane
    proj_coords = torch.bmm(intrinsics, cam_coords)  # [bs, 3, n_points]
    proj_coords = proj_coords.transpose(1, 2)  # [bs, n_points, 3]

    # Normalize by depth to get UV coordinates
    uv_coords = proj_coords[..., :2] / proj_coords[..., 2:3]  # [bs, n_points, 2]

    if return_depth:
        return uv_coords, proj_coords[..., 2:3]
    return uv_coords


def proj_view_depth_to_world(
    depth: torch.Tensor, intrinsics: torch.Tensor, w2c: torch.Tensor
) -> torch.Tensor:
    """Project depth maps to 3D world coordinates (OpenGL convention).

    Args:
        depth: Depth maps [bs, H, W].
        intrinsics: Camera intrinsics [bs, 3, 3].
        w2c: World-to-camera matrices [bs, 4, 4].

    Returns:
        World coordinates [bs, H, W, 3].
    """
    bs, height, width = depth.shape
    device = depth.device

    # Create pixel grid
    x, y = torch.meshgrid(
        torch.arange(height, device=device) / height,
        torch.arange(width, device=device) / width,
        indexing="xy",
    )
    pixels = torch.stack([x, y], dim=-1).reshape(-1, 2)
    pixels = pixels.unsqueeze(0).repeat(bs, 1, 1)  # [bs, H*W, 2]

    # Create homogeneous pixel coordinates scaled by depth
    depths = depth.view(bs, -1, 1)  # [bs, H*W, 1]
    homo_pixels = torch.cat(
        [pixels, torch.ones(bs, height * width, 1, device=device)], dim=-1
    )
    homo_pixels = homo_pixels * depths  # [bs, H*W, 3]

    # Unproject to camera coordinates: K_inv @ point
    intrinsics_inv = torch.inverse(intrinsics.cpu()).to(device)
    homo_pixels = homo_pixels.transpose(1, 2)  # [bs, 3, H*W]
    cam_coords = torch.bmm(intrinsics_inv, homo_pixels)  # [bs, 3, H*W]

    # Transform to world coordinates: w2c_inv @ cam_coord
    cam_coords = torch.cat(
        [cam_coords, torch.ones([bs, 1, height * width], device=device)], dim=1
    )
    w2c_inv = torch.inverse(w2c.cpu()).to(device)
    world_coords = torch.bmm(w2c_inv, cam_coords)[:, :3]  # [bs, 3, H*W]
    world_coords = world_coords.transpose(1, 2)  # [bs, H*W, 3]
    world_coords = world_coords.view(bs, height, width, 3)  # [bs, H, W, 3]
    return world_coords


def convert_all_depths(all_depths: List[str]) -> List[str]:
    """Convert depth .exr files to .npy format.

    Args:
        all_depths: List of .exr depth file paths.

    Returns:
        List of converted .npy file paths.
    """
    for depth_file in all_depths:
        depth = cv2.imread(
            depth_file,
            cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH,
        )[..., 0]
        depth[depth > 1e6] = -1  # Mark invalid depths
        np.save(
            depth_file.replace(".exr", ".npy"),
            depth.astype(np.float16),
        )
    return [depth_file.replace(".exr", ".npy") for depth_file in all_depths]


def convert_cam_poses(cam_json: str) -> Tuple[np.ndarray, np.ndarray]:
    """Convert camera poses from NeRF-style JSON to intrinsics and extrinsics.

    Args:
        cam_json: Path to transforms.json file.

    Returns:
        cam_intr_list: Camera intrinsics [N, 3, 3].
        cam_extr_list: Camera extrinsics (world-to-camera) [N, 4, 4].
    """
    with open(cam_json, "r", encoding="utf-8") as json_file:
        cams = json.load(json_file)
    num_total_imgs = len(cams["frames"])

    cam_intr_list = []
    cam_extr_list = []
    for frame_idx in range(num_total_imgs):
        frame_cam = cams["frames"][frame_idx]
        assert (
            frame_cam["file_path"] == f"{frame_idx:03d}.png"
        ), f"Expected {frame_idx:03d}.png, got {frame_cam['file_path']}"

        # Convert camera-to-world to world-to-camera
        c2w = torch.tensor(frame_cam["transform_matrix"])
        c2w[:3, 1:3] *= -1  # Flip Y and Z axes for coordinate system conversion
        extrinsics = torch.inverse(c2w)

        # Get intrinsics from field of view
        fov = frame_cam["camera_angle_x"]
        intrinsics = utils3d.torch.intrinsics_from_fov_xy(
            torch.tensor(fov), torch.tensor(fov)
        )

        cam_intr_list.append(intrinsics)
        cam_extr_list.append(extrinsics)

    return np.stack(cam_intr_list), np.stack(cam_extr_list)


class MvFeatVol:
    """Multi-view feature volume builder.

    This class takes multi-view images, depths, and camera parameters to build
    a 3D feature volume representation. The volume is a sparse voxel grid where
    each occupied voxel contains features aggregated from multiple views.

    Args:
        all_images: List of image paths or PIL images.
        all_depth: Depth maps [N, H, W].
        all_w2c: World-to-camera matrices [N, 4, 4].
        all_intrinsics: Camera intrinsics [N, 3, 3].
        get_feature_fn: Function to extract features from images.
        vol_resolution: Voxel grid resolution (default: 64).
        bounding_radius: Half-size of the bounding box (default: 0.5).
    """

    max_surface_points = 500_000  # Limit for memory efficiency

    def __init__(
        self,
        all_images: List[Union[str, Image.Image]],
        all_depth: np.ndarray,
        all_w2c: np.ndarray,
        all_intrinsics: np.ndarray,
        get_feature_fn: Callable,
        vol_resolution: int = 64,
        bounding_radius: float = 0.5,
    ):
        self.all_images = all_images
        self.get_feature_fn = get_feature_fn
        self.vol_resolution = vol_resolution

        # Extract features from all images
        if isinstance(all_images[0], str):
            all_images = [Image.open(image_name) for image_name in all_images]
        else:
            assert isinstance(
                all_images[0], Image.Image
            ), f"Unknown image type: {type(all_images[0])}."

        self.all_feat = get_feature_fn(all_images).contiguous()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.all_feat = self.all_feat.to(self.device)
        self.dtype = self.all_feat.dtype

        # Initialize camera and depth data
        self.all_depth = torch.tensor(all_depth, device=self.device, dtype=self.dtype)
        self.all_w2c = torch.tensor(all_w2c, device=self.device, dtype=self.dtype)
        self.all_intrinsics = torch.tensor(
            all_intrinsics, device=self.device, dtype=self.dtype
        )

        # Initialize voxel grid
        self.grid_coord_tl, self.grid_coord_br = self._init_volume_grid_coord(
            vol_resolution, bounding_radius
        )
        self.grid_coord_center = (self.grid_coord_tl + self.grid_coord_br) / 2
        self.grid_radius = (
            self.grid_coord_br[0, 0, 0] - self.grid_coord_tl[0, 0, 0]
        ) / 2

        # Determine occupancy from surface points
        self.all_surface_points = self._init_surface_points()
        self.grid_occupied = self._init_grid_occupied_by_surface_points()

    def _init_surface_points(self) -> torch.Tensor:
        """Extract 3D surface points from depth maps.

        Returns:
            Surface points [N_points, 3].
        """
        empty_depth_mask = 0 > self.all_depth
        all_world_coord = proj_view_depth_to_world(
            self.all_depth, self.all_intrinsics, self.all_w2c
        )
        all_points = all_world_coord[~empty_depth_mask, :]

        # Subsample if too many points
        if all_points.shape[0] > self.max_surface_points:
            all_points = all_points[
                torch.randperm(all_points.shape[0])[: self.max_surface_points]
            ]
        return all_points

    def _init_volume_grid_coord(
        self, vol_resolution: int, bounding_radius: float
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Initialize voxel grid coordinates.

        Args:
            vol_resolution: Grid resolution.
            bounding_radius: Half-size of bounding box.

        Returns:
            grid_coord_topleft: Top-left corners [vol_res, vol_res, vol_res, 3].
            grid_coord_bottomright: Bottom-right corners [vol_res, vol_res, vol_res, 3].
        """
        grid_interval = 2 * bounding_radius / vol_resolution
        grid_coord_topleft = torch.stack(
            torch.meshgrid(
                *[
                    torch.linspace(
                        -bounding_radius,
                        bounding_radius - grid_interval,
                        vol_resolution,
                        device=self.device,
                    )
                ]
                * 3,
                indexing="ij",
            ),
            dim=-1,
        )
        grid_coord_bottomright = grid_coord_topleft + grid_interval
        return grid_coord_topleft, grid_coord_bottomright

    def _init_grid_occupied_by_surface_points(
        self, processing_batch_size: int = 512
    ) -> torch.Tensor:
        """Determine which voxels are occupied based on surface points.

        A voxel is considered occupied if any surface point falls within it.

        Args:
            processing_batch_size: Batch size for processing.

        Returns:
            Grid occupancy mask [vol_resolution, vol_resolution, vol_resolution].
        """
        start = 0
        grid_center_in_range = torch.zeros(
            self.vol_resolution,
            self.vol_resolution,
            self.vol_resolution,
            device=self.device,
        ).bool()
        to_process_mask = torch.ones_like(grid_center_in_range).bool()

        while start < self.all_surface_points.shape[0]:
            end = min(start + processing_batch_size, self.all_surface_points.shape[0])

            # Check which grid cells contain surface points
            grid_center_to_all_point_xyz = (
                self.grid_coord_center[to_process_mask, :].unsqueeze(0)
                - self.all_surface_points[start:end, None, None, None]
            ).abs()

            grid_center_in_range_batch = (
                grid_center_to_all_point_xyz <= self.grid_radius
            ).all(dim=-1)
            grid_center_in_range_batch = grid_center_in_range_batch.any(dim=0)

            grid_center_in_range[to_process_mask] = (
                grid_center_in_range[to_process_mask] | grid_center_in_range_batch
            )
            to_process_mask = ~grid_center_in_range
            start = end

        return grid_center_in_range

    @torch.inference_mode()
    def get_feature_volume(self, processing_batch_size: int = 512) -> torch.Tensor:
        """Compute feature volume by projecting features to occupied voxels.

        For each occupied voxel:
        1. Project its center to all camera views
        2. Sample features from the nearest view
        3. Assign the sampled feature to the voxel

        Args:
            processing_batch_size: Batch size for processing.

        Returns:
            Features for occupied voxels [num_valid_grid, channel].
        """
        valid_world_coord = self.grid_coord_center[self.grid_occupied]
        all_view_uv_coord, all_view_proj_depth = proj_world_to_view_uv(
            valid_world_coord.unsqueeze(0).repeat(self.all_depth.shape[0], 1, 1),
            self.all_intrinsics,
            self.all_w2c,
            return_depth=True,
        )
        # all_view_uv_coord: [num_cam, num_valid_grid, 2]
        # all_view_proj_depth: [num_cam, num_valid_grid, 1]

        # Find nearest view for each voxel
        all_nearest_view = all_view_proj_depth[..., 0].argmin(dim=0)  # [num_valid_grid]

        # Normalize UV coordinates to [-1, 1] for grid_sample
        canonical_coord = (all_view_uv_coord * 2 - 1).unsqueeze(1).contiguous()
        # Shape: [num_cam, 1, num_valid_grid, 2]

        # Sample features from nearest views
        all_nearest_feat = []
        start = 0
        while start < canonical_coord.shape[2]:
            end = min(start + processing_batch_size, canonical_coord.shape[2])

            view_feat = torch.nn.functional.grid_sample(
                self.all_feat,
                canonical_coord[:, :, start:end],
                align_corners=True,
                padding_mode="zeros",
                mode="nearest",
            ).squeeze(2)  # [num_cam, channel, batch_size]

            nearest = all_nearest_view[start:end]
            view_feat_nearest = view_feat.gather(
                0, nearest.view(1, 1, -1).expand(-1, view_feat.shape[1], -1)
            )[0]
            all_nearest_feat.append(view_feat_nearest)
            start = end

        all_nearest_feat = torch.cat(all_nearest_feat, dim=1)
        return all_nearest_feat.permute(1, 0)  # [num_valid_grid, channel]


def compute_feat_coords_pair(
    all_images: List[Union[str, Image.Image]],
    all_depth: np.ndarray,
    all_w2c: np.ndarray,
    all_intrinsics: np.ndarray,
    vol_resolution: int = 64,
    bounding_radius: float = 0.5,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Helper function to compute feature and coordinate pairs for a single object.

    This is the main entry point for processing a single object.

    Args:
        all_images: List of image paths or PIL images.
        all_depth: Depth maps [N, H, W].
        all_w2c: World-to-camera matrices [N, 4, 4].
        all_intrinsics: Camera intrinsics [N, 3, 3].
        vol_resolution: Voxel grid resolution (default: 64).
        bounding_radius: Half-size of bounding box (default: 0.5).

    Returns:
        avg_feat: Features for occupied voxels [num_valid_grid, channel].
        coords: Integer voxel coordinates [num_valid_grid, 3].
    """
    mv_feat_vol = MvFeatVol(
        all_images=all_images,
        all_depth=all_depth,
        all_w2c=all_w2c,
        all_intrinsics=all_intrinsics,
        get_feature_fn=MvFeatExtractor(),
        vol_resolution=vol_resolution,
        bounding_radius=bounding_radius,
    )
    avg_feat = mv_feat_vol.get_feature_volume()
    coords = torch.argwhere(mv_feat_vol.grid_occupied).int()
    return avg_feat, coords


if __name__ == "__main__":
    # Example usage for a single object
    import os

    # Example paths (adjust to your data)
    data_folder = "path/to/your/data"
    num_views = 16

    # Load images and depths
    all_images = [
        Image.open(os.path.join(data_folder, f"{i:03d}.png"))
        for i in range(num_views)
    ]
    all_depths = [
        os.path.join(data_folder, f"{i:03d}_depth.exr")
        for i in range(num_views)
    ]
    all_depths = convert_all_depths(all_depths)
    depth = np.stack([np.load(depth_file) for depth_file in all_depths])

    # Load camera parameters
    intr, w2c = convert_cam_poses(os.path.join(data_folder, "transforms.json"))

    # Compute feature volume
    print("Computing feature volume...")
    feats, coords = compute_feat_coords_pair(
        all_images=all_images,
        all_depth=depth,
        all_w2c=w2c,
        all_intrinsics=intr,
        vol_resolution=64,
        bounding_radius=0.5,
    )

    # Save results
    print(f"Occupied voxels: {coords.shape[0]}")
    print(f"Feature shape: {feats.shape}")
    np.save("object.feats.npy", feats.cpu().numpy().astype(np.float16))
    np.save("object.coords.npy", coords.int().cpu().numpy())
    print("Saved object.feats.npy and object.coords.npy")
