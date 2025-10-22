#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2025 Apple Inc. All Rights Reserved.
#
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from .data_utils import decode_video_decord
from .sparse_preprocessors import process_image


class FolderDataset(Dataset):
    """Dataset that loads videos from a folder structure.

    Supports multiple folder structures:
    1. Simple: videos directly in folder
    2. With captions: videos + matching .txt files with captions
    4. Nested: videos in subfolders
    """

    def __init__(
        self,
        folder_path: str,
        caption_file: Optional[str] = None,
        video_extensions: Tuple[str, ...] = (".mp4", ".avi", ".mov", ".mkv", ".webm"),
        image_extensions: Tuple[str, ...] = (
            ".jpg",
            ".jpeg",
            ".png",
            ".bmp",
            ".tiff",
            ".tif",
            ".webp",
            ".gif",
        ),
        recursive: bool = False,
        max_frames: int = -1,
        max_words: int = 64,
        patch_size: Tuple[int, int, int] = (4, 16, 16),
        min_resolution: Tuple[int, int] = None,
        max_resolution: Tuple[int, int] = None,
        size_factor: int = 16,
        random_crop: bool = False,
        patch_sample_ratio: float = 1.0,
        max_seq_len: Optional[int] = None,
        is_training: bool = False,
        min_stride: int = 1,
        max_stride: int = 0,
        random_stride: Optional[List] = None,
        random_frame: Optional[List] = None,
    ):
        """Initialize FolderDataset.

        Args:
            video_folder: Path to folder containing videos
            caption_file: Optional path to JSON file with captions, or 'auto' to look for .txt files
            video_extensions: Tuple of valid video file extensions
            recursive: Whether to search subfolders recursively
            max_frames: Maximum number of video frames to extract
            max_words: Maximum number of text tokens
            patch_size: Patch size for sparse processing (temporal, height, width)
            min_resolution: Minimum resolution (height, width)
            max_resolution: Maximum resolution (height, width)
            size_factor: Size factor for processing
            random_crop: Whether to use random cropping
            patch_sample_ratio: Patch sampling ratio
            max_seq_len: Maximum sequence length
            is_training: Whether in training mode (affects video decoding)
            min_stride: Minimum stride between frames
            max_stride: Maximum stride between frames
            random_stride: Whether to use random stride
            random_frame: Whether to use random frame sampling
        """
        self.folder_path = folder_path
        self.caption_file = caption_file
        self.video_extensions = video_extensions
        self.image_extensions = image_extensions
        self.recursive = recursive

        # Video processing parameters
        self.max_frames = max_frames
        self.max_words = max_words
        self.patch_size = patch_size
        self.min_resolution = min_resolution
        self.max_resolution = max_resolution
        self.size_factor = size_factor
        self.random_crop = random_crop
        self.patch_sample_ratio = patch_sample_ratio
        self.max_seq_len = max_seq_len

        # Video decoding parameters
        self.is_training = is_training
        self.min_stride = min_stride
        self.max_stride = max_stride
        self.random_stride = random_stride
        self.random_frame = random_frame

        # Load dataset
        self.data = self._load_dataset()

    def _load_dataset(self) -> List[Dict]:
        """Load all images and videos from folder."""
        if not os.path.exists(self.folder_path):
            raise FileNotFoundError(f"Folder not found: {self.folder_path}")

        # Find all media files
        media_files = self._find_media_files()

        if not media_files:
            raise ValueError(f"No image or video files found in {self.folder_path}")

        # Build dataset
        data = []
        for file_path, file_type in media_files:
            # Get file ID from filename (without extension)
            file_id = os.path.splitext(os.path.basename(file_path))[0]

            data.append(
                {
                    "file_id": file_id,
                    "file_path": file_path,
                    "file_type": file_type,  # "image" or "video"
                }
            )

        # Count by type
        video_count = sum(1 for item in data if item["file_type"] == "video")
        image_count = sum(1 for item in data if item["file_type"] == "image")

        print(f"Loaded {len(data)} files from {self.folder_path}")
        print(f"  - {video_count} videos")
        print(f"  - {image_count} images")

        return data

    def _find_media_files(self) -> List[Tuple[str, str]]:
        """Find all image and video files in the folder.

        Returns:
            List of tuples (file_path, file_type) where file_type is "image" or "video"
        """
        media_files = []

        if self.recursive:
            # Search recursively
            for root, dirs, files in os.walk(self.folder_path):
                for file in files:
                    file_path = os.path.join(root, file)
                    file_type = self._get_file_type(file)
                    if file_type:
                        media_files.append((file_path, file_type))
        else:
            # Search only in the main folder
            for file in os.listdir(self.folder_path):
                file_path = os.path.join(self.folder_path, file)
                if os.path.isfile(file_path):
                    file_type = self._get_file_type(file)
                    if file_type:
                        media_files.append((file_path, file_type))

        # Sort for consistent ordering
        media_files.sort(key=lambda x: x[0])
        return media_files

    def _get_file_type(self, filename: str) -> Optional[str]:
        """Determine if file is image or video based on extension.

        Returns:
            "image", "video", or None if not a supported media file
        """
        filename_lower = filename.lower()

        if any(filename_lower.endswith(ext) for ext in self.video_extensions):
            return "video"
        elif any(filename_lower.endswith(ext) for ext in self.image_extensions):
            return "image"
        else:
            return None

    def _load_image(self, file_path: str) -> torch.Tensor:
        image = Image.open(file_path).convert("RGB")
        image = np.array(image)
        image = torch.from_numpy(image)

        return image

    def _load_video(self, file_path: str) -> torch.Tensor:
        # Extract video frames
        with open(file_path, "rb") as file:
            video_byte = file.read()

        data, _, _ = decode_video_decord(
            is_training=self.is_training,
            data=video_byte,
            num_to_select=self.max_frames,
            min_stride=self.min_stride,
            max_stride=self.max_stride,
            random_stride=self.random_stride,
            random_frame=self.random_frame,
        )
        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        """Get a single video sample."""
        item = self.data[idx]
        file_id = item["file_id"]
        file_path = item["file_path"]
        file_type = item["file_type"]

        # Load media based on type
        if file_type == "image":
            data = self._load_image(file_path)
        else:  # video
            data = self._load_video(file_path)

        data = data.float()  # Ensure data is float tensor
        # Process video using AToken sparse processing
        feats, coords = process_image(
            data,
            patch_size=self.patch_size,
            min_resolution=self.min_resolution,
            max_resolution=self.max_resolution,
            square_image_length=None,
            size_factor=self.size_factor,
            random_crop=self.random_crop,
            patch_sample_ratio=self.patch_sample_ratio,
            padding_type="zero",
            temporal_padding_to=self.patch_size[0],
            random_resolution=False,
        )

        # Check sequence length
        if self.max_seq_len and coords.shape[0] > self.max_seq_len:
            print(
                f"Warning: Sequence length {coords.shape[0]} exceeds maximum {self.max_seq_len} for video {file_id}"
            )
            # Truncate if needed
            feats = feats[: self.max_seq_len]
            coords = coords[: self.max_seq_len]

        return {
            "feats": feats,
            "coords": coords,
            "file_id": file_id,
            "file_path": file_path,
            "file_type": file_type,  # "image" or "video"
        }


def create_folder_dataloader(
    folder_path: str,
    batch_size: int = 8,
    num_workers: int = 0,
    shuffle: bool = False,
    video_extensions: Tuple[str, ...] = (".mp4", ".avi", ".mov", ".mkv", ".webm"),
    recursive: bool = False,
    max_frames: int = -1,
    max_words: int = 64,
    patch_size: Tuple[int, int, int] = (4, 16, 16),
    min_resolution: Tuple[int, int] = None,
    max_resolution: Tuple[int, int] = None,
    size_factor: int = 16,
    random_crop: bool = False,
    patch_sample_ratio: float = 1.0,
    max_seq_len: Optional[int] = None,
    is_training: bool = False,
    min_stride: int = 1,
    max_stride: int = 0,
    random_stride: Optional[List] = None,
    random_frame: Optional[List] = None,
) -> DataLoader:
    """Create a dataloader for videos in a folder.

    Args:
        folder_path: Path to folder containing videos
        caption_file: Path to caption file or 'auto' for .txt files
        batch_size: Batch size
        num_workers: Number of worker processes
        shuffle: Whether to shuffle the data
        video_extensions: Valid video file extensions
        recursive: Search subfolders recursively
        max_frames: Maximum number of video frames to extract
        max_words: Maximum number of text tokens
        patch_size: Patch size for sparse processing
        min_resolution: Minimum resolution
        max_resolution: Maximum resolution
        size_factor: Size factor for processing
        random_crop: Whether to use random cropping
        patch_sample_ratio: Patch sampling ratio
        max_seq_len: Maximum sequence length
        is_training: Whether in training mode (affects video decoding)
        min_stride: Minimum stride between frames
        max_stride: Maximum stride between frames
        random_stride: Whether to use random stride
        random_frame: Whether to use random frame sampling

    Returns:
        DataLoader instance

    """

    dataset = FolderDataset(
        folder_path=folder_path,
        video_extensions=video_extensions,
        recursive=recursive,
        max_frames=max_frames,
        max_words=max_words,
        patch_size=patch_size,
        min_resolution=min_resolution,
        max_resolution=max_resolution,
        size_factor=size_factor,
        random_crop=random_crop,
        patch_sample_ratio=patch_sample_ratio,
        max_seq_len=max_seq_len,
        is_training=is_training,
        min_stride=min_stride,
        max_stride=max_stride,
        random_stride=random_stride,
        random_frame=random_frame,
    )

    def collate_fn(batch):
        """Custom collate function for sparse tensors."""
        # Separate different components
        feats_list = [item["feats"] for item in batch]
        coords_list = [item["coords"] for item in batch]
        file_ids = [item["file_id"] for item in batch]
        file_paths = [item["file_path"] for item in batch]
        file_types = [item["file_type"] for item in batch]  # "image" or "video"

        return {
            "feats_list": feats_list,
            "coords_list": coords_list,
            "file_ids": file_ids,
            "file_paths": file_paths,
            "file_types": file_types,  # Include file types
        }

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    return dataloader
