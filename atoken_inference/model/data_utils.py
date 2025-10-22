#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2025 Apple Inc. All Rights Reserved.
#
# pylint: skip-file
# ruff: noqa

from typing import List, Union, Tuple, Optional
import random
import warnings

import io
import os

from enum import Enum
import math

import numpy as np
import torch
from einops import rearrange
from webdataset.pytorch import IterableDataset
import decord
import contextlib
import os
import sys
import decord
import imageio
from PIL import Image


class TestSampleMode(Enum):
    AUTO_STRIDE = "auto_stride"
    MIN_STRIDE = "min_stride"
    MAX_STRIDE = "max_stride"


def save_image(tensor, save_path, denormalize=True):
    """
    Save a tensor as an image file.

    Args:
        tensor: Tensor of shape (T, C, H, W) or (T, H, W, C) with values in [-1, 1] or [0, 1]
        save_path: Path to save the image
        denormalize: Whether to denormalize from [-1, 1] to [0, 255]
    """
    # Move to CPU and convert to numpy
    if isinstance(tensor, torch.Tensor):
        tensor = tensor.detach().float().cpu()

    if tensor.ndim == 4:
        tensor = tensor[0]

    # Handle different tensor shapes
    if tensor.ndim == 3:
        # Check if (C, H, W) or (H, W, C)
        if tensor.shape[0] == 3 or tensor.shape[0] == 1:
            # Assume (C, H, W), convert to (H, W, C)
            tensor = tensor.permute(1, 2, 0)

    # Convert to numpy
    img_np = tensor.numpy()

    # Denormalize if needed
    if denormalize:
        # Assume values are in [-1, 1], convert to [0, 255]
        img_np = ((img_np + 1) / 2 * 255).clip(0, 255).astype(np.uint8)
    else:
        # Assume values are in [0, 1], convert to [0, 255]
        img_np = (img_np * 255).clip(0, 255).astype(np.uint8)

    # Handle grayscale
    if img_np.shape[-1] == 1:
        img_np = img_np.squeeze(-1)

    # Save image
    img_pil = Image.fromarray(img_np)
    img_pil.save(save_path)
    print(f"Saved image to {save_path}")


def save_video(tensor, save_path, fps=30, denormalize=True):
    """
    Save tensor as a video file.

    Args:
        tensor: Tensor with shape (T, C, H, W) or (T, H, W, C) or list of (C, H, W)
        save_path: Path to save the video
        fps: Frames per second
        denormalize: Whether to denormalize from [-1, 1]
    """
    # Handle list input (original behavior)
    if isinstance(tensor, (list, tuple)):
        frames = []
        for t in tensor:
            if isinstance(t, torch.Tensor):
                t = t.detach().float().cpu()

            # Convert (C, H, W) to (H, W, C)
            if t.shape[0] == 3 or t.shape[0] == 1:
                t = t.permute(1, 2, 0)

            img_np = t.numpy()

            # Denormalize
            if denormalize:
                img_np = ((img_np + 1) / 2 * 255).clip(0, 255).astype(np.uint8)
            else:
                img_np = (img_np * 255).clip(0, 255).astype(np.uint8)

            frames.append(img_np)

    # Handle 4D tensor input
    else:
        if isinstance(tensor, torch.Tensor):
            tensor = tensor.detach().float().cpu()

        # Check shape: (T, C, H, W) or (T, H, W, C)
        if tensor.ndim == 4:
            # If second dimension is 1 or 3, it's (T, C, H, W)
            if tensor.shape[1] in [1, 3]:
                # Convert to (T, H, W, C)
                tensor = tensor.permute(0, 2, 3, 1)
            # Otherwise assume it's already (T, H, W, C)

        # Convert to numpy
        frames_np = tensor.numpy()

        # Denormalize
        if denormalize:
            frames_np = ((frames_np + 1) / 2 * 255).clip(0, 255).astype(np.uint8)
        else:
            frames_np = (frames_np * 255).clip(0, 255).astype(np.uint8)

        # Convert to list of frames for imageio
        frames = [frames_np[i] for i in range(frames_np.shape[0])]

    # Save as video
    imageio.mimsave(save_path, frames, fps=fps)
    print(f"Saved video with {len(frames)} frames to {save_path}")


def random_samples(sources, probs=None, longest=False):
    """Yield samples randomly from multiple sources based on given probabilities.

    Args:
        sources (list): List of iterable sources to draw samples from.
        probs (list, optional): List of probabilities for each source. Defaults to None.
        longest (bool): If True, continue until all sources are exhausted. Defaults to False.

    Yields:
        Sample randomly selected from one of the sources.
    """
    if probs is None:
        probs = [1] * len(sources)
    else:
        probs = list(probs)
    while len(sources) > 0:
        cum = (np.array(probs) / np.sum(probs)).cumsum()
        r = random.random()
        i = np.searchsorted(cum, r)
        try:
            yield next(sources[i])
        except StopIteration:
            if longest:
                del sources[i]
                del probs[i]
            else:
                break


class RandomMix(IterableDataset):
    """Iterate over multiple datasets by randomly selecting samples based on given probabilities."""

    def __init__(self, datasets, probs=None, longest=False, batch_size=None):
        """Initialize the RandomMix iterator.

        Args:
            datasets (list): List of datasets to iterate over.
            probs (list, optional): List of probabilities for each dataset. Defaults to None.
            longest (bool): If True, continue until all datasets are exhausted. Defaults to False.
        """
        self.datasets = datasets
        self.probs = probs
        self.longest = longest
        self.batch_size = batch_size

    def __iter__(self):
        """Return an iterator over the sources.

        Returns:
            iterator: An iterator that yields samples randomly from the datasets.
        """
        sources = [iter(d) for d in self.datasets]
        return random_samples(sources, self.probs, longest=self.longest)


def select_frame_indices(
    is_training: bool,
    num_frames_total: int,
    num_to_select: int,
    min_stride: int,
    max_stride: int = 0,
    random_frame: Optional[List] = None,
    test_sample_mode: TestSampleMode = TestSampleMode.MAX_STRIDE,
) -> Tuple[np.ndarray, int]:
    """
    Select frame indices for video sampling.

    Args:
        is_training: Whether in training mode
        num_frames_total: Total number of frames in video
        num_to_select: Number of frames to select
        min_stride: Minimum stride between frames
        max_stride: Maximum stride between frames
        test_sample_mode: Sampling mode for testing

    Returns:
        Tuple of (selected indices array, stride value)
    """
    if num_to_select == -1:
        num_to_select = num_frames_total

    if random_frame is not None:
        eligible_frames = [frame for frame in random_frame if frame < num_frames_total]
        if eligible_frames:
            num_to_select = random.choice(eligible_frames)
        else:
            num_to_select = num_frames_total

    if max_stride <= 0:
        # If max_stride is not provided, infer from the video
        max_stride = int(num_frames_total // num_to_select)

    if is_training:
        max_stride = max(1, min(max_stride, int(num_frames_total // num_to_select)))
        stride = np.random.randint(1, max_stride + 1)
        stride = max(stride, min_stride)

        start_limit = max(num_frames_total - num_to_select * stride, 1)
        start = np.random.randint(0, start_limit)
        indices = start + np.arange(num_to_select) * stride
    else:  # Testing mode
        if test_sample_mode == TestSampleMode.AUTO_STRIDE:
            # Create evenly spaced points
            bounds = np.linspace(0, num_frames_total - 1, num_to_select + 1)
            indices = (bounds[1:] + bounds[:-1]) / 2.0
            indices = np.round(indices).astype(np.int32)
        else:
            stride = int(num_frames_total // num_to_select)

            if test_sample_mode == TestSampleMode.MIN_STRIDE:
                stride = max(1, min_stride)
            elif test_sample_mode == TestSampleMode.MAX_STRIDE:
                stride = max(1, max_stride)

            indices = np.arange(num_to_select) * stride

        # Calculate stride
        stride = int(indices[1] - indices[0]) if num_to_select > 1 else 0

    # Ensure indices are within valid range
    indices = np.clip(indices, 0, num_frames_total - 1)
    return indices.astype(np.int32), stride


def get_valid_frame_mask(indices: np.ndarray) -> np.ndarray:
    """
    Create a mask for valid frames.

    Args:
        indices: Array of frame indices

    Returns:
        Boolean mask array of same shape as indices
    """
    return np.ones_like(indices, dtype=bool)


def decode_video_decord(
    is_training: bool,
    data: Union[bytes, np.ndarray],
    num_to_select: int,
    min_stride: int,
    max_stride: int = 0,
    random_stride: Optional[List] = None,
    random_frame: Optional[List] = None,
    test_sample_mode: TestSampleMode = TestSampleMode.MAX_STRIDE,
) -> Tuple[np.ndarray, int, np.ndarray]:
    """
    Decode video using decord library.

    Args:
        is_training: Whether in training mode
        data: Video data as bytes or numpy array
        num_to_select: Number of frames to select
        min_stride: Minimum stride between frames
        max_stride: Maximum stride between frames
        test_sample_mode: Sampling mode for testing
        random_frame: random select the num_to_select from the random_frame.

    Returns:
        Tuple of (selected frames array, stride value, valid frame mask)
    """
    # Convert data to bytes if it's numpy array
    if isinstance(data, np.ndarray):
        data = data.tobytes()

    buf = io.BytesIO(data)
    reader = decord.VideoReader(buf, num_threads=1)

    if random_stride is not None:
        assert min_stride == 0 and max_stride == 0
        sample_stride = random.choice(random_stride)
        min_stride = sample_stride
        max_stride = sample_stride

    indices, stride = select_frame_indices(
        is_training,
        len(reader),
        num_to_select,
        min_stride,
        max_stride,
        random_frame,
        TestSampleMode(test_sample_mode),
    )

    valid_frame_mask = get_valid_frame_mask(indices)
    # Use decord's native torch bridge
    decord.bridge.set_bridge("torch")
    frames = reader.get_batch(indices.tolist()).to("cpu")

    frames = frames.float()
    return frames, torch.tensor(stride, device="cpu"), valid_frame_mask


def decode_video_text_decord(
    data: Union[bytes, np.ndarray],
    num_to_select: int,
    min_stride: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Optimized function to decode video using decord library.

    Args:
        data: Video data as bytes or numpy array
        num_to_select: Number of frames to select
        min_stride: Minimum stride between frames

    Returns:
        Tuple of (selected frames tensor, stride value, valid frame mask)
    """
    # Use context manager to suppress specific warnings
    # Convert data to bytes if it's numpy array
    if isinstance(data, np.ndarray):
        data = data.tobytes()

    # Create buffer and reader
    buf = io.BytesIO(data)

    # Import decord here to allow for error handling if it can't be imported
    reader = decord.VideoReader(buf, num_threads=1)

    # Get video properties
    fps = max(1.0, reader.get_avg_fps())  # Ensure fps is at least 1
    total_frames = len(reader)

    # Calculate target frames
    target_frames = int(total_frames // fps)

    # Determine how many frames to sample
    if target_frames > num_to_select:
        sample_frames = num_to_select
    else:
        sample_frames = math.ceil(target_frames / min_stride) * min_stride

    # Safeguard against empty videos or division by zero
    if total_frames == 0 or sample_frames == 0:
        ValueError("No frames to sample from.")

    # Calculate frame indices
    indices = np.linspace(0, total_frames - 1, sample_frames, dtype=int)

    # Calculate stride
    stride = total_frames / max(1, sample_frames)

    # Get valid frame mask
    valid_frame_mask = get_valid_frame_mask(indices)

    # Set bridge and get frames
    decord.bridge.set_bridge("torch")

    # Use try/except to handle potential decoding errors
    frames = reader.get_batch(indices.tolist()).to("cpu").float()

    return frames, torch.tensor(stride, device="cpu"), valid_frame_mask


def parse_reduction_config(config: str) -> List[int]:
    reduction = []
    for item in config.split("-"):
        if "{" in item:
            factor, repeat = item.split("{")
            repeat = int(repeat[:-1])
            reduction.extend([int(factor)] * repeat)
        else:
            reduction.append(int(item))

    return reduction


def get_encoder_sampler_ids(
    num_patches,
    sr_cfg,
    tr_cfg,
    sampling_methods,
):
    """
    Given the 4 x 4 grid, we want to sampler the rel_ids that indice for each layer to
    mimic the down-sampling.


    """

    relative_ids_list = []
    absolute_ids_list = []
    feat_map_shape_list = []
    rel_reduction_list = []

    prev_sr = sr_cfg[0]
    prev_tr = tr_cfg[0]
    t, h, w = num_patches

    # create index map.
    index_map = np.arange(np.prod(num_patches)).reshape(num_patches)
    absolute_map = index_map
    weight_map = np.ones((t, h, w))

    for _, (sr, tr) in enumerate(zip(sr_cfg, tr_cfg)):
        if h == 1 or w == 1:
            rel_sr = 1
        else:
            rel_sr = sr // prev_sr

        if t == 1:
            rel_tr = 1
        else:
            rel_tr = tr // prev_tr

        rel_reduction_list.append([rel_sr, rel_tr])

        reshape_index = rearrange(
            index_map,
            "(t pt) (h ph) (w pw) -> (pt ph pw) t h w",
            pt=rel_tr,
            ph=rel_sr,
            pw=rel_sr,
        )
        reshape_weights = rearrange(
            weight_map,
            "(t pt) (h ph) (w pw) -> (pt ph pw) t h w",
            pt=rel_tr,
            ph=rel_sr,
            pw=rel_sr,
        )

        d, t, h, w = reshape_index.shape
        reshape_index = reshape_index.reshape(d, -1)
        reshape_weights = reshape_weights.reshape(d, -1)

        # get the relative index:
        if sampling_methods == "one":
            relative_ids = reshape_index[0]
        elif sampling_methods == "random":
            random_indices = np.random.randint(
                0, reshape_index.shape[0], size=reshape_index.shape[1]
            )
            relative_ids = reshape_index[random_indices, np.arange(reshape_index.shape[1])]
        else:
            raise ValueError("Not implement yet")

        # aboslute ids should be
        absolute_map = absolute_map.reshape(-1)[relative_ids]
        index_map = np.arange(np.prod([t, h, w])).reshape([t, h, w])
        prev_tr = tr
        prev_sr = sr

        relative_ids_list.append(relative_ids)
        absolute_ids_list.append(absolute_map)
        feat_map_shape_list.append(index_map.shape)

    return relative_ids_list, absolute_ids_list, feat_map_shape_list, rel_reduction_list


def get_decoder_sampler_ids(
    enc_rel_ids, enc_abs_ids, enc_shape, rel_reduction_list, sampling_methods
):
    # reverse need to copy the first layer and drop last.
    absolute_ids_list = enc_abs_ids[::-1]
    enc_shape_reverse = enc_shape[::-1]

    relative_ids_list = []
    for i, (rel_sr, rel_tr) in enumerate(rel_reduction_list[::-1]):
        index_map = np.arange(np.prod(enc_shape_reverse[i])).reshape(enc_shape_reverse[i])

        if i == 0:
            relative_ids_list.append(index_map.flatten())
        t, h, w = index_map.shape

        # expand the index_map based on sr and tr.
        index_map_expand = index_map.reshape(1, -1).repeat(rel_tr * rel_sr**2, 0)
        index_map_expand = rearrange(
            index_map_expand,
            "(pt ph pw) (t h w) -> (t pt) (h ph) (w pw)",
            pt=rel_tr,
            ph=rel_sr,
            pw=rel_sr,
            t=t,
            h=h,
            w=w,
        )
        index_map_expand = index_map_expand.flatten()
        relative_ids_list.append(index_map_expand)

    relative_ids_list = relative_ids_list[:-1]
    return relative_ids_list, absolute_ids_list


def pad_enc_or_abs_indices(enc_indices, target_seq_length, max_row, is_abs_index=False):
    enc_indices_padded = []
    for layer in enc_indices:
        # accumulated rel indices
        current_length = 0
        padded_layer = []
        for indice in layer:
            if not is_abs_index:
                # shift the relateive index
                indice = indice + current_length

            # pad with -1 on row
            indice = np.pad(
                indice,
                pad_width=(
                    (0, max_row - indice.shape[0]),
                    (0, 0),
                ),  # Add 3 rows below, no padding for columns
                mode="constant",
                constant_values=-1,
            )
            padded_layer.append(indice)
            current_length += indice.shape[1]

        padded_layer = np.concatenate(padded_layer, axis=-1)
        # pad to maximum seq length
        pad_length = target_seq_length - padded_layer.shape[1]

        padded_layer = np.concatenate([padded_layer, np.zeros([max_row, pad_length]) - 1], axis=-1)
        enc_indices_padded.append(padded_layer)

    enc_indices_padded = np.stack(enc_indices_padded, axis=0)

    return enc_indices_padded


def pad_dec_and_rel_indices(dec_indices, target_seq_length, is_abs_index=False):
    dec_indices_padded = []
    for layer in dec_indices:
        # accumulated rel indices
        current_length = 0
        padded_layer = []
        for indice in layer:
            if not is_abs_index:
                # shift the relateive index x only
                indice = indice + [[current_length], [0]]
            padded_layer.append(indice)
            current_length += np.max(indice[0]) + 1

        padded_layer = np.concatenate(padded_layer, axis=-1)
        # pad to maximum seq length
        pad_length = target_seq_length - padded_layer.shape[1]

        padded_layer = np.concatenate([padded_layer, np.zeros([2, pad_length]) - 1], axis=-1)
        dec_indices_padded.append(padded_layer)

    dec_indices_padded = np.stack(dec_indices_padded, axis=0)
    return dec_indices_padded


def pad_indices(indices, target_seq_length):
    indices_padded = []
    offsets = []
    for layer in indices:
        current_length = 0
        offset = [
            0,
        ]
        for seq in layer:
            current_length = seq.shape[0] + current_length
            offset.append(current_length)
        offsets.append(np.array(offset))
    # repeat the first
    offsets.insert(0, offsets[0])

    for layer_id, layer in enumerate(indices):
        # accumulated rel indices
        padded_layer = []
        for seq_id, seq in enumerate(layer):
            # if not is_abs_index:
            # shift the relateive index
            seq = seq + offsets[layer_id][seq_id]

            padded_layer.append(seq)

        padded_layer = np.concatenate(padded_layer, axis=-1)
        # pad to maximum seq length
        pad_length = target_seq_length - padded_layer.shape[0]
        padded_layer = np.concatenate([padded_layer, np.zeros([pad_length]) - 1], axis=-1)
        indices_padded.append(padded_layer)

    indices_padded = np.stack(indices_padded, axis=0)
    return indices_padded
