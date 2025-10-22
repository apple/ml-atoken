#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2025 Apple Inc. All Rights Reserved.
#
# ruff: noqa
# pylint: skip-file
from typing import Dict, Callable, List, Optional, Union
from functools import partial

import random
import ftfy
import numpy as np
import regex as re
import html
import string

import io
import json
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from einops import rearrange

from .data_utils import (
    decode_video_decord,
    decode_video_text_decord,
)

from transformers import AutoTokenizer


class SigLipTokenizer:
    """HuggingFace tokenizer wrapper for SigLIP T5 compatible sentencepiece vocabs

    NOTE: this is not needed in normal library use, but is used to import new sentencepiece tokenizers
    into OpenCLIP. Leaving code here in case future models use new tokenizers.
    """

    VOCAB_FILES = {
        # used in SigLIP2 models, vocab_size=256000
        "gemma": "google/siglip2-base-patch16-naflex",
    }

    def __init__(
        self,
        tokenizer_name: str,
        context_length: Optional[int] = 64,
    ):
        self.tokenizer = AutoTokenizer.from_pretrained(self.VOCAB_FILES[tokenizer_name])

        self.tokenizer.pad_token_id = 0 if "gemma" in tokenizer_name else 1
        self.tokenizer.eos_token_id = 1
        self.context_length = context_length

    def save_pretrained(self, dest):
        self.tokenizer.save_pretrained(dest)

    def __call__(
        self, texts: Union[str, List[str]], context_length: Optional[int] = None
    ) -> torch.Tensor:
        # same cleaning as for default tokenizer, except lowercasing
        # adding lower (for case-sensitive tokenizers) will make it more robust but less sensitive to nuance
        if isinstance(texts, str):
            texts = [texts]

        context_length = context_length or self.context_length
        assert context_length, "Please set a valid context length in class init or call."

        texts = [canonicalize_text(basic_clean(text)) for text in texts]
        output = self.tokenizer(
            texts,
            return_tensors="pt",
            max_length=context_length,
            padding="max_length",
            truncation=True,
        )
        return output.input_ids


def filter_video_samples(sample):
    """
    Filter out samples with 0 bytes video
    """
    return len(sample.get("video", b"")) > 0


def special_decode_bytes_tensor(tensor_bytes):
    # If we're dealing with a TensorFlow tensor, convert to numpy
    if hasattr(tensor_bytes, "numpy"):
        tensor_bytes = tensor_bytes.numpy()

    # If it's a bytes object, decode it directly
    if isinstance(tensor_bytes, bytes):
        return tensor_bytes.decode("utf-8")

    # If it's already a list of ASCII values as in your output
    if isinstance(tensor_bytes, list) and all(isinstance(x, int) for x in tensor_bytes):
        return "".join(chr(x) for x in tensor_bytes)

    return tensor_bytes


def basic_clean(text):
    text = ftfy.fix_text(text)
    text = html.unescape(html.unescape(text))
    return text.strip()


def canonicalize_text(
    text,
    *,
    keep_punctuation_exact_string=None,
    trans_punctuation: dict = str.maketrans("", "", string.punctuation),
):
    """Returns canonicalized `text` (lowercase and punctuation removed).

    From: https://github.com/google-research/big_vision/blob/53f18caf27a9419231bbf08d3388b07671616d3d/big_vision/evaluators/proj/image_text/prompt_engineering.py#L94

    Args:
      text: string to be canonicalized.
      keep_punctuation_exact_string: If provided, then this exact string kept.
        For example providing '{}' will keep any occurrences of '{}' (but will
        still remove '{' and '}' that appear separately).
    """
    text = text.replace("_", " ")
    if keep_punctuation_exact_string:
        text = keep_punctuation_exact_string.join(
            part.translate(trans_punctuation) for part in text.split(keep_punctuation_exact_string)
        )
    else:
        text = text.translate(trans_punctuation)
    text = text.lower()
    text = " ".join(text.split())
    return text.strip()


def process_image(
    image,
    patch_size,
    min_resolution,
    max_resolution,
    square_image_length,
    size_factor,
    random_crop=True,
    random_time=0,
    patch_sample_ratio=[1.0, 1.0],
    padding_type="zero",
    temporal_padding_to=1,
    random_resolution=True,
):
    if len(image.shape) == 3:  # Single image: (H, W, C)
        image = image.unsqueeze(0)  # Add batch dimension: (1, H, W, C)

    T, H, W, D = image.shape
    Pt, Ph, Pw = patch_size  # Unpack Ph and Pw from patch_size

    if T < temporal_padding_to:
        if padding_type == "zero":
            # Zero padding
            pad = (torch.zeros((temporal_padding_to - T, H, W, D), dtype=image.dtype) + 127.5).to(
                image
            )
            image = torch.cat([image, pad], dim=0)
        else:
            # T is always 1, so just repeat the frame
            image = image.repeat(temporal_padding_to, 1, 1, 1)
        T = temporal_padding_to

    # Ensure dimensions are divisible by size_factor Pt.
    T = T // Pt * Pt
    image = image[:T]

    # Only resize if dimensions are outside min/max bounds
    resize_needed = False

    if min_resolution is None and max_resolution is None:
        min_resolution = (H, W)
        max_resolution = (H, W)

    if square_image_length is None:
        if random_resolution:
            if H < min_resolution[0] or W < min_resolution[1]:
                h_scale = min_resolution[0] / H
                w_scale = min_resolution[1] / W
                scale_factor = max(h_scale, w_scale)
            else:
                # Randomly select a resolution scaling factor
                h_scale = np.random.uniform(min_resolution[0] / H, 1.0)
                w_scale = np.random.uniform(min_resolution[1] / W, 1.0)
                if (
                    round(H * min(h_scale, w_scale)) > min_resolution[0]
                    and round(W * min(h_scale, w_scale)) > min_resolution[1]
                ):
                    scale_factor = min(h_scale, w_scale)
                else:
                    scale_factor = max(h_scale, w_scale)

            resize_H = round(H * scale_factor)
            resize_W = round(W * scale_factor)
            resize_needed = True

        else:
            if H < min_resolution[0] or W < min_resolution[1]:
                h_scale = min_resolution[0] / H
                w_scale = min_resolution[1] / W
                scale_factor = max(h_scale, w_scale)

                resize_H = round(H * scale_factor)
                resize_W = round(W * scale_factor)

                resize_needed = True
            elif H > max_resolution[0] or W > max_resolution[1]:
                h_scale = max_resolution[0] / H
                w_scale = max_resolution[1] / W

                if (
                    round(H * min(h_scale, w_scale)) > min_resolution[0]
                    and round(W * min(h_scale, w_scale)) > min_resolution[1]
                ):
                    scale_factor = min(h_scale, w_scale)
                else:
                    scale_factor = max(h_scale, w_scale)

                resize_H = round(H * scale_factor)
                resize_W = round(W * scale_factor)
                resize_needed = True
            else:
                resize_H = H
                resize_W = W
    else:
        resize_W = np.sqrt(W / H) * square_image_length
        resize_H = resize_W * H / W
        resize_H = int(resize_H // size_factor * size_factor)
        resize_W = int(resize_W // size_factor * size_factor)
        resize_needed = True

    # Only perform resize if needed
    if resize_needed:
        image = image.permute(0, 3, 1, 2)  # (T, H, W, C) -> (T, C, H, W)
        image = F.interpolate(
            image.float(),
            size=(resize_H, resize_W),
            mode="bilinear",
            align_corners=False,
        )
        image = image.permute(0, 2, 3, 1)  # (T, C, H, W) -> (T, H, W, C)

    # Calculate target size for potential cropping
    target_H = resize_H
    target_W = resize_W

    # Ensure target size is within bounds
    if square_image_length is None:
        target_H = min(max(target_H, min_resolution[0]), max_resolution[0])
        target_W = min(max(target_W, min_resolution[1]), max_resolution[1])

    # Make target size divisible by size_factor
    target_H = target_H // size_factor * size_factor
    target_W = target_W // size_factor * size_factor

    # Crop if the image is larger than target size
    if resize_H > target_H or resize_W > target_W:
        # Calculate valid ranges for crop
        h_start_max = resize_H - target_H
        w_start_max = resize_W - target_W

        if random_crop:
            # Random crop positions
            h_start = torch.randint(0, h_start_max + 1, (1,)).item()
            w_start = torch.randint(0, w_start_max + 1, (1,)).item()
        else:
            # Center crop for evaluation
            h_start = h_start_max // 2
            w_start = w_start_max // 2
        # Perform crop
        image = image[:, h_start : h_start + target_H, w_start : w_start + target_W, :]

    # Normalize image
    image = image.float() / 127.5 - 1.0
    T, H, W, _ = image.shape

    # Verify dimensions are correct before rearrange
    assert H % Ph == 0, f"Height {H} not divisible by patch size {Ph}"
    assert W % Pw == 0, f"Width {W} not divisible by patch size {Pw}"
    assert T % Pt == 0, f"Time {T} not divisible by patch size {Pt}"

    # Z, Pz make to 1 for images
    Z = 1
    Pz = 1

    # reshape to (T//Pt, H//Ph, W//Pw, Pt*Ph*Pw*C)
    feats = rearrange(
        image,
        "(nt pt) (nh ph) (nw pw) c -> nt nh nw (pt ph pw c)",
        nt=T // Pt,
        nh=H // Ph,
        nw=W // Pw,
        pt=Pt,
        ph=Ph,
        pw=Pw,
    )

    feats = rearrange(feats, "t h w c -> (t h w) c")

    if random_time > 0:
        random_time_step = random.randint(0, random_time)
        assert T // Pt == 1
    else:
        random_time_step = 0

    coords = (
        torch.stack(
            torch.meshgrid(
                torch.arange(T // Pt) + random_time_step,
                torch.arange(H // Ph),
                torch.arange(W // Pw),
                torch.arange(Z // Pz),
                indexing="ij",
            ),
            dim=-1,
        )
        .reshape(-1, 4)
        .int()
    )

    # Check if patch_sample_ratio is a list or float
    if isinstance(patch_sample_ratio, list):
        # If it's a list [min_ratio, max_ratio], sample a ratio from this interval
        min_ratio, max_ratio = patch_sample_ratio
        # Randomly sample a ratio from the interval
        patch_sample_ratio = np.random.uniform(min_ratio, max_ratio)

    # Apply sampling if ratio is less than 1.0
    if patch_sample_ratio < 1.0:
        num_patch = int(feats.shape[0] * patch_sample_ratio)
        rand_index = np.random.permutation(feats.shape[0])[:num_patch]
        feats = feats[rand_index]
        coords = coords[rand_index]

    return feats, coords


def keep_aspect_ratio_image_process_fn(
    patch_size=(1, 8, 8),
    min_resolution=(128, 128),
    max_resolution=(384, 384),
    square_image_length=None,
    size_factor=16,
    temporal_padding_to=1,
    padding_type="zero",
    random_crop=True,
    random_time=0,
    patch_sample_ratio=1.0,
    max_seq_len=None,
    random_resolution=True,
) -> Callable:
    def get_image(data: Dict) -> Dict:
        """Get images webdataset and keep the aspect ratio.
        Args:
            data (Dict): Data dictionary.
        Returns:
        """
        image = Image.open(io.BytesIO(data["image"])).convert("RGB")
        image = torch.from_numpy(np.array(image))

        feats, coords = process_image(
            image,
            patch_size,
            min_resolution,
            max_resolution,
            square_image_length,
            size_factor,
            random_crop,
            random_time,
            patch_sample_ratio,
            padding_type=padding_type,
            temporal_padding_to=temporal_padding_to,
            random_resolution=random_resolution,
        )

        # Check if coords is empty
        if coords.numel() == 0 or feats.numel() == 0:
            raise ValueError("Generated empty coordinates or features tensor")

        # Check token length if max_token_length is specified
        if (max_seq_len is not None and coords.shape[0] > max_seq_len) or len(coords) == 0:
            raise ValueError(
                f"Token length {coords.shape[0]} exceeds maximum {max_seq_len} or empty"
            )

        return {
            "feats": feats,
            "coords": coords,
        }

    return get_image


def keep_aspect_ratio_video_process_fn(
    is_training,
    num_to_select,
    min_stride,
    random_frame=None,
    max_stride=0,
    random_stride=None,
    patch_size=(1, 1, 1),
    min_resolution=(128, 128),
    max_resolution=(384, 384),
    square_image_length=None,
    size_factor=16,
    random_crop=True,
    max_seq_len=None,
    random_resolution=True,
) -> Callable:
    def get_video(data: Dict) -> Dict:
        """Get images webdataset and keep the aspect ratio.
        Args:
            data (Dict): Data dictionary.
        Returns:
        """
        data, _, _ = decode_video_decord(
            is_training=is_training,
            data=data["video_bytes"],
            num_to_select=num_to_select,
            min_stride=min_stride,
            max_stride=max_stride,
            random_stride=random_stride,
            random_frame=random_frame,
        )

        feats, coords = process_image(
            data,
            patch_size,
            min_resolution,
            max_resolution,
            square_image_length,
            size_factor,
            random_crop=random_crop,
            random_resolution=random_resolution,
        )

        # Check if coords is empty
        if coords.numel() == 0 or feats.numel() == 0:
            raise ValueError("Generated empty coordinates or features tensor")

        # Check token length if max_token_length is specified
        if (max_seq_len is not None and coords.shape[0] > max_seq_len) or len(coords) == 0:
            raise ValueError(
                f"Token length {coords.shape[0]} exceeds maximum {max_seq_len} or empty"
            )

        return {
            "feats": feats,
            "coords": coords,
        }

    return get_video


def keep_aspect_ratio_video_text_process_fn_new(
    num_to_select,
    patch_size=(1, 8, 8),
    min_resolution=(128, 128),
    max_resolution=(384, 384),
    square_image_length=None,
    size_factor=16,
    text_field=None,
    text_field_rate=None,
    default_text_field="caption",
    default_video_field="video",
    text_context_length=512,
    random_crop=True,
    patch_sample_ratio=1.0,
    max_seq_len=None,
    text_decoder=False,
    random_resolution=True,
) -> Callable:
    tokenizer = SigLipTokenizer(tokenizer_name="gemma", context_length=text_context_length)

    # Normalize the text_field_rate to ensure it sums to 1
    text_field_rate = np.array(text_field_rate)
    normalized_rates = text_field_rate / text_field_rate.sum()

    def get_image(data: Dict) -> Dict:
        """Get images webdataset and keep the aspect ratio.
        Args:
            data (Dict): Data dictionary.
        Returns:
        """
        if text_field is not None:
            chosen_field = np.random.choice(text_field, p=normalized_rates)
        else:
            chosen_field = default_text_field

        decoded_str = data[chosen_field]
        if isinstance(decoded_str, bytes):
            decoded_str = decoded_str.decode("utf-8")

        encoded_input = tokenizer(decoded_str)[0]
        encoded_target = encoded_input

        if text_decoder:
            # add bos token.
            encoded_input = torch.concat(
                [torch.tensor([tokenizer.tokenizer.bos_token_id]), encoded_input[:-1]],
                axis=0,
            )

        # truncate sequence to context length
        if len(encoded_input) > text_context_length:
            encoded_input = encoded_input[:text_context_length]

        if len(encoded_target) > text_context_length:
            encoded_target = encoded_target[:text_context_length]

        # pad to sequence length
        if len(encoded_input) < text_context_length:
            encoded_input = F.pad(encoded_input, (0, text_context_length - len(encoded_input)))
            encoded_target = F.pad(encoded_target, (0, text_context_length - len(encoded_target)))

        data, _, _ = decode_video_text_decord(
            data=data[default_video_field],
            num_to_select=num_to_select,
            min_stride=patch_size[0],
        )

        feats, coords = process_image(
            data,
            patch_size,
            min_resolution,
            max_resolution,
            square_image_length,
            size_factor,
            random_crop,
            patch_sample_ratio=patch_sample_ratio,
            padding_type="zero",
            temporal_padding_to=patch_size[0],
            random_resolution=random_resolution,
        )

        # Check if coords is empty
        if coords.numel() == 0 or feats.numel() == 0:
            raise ValueError("Generated empty coordinates or features tensor")

        # Check token length if max_token_length is specified
        if max_seq_len is not None and coords.shape[0] > max_seq_len:
            raise ValueError(f"Token length {coords.shape[0]} exceeds maximum {max_seq_len}")

        return {
            "feats": feats,
            "coords": coords,
            "text_inputs": encoded_input,
            "text_targets": encoded_target,
        }

    return get_image


def keep_aspect_ratio_image_text_process_fn(
    patch_size=(1, 8, 8),
    min_resolution=(128, 128),
    max_resolution=(384, 384),
    size_factor=16,
    temporal_padding_to=1,
    padding_type="zero",
    text_field=None,
    text_field_rate=None,
    default_text_field="caption",
    text_context_length=64,
    random_crop=True,
    patch_sample_ratio=[1.0, 1.0],
    max_seq_len=None,
    square_image_length=None,
    text_decoder=False,
    random_resolution=True,
) -> Callable:
    tokenizer = SigLipTokenizer(tokenizer_name="gemma", context_length=text_context_length)

    # Normalize the text_field_rate to ensure it sums to 1
    text_field_rate = np.array(text_field_rate)
    normalized_rates = text_field_rate / text_field_rate.sum()

    # if use_image_augmentation:
    #     image_transform_fn = image_transform_v2(
    #         is_train=True,
    #     )
    # else:
    image_transform_fn = None

    def get_image(data: Dict) -> Dict:
        """Get images webdataset and keep the aspect ratio.
        Args:
            data (Dict): Data dictionary.
        Returns:
        """
        # first filter out those empty
        valid_field_idx = [
            idx
            for idx, value in enumerate(text_field)
            if len(json.loads(data[value].decode("utf-8"))) > 0
        ]
        if len(valid_field_idx) == 0:
            raise ValueError("All text fields are empty")

        valid_text_field = [text_field[idx] for idx in valid_field_idx]
        valid_normalized_rates = normalized_rates[np.array(valid_field_idx)]
        valid_normalized_rates = valid_normalized_rates / valid_normalized_rates.sum()

        if text_field is not None:
            chosen_field = np.random.choice(valid_text_field, p=valid_normalized_rates)
        else:
            chosen_field = default_text_field

        decoded_str = data[chosen_field].decode("utf-8")
        decoded_str = json.loads(decoded_str)

        if len(decoded_str[0]) == 0:
            decoded_str = data[default_text_field].decode("utf-8")
            decoded_str = json.loads(decoded_str)

        encoded_input = tokenizer(decoded_str[0])[0]
        encoded_target = encoded_input

        if text_decoder:
            # add bos token.
            encoded_input = torch.concat(
                [torch.tensor([tokenizer.tokenizer.bos_token_id]), encoded_input[:-1]],
                axis=0,
            )

        # truncate sequence to context length
        if len(encoded_input) > text_context_length:
            encoded_input = encoded_input[:text_context_length]

        if len(encoded_target) > text_context_length:
            encoded_target = encoded_target[:text_context_length]

        # attention_mask = torch.ones_like(encoded_input)

        # pad to sequence length
        if len(encoded_input) < text_context_length:
            encoded_input = F.pad(encoded_input, (0, text_context_length - len(encoded_input)))
            encoded_target = F.pad(encoded_target, (0, text_context_length - len(encoded_target)))

        img = Image.open(io.BytesIO(data["image"]))

        if image_transform_fn is not None:
            # Apply image transformations if provided, current only support image.
            image = image_transform_fn(img)
        else:
            if img.mode == "P" and "transparency" in img.info:
                image = img.convert("RGBA").convert("RGB")
            else:
                image = img.convert("RGB")

        image = torch.from_numpy(np.array(image))

        feats, coords = process_image(
            image,
            patch_size,
            min_resolution,
            max_resolution,
            square_image_length,
            size_factor,
            random_crop,
            patch_sample_ratio=patch_sample_ratio,
            padding_type=padding_type,
            temporal_padding_to=temporal_padding_to,
            random_resolution=random_resolution,
        )

        # Check if coords is empty
        if coords.numel() == 0 or feats.numel() == 0:
            raise ValueError("Generated empty coordinates or features tensor")

        # Check token length if max_token_length is specified
        if (max_seq_len is not None and coords.shape[0] > max_seq_len) or len(coords) == 0:
            raise ValueError(
                f"Token length {coords.shape[0]} exceeds maximum {max_seq_len} or empty"
            )

        return {
            "feats": feats,
            "coords": coords,
            "text_inputs": encoded_input,
            "text_targets": encoded_target,
        }

    return get_image


def imagenet_zeroshot_process_fn(
    patch_size=(1, 8, 8),
    min_resolution=(128, 128),
    max_resolution=(384, 384),
    size_factor=16,
    temporal_padding_to=1,
    padding_type="zero",
    random_crop=False,
    square_image_length=None,
) -> Callable:
    def get_image(data: Dict) -> Dict:
        """Get images webdataset and keep the aspect ratio.
        Args:
            data (Dict): Data dictionary.
        Returns:
        """
        label = torch.tensor(int(data["meta.json"]["class_id"])).long()
        image = Image.open(io.BytesIO(data["image"])).convert("RGB")
        image = torch.from_numpy(np.array(image))
        feats, coords = process_image(
            image,
            patch_size,
            min_resolution,
            max_resolution,
            square_image_length,
            size_factor,
            random_crop,
            padding_type=padding_type,
            temporal_padding_to=temporal_padding_to,
        )

        return {
            "feats": feats,
            "coords": coords,
            "label": label,
        }

    return get_image


def decode_gs_process_fn(
    num_image: int,
    image_size: int,
    bg_color: float = 0,
    with_normal_depth: bool = False,
    max_seq_len: int = None,
    target_feat_dims: int = 768,
) -> Callable:
    """Get feats, coords, images, intrinsics, w2c from data.

    Args:
        num_image (int): Number of images to sample.
        image_size (int): Image size.
        bg_color (float): Background color. Default: 0.
        with_normal_depth (bool): Whether to include normal and depth images. Default: False.

    Returns:
        Callable: Function to get feats, coords, images, intrinsics, w2c from data.
    """

    def get_feat_vol_image(data: Dict) -> Dict:
        """Get feats, coords, images, intrinsics, w2c from data.

        Args:
            data (Dict): Data dictionary.

        Returns:
            Dict: Data dictionary with
        """
        all_image_keys = [
            key
            for key in data.keys()
            if (".png" in key or ".webp" in key) and ("normal" not in key)
        ]
        assert (
            len(all_image_keys) >= num_image
        ), f"len(all_image_keys): {len(all_image_keys)}, expected: >= {num_image}"
        image_keys = random.sample(all_image_keys, num_image)
        image_ids = np.array([int(key.split(".")[0]) for key in image_keys])
        all_images = []
        for key in image_keys:
            with io.BytesIO(data[key]) as stream:
                image = Image.open(stream)
                image.load()
            image = np.array(image.resize((image_size, image_size))) / 255.0
            all_images.append(image)
        images = np.stack(all_images, axis=0)
        if images.shape[-1] == 4:
            images = images[..., :3] * images[..., -1:] + bg_color * (1 - images[..., -1:])

        if "feats.npy" in data:
            feats, coords = data["feats.npy"], data["coords.npy"]
        else:
            feats, coords = data["feats.npz"]["arr_0"], data["coords.npz"]["arr_0"]
        coords = torch.tensor(coords).int()

        # Check token length if max_token_length is specified
        if (max_seq_len is not None and coords.shape[0] > max_seq_len) or len(coords) == 0:
            raise ValueError(
                f"Token length {coords.shape[0]} exceeds maximum {max_seq_len} or empty"
            )

        # add t dim to coords
        coords = torch.cat([torch.ones(coords.shape[0], 1, dtype=coords.dtype), coords], dim=1)
        # remove nan in feats
        if np.isnan(feats).any():
            feats = np.nan_to_num(feats)

        feats = torch.tensor(feats, dtype=torch.float32) * 2 - 1

        # zero padding to target feature dimensions
        if feats.shape[1] < target_feat_dims:
            padding = torch.zeros(
                (feats.shape[0], target_feat_dims - feats.shape[1]),
                dtype=feats.dtype,
            )
            feats = torch.cat([feats, padding], dim=1)

        out_dict = {
            "feats": feats,
            "coords": coords,
            "images": torch.tensor(images, dtype=torch.float32),
        }

        if "intrinsics.npy" in data:
            out_dict.update(
                {
                    "intrinsics": torch.tensor(
                        data["intrinsics.npy"][image_ids], dtype=torch.float32
                    ),
                    "w2c": torch.tensor(data["w2c.npy"][image_ids], dtype=torch.float32),
                }
            )

        if with_normal_depth:
            filetype = all_image_keys[0].split(".")[-1]
            if image_keys[0].replace(f".{filetype}", "_normal.webp") in data:
                normal_filetype = "webp"
            else:
                normal_filetype = "png"
            all_normals = [
                np.array(
                    data[key.replace(f".{filetype}", f"_normal.{normal_filetype}")].resize(
                        (image_size, image_size)
                    )
                )
                / 255.0
                for key in image_keys
            ]
            normals = np.stack(all_normals, axis=0)
            if normals.shape[-1] == 4:
                normals = normals[..., :3] * normals[..., -1:] + 0.5 * (1 - normals[..., -1:])

            if "depth.npy" in data:
                depths = data["depth.npy"][image_ids]
            else:
                all_depths = []
                for key in image_keys:
                    depth_key = key.replace(f".{filetype}", "_depth.npy")
                    depth_raw = data[depth_key]
                    all_depths.append(depth_raw)
                depths = np.stack(all_depths, axis=0)
            depths = torch.tensor(depths, dtype=torch.float32)  # (N, H, W)
            if image_size != depths.shape[-1]:
                depths = torch.nn.functional.interpolate(
                    depths[:, None], size=(image_size, image_size), mode="bilinear"
                ).squeeze(1)

            out_dict.update(
                {
                    "normals": torch.tensor(normals, dtype=torch.float32),
                    "depths": depths,
                }
            )

        return out_dict

    return get_feat_vol_image


def decode_gs_text_process_fn(
    num_image: int,
    image_size: int,
    bg_color: float = 0,
    with_normal_depth: bool = False,
    max_seq_len: int = None,
    text_context_length=150,
    target_feat_dims: int = 768,
) -> Callable:
    """Get feats, coords, images, intrinsics, w2c from data.

    Args:
        num_image (int): Number of images to sample.
        image_size (int): Image size.
        bg_color (float): Background color. Default: 0.
        with_normal_depth (bool): Whether to include normal and depth images. Default: False.

    Returns:
        Callable: Function to get feats, coords, images, intrinsics, w2c from data.
    """
    tokenizer = SigLipTokenizer(tokenizer_name="gemma", context_length=text_context_length)

    def get_feat_vol_image(data: Dict) -> Dict:
        """Get feats, coords, images, intrinsics, w2c from data.

        Args:
            data (Dict): Data dictionary.

        Returns:
            Dict: Data dictionary with
        """

        decoded_str = data["caption.txt"]
        encoded_input = tokenizer(decoded_str)[0]
        encoded_target = encoded_input
        encoded_input = torch.concat(
            [torch.tensor([tokenizer.tokenizer.bos_token_id]), encoded_input[:-1]],
            axis=0,
        )

        # truncate sequence to context length
        if len(encoded_input) > text_context_length:
            encoded_input = encoded_input[:text_context_length]

        if len(encoded_target) > text_context_length:
            encoded_target = encoded_target[:text_context_length]

        # truncate sequence to context length
        if len(encoded_input) > text_context_length:
            encoded_input = encoded_input[:text_context_length]

        if len(encoded_target) > text_context_length:
            encoded_target = encoded_target[:text_context_length]

        # pad to sequence length
        if len(encoded_input) < text_context_length:
            encoded_input = F.pad(encoded_input, (0, text_context_length - len(encoded_input)))
            encoded_target = F.pad(encoded_target, (0, text_context_length - len(encoded_target)))

        all_image_keys = [
            key
            for key in data.keys()
            if (".png" in key or ".webp" in key) and ("normal" not in key)
        ]
        assert (
            len(all_image_keys) >= num_image
        ), f"len(all_image_keys): {len(all_image_keys)}, expected: >= {num_image}"
        image_keys = random.sample(all_image_keys, num_image)
        image_ids = np.array([int(key.split(".")[0]) for key in image_keys])
        all_images = []
        for key in image_keys:
            with io.BytesIO(data[key]) as stream:
                image = Image.open(stream)
                image.load()
            image = np.array(image.resize((image_size, image_size))) / 255.0
            all_images.append(image)
        images = np.stack(all_images, axis=0)
        if images.shape[-1] == 4:
            images = images[..., :3] * images[..., -1:] + bg_color * (1 - images[..., -1:])

        if "feats.npy" in data:
            feats, coords = data["feats.npy"], data["coords.npy"]
        else:
            feats, coords = data["feats.npz"]["arr_0"], data["coords.npz"]["arr_0"]
        coords = torch.tensor(coords).int()

        # Check if coords is empty
        if coords.numel() == 0:
            raise ValueError("Generated empty coordinates or features tensor")

        # Check token length if max_token_length is specified
        if (max_seq_len is not None and coords.shape[0] > max_seq_len) or len(coords) == 0:
            raise ValueError(
                f"Token length {coords.shape[0]} exceeds maximum {max_seq_len} or empty"
            )

        # add t dim to coords
        coords = torch.cat([torch.ones(coords.shape[0], 1, dtype=coords.dtype), coords], dim=1)
        # remove nan in feats
        if np.isnan(feats).any():
            feats = np.nan_to_num(feats)

        feats = torch.tensor(feats, dtype=torch.float32) * 2 - 1

        # zero padding to target feature dimensions
        if feats.shape[1] < target_feat_dims:
            padding = torch.zeros(
                (feats.shape[0], target_feat_dims - feats.shape[1]),
                dtype=feats.dtype,
            )
            feats = torch.cat([feats, padding], dim=1)

        out_dict = {
            "feats": feats,
            "coords": coords,
            "images": torch.tensor(images, dtype=torch.float32),
            "text_inputs": encoded_input,
            "text_targets": encoded_target,
        }

        if "intrinsics.npy" in data:
            out_dict.update(
                {
                    "intrinsics": torch.tensor(
                        data["intrinsics.npy"][image_ids], dtype=torch.float32
                    ),
                    "w2c": torch.tensor(data["w2c.npy"][image_ids], dtype=torch.float32),
                }
            )

        if with_normal_depth:
            filetype = all_image_keys[0].split(".")[-1]
            if image_keys[0].replace(f".{filetype}", "_normal.webp") in data:
                normal_filetype = "webp"
            else:
                normal_filetype = "png"
            all_normals = [
                np.array(
                    data[key.replace(f".{filetype}", f"_normal.{normal_filetype}")].resize(
                        (image_size, image_size)
                    )
                )
                / 255.0
                for key in image_keys
            ]
            normals = np.stack(all_normals, axis=0)
            if normals.shape[-1] == 4:
                normals = normals[..., :3] * normals[..., -1:] + 0.5 * (1 - normals[..., -1:])

            if "depth.npy" in data:
                depths = data["depth.npy"][image_ids]
            else:
                all_depths = []
                for key in image_keys:
                    depth_key = key.replace(f".{filetype}", "_depth.npy")
                    depth_raw = data[depth_key]
                    all_depths.append(depth_raw)
                depths = np.stack(all_depths, axis=0)
            depths = torch.tensor(depths, dtype=torch.float32)  # (N, H, W)
            if image_size != depths.shape[-1]:
                depths = torch.nn.functional.interpolate(
                    depths[:, None], size=(image_size, image_size), mode="bilinear"
                ).squeeze(1)

            out_dict.update(
                {
                    "normals": torch.tensor(normals, dtype=torch.float32),
                    "depths": depths,
                }
            )

        return out_dict

    return get_feat_vol_image
