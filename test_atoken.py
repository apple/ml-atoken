#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2025 Apple Inc. All Rights Reserved.
#
#!/usr/bin/env python3
"""
Test script to load a real AToken model using actual config and S3 checkpoint.
"""

import os
import urllib.request

import imageio
import torch
from PIL import Image

from atoken_inference.atoken_wrapper import ATokenWrapper
from atoken_inference.model.basic import SparseTensor
from atoken_inference.model.dataloader import create_folder_dataloader
from atoken_inference.model.utils import sparse_to_img_list


def unnormalize(img):
    img = img.permute(0, 2, 3, 1)  # Change to (B, H, W, C)
    img = (img + 1) / 2
    img = (img * 255).clamp(0, 255).to(torch.uint8)  # Convert to uint8
    img = img.cpu().numpy()  # Convert to numpy array
    return img


def test_atoken_wrapper_rec():
    # Real model paths - use available config

    model_path = "checkpoints/atoken-soc.pt"
    config_path = "configs/atoken-soc.yaml"

    print(f"Testing with config: {config_path}")
    print(f"Testing with model: {model_path}")

    # Check if files exist
    if not os.path.exists(config_path):
        print(f"Config file not found: {config_path}")
        return

    wrapper = ATokenWrapper(config_path, model_path).cuda().to(torch.bfloat16)

    print("Model loaded successfully!")

    # download example video for rec
    example_url = (
        "https://commondatastorage.googleapis.com/gtv-videos-bucket/sample/ForBiggerEscapes.mp4"
    )

    # Create assets/examples directory if it doesn't exist
    os.makedirs("assets/examples", exist_ok=True)

    # Download video if not present
    video_filename = "ForBiggerEscapes.mp4"
    video_path = os.path.join("assets/examples", video_filename)

    if os.path.exists(video_path):
        print(f"Video already exists at {video_path}, skipping download")
    else:
        print(f"Downloading video from {example_url}...")
        urllib.request.urlretrieve(example_url, video_path)
        print(f"Video downloaded to {video_path}")

    dataloader = create_folder_dataloader(
        folder_path="assets/examples/",
        batch_size=1,
        max_frames=-1,
        patch_size=(4, 16, 16),
    )
    count = 0
    for batch in dataloader:
        batch_coords = []
        batch_feats = []

        for i, (feats, coords) in enumerate(zip(batch["feats_list"], batch["coords_list"])):
            bcoords = (
                torch.zeros([coords.shape[0], 1]).to(device=coords.device, dtype=coords.dtype) + i
            )
            new_coords = torch.cat([bcoords, coords], dim=1)
            batch_coords.append(new_coords)
            batch_feats.append(feats)

        coords = torch.cat(batch_coords, dim=0)
        feats = torch.cat(batch_feats, dim=0)

        file_ids = batch["file_ids"]
        file_types = batch["file_types"]  # "image" or "video"

        # Process the video
        video_sparse = SparseTensor(feats, coords)
        task_types = ["video"]
        kwargs = {"task_types": task_types}
        rec, image_feat, x_no_proj = wrapper.inference(video_sparse, **kwargs)

        img_list = sparse_to_img_list(video_sparse.cpu(), [4, 16, 16], task_types=task_types)
        rec_list = sparse_to_img_list(rec.cpu(), [4, 16, 16], task_types=task_types)

        output_dir = "assets/examples_video"
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # Save the reconstructed images or video
        for img, rec, file_type, file_id in zip(img_list, rec_list, file_types, file_ids):
            img = unnormalize(img)
            rec = unnormalize(rec)

            if file_type == "image":
                # directly save image
                img_save_path = os.path.join(output_dir, f"{file_id}_img.png")
                rec_save_path = os.path.join(output_dir, f"{file_id}_rec.png")

                image = Image.fromarray(img[0])
                image.save(img_save_path)

                reconstruction = Image.fromarray(rec[0])
                reconstruction.save(rec_save_path)

            elif file_type == "video":
                # Write video
                imageio.mimsave(os.path.join(output_dir, f"{file_id}_img.mp4"), img, fps=30)
                imageio.mimsave(os.path.join(output_dir, f"{file_id}_rec.mp4"), rec, fps=30)

            print(f"saved {file_ids}...")
            count += 1

        if count >= 10:
            break


if __name__ == "__main__":
    test_atoken_wrapper_rec()
