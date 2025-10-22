#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2025 Apple Inc. All Rights Reserved.
#
#!/bin/bash

# Create checkpoints directory if it doesn't exist
mkdir -p checkpoints

echo "Downloading AToken checkpoints..."

# Base URL
BASE_URL="https://ml-site.cdn-apple.com/models/atoken"

# Download main models
echo "Downloading AToken-So/C..."
wget -O checkpoints/atoken-soc.pt "${BASE_URL}/atoken-soc.pt"

echo "Downloading AToken-So/D..."
wget -O checkpoints/atoken-sod.pt "${BASE_URL}/atoken-sod.pt"

echo "Downloading 3D Decode GS..."
wget -O checkpoints/3d_decode_gs.pt "${BASE_URL}/3d_decode_gs.pt"

# Download early stage pretrained weights
echo "Downloading AToken-So/C-s1..."
wget -O checkpoints/atoken-soc-s1.pt "${BASE_URL}/atoken-soc-s1.pt"

echo "Downloading AToken-So/C-s2..."
wget -O checkpoints/atoken-soc-s2.pt "${BASE_URL}/atoken-soc-s2.pt"

echo "All checkpoints downloaded successfully to ./checkpoints/"
