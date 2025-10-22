#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2025 Apple Inc. All Rights Reserved.
#
"""Base module for AToken models."""

import torch
import torch.nn as nn


class BaseModule(nn.Module):
    """Base module class for AToken models."""

    def __init__(self):
        super().__init__()

    def save(self, path):
        """Save the model to a path."""
        torch.save(self.state_dict(), path)

    def load(self, path):
        """Load the model from a path."""
        self.load_state_dict(torch.load(path, map_location="cpu"))
