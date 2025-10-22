#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2025 Apple Inc. All Rights Reserved.
#
"""Sparse linear layer for AToken."""

import torch.nn as nn

from .basic import SparseTensor


class SparseLinear(nn.Linear):
    """Sparse linear layer that works with SparseTensor."""

    def __init__(self, in_features, out_features, bias=False):
        super().__init__(in_features, out_features, bias)

    def forward(self, input):
        if isinstance(input, SparseTensor):
            return input.replace(super().forward(input.feats))
        else:
            return super().forward(input)
