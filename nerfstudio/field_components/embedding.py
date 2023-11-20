# Copyright 2022 the Regents of the University of California, Nerfstudio Team and contributors. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Code for embeddings.
"""
from typing import Optional, Callable

import torch
from jaxtyping import Shaped, Int
from torch import Tensor, nn

from nerfstudio.field_components.base_field_component import FieldComponent


class Embedding(FieldComponent):
    """Index into embeddings.
    # TODO: add different types of initializations

    Args:
        in_dim: Number of embeddings
        out_dim: Dimension of the embedding vectors
    """

    def __init__(self, in_dim: int, out_dim: int) -> None:
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.build_nn_modules()

    def build_nn_modules(self) -> None:
        self.embedding = nn.Embedding(self.in_dim, self.out_dim)

    def mean(self, dim=0):
        """Return the mean of the embedding weights along a dim."""
        return self.embedding.weight.mean(dim)

    def forward(self, in_tensor: Shaped[Tensor, "*batch input_dim"]) -> Shaped[Tensor, "*batch output_dim"]:
        """Call forward

        Args:
            in_tensor: input tensor to process
        """
        return self.embedding(in_tensor)

    def regularize_appearence(
        self,
        regularize_fn: Callable[[Tensor], Tensor] = torch.abs,
        camera_indices: Optional[Int[Tensor, "num_trainable_cameras"]] = None,
    ) -> Shaped[Tensor, "*batch output_dim"]:
        """Additional regularisation for training stabilisation

        Args:
            regularize_fn: function to regularize weights
        """

        weights = self.embedding.weight

        if camera_indices is not None:
            weights = weights[camera_indices]

        return regularize_fn(weights)
