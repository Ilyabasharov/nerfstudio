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
Gradient scalers.
"""
from typing import Dict, FrozenSet, Tuple, cast
from jaxtyping import Float

import torch
from torch import Tensor

from nerfstudio.field_components.field_heads import FieldHeadNames
from nerfstudio.cameras.rays import RaySamples


class _GradientScaler(torch.autograd.Function):  # typing: ignore
    """
    Scale gradients by a constant factor.
    Ideas taken from https://gradient-scaling.github.io
    """

    @staticmethod
    def forward(ctx, value, scaling):
        ctx.save_for_backward(scaling)
        return value, scaling

    @staticmethod
    def backward(ctx, output_grad, grad_scaling):
        (scaling,) = ctx.saved_tensors
        return output_grad * scaling, grad_scaling


def scale_gradients_by_distance_squared(
    field_outputs: Dict[FieldHeadNames, Tensor],
    ray_samples: RaySamples,
    field_names_exclude: FrozenSet[FieldHeadNames] = frozenset(),
) -> Dict[FieldHeadNames, Tensor]:
    """
    Scale gradients by the ray distance to the pixel
    as suggested in `Radiance Field Gradient Scaling for Unbiased Near-Camera Training` paper

    Note: The scaling is applied on the interval of [0, 1] along the ray!

    Example:
        GradientLoss should be called right after obtaining the densities and colors from the field. ::
            >>> field_outputs = scale_gradient_by_distance_squared(field_outputs, ray_samples)
    """
    out = {}
    ray_dist = (ray_samples.frustums.starts + ray_samples.frustums.ends) / 2
    scaling = ray_dist.square_().clamp_(0, 1)
    for key, value in field_outputs.items():
        if key in field_names_exclude:
            out[key] = value
        else:
            out[key], _ = cast(Tuple[Tensor, Tensor], _GradientScaler.apply(value, scaling))
    return out


class _HashGradientScaler(torch.autograd.Function):  # typing: ignore
    """
    Scales the gradients of hash features based on a provided mask
    Ideas taken from https://camp-nerf.github.io
    """

    @staticmethod
    def forward(
        ctx,
        value: Float[Tensor, "bs num_levels features_per_level"],
        mask: Float[Tensor, "bs num_levels features_per_level"],
    ):
        ctx.save_for_backward(mask)
        return value, mask

    @staticmethod
    def backward(ctx, output_grad, grad_scaling):
        (mask,) = ctx.saved_tensors
        return output_grad * mask, grad_scaling
