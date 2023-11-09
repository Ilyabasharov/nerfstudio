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
Bundle adjustment for high frequiency encoder features
"""

from enum import Enum
from typing import Optional, Tuple
from jaxtyping import Float

import torch
from torch import Tensor
from torch import nn

from nerfstudio.model_components.scalers import _HashGradientScaler
from nerfstudio.utils.writer import GLOBAL_BUFFER


class BundleAdjustment(nn.Module):
    """Bundle adjustment for high frequiency features

    Args:
        use_bundle_adjust: whether to use bundle adjustment
        coarse_to_fine_iters: iterations (percentage of max iters) at which bundle adjustment is active
    """

    def __init__(
        self,
        use_bundle_adjust: bool,
        coarse_to_fine_iters: Optional[Tuple[float, float]] = None,
    ) -> None:
        super().__init__()

        self.use_bundle_adjust = use_bundle_adjust
        self.coarse_to_fine_iters = coarse_to_fine_iters
        self.step = 0
        self.max_iters: int = GLOBAL_BUFFER.get("max_iter", 10000)

        if not use_bundle_adjust:
            self.forward = lambda x: x
        else:
             # check that inputs args are correct
            assert coarse_to_fine_iters is not None and (
                0 <= coarse_to_fine_iters[0] <= 1 and 0 <= coarse_to_fine_iters[1] <= 1
            ), f"start and end iterations for bundle adjustment have to be not None and positive, got {coarse_to_fine_iters}"

            assert (coarse_to_fine_iters[0] < coarse_to_fine_iters[1]
            ), f"start should be less than end iterations for bundle adjustment, got {coarse_to_fine_iters}"

            self.forward = self.mask_freqs

    def step_cb(self, step: int) -> None:
        """Record current step"""
        self.step = step

        # Turn off bundle adjustment
        if self.get_progress() >= 1:
            self.use_bundle_adjust = False

    def mask_freqs(
        self,
        input_features: Float[Tensor, "*bs features"],
    ) -> Float[Tensor, "*bs features"]:
        """Masking input features"""

    def get_progress(self) -> float:
        """Log unlocked features"""

        if not self.use_bundle_adjust:
            return 1.0

        start, end = self.coarse_to_fine_iters # type: ignore
        return min(((self.step / self.max_iters) - start) / (end - start), 1)


class HashBundleAdjustment(BundleAdjustment):
    """Bundle adjustment for Hash Encoder"""

    def _get_masks(
        self,
        input_features: Float[Tensor, "*bs num_levels features_per_level"],
    ) -> Float[Tensor, "*bs num_levels features_per_level"]:
        """Get frequency masks
        Args:
            input_features: array of the hashgrid values
        """
        B, (L, F) = input_features.shape[:-2], input_features.shape[-2:]
        start, end = self.coarse_to_fine_iters # type: ignore
        # From https://arxiv.org/pdf/2104.06405.pdf equation 14
        # to prevent all weights to be 0, we leave the features from the first grid alone.
        alpha = ((self.step / self.max_iters) - start) / (end - start) * (L - 1)

        if alpha > L or L == 1:
            return input_features.new_ones(L)

        k = torch.arange(L - 1, dtype=input_features.dtype, device=input_features.device)
        mask_vals = (1 - (alpha - k).clamp_(min=0, max=1).mul_(torch.pi).cos_()) / 2
        mask_vals = torch.cat([mask_vals.new_ones(1), mask_vals])

        return mask_vals

    def mask_freqs(
        self,
        input_features: Float[Tensor, "*bs num_levels features_per_level"],
    ) -> Float[Tensor, "*bs num_levels features_per_level"]:
        """Masking input features per each hash level
        Args:
            input_features: array of the hashgrid values
        """
        mask_vals = self._get_masks(input_features)[None, ..., None]
        return mask_vals * input_features


class HashGradBundleAdjustment(HashBundleAdjustment):
    """Gradient bundle adjustment for Hash Encoder.
    Proposed in https://arxiv.org/abs/2302.01571

    Args:
        bundle_adjust: whether to use bundle adjustment
        coarse_to_fine_iters: iterations (percentage of max iters) at which bundle adjustment is active
    """

    def mask_freqs(
        self,
        input_features: Float[Tensor, "*bs num_levels features_per_level"],
    ) -> Float[Tensor, "*bs num_levels features_per_level"]:

        mask_vals = self._get_masks(input_features)[None, ..., None]

        masked = input_features
        if self.training and mask_vals is not None:
            masked, _ = _HashGradientScaler.apply(input_features, mask_vals) # type: ignore

        return masked


class HashAcceleratedBundleAdjustment(HashBundleAdjustment):
    """Bundle-Adjusting Accelerated Neural Graphics Primitives
    Proposed in https://arxiv.org/abs/2306.04166

    Args:
        bundle_adjust: whether to use bundle adjustment
        coarse_to_fine_iters: iterations (percentage of max iters) at which bundle adjustment is active
    """

    def mask_freqs(
        self,
        input_features: Float[Tensor, "*bs num_levels features_per_level"],
    ) -> Float[Tensor, "*bs num_levels features_per_level"]:
        """Masking input features per each hash level
        Args:
            input_features: array of the hashgrid values
        """
        mask_vals = self._get_masks(input_features)
        available_features = input_features[:, mask_vals > 0]
        mask_vals = mask_vals[None, :, None]

        # the set of features that has the highest grid level with a nonzero weight
        coarse_features = available_features[:, -1:]
        coarse_repeats = coarse_features.expand_as(input_features)

        # see eq. 8
        encoded_x = input_features * mask_vals + coarse_repeats * (1 - mask_vals)
        return encoded_x


class BundleAdjustmentType(Enum):
    """Types of bundle adjustment for pass gradients through encoding."""

    BARF = HashBundleAdjustment
    CAMP = HashGradBundleAdjustment
    BAANGP = HashGradBundleAdjustment
