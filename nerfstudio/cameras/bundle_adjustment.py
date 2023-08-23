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

from typing import Optional, Tuple
from jaxtyping import Float
from torch import Tensor

import torch
import torch.nn as nn

from nerfstudio.utils.writer import GLOBAL_BUFFER


class BundleAdjustment(nn.Module):
    """Bundle adjustment for high frequiency features

    Args:
        bundle_adjust: whether to use bundle adjustment
        coarse_to_fine_iters: iterations (percentage of max iters) at which bundle adjustment is active
    """

    def __init__(
        self,
        bundle_adjust: bool,
        coarse_to_fine_iters: Optional[Tuple[float, float]] = None,
    ) -> None:
        super().__init__()

        self.bundle_adjust = bundle_adjust
        self.coarse_to_fine_iters = coarse_to_fine_iters
        self.step = 0
        self.max_iters: int = GLOBAL_BUFFER.get("max_iter", 10000)

        if not bundle_adjust:
            self.forward = lambda x: x
        else:
             # check that inputs args are correct
            assert coarse_to_fine_iters is not None and (
                coarse_to_fine_iters[0] >= 0 and coarse_to_fine_iters[1] >= 0
            ), f"start and end iterations for bundle adjustment have to be not None and positive, got {coarse_to_fine_iters}"

            self.forward = self.mask_freqs

    def step_cb(self, step: int) -> None:
        """Record current step"""
        self.step = step

    def mask_freqs(self, input_features: Float[Tensor, "*bs features"]) -> Float[Tensor, "*bs features"]:
        """Masking input features"""

    def get_progress(self) -> float:
        """Log unlocked features"""

        if not self.bundle_adjust:
            return 1.0
        
        start, end = self.coarse_to_fine_iters # type: ignore
        return ((self.step / self.max_iters) - start) / (end - start)
    

class HashBundleAdjustment(BundleAdjustment):
    """Bundle adjustment for Hash Encoder

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
        B, (L, F) = input_features.shape[:-2], input_features.shape[-2:]
        start, end = self.coarse_to_fine_iters # type: ignore
        # From https://arxiv.org/pdf/2104.06405.pdf equation 14
        alpha = ((self.step / self.max_iters) - start) / (end - start) * L

        # if all features are unlocked
        if alpha > L:
            return input_features
        
        k = torch.arange(L, dtype=input_features.dtype, device=input_features.device)
        mask_vals = (1 - (alpha - k).clamp_(min=0, max=1).mul_(torch.pi).cos_()) / 2
        mask_vals = mask_vals[None, ..., None].repeat((*B, 1, F))
        masked = mask_vals * input_features
        return masked