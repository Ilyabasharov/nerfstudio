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
Special activation functions.
"""

from typing import TYPE_CHECKING

import torch
from torch import nn
from jaxtyping import Float
from torch import Tensor
from torch.autograd import Function
from torch.cuda.amp import custom_bwd, custom_fwd


class _TruncExp(Function):
    # Implementation from torch-ngp:
    # https://github.com/ashawkey/torch-ngp/blob/93b08a0d4ec1cc6e69d85df7f0acdfb99603b628/activation.py
    @staticmethod
    @custom_fwd(cast_inputs=torch.float32)
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return x.exp()

    @staticmethod
    @custom_bwd
    def backward(ctx, g):
        x = ctx.saved_tensors[0]
        return g * x.clamp(-15, 15).exp()


class ShiftedSoftplus(nn.Softplus):
    """Shifted version of softplus activation."""

    __constants__ = ['beta', 'threshold', 'shift']
    beta: int
    threshold: int
    shift: float

    def __init__(
        self,
        beta: int = 1,
        threshold: int = 20,
        shift: float = 0.001,
    ) -> None:
        super().__init__(beta=beta, threshold=threshold)
        self.shift = shift

    def forward(self, input: Tensor) -> Tensor:
        return super().forward(input + self.shift)
    
    def extra_repr(self) -> str:
        return f'beta={self.beta}, threshold={self.threshold}, shift={self.shift}'


if TYPE_CHECKING:

    def trunc_exp(_: Float[Tensor, "*bs"], /) -> Float[Tensor, "*bs"]:
        """Same as torch.exp, but with the backward pass clipped to prevent vanishing/exploding
        gradients."""
        raise NotImplementedError()

else:
    trunc_exp = _TruncExp.apply
    """Same as torch.exp, but with the backward pass clipped to prevent vanishing/exploding
    gradients."""
