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
RefNerfacto implementation with depth losses.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Type

from nerfstudio.models.refnerfacto import RefNerfactoModel, RefNerfactoModelConfig
from nerfstudio.models.depth_based.depth_nerfacto import (
    DepthNerfactoModel,
    DepthNerfactoModelConfig,
)


@dataclass
class DepthRefNerfactoModelConfig(RefNerfactoModelConfig, DepthNerfactoModelConfig):
    """Additional parameters for depth supervision."""

    _target: Type = field(default_factory=lambda: DepthRefNerfactoModel)


class DepthRefNerfactoModel(RefNerfactoModel, DepthNerfactoModel):
    """Depth RefNerfacto model.

    Args:
        config: DepthRefNerfactoModel configuration to instantiate model
    """

    config: DepthRefNerfactoModelConfig
