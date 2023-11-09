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
Pose and Intrinsics Optimizers
"""

from __future__ import annotations

import functools
from dataclasses import dataclass, field
from typing import Literal, Optional, Type, Union, Tuple, Callable

import torch
import tyro
from jaxtyping import Float, Int
from torch import Tensor, nn
from typing_extensions import assert_never

from nerfstudio.cameras.lie_groups import exp_map_SE3, exp_map_SO3xR3
from nerfstudio.configs.base_config import InstantiateConfig
from nerfstudio.engine.optimizers import AdamOptimizerConfig, OptimizerConfig
from nerfstudio.engine.schedulers import (
    ExponentialDecaySchedulerConfig,
    SchedulerConfig,
)
from nerfstudio.utils import poses as pose_utils


@dataclass
class PoseOptimizerConfig(InstantiateConfig):
    """Configuration of optimization for camera poses."""

    _target: Type = field(default_factory=lambda: PoseOptimizer)

    mode: Literal["off", "SO3xR3", "SE3"] = "off"
    """Pose optimization strategy to use. If enabled, we recommend SO3xR3."""
    position_noise_std: float = 0.0
    """Noise to add to initial positions. Useful for debugging."""
    orientation_noise_std: float = 0.0
    """Noise to add to initial orientations. Useful for debugging."""
    optimizer: OptimizerConfig = field(default_factory=lambda: AdamOptimizerConfig(lr=6e-4, eps=1e-15))
    """Optimizer for poses."""
    scheduler: SchedulerConfig = field(default_factory=lambda: ExponentialDecaySchedulerConfig(max_steps=10000))
    """Learning rate scheduler for pose optimizer.."""
    param_group: tyro.conf.Suppress[str] = "pose_opt"
    """Name of the parameter group used for pose optimization.
    Can be any string that doesn't conflict with other groups."""


class PoseOptimizer(nn.Module):
    """Layer that modifies camera poses to be optimized as well as the field during training."""

    config: PoseOptimizerConfig

    def __init__(
        self,
        config: PoseOptimizerConfig,
        num_cameras: int,
        device: Union[torch.device, str],
        non_trainable_camera_indices: Optional[Int[Tensor, "num_non_trainable_cameras"]] = None,
        **kwargs,
    ) -> None:
        super().__init__()
        self.config = config
        self.num_cameras = num_cameras
        self.device = device
        self.non_trainable_camera_indices = non_trainable_camera_indices

        # Initialize learnable parameters.
        if self.config.mode == "off":
            pass
        elif self.config.mode in ("SO3xR3", "SE3"):
            self.pose_adjustment = torch.nn.Parameter(torch.zeros((num_cameras, 6), device=device))
        else:
            assert_never(self.config.mode)

        # Initialize pose noise; useful for debugging.
        if config.position_noise_std != 0.0 or config.orientation_noise_std != 0.0:
            assert config.position_noise_std >= 0.0 and config.orientation_noise_std >= 0.0
            std_vector = torch.tensor(
                [config.position_noise_std] * 3 + [config.orientation_noise_std] * 3, device=device
            )
            self.pose_noise = exp_map_SE3(torch.normal(torch.zeros((num_cameras, 6), device=device), std_vector))
        else:
            self.pose_noise = None

    def forward(
        self,
        indices: Int[Tensor, "camera_indices"],
    ) -> Float[Tensor, "camera_indices 3 4"]:
        """Indexing into camera adjustments.
        Args:
            indices: indices of Cameras to optimize.
        Returns:
            Transformation matrices from optimized camera coordinates
            to given camera coordinates.
        """
        outputs = []

        # Apply learned transformation delta.
        if self.config.mode == "off":
            pass
        elif self.config.mode == "SO3xR3":
            outputs.append(exp_map_SO3xR3(self.pose_adjustment[indices, :]))
        elif self.config.mode == "SE3":
            outputs.append(exp_map_SE3(self.pose_adjustment[indices, :]))
        else:
            assert_never(self.config.mode)
        # Detach non-trainable indices by setting to identity transform
        if self.non_trainable_camera_indices is not None:
            outputs[0][self.non_trainable_camera_indices] = torch.eye(4, device=self.device)[:3, :4]

        # Apply initial pose noise.
        if self.pose_noise is not None:
            outputs.append(self.pose_noise[indices, :, :])
        # Return: identity if no transforms are needed, otherwise multiply transforms together.
        if len(outputs) == 0:
            # Note that using repeat() instead of tile() here would result in unnecessary copies.
            return torch.eye(4, device=self.device)[None, :3, :4].tile(indices.shape[0], 1, 1)
        return functools.reduce(pose_utils.multiply, outputs)


@dataclass
class IntrinsicOptimizerConfig(InstantiateConfig):
    """Configuration of optimization for camera intrinsics."""

    _target: Type = field(default_factory=lambda: IntrinsicOptimizer)

    mode: Literal["off", "shift", "scale+shift", "square_scale", "square_scale+shift"] = "off"
    """Intrinsics optimization strategy to use. If enabled, we recommend square_scale."""
    optimizer: OptimizerConfig = field(default_factory=lambda: AdamOptimizerConfig(lr=6e-4, eps=1e-15))
    """Optimizer parameters for intrinsic optimization."""
    scheduler: SchedulerConfig = field(default_factory=lambda: ExponentialDecaySchedulerConfig(max_steps=10000))
    """Learning rate scheduler for intrinsic optimizer."""
    param_group: tyro.conf.Suppress[str] = "intrinsic_opt"
    """Name of the parameter group used for intrinsic optimization.
    Can be any string that doesn't conflict with other groups."""


class IntrinsicOptimizer(nn.Module):
    """Layer that modifies camera intrinsics to be
    optimized as well as the field during training."""

    config: IntrinsicOptimizerConfig
    num_params: int = 4

    def __init__(
        self,
        config: IntrinsicOptimizerConfig,
        device: Union[torch.device, str],
        non_trainable_camera_indices: Optional[Int[Tensor, "num_non_trainable_cameras"]] = None,
        num_cameras: int = 1,
        **kwargs,
    ) -> None:
        assert num_cameras == 1, "More than 1 intrinsic optimizer are not supported yet."
        super().__init__()
        self.config = config
        self.num_cameras = num_cameras
        self.device = device
        self.non_trainable_camera_indices = non_trainable_camera_indices

        # Initialize learnable parameters.
        if self.config.mode == "off":
            pass
        elif self.config.mode in ("shift", "scale+shift", "square_scale", "square_scale+shift"):
            for param in ("scale", "shift"):
                num_scale_param = self.num_params if param in self.config.mode else 0
                init_func: Callable = torch.ones if param == "scale" else torch.zeros

                self.register_parameter(
                    name=f"{config.param_group}_adjustment_{param}",
                    param=torch.nn.Parameter(
                        init_func(num_scale_param, device=device),
                        requires_grad=True,
                    )
                )

            self.scale_tf = lambda x: x ** 2 \
                if "square" in self.config.mode \
                else lambda x: x
        else:
            assert_never(self.config.mode)

    def forward(
        self,
        fx: Float[Tensor, "1"],
        fy: Float[Tensor, "1"],
        cx: Float[Tensor, "1"],
        cy: Float[Tensor, "1"],
        indices: Optional[Int[Tensor, "camera_indices"]] = None,
    ) -> Tuple[
            Float[Tensor, "1"],
            Float[Tensor, "1"],
            Float[Tensor, "1"],
            Float[Tensor, "1"],
        ]:
        """Intrinsics correction
        Args:
            fx: focals at x coordinate to optimize.
            fy: focals at y coordinate to optimize
            cx: principal point at x coordinate to optimize.
            cy: principal point at y coordinate to optimize
        Returns:
            Corrected fx, fy, cx, cy
        """

        # Apply learned transformation delta.
        if self.config.mode == "off":
            pass
        elif "shift" in self.config.mode or "scale" in self.config.mode:
            if "scale" in self.config.mode:
                fx = fx * self.scale_tf(self.intrinsic_opt_adjustment_scale[0])
                fy = fy * self.scale_tf(self.intrinsic_opt_adjustment_scale[1])
                cx = cx * self.scale_tf(self.intrinsic_opt_adjustment_scale[2])
                cy = cy * self.scale_tf(self.intrinsic_opt_adjustment_scale[3])

            if "shift" in self.config.mode:
                fx = fx + self.intrinsic_opt_adjustment_shift[0]
                fy = fy + self.intrinsic_opt_adjustment_shift[1]
                cx = cx + self.intrinsic_opt_adjustment_shift[2]
                cy = cy + self.intrinsic_opt_adjustment_shift[3]
        else:
            assert_never(self.config.mode)

        return fx, fy, cx, cy


@dataclass
class DistortionOptimizerConfig(IntrinsicOptimizerConfig):
    """Configuration of optimization for camera distortion."""

    _target: Type = field(default_factory=lambda: DistortionOptimizer)

    mode: Literal["off", "shift"] = "off"
    """Distortion optimization strategy to use. If enabled, we recommend shift."""
    param_group: tyro.conf.Suppress[str] = "distortion_opt"
    """Name of the parameter group used for distortion optimization.
    Can be any string that doesn't conflict with other groups."""


class DistortionOptimizer(IntrinsicOptimizer):
    """Layer that modifies camera distortion to be
    optimized as well as the field during training."""

    config: DistortionOptimizerConfig
    num_params: int = 6

    def __init__(
        self,
        config: DistortionOptimizerConfig,
        device: Union[torch.device, str],
        non_trainable_camera_indices: Optional[Int[Tensor, "num_non_trainable_cameras"]] = None,
        num_cameras: int = 1,
        **kwargs,
    ) -> None:
        assert num_cameras == 1, "More than 1 distortion optimizer are not supported yet."
        super().__init__(
            config=config,
            device=device,
            non_trainable_camera_indices=non_trainable_camera_indices,
            num_cameras=num_cameras,
        )

    def forward(
        self,
        indices: Int[Tensor, "camera_indices"],
        distortion_params: Optional[Float[Tensor, "*batch_dist_params 6"]] = None,
    ) -> Float[Tensor, "*batch_dist_params 6"]:
        """Distortion correction
        Args:
            indices: used for stole shape of input rays
            distortion_params: the distortion parameters [k1, k2, k3, k4, p1, p2].
                Can be None.
        Returns:
            Corrected distortion params.
        """
        corrected_distortion = None
        # Apply learned transformation delta.
        if self.config.mode == "off":
            pass
        elif "shift" in self.config.mode:
            corrected_distortion = getattr(
                self,
                f"{self.config.param_group}_adjustment_shift",
            ).expand(indices.shape[0], self.num_params)
            if distortion_params is not None:
                corrected_distortion = distortion_params + corrected_distortion
        else:
            assert_never(self.config.mode)

        return corrected_distortion
