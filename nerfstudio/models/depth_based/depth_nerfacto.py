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
Nerfacto augmented with depth supervision.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Tuple, Type, List, Union

import torch
from torch import Tensor
import numpy as np
from matplotlib.figure import Figure

from nerfstudio.cameras.rays import RayBundle
from nerfstudio.model_components import losses
from nerfstudio.model_components.losses import DepthLossType, depth_ranking_loss
from nerfstudio.models.nerfacto import NerfactoModel, NerfactoModelConfig
from nerfstudio.engine.callbacks import (
    TrainingCallback,
    TrainingCallbackAttributes,
    TrainingCallbackLocation,
)
from nerfstudio.utils import colormaps, matplotlib_utis


@dataclass
class DepthNerfactoModelConfig(NerfactoModelConfig):
    """Additional parameters for depth supervision."""

    _target: Type = field(default_factory=lambda: DepthNerfactoModel)
    depth_loss_mult: float = 1e-2
    """Lambda of the depth loss."""
    depth_ranking_loss_mult: float = 1.0
    """Lambda of the depth loss."""
    is_euclidean_depth: bool = False
    """Whether input depth maps are Euclidean distances (or z-distances)."""
    depth_sigma: float = 0.02
    """Uncertainty around depth values in meters (defaults to 1cm)."""
    should_decay_sigma: bool = True
    """Whether to exponentially decay sigma."""
    starting_depth_sigma: float = 0.2
    """Starting uncertainty around depth values in meters (defaults to 0.2m)."""
    sigma_decay_rate: float = 0.99985
    """Rate of exponential decay."""
    use_depth_loss: bool = True
    """Whether to use depth ranking loss for absolute depth."""
    depth_loss_type: DepthLossType = DepthLossType.URF
    """Depth loss type."""
    use_depth_ranking_loss: bool = True
    """Whether to use depth ranking loss for relative depth."""


class DepthNerfactoModel(NerfactoModel):
    """Depth loss augmented nerfacto model.

    Args:
        config: Nerfacto configuration to instantiate model
    """

    config: DepthNerfactoModelConfig

    def populate_modules(self):
        """Set the fields and modules."""
        super().populate_modules()

        self.register_buffer(
            name="depth_sigma_current",
            tensor=torch.tensor(
                self.config.starting_depth_sigma
                if self.config.should_decay_sigma
                else self.config.depth_sigma
            ),
        )

        self.register_buffer(
            name="depth_sigma_base",
            tensor=torch.tensor(self.config.depth_sigma),
        )

        self.use_depth_ranking_loss = self.config.use_depth_ranking_loss or losses.FORCE_PSEUDODEPTH_LOSS
        self.ranking_loss_multiplier = 0.0
        
        if self.config.use_depth_loss:
            self.depth_loss = self.config.depth_loss_type.value(self.config.is_euclidean_depth)

    def get_outputs(self, ray_bundle: RayBundle):
        outputs = super().get_outputs(ray_bundle)
        if ray_bundle.metadata is not None and "directions_norm" in ray_bundle.metadata:
            outputs["directions_norm"] = ray_bundle.metadata["directions_norm"]
        return outputs

    def get_metrics_dict(self, outputs, batch):
        metrics_dict = super().get_metrics_dict(outputs, batch)
        if self.training:
            if self.use_depth_ranking_loss:
                assert "depth_ranking_image" in batch, (
                    "Check dataset or turn off `depth_ranking_loss`")
                metrics_dict["depth_ranking_loss"] = 0.0
                metrics_dict["ranking_loss_multiplier"] = self.ranking_loss_multiplier
                ranking_depth = batch["depth_ranking_image"].to(self.device)
            if self.config.use_depth_loss:
                assert "depth_image" in batch, (
                    "Check dataset or turn off `depth_loss`")
                metrics_dict["depth_loss"] = 0.0
                metrics_dict["sigma"] = self.depth_sigma_current.item()
                termination_depth = batch["depth_image"].to(self.device)
                
            for i in range(len(outputs["weights_list"])):
                if self.config.use_depth_loss:
                    metrics_dict["depth_loss"] += self.depth_loss(
                        density=outputs["density_list"][i],
                        weights=outputs["weights_list"][i],
                        ray_samples=outputs["ray_samples_list"][i],
                        termination_depth=termination_depth,
                        predicted_depth=outputs[f"prop_depth_{i}"],
                        sigma=self.depth_sigma_current,
                        directions_norm=outputs["directions_norm"],
                    ) / len(outputs["weights_list"])

                if self.use_depth_ranking_loss:
                    expected_depth = self.renderer_expected_depth(
                        weights=outputs["weights_list"][i],
                        ray_samples=outputs["ray_samples_list"][i],
                    )
                    metrics_dict["depth_ranking_loss"] += depth_ranking_loss(
                        rendered_depth=expected_depth,
                        gt_depth=ranking_depth,
                    ) / len(outputs["weights_list"])

        return metrics_dict

    def get_training_callbacks(
        self, training_callback_attributes: TrainingCallbackAttributes,
    ) -> List[TrainingCallback]:
        callbacks = super().get_training_callbacks(training_callback_attributes)

        callbacks.append(
            TrainingCallback(
                where_to_run=[TrainingCallbackLocation.BEFORE_TRAIN_ITERATION],
                update_every_num_iters=1,
                func=self._set_ranking_loss_multiplier,
            )
        )

        if self.config.should_decay_sigma:
            callbacks.append(
                TrainingCallback(
                    where_to_run=[TrainingCallbackLocation.BEFORE_TRAIN_ITERATION],
                    update_every_num_iters=1,
                    func=self._set_depth_sigma,
                )
            )

        return callbacks

    def get_loss_dict(self, outputs, batch, metrics_dict=None):
        loss_dict = super().get_loss_dict(outputs, batch, metrics_dict)
        if self.training:
            assert metrics_dict is not None
            if "depth_ranking_loss" in metrics_dict:
                loss_dict["depth_ranking_loss"] = self.ranking_loss_multiplier * metrics_dict["depth_ranking_loss"]
            if "depth_loss" in metrics_dict:
                loss_dict["depth_loss"] = self.config.depth_loss_mult * metrics_dict["depth_loss"]

        return loss_dict

    @torch.no_grad()
    def _visualise_weights(
        self,
        outputs: Dict[str, Union[Tensor, List[Tensor]]],
        batch: Dict[str, Tensor],
    ) -> Dict[str, List[Figure]]:
        figures_dict = {}
        if self.vis_weights_dist:
            vis_indexes = outputs.get("vis_indexes", None)
            termination_depth = batch.get("depth_image", None)

            if termination_depth is not None and vis_indexes is not None:
                termination_depth = termination_depth[
                    vis_indexes[0][..., 0],
                    vis_indexes[0][..., 1],
                ].cpu()

            figures_dict["weights_dist"] = [
                matplotlib_utis.plot_weights_distribution_multiprop(
                    weights=outputs["weights_list"],
                    steps=outputs["ray_steps_list"],
                    termination_depth=termination_depth,
                    i=i,
                )
                for i in range(self.config.num_rays_to_visualize)
            ]
        return figures_dict

    def get_image_metrics_and_images(
        self,
        outputs: Dict[str, Union[Tensor, List[Tensor]]],
        batch: Dict[str, Tensor],
    ) -> Tuple[
        Dict[str, float],
        Dict[str, Tensor],
        Dict[str, List[Figure]],
    ]:
        """Appends ground truth depth to the depth image."""
        metrics, images, figures = super().get_image_metrics_and_images(outputs, batch)

        depths_show = []
        near_plane, far_plane = None, None
        if self.config.use_depth_loss:
            termination_depth = batch["depth_image"].to(self.device)
            if not self.config.is_euclidean_depth:
                termination_depth = termination_depth * outputs["directions_norm"]

            near_plane = torch.min(termination_depth).cpu().item()
            far_plane = torch.max(termination_depth).cpu().item()

            depths_show.append(colormaps.apply_depth_colormap(termination_depth))

        if self.use_depth_ranking_loss:
            ranking_depth = batch["depth_ranking_image"].to(self.device)
            depths_show.append(colormaps.apply_depth_colormap(ranking_depth))

        predicted_depth = outputs[f"prop_depth_{self.config.num_proposal_iterations}"]

        predicted_depth_colormap = colormaps.apply_depth_colormap(
            depth=predicted_depth,
            accumulation=outputs["accumulation"],
            near_plane=near_plane,
            far_plane=far_plane,
        )
        depths_show.append(predicted_depth_colormap)
        images["depth"] = torch.cat(depths_show, dim=1)

        if self.config.use_depth_loss:
            depth_mask = termination_depth > 0
            metrics["depth_mse"] = torch.nn.functional.mse_loss(
                predicted_depth[depth_mask],
                termination_depth[depth_mask]
            ).cpu().item()

        return metrics, images, figures

    def _set_depth_sigma(self, step: int) -> None:
        """Sets up ranking loss multiplier."""

        self.depth_sigma_current = torch.maximum(
            self.config.sigma_decay_rate * self.depth_sigma_current,
            self.depth_sigma_base,
        )

    def _set_ranking_loss_multiplier(self, step: int) -> None:
        """Sets up ranking loss multiplier."""
        self.ranking_loss_multiplier: float = self.config.depth_ranking_loss_mult * np.interp(
            step, [0, 5000], [0, 0.2]
        ).item()
