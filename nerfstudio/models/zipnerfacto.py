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
ZipNerfacto implementation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Literal, Tuple, Type

import torch

from nerfstudio.models.nerfacto import NerfactoModel, NerfactoModelConfig
from nerfstudio.field_components.spatial_distortions import LinearizedSceneContraction
from nerfstudio.model_components.losses import (
    zipnerf_loss,
    CharbonnierLoss,
)
from nerfstudio.fields.zipnerfacto_field import ZipNerfactoField
from nerfstudio.model_components.ray_samplers import PowerSampler
from nerfstudio.fields.density_fields import HashMLPGaussianDensityField


@dataclass
class ZipNerfactoModelConfig(NerfactoModelConfig):
    """ZipNerfacto Model Config"""

    _target: Type = field(default_factory=lambda: ZipNerfactoModel)

    proposal_weights_anneal_max_num_iters: int = 1
    """Max num iterations for the annealing function."""
    use_same_proposal_network: bool = False
    """Use the same proposal network. Otherwise use different ones."""
    proposal_net_args_list: List[Dict] = field(
        default_factory=lambda: [
            {"hidden_dim": 16, "log2_hashmap_size": 17, "num_levels": 5, "max_res": 256, "use_linear": False, "features_per_level": 2},
            {"hidden_dim": 16, "log2_hashmap_size": 17, "num_levels": 7, "max_res": 4096, "use_linear": False, "features_per_level": 4},
        ]
    )
    """Arguments for the proposal density fields."""
    max_res: int = 8192
    """Maximum resolution of the hashmap for the base mlp."""
    log2_hashmap_size: int = 22
    """Size of the hashmap for the base mlp"""
    features_per_level: int = 4
    """How many hashgrid features per level"""
    num_nerf_samples_per_ray: int = 64
    """Number of samples per ray for the nerf network."""
    num_proposal_samples_per_ray: Tuple[int, ...] = (256, 128)
    """Number of samples per ray for each proposal network."""
    interlevel_loss_mult: float = 1
    """Proposal loss multiplier."""
    scale_featurization: bool = True
    """Scale featurization from appendix of ZipNeRF."""
    regularize_function: Literal["abs", "square"] = "square"
    """Type of regularization."""
    compute_hash_regularization: bool = True
    """Whether to compute regularization on hash weights."""
    proposal_initial_sampler: Literal["power"] = "power"
    """Initial sampler for the proposal network."""
    interlevel_loss_type: Literal["zipnerf"] = "zipnerf"
    """Type of interlevel loss."""
    implementation: Literal["tcnn", "torch"] = "tcnn"
    """Which implementation to use for the model."""


class ZipNerfactoModel(NerfactoModel):
    """ZipNerfacto model

    Args:
        config: ZipNerfactoModel configuration to instantiate model
    """

    config: ZipNerfactoModelConfig

    def populate_modules(self):
        """Set the fields and modules."""
        super().populate_modules()

        self.scene_contraction = LinearizedSceneContraction(order=float("inf"))

        # Fields
        self.field = ZipNerfactoField(
            self.scene_box.aabb,
            self.bundle_adjustment,
            hidden_dim=self.config.hidden_dim,
            num_levels=self.config.num_levels,
            max_res=self.config.max_res,
            log2_hashmap_size=self.config.log2_hashmap_size,
            hidden_dim_color=self.config.hidden_dim_color,
            hidden_dim_transient=self.config.hidden_dim_transient,
            spatial_distortion=self.scene_contraction,
            num_images=self.num_train_data,
            use_pred_normals=self.config.predict_normals,
            use_average_appearance_embedding=self.config.use_average_appearance_embedding,
            appearance_embedding_dim=self.config.appearance_embed_dim,
            scale_featurization=self.config.scale_featurization,
            regularize_function=self.regularize_function,
            compute_hash_regularization=self.config.compute_hash_regularization,
            implementation=self.config.implementation,
        )

        self.density_fns = []
        num_prop_nets = self.config.num_proposal_iterations
        # Build the proposal network(s)
        self.proposal_networks = torch.nn.ModuleList()
        if self.config.use_same_proposal_network:
            assert len(self.config.proposal_net_args_list) == 1, "Only one proposal network is allowed."
            prop_net_args = self.config.proposal_net_args_list[0]
            network = HashMLPGaussianDensityField(
                self.scene_box.aabb,
                self.bundle_adjustment,
                spatial_distortion=self.scene_contraction,
                **prop_net_args,
                scale_featurization=self.config.scale_featurization,
                regularize_function=self.regularize_function,
                compute_hash_regularization=self.config.compute_hash_regularization,
                implementation=self.config.implementation,
            )
            self.proposal_networks.append(network)
            self.density_fns.extend([network.density_fn for _ in range(num_prop_nets)])
        else:
            for i in range(num_prop_nets):
                prop_net_args = self.config.proposal_net_args_list[min(i, len(self.config.proposal_net_args_list) - 1)]
                network = HashMLPGaussianDensityField(
                    self.scene_box.aabb,
                    self.bundle_adjustment,
                    spatial_distortion=self.scene_contraction,
                    **prop_net_args,
                    scale_featurization=self.config.scale_featurization,
                    regularize_function=self.regularize_function,
                    compute_hash_regularization=self.config.compute_hash_regularization,
                    implementation=self.config.implementation,
                )
                self.proposal_networks.append(network)
            self.density_fns.extend([network.density_fn for network in self.proposal_networks])

        # Samplers
        self.proposal_sampler.initial_sampler = PowerSampler(
            single_jitter=self.config.use_single_jitter,
        )

        # Losses
        self.rgb_loss = CharbonnierLoss()

        self.interlevel_loss = zipnerf_loss
