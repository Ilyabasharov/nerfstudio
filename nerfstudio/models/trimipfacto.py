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
RefNerfacto implementation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Type

from nerfstudio.models.nerfacto import NerfactoModel, NerfactoModelConfig
from nerfstudio.model_components.losses import CharbonnierLoss
from nerfstudio.fields.trimipfacto_field import TriMipfactoField


@dataclass
class TriMipfactoModelConfig(NerfactoModelConfig):
    """RefNerfacto Model Config"""

    _target: Type = field(default_factory=lambda: TriMipfactoModel)
    geo_feat_dim: int = 127
    """Geo feature vector dim."""
    predict_normals: bool = True
    """Whether to predict normals or not."""
    supervise_pred_normals_by_density: bool = True
    """Whether to supervise predicted normals by density."""
    occ_grid_resolution: int = 128
    """Whether to supervise predicted normals by density."""



class TriMipfactoModel(NerfactoModel):
    """RefNerfacto model

    Args:
        config: RefNerfactoModel configuration to instantiate model
    """

    config: TriMipfactoModelConfig

    def populate_modules(self):
        """Set the fields and modules."""
        super().populate_modules()

        # Fields
        self.field = TriMipfactoField(
            self.scene_box.aabb,
            geo_feat_dim=self.config.geo_feat_dim,
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
            regularize_function=self.regularize_function,
            compute_hash_regularization=self.config.compute_hash_regularization,
            compute_appearence_regularization=self.config.compute_appearence_regularization,
            implementation=self.config.implementation,
        )

        # Losses
        self.rgb_loss = CharbonnierLoss()
