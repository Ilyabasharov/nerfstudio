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
ZipRefNerfacto implementation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Type

from nerfstudio.models.zipnerfacto import ZipNerfactoModel, ZipNerfactoModelConfig
from nerfstudio.models.refnerfacto import RefNerfactoModelConfig
from nerfstudio.fields.ziprefnerfacto_field import ZipRefNerfactoField
from nerfstudio.model_components.renderers import RoughnessRenderer


@dataclass
class ZipRefNerfactoModelConfig(ZipNerfactoModelConfig, RefNerfactoModelConfig):
    """ZipRefNerfacto Model Config"""

    _target: Type = field(default_factory=lambda: ZipRefNerfactoModel)

    predict_normals: bool = True
    """Whether to predict normals or not."""
    supervise_pred_normals_by_density: bool = True
    """Whether to supervise predicted normals by density."""
    use_appearance_embedding: bool = False
    """whether to use use_appearance_embedding or predict scales to bottleneck vector."""


class ZipRefNerfactoModel(ZipNerfactoModel):
    """ZipRefNerfacto model. Combines RefField with multisampling ZipNerf.

    Args:
        config: ZipRefNerfactoModel configuration to instantiate model
    """

    config: ZipRefNerfactoModelConfig

    def populate_modules(self):
        """Set the fields and modules."""
        super().populate_modules()

        # Make sure that normals are computed if reflection direction is used.
        if self.config.use_reflections and not self.config.predict_normals:
            raise ValueError(
                'Normals must be computed for reflection directions.')
        
        if not (self.config.use_ide_enc and self.config.use_pred_roughness):
            raise ValueError(
                'IDE cannot be computed without roughness.')
        
        # combined field
        self.field = ZipRefNerfactoField(
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
            use_appearance_embedding=self.config.use_appearance_embedding,
            appearance_embedding_dim=self.config.appearance_embed_dim,
            scale_featurization=self.config.scale_featurization,
            regularize_function=self.regularize_function,
            compute_hash_regularization=self.config.compute_hash_regularization,
            bottleneck_width=self.config.bottleneck_width,
            deg_view=self.config.deg_view,
            use_reflections=self.config.use_reflections,
            use_ide_enc=self.config.use_ide_enc,
            use_pred_roughness=self.config.use_pred_roughness,
            roughness_bias=self.config.roughness_bias,
            use_pred_diffuse_color=self.config.use_pred_diffuse_color,
            use_pred_specular_tint=self.config.use_pred_specular_tint,
            use_n_dot_v=self.config.use_n_dot_v,
            rgb_premultiplier=self.config.rgb_premultiplier,
            rgb_bias=self.config.rgb_bias,
            rgb_padding=self.config.rgb_padding,
            implementation=self.config.implementation,
        )

        self.roughness_renderer = RoughnessRenderer()
