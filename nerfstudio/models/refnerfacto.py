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
from typing import Type, Optional, Tuple

from nerfstudio.models.nerfacto import NerfactoModel, NerfactoModelConfig
from nerfstudio.model_components.losses import CharbonnierLoss
from nerfstudio.fields.refnerfacto_field import RefNerfactoField
from nerfstudio.model_components.renderers import RoughnessRenderer


@dataclass
class RefNerfactoModelConfig(NerfactoModelConfig):
    """RefNerfacto Model Config"""

    _target: Type = field(default_factory=lambda: RefNerfactoModel)
    bottleneck_width: int = 256
    """The width of the bottleneck vector. Defaults to 256."""
    deg_view: int = 4
    """Degree of encoding for viewdirs or refdirs."""
    use_reflections: bool = True
    """If True, use refdirs instead of viewdirs."""
    use_ide_enc: bool = True
    """If True, use IDE to encode directions."""
    use_pred_roughness: bool = True
    """If False and if use_directional_enc is True, use zero roughness in IDE."""
    roughness_bias: float = -1.0
    """Shift added to raw roughness pre-activation."""
    use_pred_diffuse_color: bool = True
    """If True, predict the diffuse & specular colors."""
    use_pred_specular_tint: bool = True
    """If True, predict the specular tint."""
    use_n_dot_v: bool = True
    """If True, feed dot(n * viewdir) to 2nd MLP."""
    rgb_premultiplier: float = 1.0
    """Premultiplier on RGB before activation."""
    rgb_bias: float = 0.0
    """The shift added to raw colors pre-activation."""
    rgb_padding: float = 0.001
    """Padding added to the RGB outputs."""
    predict_normals: bool = True
    """Whether to predict normals or not."""
    use_bundle_adjust: bool = False
    """Whether to bundle adjust (BARF)"""
    coarse_to_fine_iters: Optional[Tuple[float, float]] = (0.0, 0.1)
    """Iterations (as a percentage of total iterations) at which coarse to fine hash grid optimization starts and ends.
    Linear interpolation between (start, end) and full activation of hash grid from end onwards."""
    compute_hash_regularization: bool = True
    """Whether to compute regularization on hash weights."""
    supervise_pred_normals_by_density: bool = True
    """Whether to supervise predicted normals by density."""


class RefNerfactoModel(NerfactoModel):
    """RefNerfacto model

    Args:
        config: RefNerfactoModel configuration to instantiate model
    """

    config: RefNerfactoModelConfig

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

        # Fields
        self.field = RefNerfactoField(
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

        # Losses
        self.rgb_loss = CharbonnierLoss()

        self.roughness_renderer = RoughnessRenderer()