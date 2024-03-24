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
Field for compound nerf model, adds scene contraction and image embeddings to instant ngp
"""

from typing import Literal, Optional, Tuple, Callable

import math
import torch
from torch import Tensor, nn

from nerfstudio.cameras.rays import RaySamples
from nerfstudio.data.scene_box import SceneBox
from nerfstudio.field_components.activations import trunc_exp
from nerfstudio.field_components.encodings import TriMipEncoding
from nerfstudio.field_components.mlp import MLP
from nerfstudio.field_components.spatial_distortions import SpatialDistortion
from nerfstudio.cameras.bundle_adjustment import HashBundleAdjustment
from nerfstudio.fields.nerfacto_field import NerfactoField


class TriMipfactoField(NerfactoField):
    """Compound Field that uses TCNN

    Args:
        aabb: parameters of scene aabb bounds
        bundle_adjustment: BARF technique for adjust poses
        num_images: number of images in the dataset
        num_layers: number of hidden layers
        hidden_dim: dimension of hidden layers
        geo_feat_dim: output geo feat dimensions
        num_levels: number of levels of the hashmap for the base mlp
        base_res: base resolution of the hashmap for the base mlp
        max_res: maximum resolution of the hashmap for the base mlp
        log2_hashmap_size: size of the hashmap for the base mlp
        num_layers_color: number of hidden layers for color network
        num_layers_transient: number of hidden layers for transient network
        features_per_level: number of features per level for the hashgrid
        hidden_dim_color: dimension of hidden layers for color network
        hidden_dim_transient: dimension of hidden layers for transient network
        appearance_embedding_dim: dimension of appearance embedding
        transient_embedding_dim: dimension of transient embedding
        use_transient_embedding: whether to use transient embedding
        use_semantics: whether to use semantic segmentation
        num_semantic_classes: number of semantic classes
        use_pred_normals: whether to use predicted normals
        use_average_appearance_embedding: whether to use average appearance embedding or zeros for inference
        use_appearance_embedding: whether to use use_appearance_embedding or predict scales to bottleneck vector
        shift_scale_out_dim: if not use use_appearance_embedding, dim of out shift and scale vectors
        spatial_distortion: spatial distortion to apply to the scene
        compute_hash_regularization: whether to compute regularization on hash weights
        compute_appearence_regularization: whether to appearence regularization on embeddings
        regularize_function: type of regularization
    """

    aabb: Tensor

    def __init__(
        self,
        aabb: Tensor,
        num_images: int,
        num_layers: int = 2,
        hidden_dim: int = 64,
        geo_feat_dim: int = 15,
        num_levels: int = 16,
        base_res: int = 16,
        max_res: int = 2048,
        log2_hashmap_size: int = 19,
        num_layers_color: int = 3,
        num_layers_transient: int = 2,
        features_per_level: int = 2,
        hidden_dim_color: int = 64,
        hidden_dim_transient: int = 64,
        appearance_embedding_dim: int = 32,
        use_appearance_embedding: bool = True,
        shift_scale_out_dim: Optional[int] = None,
        transient_embedding_dim: int = 16,
        use_transient_embedding: bool = False,
        use_semantics: bool = False,
        num_semantic_classes: int = 100,
        pass_semantic_gradients: bool = False,
        use_pred_normals: bool = False,
        use_average_appearance_embedding: bool = False,
        spatial_distortion: Optional[SpatialDistortion] = None,
        compute_hash_regularization: bool = False,
        compute_appearence_regularization: bool = False,
        regularize_function: Callable[[Tensor], Tensor] = torch.square,
        density_activation: Callable[[Tensor], Tensor] = lambda x: trunc_exp(x - 1),
        implementation: Literal["tcnn", "torch"] = "tcnn",
        plane_size: int = 512,
        feature_dim: int = 16,
        levels: int = 8,
    ) -> None:
        super().__init__(
            aabb=aabb,
            bundle_adjustment=HashBundleAdjustment(use_bundle_adjust=False),
            num_images=num_images,
            num_layers=num_layers,
            hidden_dim=hidden_dim,
            geo_feat_dim=geo_feat_dim,
            num_levels=num_levels,
            base_res=base_res,
            max_res=max_res,
            log2_hashmap_size=log2_hashmap_size,
            num_layers_color=num_layers_color,
            num_layers_transient=num_layers_transient,
            features_per_level=features_per_level,
            hidden_dim_color=hidden_dim_color,
            hidden_dim_transient=hidden_dim_transient,
            appearance_embedding_dim=appearance_embedding_dim,
            use_appearance_embedding=use_appearance_embedding,
            transient_embedding_dim=transient_embedding_dim,
            use_transient_embedding=use_transient_embedding,
            use_semantics=use_semantics,
            num_semantic_classes=num_semantic_classes,
            pass_semantic_gradients=pass_semantic_gradients,
            use_pred_normals=use_pred_normals,
            use_average_appearance_embedding=use_average_appearance_embedding,
            spatial_distortion=spatial_distortion,
            compute_hash_regularization=compute_hash_regularization,
            regularize_function=regularize_function,
            implementation=implementation,
            shift_scale_out_dim=shift_scale_out_dim,
            compute_appearence_regularization=compute_appearence_regularization,
            density_activation=density_activation,
        )

        self.log2_plane_size = math.log2(plane_size)

        aabb_min, aabb_max = torch.split(self.aabb, 3, dim=-1)
        self.aabb_size = aabb_max - aabb_min
        assert (
            self.aabb_size[0] == self.aabb_size[1] == self.aabb_size[2]
        ), "Current implementation only supports cube aabb"

        self.feature_vol_radii = self.aabb_size[0] / 2.0

        self.mlp_base_grid = TriMipEncoding(
            n_levels=levels,
            plane_size=plane_size,
            feature_dim=feature_dim,
        )
        self.mlp_base_mlp = MLP(
            in_dim=self.mlp_base_grid.get_out_dim(),
            num_layers=num_layers,
            layer_width=hidden_dim,
            out_dim=1 + self.geo_feat_dim,
            activation=nn.ReLU(),
            out_activation=None,
            implementation=implementation,
        )

    def get_density(self, ray_samples: RaySamples) -> Tuple[Tensor, Tensor]:
        """Computes and returns the densities."""
        positions = ray_samples.frustums.get_positions()
        if self.spatial_distortion is not None:
            positions = self.spatial_distortion(positions)
            positions = (positions + 2.0) / 4.0
        else:
            positions = SceneBox.get_normalized_positions(positions, self.aabb)

        distance = ray_samples.frustums.get_steps()
        radiis = ray_samples.frustums.radii

        assert ray_samples.metadata

        sample_ball_radii = self.compute_ball_radii(
            distance, radiis, cos
        )
        level_vol = torch.log2(
            sample_ball_radii / self.feature_vol_radii
        )  # real level should + log2(feature_resolution)

        # Make sure the tcnn gets inputs between 0 and 1.
        selector = ((positions > 0.0) & (positions < 1.0)).all(dim=-1)
        positions = positions * selector[..., None]
        self._sample_locations = positions
        if not self._sample_locations.requires_grad:
            self._sample_locations.requires_grad = True
        hashgrid_vecs = self.mlp_base_grid(
            positions.view(-1, 3),
            level=torch.empty_like(positions[..., 0]).fill_(self.occ_level_vol),
        )
        h = self.mlp_base_mlp(hashgrid_vecs).view(*ray_samples.frustums.shape, -1)
        density_before_activation, base_mlp_out = torch.split(h, [1, self.geo_feat_dim], dim=-1)
        self._density_before_activation = density_before_activation

        # Rectifying the density with an exponential is much more stable than a ReLU or
        # softplus, because it enables high post-activation (float32) density outputs
        # from smaller internal (float16) parameters.
        density = self.density_activation(density_before_activation.to(positions))
        density = density * selector[..., None]

        return density, base_mlp_out
    
    @staticmethod
    def compute_ball_radii(distance: Tensor, radiis: Tensor, cos: Tensor) -> Tensor:
        inverse_cos = 1.0 / cos
        tmp = (inverse_cos * inverse_cos - 1).sqrt() - radiis
        sample_ball_radii = distance * radiis * cos / (tmp * tmp + 1.0).sqrt()
        return sample_ball_radii
