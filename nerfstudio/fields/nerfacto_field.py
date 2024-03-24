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

from typing import Dict, Literal, Optional, Tuple, Callable, Union
from jaxtyping import Int, Float

import torch
from torch import Tensor, nn

from nerfstudio.cameras.rays import RaySamples, RayBundle
from nerfstudio.data.scene_box import SceneBox
from nerfstudio.field_components.activations import trunc_exp
from nerfstudio.field_components.embedding import Embedding
from nerfstudio.field_components.encodings import HashEncoding, NeRFEncoding, SHEncoding
from nerfstudio.field_components.field_heads import (
    FieldHeadNames,
    PredNormalsFieldHead,
    SemanticFieldHead,
    TransientDensityFieldHead,
    TransientRGBFieldHead,
    UncertaintyFieldHead,
)
from nerfstudio.field_components.mlp import MLP
from nerfstudio.field_components.spatial_distortions import SpatialDistortion
from nerfstudio.fields.base_field import Field, get_normalized_directions
from nerfstudio.cameras.bundle_adjustment import HashBundleAdjustment


class NerfactoField(Field):
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
        bundle_adjustment: HashBundleAdjustment,
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
    ) -> None:
        super().__init__()

        self.register_buffer("aabb", aabb)
        self.geo_feat_dim = geo_feat_dim

        self.register_buffer("max_res", torch.tensor(max_res))
        self.register_buffer("num_levels", torch.tensor(num_levels))
        self.register_buffer("features_per_level", torch.tensor(features_per_level))
        self.register_buffer("log2_hashmap_size", torch.tensor(log2_hashmap_size))

        self.spatial_distortion = spatial_distortion
        self.num_images = num_images
        self.appearance_embedding_dim = appearance_embedding_dim
        self.embedding_appearance = Embedding(self.num_images, self.appearance_embedding_dim)
        self.use_average_appearance_embedding = use_average_appearance_embedding
        self.use_appearance_embedding = use_appearance_embedding
        self.shift_scale_out_dim = shift_scale_out_dim
        self.use_transient_embedding = use_transient_embedding
        self.use_semantics = use_semantics
        self.use_pred_normals = use_pred_normals
        self.pass_semantic_gradients = pass_semantic_gradients
        self.base_res = base_res
        self.compute_hash_regularization = compute_hash_regularization
        self.compute_appearence_regularization = compute_appearence_regularization
        self.regularize_function = regularize_function
        self.hidden_dim = hidden_dim
        self.density_activation = density_activation

        self.direction_encoding = SHEncoding(
            levels=4,
            implementation=implementation,
        )

        self.position_encoding = NeRFEncoding(
            in_dim=3,
            num_frequencies=2,
            min_freq_exp=0,
            max_freq_exp=2 - 1,
            implementation=implementation,
        )

        self.mlp_base_grid = HashEncoding(
            bundle_adjustment=bundle_adjustment,
            num_levels=num_levels,
            min_res=base_res,
            max_res=max_res,
            log2_hashmap_size=log2_hashmap_size,
            features_per_level=features_per_level,
            implementation=implementation,
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

        # transients
        if self.use_transient_embedding:
            self.transient_embedding_dim = transient_embedding_dim
            self.embedding_transient = Embedding(self.num_images, self.transient_embedding_dim)
            self.mlp_transient = MLP(
                in_dim=self.geo_feat_dim + self.transient_embedding_dim,
                num_layers=num_layers_transient,
                layer_width=hidden_dim_transient,
                out_dim=hidden_dim_transient,
                activation=nn.ReLU(),
                out_activation=None,
                implementation=implementation,
            )
            self.field_head_transient_uncertainty = UncertaintyFieldHead(in_dim=self.mlp_transient.get_out_dim())
            self.field_head_transient_rgb = TransientRGBFieldHead(in_dim=self.mlp_transient.get_out_dim())
            self.field_head_transient_density = TransientDensityFieldHead(in_dim=self.mlp_transient.get_out_dim())

        # semantics
        if self.use_semantics:
            self.mlp_semantics = MLP(
                in_dim=self.geo_feat_dim,
                num_layers=2,
                layer_width=64,
                out_dim=hidden_dim_transient,
                activation=nn.ReLU(),
                out_activation=None,
                implementation=implementation,
            )
            self.field_head_semantics = SemanticFieldHead(
                in_dim=self.mlp_semantics.get_out_dim(), num_classes=num_semantic_classes
            )

        # predicted normals
        if self.use_pred_normals:
            self.mlp_pred_normals = MLP(
                in_dim=self.geo_feat_dim + self.position_encoding.get_out_dim(),
                num_layers=3,
                layer_width=64,
                out_dim=hidden_dim_transient,
                activation=nn.ReLU(),
                out_activation=None,
                implementation=implementation,
            )
            self.field_head_pred_normals = PredNormalsFieldHead(in_dim=self.mlp_pred_normals.get_out_dim())

        mlp_head_in_dim = self.direction_encoding.get_out_dim() + self.geo_feat_dim
        if self.use_appearance_embedding:
            mlp_head_in_dim += self.appearance_embedding_dim

        self.mlp_head = MLP(
            in_dim=mlp_head_in_dim,
            num_layers=num_layers_color,
            layer_width=hidden_dim_color,
            out_dim=3,
            activation=nn.ReLU(),
            out_activation=nn.Sigmoid(),
            implementation=implementation,
        )

        if not self.use_appearance_embedding:
            if self.shift_scale_out_dim is None:
                self.shift_scale_out_dim = self.geo_feat_dim

            self.affine_mlp = MLP(
                in_dim=self.appearance_embedding_dim,
                num_layers=2,
                layer_width=128,
                out_dim=self.shift_scale_out_dim * 2,
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
        # Make sure the tcnn gets inputs between 0 and 1.
        selector = ((positions > 0.0) & (positions < 1.0)).all(dim=-1)
        positions = positions * selector[..., None]
        self._sample_locations = positions
        if not self._sample_locations.requires_grad:
            self._sample_locations.requires_grad = True
        hashgrid_vecs = self.mlp_base_grid(positions.view(-1, 3))
        h = self.mlp_base_mlp(hashgrid_vecs).view(*ray_samples.frustums.shape, -1)
        density_before_activation, base_mlp_out = torch.split(h, [1, self.geo_feat_dim], dim=-1)
        self._density_before_activation = density_before_activation

        # Rectifying the density with an exponential is much more stable than a ReLU or
        # softplus, because it enables high post-activation (float32) density outputs
        # from smaller internal (float16) parameters.
        density = self.density_activation(density_before_activation.to(positions))
        density = density * selector[..., None]

        return density, base_mlp_out

    def _get_appearance_embedding(
        self,
        camera_indices: Int[Tensor, "*bs"],
    ) -> Tuple[
        Float[Tensor, "*bs appearance_embedding_dim"],
        Union[float, Float[Tensor, "*bs geo_feat_dim"]],
        Union[float, Float[Tensor, "*bs geo_feat_dim"]],
    ]:
        """Appearance prediction."""
        scale, shift = 1, 0
        if self.training:
            embedded_appearance = self.embedding_appearance(camera_indices)
            if not self.use_appearance_embedding:
                output = self.affine_mlp(embedded_appearance.view(-1, self.appearance_embedding_dim))
                scale, shift = torch.split(output, [self.shift_scale_out_dim, self.shift_scale_out_dim], dim=-1)  # type: ignore
                scale = trunc_exp(scale)
        else:
            multiplier = self.embedding_appearance.mean(dim=0) if self.use_average_appearance_embedding else 0.0
            embedded_appearance = torch.ones(
                (*camera_indices.shape, self.appearance_embedding_dim),
                device=camera_indices.device,
            ) * multiplier
            
        return embedded_appearance, scale, shift

    def get_outputs(
        self,
        ray_samples: RaySamples,
        ray_bundle: Optional[RayBundle] = None,
        density_embedding: Optional[Tensor] = None
    ) -> Dict[FieldHeadNames, Tensor]:
        assert density_embedding is not None
        outputs = {}
        if ray_samples.camera_indices is None:
            raise AttributeError("Camera indices are not provided.")
        camera_indices = ray_samples.camera_indices.squeeze()
        directions = get_normalized_directions(ray_samples.frustums.directions)
        directions_flat = directions.view(-1, 3)
        d = self.direction_encoding(directions_flat)

        outputs_shape = ray_samples.frustums.directions.shape[:-1]

        # appearance
        embedded_appearance, scale, shift = self._get_appearance_embedding(camera_indices)

        # transients
        if self.use_transient_embedding and self.training:
            embedded_transient = self.embedding_transient(camera_indices)
            transient_input = torch.cat(
                [
                    density_embedding.view(-1, self.geo_feat_dim),
                    embedded_transient.view(-1, self.transient_embedding_dim),
                ],
                dim=-1,
            )
            x = self.mlp_transient(transient_input).view(*outputs_shape, -1).to(directions)
            outputs[FieldHeadNames.UNCERTAINTY] = self.field_head_transient_uncertainty(x)
            outputs[FieldHeadNames.TRANSIENT_RGB] = self.field_head_transient_rgb(x)
            outputs[FieldHeadNames.TRANSIENT_DENSITY] = self.field_head_transient_density(x)

        # semantics
        if self.use_semantics:
            semantics_input = density_embedding.view(-1, self.geo_feat_dim)
            if not self.pass_semantic_gradients:
                semantics_input = semantics_input.detach()

            x = self.mlp_semantics(semantics_input).view(*outputs_shape, -1).to(directions)
            outputs[FieldHeadNames.SEMANTICS] = self.field_head_semantics(x)

        # predicted normals
        if self.use_pred_normals:
            positions = ray_samples.frustums.get_positions()

            positions_flat = self.position_encoding(positions.view(-1, 3))
            pred_normals_inp = torch.cat([positions_flat, density_embedding.view(-1, self.geo_feat_dim)], dim=-1)

            x = self.mlp_pred_normals(pred_normals_inp).view(*outputs_shape, -1).to(directions)
            outputs[FieldHeadNames.PRED_NORMALS] = self.field_head_pred_normals(x)

        inputs_mlp_head = [
            d,
            density_embedding.view(-1, self.geo_feat_dim) * scale + shift,
        ]

        if self.use_appearance_embedding:
            inputs_mlp_head.append(
                embedded_appearance.view(-1, self.appearance_embedding_dim),
            )

        outputs[FieldHeadNames.RGB] = self.mlp_head(
            torch.cat(inputs_mlp_head, dim=-1),
        ).view(*outputs_shape, -1).to(directions)

        # finally, compute regularisations for stable training
        self._compute_regularisations(outputs, camera_indices)

        return outputs

    def _compute_regularisations(
        self,
        outputs: Dict[FieldHeadNames, Tensor],
        camera_indices: Int[Tensor, "num_trainable_cameras"],
    ) -> None:
        if self.training:
            if self.compute_hash_regularization:
                outputs[FieldHeadNames.HASH_DECAY] = \
                    self.mlp_base_grid.regularize_hash_pyramid(self.regularize_function)
            if self.compute_appearence_regularization:
                outputs[FieldHeadNames.APPEARENCE_DECAY] = \
                    self.embedding_appearance.regularize_appearence(
                        self.regularize_function,
                        camera_indices=camera_indices,
                    )
