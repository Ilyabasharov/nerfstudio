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
Advanced Field from RefNeRF.
"""

from typing import Literal, Optional, Callable, Dict

import torch
import math
from torch import Tensor, nn

from nerfstudio.fields.base_field import get_normalized_directions
from nerfstudio.cameras.rays import RaySamples, RayBundle
from nerfstudio.field_components.mlp import MLP
from nerfstudio.field_components.spatial_distortions import SpatialDistortion
from nerfstudio.fields.nerfacto_field import NerfactoField
from nerfstudio.cameras.bundle_adjustment import HashBundleAdjustment
from nerfstudio.field_components.encodings import IDEncoding
from nerfstudio.field_components.field_heads import FieldHeadNames
from nerfstudio.utils.math import reflect, linear_to_srgb
from nerfstudio.field_components.activations import trunc_exp

REF_CONSTANT = math.log(3.0)


class RefNerfactoField(NerfactoField):
    """Compound Field that uses TCNN

    Args:
        aabb: parameters of scene aabb bounds
        bundle_adjustment: BARF technique for adjust poses
        num_images: number of images in the dataset
        num_layers: number of hidden layers
        hidden_dim: dimension of hidden layers
        geo_feat_dim: output geo feat dimensions
        num_levels: number of levels of the hashmap for the base mlp
        max_res: maximum resolution of the hashmap for the base mlp
        log2_hashmap_size: size of the hashmap for the base mlp
        num_layers_color: number of hidden layers for color network
        features_per_level: number of features per level for the hashgrid
        num_layers_transient: number of hidden layers for transient network
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
        spatial_distortion: spatial distortion to apply to the scene
        regularize_function: type of regularization
        deg_view: degree of encoding for viewdirs or refdirs
        use_ide_enc: if true, use IDE to encode directions
        use_pred_roughness: if False and if use_ide_enc is True, use zero roughness in IDE
        roughness_bias: shift added to raw roughness pre-activation
        use_pred_diffuse_color: if True, predict the diffuse & specular colors
        use_pred_specular_tint: if True, predict the specular tint
        use_n_dot_v: if True, feed dot(n * viewdir) to 2nd MLP
        bottleneck_width: the width of the bottleneck vector
        rgb_padding: padding added to the RGB outputs
        rgb_premultiplier: premultiplier on RGB before activation
        rgb_bias: the shift added to raw colors pre-activation
        use_reflections: whether to use reflections or not
        compute_hash_regularization: whether to compute regularization on hash weights
    """

    aabb: Tensor
    bundle_adjustment: HashBundleAdjustment

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
        transient_embedding_dim: int = 16,
        use_transient_embedding: bool = False,
        use_semantics: bool = False,
        num_semantic_classes: int = 100,
        pass_semantic_gradients: bool = False,
        use_pred_normals: bool = False,
        use_average_appearance_embedding: bool = False,
        use_appearance_embedding: bool = False,
        spatial_distortion: Optional[SpatialDistortion] = None,
        regularize_function: Callable[[Tensor], Tensor] = torch.square,
        compute_hash_regularization: bool = True,
        use_reflections: bool = True,
        deg_view: int = 4,
        use_ide_enc: bool = True,
        use_pred_roughness: bool = True,
        roughness_bias: float = -1.0,
        use_pred_diffuse_color: float = True,
        use_pred_specular_tint: float = True,
        use_n_dot_v: float = True,
        bottleneck_width: int = 256,
        rgb_padding: float = 0.001,
        rgb_premultiplier: float = 1.0,
        rgb_bias: float = 0.0,
        implementation: Literal["tcnn", "torch"] = "tcnn",
    ) -> None:
        super().__init__(
            aabb=aabb,
            bundle_adjustment=bundle_adjustment,
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
            transient_embedding_dim=transient_embedding_dim,
            use_transient_embedding=use_transient_embedding,
            use_semantics=use_semantics,
            num_semantic_classes=num_semantic_classes,
            pass_semantic_gradients=pass_semantic_gradients,
            use_pred_normals=use_pred_normals,
            use_average_appearance_embedding=use_average_appearance_embedding,
            use_appearance_embedding=use_appearance_embedding,
            shift_scale_out_dim=bottleneck_width,
            spatial_distortion=spatial_distortion,
            compute_hash_regularization=compute_hash_regularization,
            regularize_function=regularize_function,
            implementation=implementation,
        )

        self.use_ide_enc = use_ide_enc
        self.use_reflections = use_reflections
        self.use_pred_roughness = use_pred_roughness
        self.use_pred_diffuse_color = use_pred_diffuse_color
        self.use_pred_specular_tint = use_pred_specular_tint
        self.use_n_dot_v = use_n_dot_v
        self.rgb_padding = rgb_padding
        self.rgb_premultiplier = rgb_premultiplier
        self.rgb_bias = rgb_bias
        self.bottleneck_width = bottleneck_width

        in_dim_mlps = self.geo_feat_dim + self.position_encoding.get_out_dim()

        self.mlp_pred_bottleneck = MLP(
            in_dim=in_dim_mlps,
            num_layers=3,
            layer_width=64,
            out_dim=bottleneck_width,
            activation=nn.ReLU(),
            out_activation=None,
            implementation=implementation,
        )

        if use_pred_roughness:
            self.roughness_bias = roughness_bias
            self.mlp_pred_raw_roughness = MLP(
                in_dim=in_dim_mlps,
                num_layers=3,
                layer_width=64,
                out_dim=1,
                activation=nn.ReLU(),
                out_activation=None,
                implementation=implementation,
            )

        if use_pred_diffuse_color:
            self.mlp_pred_raw_diffuse_color = MLP(
                in_dim=in_dim_mlps,
                num_layers=3,
                layer_width=64,
                out_dim=3,
                activation=nn.ReLU(),
                out_activation=None,
                implementation=implementation,
            )

        if use_pred_specular_tint:
            self.mlp_pred_specular_tint = MLP(
                in_dim=in_dim_mlps,
                num_layers=3,
                layer_width=64,
                out_dim=3,
                activation=nn.ReLU(),
                out_activation=nn.Sigmoid(),
                implementation=implementation,
            )

        mlp_head_in_dim = bottleneck_width
        
        if self.use_appearance_embedding:
            mlp_head_in_dim += appearance_embedding_dim

        if use_n_dot_v:
            mlp_head_in_dim += 1
        
        if use_ide_enc:
            self.ide_encoding = IDEncoding(deg_view=deg_view)
            mlp_head_in_dim += self.ide_encoding.get_out_dim()
        else:
            mlp_head_in_dim += self.direction_encoding.get_out_dim()

        self.mlp_head = MLP(
            in_dim=mlp_head_in_dim,
            num_layers=num_layers_color,
            layer_width=hidden_dim_color,
            out_dim=3,
            activation=nn.ReLU(),
            out_activation=None,
            implementation=implementation,
        )

    def get_outputs(
        self,
        ray_samples: RaySamples,
        ray_bundle: Optional[RayBundle] = None,
        density_embedding: Optional[Tensor] = None,
    ) -> Dict[FieldHeadNames, Tensor]:
        assert density_embedding is not None
        outputs = {}
        if ray_samples.camera_indices is None:
            raise AttributeError("Camera indices are not provided.")
        camera_indices = ray_samples.camera_indices.squeeze()

        directions_flat = get_normalized_directions(ray_samples.frustums.directions).view(-1, 3)

        positions_encoded_flat = self.position_encoding(
            ray_samples.frustums.get_positions().view(-1, 3)
        )

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
            x = self.mlp_transient(transient_input).view(*outputs_shape, -1).to(directions_flat)
            outputs[FieldHeadNames.UNCERTAINTY] = self.field_head_transient_uncertainty(x)
            outputs[FieldHeadNames.TRANSIENT_RGB] = self.field_head_transient_rgb(x)
            outputs[FieldHeadNames.TRANSIENT_DENSITY] = self.field_head_transient_density(x)

        # semantics
        if self.use_semantics:
            semantics_input = density_embedding.view(-1, self.geo_feat_dim)
            if not self.pass_semantic_gradients:
                semantics_input = semantics_input.detach()

            x = self.mlp_semantics(semantics_input).view(*outputs_shape, -1).to(directions_flat)
            outputs[FieldHeadNames.SEMANTICS] = self.field_head_semantics(x)


        inputs_mlps = torch.cat(
            [
                positions_encoded_flat,
                density_embedding.view(-1, self.geo_feat_dim),
            ],
            dim=-1,
        )

        # predicted normals
        if self.use_pred_normals:
            pred_normals = self.field_head_pred_normals(
                self.mlp_pred_normals(inputs_mlps)
                .to(directions_flat)
                .view(*outputs_shape, -1)
            )
            outputs[FieldHeadNames.PRED_NORMALS] = pred_normals

        # specular tint
        if self.use_pred_specular_tint:
            pred_specular_tint = self.mlp_pred_specular_tint(
                inputs_mlps,
            ).view(*outputs_shape, -1).to(directions_flat)
            outputs[FieldHeadNames.SPECULAR_TINT] = pred_specular_tint

        # roughness
        if self.use_pred_roughness:
            # Rectifying the density with an exponential is much more stable than a ReLU or
            # softplus, because it enables high post-activation (float32) density outputs
            # from smaller internal (float16) parameters.
            pred_roughness = trunc_exp(
                self.mlp_pred_raw_roughness(
                    inputs_mlps,
                ).view(*outputs_shape, -1)
            ).to(directions_flat)
            outputs[FieldHeadNames.ROUGHNESS] = pred_roughness

        # diffuse color
        if self.use_pred_diffuse_color:
            pred_diffuse_color = torch.nn.functional.sigmoid(
                self.mlp_pred_raw_diffuse_color(
                    inputs_mlps,
                ).view(*outputs_shape, -1)
                - REF_CONSTANT
            ).to(directions_flat)
            outputs[FieldHeadNames.DIFFUSE_COLOR] = pred_diffuse_color

        # bottleneck
        pred_bottleneck = self.mlp_pred_bottleneck(inputs_mlps)

        inputs_mlp_head = [
            pred_bottleneck * scale + shift,
        ]

        if self.use_appearance_embedding:
            inputs_mlp_head.append(
                embedded_appearance.view(-1, self.appearance_embedding_dim),
            )

        # Encode view (or reflection) directions.
        if self.use_ide_enc:
            ide_encoding_input = directions_flat
            if self.use_reflections:
                # Compute reflection directions. Note that we flip viewdirs before
                # reflecting, because they point from the camera to the point,
                # whereas `reflect` assumes they point toward the camera.
                # Returned refdirs then point from the point to the environment.
                ide_encoding_input = reflect(
                    -ray_bundle.directions[..., None, :],
                    pred_normals, # type: ignore
                )

            inputs_mlp_head.append(
                self.ide_encoding(
                    ide_encoding_input.view(-1, 3),
                    pred_roughness.view(-1, 1), # type: ignore
                )
            )

        # Append dot product between normal vectors and view directions.
        if self.use_n_dot_v:
            dotprod = torch.sum(
                pred_normals * ray_bundle.directions[..., None, :],  # type: ignore
                dim=-1, keepdims=True,
            ).view(-1, 1)

            inputs_mlp_head.append(dotprod)

        # Concatenate bottleneck, directional encoding, dot product and appearence encoding
        inputs_mlp_head = torch.cat(inputs_mlp_head, dim=-1)

        # If using diffuse/specular colors, then `rgb` is treated as linear
        # specular color. Otherwise it's treated as the color itself.
        rgb = torch.nn.functional.sigmoid(
            self.mlp_head(inputs_mlp_head) * \
            self.rgb_premultiplier + \
            self.rgb_bias
        )

        if self.use_pred_diffuse_color:
            # Initialize linear diffuse color around 0.25, so that the combined
            # linear color is initialized around 0.5.
            if self.use_pred_specular_tint:
                specular_linear = pred_specular_tint.view(-1, 3) * rgb # type: ignore
            else:
                specular_linear = 0.5 * rgb

            # Combine specular and diffuse components and tone map to sRGB.
            rgb = linear_to_srgb(
                specular_linear + pred_diffuse_color.view(-1, 3), # type: ignore
            ).clip_(0.0, 1.0) # type: ignore

        # Apply padding, mapping color to [-rgb_padding, 1+rgb_padding].
        rgb = rgb * (1 + 2 * self.rgb_padding) - self.rgb_padding

        outputs[FieldHeadNames.RGB] = rgb.view(*outputs_shape, -1).to(directions_flat)

        if self.compute_hash_regularization:
            outputs[FieldHeadNames.HASH_DECAY] = self.mlp_base_grid.regularize_hash_pyramid(self.regularize_function)

        return outputs
