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
Compound dield with zip- and ref-nerfacto's ideas.
"""

from typing import Optional, Callable, Literal

import torch
from torch import Tensor

from nerfstudio.fields.refnerfacto_field import RefNerfactoField
from nerfstudio.fields.zipnerfacto_field import ZipNerfactoField
from nerfstudio.cameras.bundle_adjustment import HashBundleAdjustment
from nerfstudio.field_components.spatial_distortions import LinearizedSceneContraction


class ZipRefNerfactoField(ZipNerfactoField, RefNerfactoField):
    """Compound Field that uses TCNN with ideas from Zip- and Ref- Nerfs

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
        bottleneck_noise: std of bottleneck noise
        bottleneck_noise_steps: number of steps applying to bottleneck noise
        rgb_padding: padding added to the RGB outputs
        rgb_premultiplier: premultiplier on RGB before activation
        rgb_bias: the shift added to raw colors pre-activation
        use_reflections: whether to use reflections or not
        scale_featurization: scale featurization from appendix of ZipNeRF
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
        use_pred_normals: bool = True,
        use_average_appearance_embedding: bool = False,
        use_appearance_embedding: bool = False,
        spatial_distortion: Optional[LinearizedSceneContraction] = None,
        regularize_function: Callable[[Tensor], Tensor] = torch.square,
        scale_featurization: bool = True,
        compute_hash_regularization: bool = True,
        use_reflections: bool = True,
        deg_view: int = 5,
        use_ide_enc: bool = True,
        use_pred_roughness: bool = True,
        roughness_bias: float = -1.0,
        use_pred_diffuse_color: float = True,
        use_pred_specular_tint: float = True,
        use_n_dot_v: float = True,
        bottleneck_noise: float = 0.001,
        bottleneck_noise_steps: int = 10000,
        rgb_padding: float = 0.001,
        rgb_premultiplier: float = 1.0,
        rgb_bias: float = 0.0,
        implementation: Literal["tcnn", "torch"] = "tcnn",
    ) -> None:

        RefNerfactoField.__init__(
            self,
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
            spatial_distortion=spatial_distortion,
            regularize_function=regularize_function,
            compute_hash_regularization=compute_hash_regularization,
            use_reflections=use_reflections,
            deg_view=deg_view,
            use_ide_enc=use_ide_enc,
            use_pred_roughness=use_pred_roughness,
            roughness_bias=roughness_bias,
            use_pred_diffuse_color=use_pred_diffuse_color,
            use_pred_specular_tint=use_pred_specular_tint,
            use_n_dot_v=use_n_dot_v,
            bottleneck_noise=bottleneck_noise,
            bottleneck_noise_steps=bottleneck_noise_steps,
            rgb_padding=rgb_padding,
            rgb_premultiplier=rgb_premultiplier,
            rgb_bias=rgb_bias,
        )

        ZipNerfactoField.__init__(
            self,
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
            spatial_distortion=spatial_distortion,
            scale_featurization=scale_featurization,
            regularize_function=regularize_function,
            compute_hash_regularization=compute_hash_regularization,
            implementation=implementation,
        )
