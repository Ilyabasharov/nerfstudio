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
Put all the method implementations in one location.
"""

from __future__ import annotations

from collections import OrderedDict
from typing import Dict

import tyro
from nerfstudio.data.pixel_samplers import PairPixelSamplerConfig

from nerfstudio.cameras.camera_optimizers import (
    PoseOptimizerConfig,
    IntrinsicOptimizerConfig,
    DistortionOptimizerConfig,
)
from nerfstudio.configs.base_config import ViewerConfig
from nerfstudio.configs.external_methods import get_external_methods

from nerfstudio.data.datamanagers.random_cameras_datamanager import RandomCamerasDataManagerConfig
from nerfstudio.data.datamanagers.base_datamanager import VanillaDataManager, VanillaDataManagerConfig

from nerfstudio.data.dataparsers.blender_dataparser import BlenderDataParserConfig
from nerfstudio.data.dataparsers.dnerf_dataparser import DNeRFDataParserConfig
from nerfstudio.data.dataparsers.instant_ngp_dataparser import InstantNGPDataParserConfig
from nerfstudio.data.dataparsers.nerfstudio_dataparser import NerfstudioDataParserConfig
from nerfstudio.data.dataparsers.phototourism_dataparser import PhototourismDataParserConfig
from nerfstudio.data.dataparsers.sdfstudio_dataparser import SDFStudioDataParserConfig
from nerfstudio.data.dataparsers.sitcoms3d_dataparser import Sitcoms3DDataParserConfig
from nerfstudio.data.datasets.depth_dataset import DepthDataset
from nerfstudio.data.datasets.sdf_dataset import SDFDataset
from nerfstudio.data.datasets.semantic_dataset import SemanticDataset
from nerfstudio.engine.optimizers import (
    AdamOptimizerConfig,
    RAdamOptimizerConfig,
    AdamWOptimizerConfig,
)
from nerfstudio.engine.schedulers import (
    CosineDecaySchedulerConfig,
    ExponentialDecaySchedulerConfig,
    MultiStepSchedulerConfig,
)
from nerfstudio.engine.trainer import TrainerConfig
from nerfstudio.field_components.temporal_distortions import TemporalDistortionKind
from nerfstudio.fields.sdf_field import SDFFieldConfig
from nerfstudio.models.depth_based.depth_nerfacto import DepthNerfactoModelConfig
from nerfstudio.models.depth_based.depth_refnerfacto import DepthRefNerfactoModelConfig
from nerfstudio.models.depth_based.depth_zipnerfacto import DepthZipNerfactoModelConfig
from nerfstudio.models.depth_based.depth_ziprefnerfacto import DepthZipRefNerfactoModelConfig
from nerfstudio.models.generfacto import GenerfactoModelConfig
from nerfstudio.models.instant_ngp import InstantNGPModelConfig
from nerfstudio.models.mipnerf import MipNerfModel
from nerfstudio.models.nerfacto import NerfactoModelConfig
from nerfstudio.models.zipnerfacto import ZipNerfactoModelConfig
from nerfstudio.models.refnerfacto import RefNerfactoModelConfig
from nerfstudio.models.ziprefnerfacto import ZipRefNerfactoModelConfig
from nerfstudio.models.neus import NeuSModelConfig
from nerfstudio.models.neus_facto import NeuSFactoModelConfig
from nerfstudio.models.semantic_nerfw import SemanticNerfWModelConfig
from nerfstudio.models.tensorf import TensoRFModelConfig
from nerfstudio.models.vanilla_nerf import NeRFModel, VanillaModelConfig
from nerfstudio.pipelines.base_pipeline import VanillaPipelineConfig
from nerfstudio.pipelines.dynamic_batch import DynamicBatchPipelineConfig
from nerfstudio.plugins.registry import discover_methods

method_configs: Dict[str, TrainerConfig] = {}
descriptions = {
    "nerfacto": "Recommended real-time model tuned for real captures. This model will be continually updated.",
    "depth-nerfacto": "Nerfacto with depth supervision.",
    "instant-ngp": "Implementation of Instant-NGP. Recommended real-time model for unbounded scenes.",
    "instant-ngp-bounded": "Implementation of Instant-NGP. Recommended for bounded real and synthetic scenes",
    "mipnerf": "High quality model for bounded scenes. (slow)",
    "semantic-nerfw": "Predicts semantic segmentations and filters out transient objects.",
    "vanilla-nerf": "Original NeRF model. (slow)",
    "tensorf": "tensorf",
    "dnerf": "Dynamic-NeRF model. (slow)",
    "phototourism": "Uses the Phototourism data.",
    "generfacto": "Generative Text to NeRF model",
    "neus": "Implementation of NeuS. (slow)",
    "neus-facto": "Implementation of NeuS-Facto. (slow)",
    "zipnerfacto": "Implementation of ZipNerfacto. (slow)",
    "refnerfacto": "Implementation of RefNerfacto.",
    "ziprefnerfacto": "Implementation of ZipRefNerfacto.",
    "depth-ziprefnerfacto": "Implementation of ZipRefNerfacto with depth supervision.",
}

method_configs["nerfacto"] = TrainerConfig(
    method_name="nerfacto",
    steps_per_eval_batch=500,
    steps_per_save=2000,
    max_num_iterations=30000,
    mixed_precision=True,
    pipeline=VanillaPipelineConfig(
        datamanager=VanillaDataManagerConfig(
            dataparser=NerfstudioDataParserConfig(),
            train_num_rays_per_batch=4096,
            eval_num_rays_per_batch=4096,
            pose_optimizer=PoseOptimizerConfig(
                mode="SO3xR3",
                optimizer=AdamWOptimizerConfig(lr=6e-4, eps=1e-8, weight_decay=1e-2),
                scheduler=ExponentialDecaySchedulerConfig(lr_final=6e-6, max_steps=200000),
            ),
        ),
        model=NerfactoModelConfig(eval_num_rays_per_chunk=1 << 15),
    ),
    optimizers={
        "proposal_networks": {
            "optimizer": AdamWOptimizerConfig(lr=1e-2, eps=1e-15),
            "scheduler": ExponentialDecaySchedulerConfig(lr_final=0.0001, max_steps=200000),
        },
        "fields": {
            "optimizer": AdamWOptimizerConfig(lr=1e-2, eps=1e-15),
            "scheduler": ExponentialDecaySchedulerConfig(lr_final=0.0001, max_steps=200000),
        },
    },
    viewer=ViewerConfig(num_rays_per_chunk=1 << 15),
    vis="viewer",
)
method_configs["nerfacto-big"] = TrainerConfig(
    method_name="nerfacto",
    steps_per_eval_batch=500,
    steps_per_save=2000,
    max_num_iterations=100000,
    mixed_precision=True,
    pipeline=VanillaPipelineConfig(
        datamanager=VanillaDataManagerConfig(
            dataparser=NerfstudioDataParserConfig(),
            train_num_rays_per_batch=8192,
            eval_num_rays_per_batch=4096,
            pose_optimizer=PoseOptimizerConfig(
                mode="SO3xR3",
                optimizer=AdamWOptimizerConfig(lr=6e-4, eps=1e-8, weight_decay=1e-3),
            ),
        ),
        model=NerfactoModelConfig(
            eval_num_rays_per_chunk=1 << 15,
            num_nerf_samples_per_ray=128,
            num_proposal_samples_per_ray=(512, 256),
            hidden_dim=128,
            hidden_dim_color=128,
            appearance_embed_dim=128,
            max_res=4096,
            proposal_weights_anneal_max_num_iters=5000,
            log2_hashmap_size=21,
        ),
    ),
    optimizers={
        "proposal_networks": {
            "optimizer": AdamWOptimizerConfig(lr=1e-2, eps=1e-15),
            "scheduler": None,
        },
        "fields": {
            "optimizer": AdamWOptimizerConfig(lr=1e-2, eps=1e-15),
            "scheduler": ExponentialDecaySchedulerConfig(lr_final=1e-4, max_steps=50000),
        },
    },
    viewer=ViewerConfig(num_rays_per_chunk=1 << 15),
    vis="viewer",
)
method_configs["nerfacto-huge"] = TrainerConfig(
    method_name="nerfacto",
    steps_per_eval_batch=500,
    steps_per_save=2000,
    max_num_iterations=100000,
    mixed_precision=True,
    pipeline=VanillaPipelineConfig(
        datamanager=VanillaDataManagerConfig(
            dataparser=NerfstudioDataParserConfig(train_split_fraction=0.998),
            train_num_rays_per_batch=10384,
            eval_num_rays_per_batch=4096,
            pose_optimizer=PoseOptimizerConfig(
                mode="SO3xR3",
                optimizer=AdamWOptimizerConfig(lr=6e-4, eps=1e-8, weight_decay=1e-3),
                scheduler=ExponentialDecaySchedulerConfig(lr_final=6e-5, max_steps=50000),
            ),
        ),
        model=NerfactoModelConfig(
            eval_num_rays_per_chunk=1 << 15,
            num_nerf_samples_per_ray=64,
            num_proposal_samples_per_ray=(256, 256),
            proposal_net_args_list=[
                {"hidden_dim": 128, "log2_hashmap_size": 17, "num_levels": 5, "max_res": 512, "use_linear": False, "features_per_level": 4},
                {"hidden_dim": 128, "log2_hashmap_size": 17, "num_levels": 7, "max_res": 2048, "use_linear": False, "features_per_level": 4},
            ],
            hidden_dim=512,
            hidden_dim_color=512,
            appearance_embed_dim=128,
            max_res=8192,
            proposal_weights_anneal_max_num_iters=5000,
            log2_hashmap_size=22,
        ),
    ),
    optimizers={
        "proposal_networks": {
            "optimizer": AdamWOptimizerConfig(lr=6e-3, eps=1e-15),
            "scheduler": ExponentialDecaySchedulerConfig(
                delay_steps=15000,
                lr_pre_warmup=6e-3,
                max_steps=120000,
                lr_final=5e-5,
            ),
        },
        "fields": {
            "optimizer": AdamWOptimizerConfig(lr=6e-3, eps=1e-15),
            "scheduler": ExponentialDecaySchedulerConfig(
                delay_steps=15000,
                lr_pre_warmup=6e-3,
                max_steps=120000,
                lr_final=5e-5,
            ),
        },
    },
    viewer=ViewerConfig(num_rays_per_chunk=1 << 15),
    vis="viewer",
)
method_configs["depth-nerfacto"] = TrainerConfig(
    method_name="depth-nerfacto",
    steps_per_eval_batch=500,
    steps_per_save=2000,
    max_num_iterations=120000,
    mixed_precision=True,
    pipeline=VanillaPipelineConfig(
        datamanager=VanillaDataManagerConfig(
            _target=VanillaDataManager[DepthDataset],
            pixel_sampler=PairPixelSamplerConfig(num_rays_per_batch=5284),
            dataparser=NerfstudioDataParserConfig(train_split_fraction=0.998, downscale_factor=2),
            train_num_rays_per_batch=5284,
            eval_num_rays_per_batch=4096,
            pose_optimizer=PoseOptimizerConfig(
                mode="SO3xR3",
                optimizer=AdamWOptimizerConfig(lr=1e-4, eps=1e-8, weight_decay=1e-3),
                scheduler=ExponentialDecaySchedulerConfig(
                    delay_steps=10000,
                    lr_pre_warmup=1e-4,
                    warmup_steps=3000,
                    lr_final=1e-6,
                    max_steps=50000,
                ),
            ),
            intrinsic_optimizer=IntrinsicOptimizerConfig(
                mode="square_scale",
                optimizer=AdamWOptimizerConfig(lr=1e-4, eps=1e-8, weight_decay=1e-4),
                scheduler=ExponentialDecaySchedulerConfig(
                    delay_steps=35000,
                    warmup_steps=3000,
                    lr_pre_warmup=1e-8,
                    lr_final=1e-5,
                    max_steps=50000,
                ),
            ),
            distortion_optimizer=DistortionOptimizerConfig(
                mode="shift",
                optimizer=AdamWOptimizerConfig(lr=1e-4, eps=1e-8, weight_decay=1e-3),
                scheduler=ExponentialDecaySchedulerConfig(
                    delay_steps=45000,
                    lr_pre_warmup=1e-8,
                    warmup_steps=3000,
                    lr_final=1e-5,
                    max_steps=50000,
                ),
            ),
        ),
        model=DepthNerfactoModelConfig(
            eval_num_rays_per_chunk=1 << 12,
            num_nerf_samples_per_ray=64,
            num_proposal_samples_per_ray=(512, 256),
            proposal_net_args_list=[
                {"hidden_dim": 64, "log2_hashmap_size": 17, "num_levels": 5, "max_res": 512, "use_linear": False, "features_per_level": 4},
                {"hidden_dim": 64, "log2_hashmap_size": 17, "num_levels": 7, "max_res": 2048, "use_linear": False, "features_per_level": 4},
            ],
            hidden_dim=512,
            hidden_dim_color=512,
            appearance_embed_dim=128,
            max_res=8192,
            proposal_weights_anneal_max_num_iters=40000,
            log2_hashmap_size=22,
            features_per_level=8,
            use_appearance_embedding=True,
            use_depth_ranking_loss=True,
            use_depth_loss=False,
            use_gradient_scaling=False,
            compute_hash_regularization=False,
            use_bundle_adjust=False,
        ),
    ),
    optimizers={
        "proposal_networks": {
            "optimizer": AdamWOptimizerConfig(lr=6e-3, eps=1e-15, weight_decay=1e-5),
            "scheduler": ExponentialDecaySchedulerConfig(
                delay_steps=20000,
                lr_pre_warmup=6e-3,
                max_steps=120000,
                lr_final=5e-5,
            ),
        },
        "fields": {
            "optimizer": AdamWOptimizerConfig(lr=6e-3, eps=1e-15, weight_decay=1e-5),
            "scheduler": ExponentialDecaySchedulerConfig(
                delay_steps=20000,
                lr_pre_warmup=6e-3,
                max_steps=120000,
                lr_final=5e-5,
            ),
        },
    },
    viewer=ViewerConfig(num_rays_per_chunk=1 << 15),
    vis="viewer",
)

method_configs["depth-refnerfacto"] = TrainerConfig(
    method_name="depth-refnerfacto",
    steps_per_eval_batch=500,
    steps_per_save=2000,
    max_num_iterations=60000,
    mixed_precision=True,
    pipeline=VanillaPipelineConfig(
        datamanager=VanillaDataManagerConfig(
            _target=VanillaDataManager[DepthDataset],
            pixel_sampler=PairPixelSamplerConfig(num_rays_per_batch=6884),
            dataparser=NerfstudioDataParserConfig(train_split_fraction=0.994),
            train_num_rays_per_batch=6884,
            eval_num_rays_per_batch=4096,
            pose_optimizer=PoseOptimizerConfig(
                mode="SO3xR3",
                optimizer=AdamWOptimizerConfig(lr=1e-4, eps=1e-8, weight_decay=1e-3),
                scheduler=ExponentialDecaySchedulerConfig(
                    delay_steps=10000,
                    lr_pre_warmup=1e-4,
                    warmup_steps=3000,
                    lr_final=1e-6,
                    max_steps=50000,
                ),
            ),
            intrinsic_optimizer=IntrinsicOptimizerConfig(
                mode="square_scale",
                optimizer=AdamWOptimizerConfig(lr=1e-4, eps=1e-8, weight_decay=1e-4),
                scheduler=ExponentialDecaySchedulerConfig(
                    delay_steps=35000,
                    warmup_steps=3000,
                    lr_pre_warmup=1e-8,
                    lr_final=1e-5,
                    max_steps=50000,
                ),
            ),
            distortion_optimizer=DistortionOptimizerConfig(
                mode="shift",
                optimizer=AdamWOptimizerConfig(lr=1e-4, eps=1e-8, weight_decay=1e-3),
                scheduler=ExponentialDecaySchedulerConfig(
                    delay_steps=45000,
                    lr_pre_warmup=1e-8,
                    warmup_steps=3000,
                    lr_final=1e-5,
                    max_steps=50000,
                ),
            ),
        ),
        model=DepthRefNerfactoModelConfig(
            eval_num_rays_per_chunk=1 << 12,
            num_nerf_samples_per_ray=64,
            num_proposal_samples_per_ray=(512, 512),
            proposal_net_args_list=[
                {"hidden_dim": 64, "log2_hashmap_size": 17, "num_levels": 5, "max_res": 512, "use_linear": False, "features_per_level": 4},
                {"hidden_dim": 64, "log2_hashmap_size": 17, "num_levels": 7, "max_res": 2048, "use_linear": False, "features_per_level": 4},
            ],
            hidden_dim=256,
            hidden_dim_color=256,
            appearance_embed_dim=128,
            max_res=8192,
            proposal_weights_anneal_max_num_iters=20000,
            log2_hashmap_size=21,
            depth_ranking_loss_mult=1.0,
            features_per_level=8,
            geo_feat_dim=127,
            use_appearance_embedding=True,
            compute_hash_regularization=True,
        ),
    ),
    optimizers={
        "proposal_networks": {
            "optimizer": AdamWOptimizerConfig(lr=6e-3, eps=1e-15, weight_decay=1e-5),
            "scheduler": ExponentialDecaySchedulerConfig(
                delay_steps=20000,
                lr_pre_warmup=6e-3,
                max_steps=120000,
                lr_final=5e-5,
            ),
        },
        "fields": {
            "optimizer": AdamWOptimizerConfig(lr=6e-3, eps=1e-15, weight_decay=1e-5),
            "scheduler": ExponentialDecaySchedulerConfig(
                delay_steps=20000,
                lr_pre_warmup=6e-3,
                max_steps=120000,
                lr_final=5e-5,
            ),
        },
    },
    viewer=ViewerConfig(num_rays_per_chunk=1 << 15),
    vis="viewer",
)

method_configs["zipnerfacto"] = TrainerConfig(
    method_name="zipnerfacto",
    steps_per_eval_batch=500,
    steps_per_save=2000,
    max_num_iterations=70000,
    mixed_precision=True,
    pipeline=VanillaPipelineConfig(
        datamanager=VanillaDataManagerConfig(
            dataparser=NerfstudioDataParserConfig(),
            train_num_rays_per_batch=6096,
            eval_num_rays_per_batch=4096,
            pose_optimizer=PoseOptimizerConfig(
                mode="SO3xR3",
                optimizer=AdamWOptimizerConfig(lr=1e-4, eps=1e-8, weight_decay=1e-3),
                scheduler=ExponentialDecaySchedulerConfig(
                    delay_steps=10000,
                    lr_pre_warmup=1e-4,
                    warmup_steps=3000,
                    lr_final=1e-6,
                    max_steps=50000,
                ),
            ),
            intrinsic_optimizer=IntrinsicOptimizerConfig(
                mode="square_scale",
                optimizer=AdamWOptimizerConfig(lr=1e-4, eps=1e-8, weight_decay=1e-4),
                scheduler=ExponentialDecaySchedulerConfig(
                    delay_steps=35000,
                    warmup_steps=3000,
                    lr_pre_warmup=1e-8,
                    lr_final=1e-5,
                    max_steps=50000,
                ),
            ),
            distortion_optimizer=DistortionOptimizerConfig(
                mode="shift",
                optimizer=AdamWOptimizerConfig(lr=1e-4, eps=1e-8, weight_decay=1e-3),
                scheduler=ExponentialDecaySchedulerConfig(
                    delay_steps=45000,
                    lr_pre_warmup=1e-8,
                    warmup_steps=3000,
                    lr_final=1e-5,
                    max_steps=50000,
                ),
            ),
        ),
        model=ZipNerfactoModelConfig(
            eval_num_rays_per_chunk=1 << 12,
            hidden_dim=128,
            hidden_dim_color=128,
            proposal_weights_anneal_max_num_iters=5000,
            appearance_embed_dim=128,
            proposal_warmup=30000,
        ),
    ),
    optimizers={
        "proposal_networks": {
            "optimizer": AdamWOptimizerConfig(lr=6e-3, eps=1e-15, weight_decay=1e-5),
            "scheduler": ExponentialDecaySchedulerConfig(
                delay_steps=20000,
                lr_pre_warmup=6e-3,
                max_steps=120000,
                lr_final=5e-5,
            ),
        },
        "fields": {
            "optimizer": AdamWOptimizerConfig(lr=6e-3, eps=1e-15, weight_decay=1e-5),
            "scheduler": ExponentialDecaySchedulerConfig(
                delay_steps=20000,
                lr_pre_warmup=6e-3,
                max_steps=120000,
                lr_final=5e-5,
            ),
        },
    },
    viewer=ViewerConfig(num_rays_per_chunk=1 << 12),
    vis="viewer",
)

method_configs["refnerfacto"] = TrainerConfig(
    method_name="refnerfacto",
    steps_per_eval_batch=500,
    steps_per_save=2000,
    max_num_iterations=70000,
    mixed_precision=True,
    pipeline=VanillaPipelineConfig(
        datamanager=VanillaDataManagerConfig(
            dataparser=NerfstudioDataParserConfig(),
            train_num_rays_per_batch=8384,
            eval_num_rays_per_batch=4096,
            pose_optimizer=PoseOptimizerConfig(
                mode="SO3xR3",
                optimizer=AdamWOptimizerConfig(lr=1e-4, eps=1e-8, weight_decay=1e-3),
                scheduler=ExponentialDecaySchedulerConfig(
                    delay_steps=10000,
                    lr_pre_warmup=1e-4,
                    warmup_steps=3000,
                    lr_final=1e-6,
                    max_steps=50000,
                ),
            ),
            intrinsic_optimizer=IntrinsicOptimizerConfig(
                mode="square_scale",
                optimizer=AdamWOptimizerConfig(lr=1e-4, eps=1e-8, weight_decay=1e-4),
                scheduler=ExponentialDecaySchedulerConfig(
                    delay_steps=35000,
                    warmup_steps=3000,
                    lr_pre_warmup=1e-8,
                    lr_final=1e-5,
                    max_steps=50000,
                ),
            ),
            distortion_optimizer=DistortionOptimizerConfig(
                mode="shift",
                optimizer=AdamWOptimizerConfig(lr=1e-4, eps=1e-8, weight_decay=1e-3),
                scheduler=ExponentialDecaySchedulerConfig(
                    delay_steps=45000,
                    lr_pre_warmup=1e-8,
                    warmup_steps=3000,
                    lr_final=1e-5,
                    max_steps=50000,
                ),
            ),
        ),
        model=RefNerfactoModelConfig(
            eval_num_rays_per_chunk=1 << 12,
            hidden_dim=256,
            hidden_dim_color=256,
            proposal_warmup=50000,
        ),
    ),
    optimizers={
        "proposal_networks": {
            "optimizer": AdamWOptimizerConfig(lr=6e-3, eps=1e-15, weight_decay=1e-5),
            "scheduler": ExponentialDecaySchedulerConfig(
                delay_steps=20000,
                lr_pre_warmup=6e-3,
                max_steps=120000,
                lr_final=5e-5,
            ),
        },
        "fields": {
            "optimizer": AdamWOptimizerConfig(lr=6e-3, eps=1e-15, weight_decay=1e-5),
            "scheduler": ExponentialDecaySchedulerConfig(
                delay_steps=20000,
                lr_pre_warmup=6e-3,
                max_steps=120000,
                lr_final=5e-5,
            ),
        },
    },
    viewer=ViewerConfig(num_rays_per_chunk=1 << 12),
    vis="viewer",
)

method_configs["ziprefnerfacto"] = TrainerConfig(
    method_name="ziprefnerfacto",
    steps_per_eval_batch=500,
    steps_per_save=2000,
    max_num_iterations=70000,
    mixed_precision=True,
    pipeline=VanillaPipelineConfig(
        datamanager=VanillaDataManagerConfig(
            dataparser=NerfstudioDataParserConfig(),
            train_num_rays_per_batch=4496,
            eval_num_rays_per_batch=4096,
            pose_optimizer=PoseOptimizerConfig(
                mode="SO3xR3",
                optimizer=AdamWOptimizerConfig(lr=1e-4, eps=1e-8, weight_decay=1e-3),
                scheduler=ExponentialDecaySchedulerConfig(
                    delay_steps=10000,
                    lr_pre_warmup=1e-4,
                    warmup_steps=3000,
                    lr_final=1e-6,
                    max_steps=50000,
                ),
            ),
            intrinsic_optimizer=IntrinsicOptimizerConfig(
                mode="square_scale",
                optimizer=AdamWOptimizerConfig(lr=1e-4, eps=1e-8, weight_decay=1e-4),
                scheduler=ExponentialDecaySchedulerConfig(
                    delay_steps=35000,
                    warmup_steps=3000,
                    lr_pre_warmup=1e-8,
                    lr_final=1e-5,
                    max_steps=50000,
                ),
            ),
            distortion_optimizer=DistortionOptimizerConfig(
                mode="shift",
                optimizer=AdamWOptimizerConfig(lr=1e-4, eps=1e-8, weight_decay=1e-3),
                scheduler=ExponentialDecaySchedulerConfig(
                    delay_steps=45000,
                    lr_pre_warmup=1e-8,
                    warmup_steps=3000,
                    lr_final=1e-5,
                    max_steps=50000,
                ),
            ),
        ),
        model=ZipRefNerfactoModelConfig(
            eval_num_rays_per_chunk=1 << 12,
            hidden_dim=256,
            hidden_dim_color=256,
            near_plane=0.0,
            proposal_weights_anneal_max_num_iters=1,
            use_bundle_adjust=False,
            use_gradient_scaling=False,
            compute_hash_regularization=True,
            proposal_warmup=50000,
        ),
    ),
    optimizers={
        "proposal_networks": {
            "optimizer": AdamWOptimizerConfig(lr=6e-3, eps=1e-15, weight_decay=1e-5),
            "scheduler": ExponentialDecaySchedulerConfig(
                delay_steps=20000,
                lr_pre_warmup=6e-3,
                max_steps=120000,
                lr_final=5e-5,
            ),
        },
        "fields": {
            "optimizer": AdamWOptimizerConfig(lr=6e-3, eps=1e-15, weight_decay=1e-5),
            "scheduler": ExponentialDecaySchedulerConfig(
                delay_steps=20000,
                lr_pre_warmup=6e-3,
                max_steps=120000,
                lr_final=5e-5,
            ),
        },
    },
    viewer=ViewerConfig(num_rays_per_chunk=1 << 12),
    vis="viewer",
)

method_configs["depth-ziprefnerfacto"] = TrainerConfig(
    method_name="depth-ziprefnerfacto",
    steps_per_eval_batch=10000,
    steps_per_eval_image=10000,
    steps_per_eval_all_images=89999,
    steps_per_save=2000,
    max_num_iterations=90000,
    mixed_precision=True,
    pipeline=VanillaPipelineConfig(
        datamanager=VanillaDataManagerConfig(
            _target=VanillaDataManager[DepthDataset],
            pixel_sampler=PairPixelSamplerConfig(num_rays_per_batch=3596),
            dataparser=NerfstudioDataParserConfig(train_split_fraction=0.998, downscale_factor=2),
            train_num_rays_per_batch=3596,
            eval_num_rays_per_batch=4096,
            pose_optimizer=PoseOptimizerConfig(
                mode="SO3xR3",
                optimizer=AdamWOptimizerConfig(lr=1e-4, eps=1e-8, weight_decay=1e-3),
                scheduler=ExponentialDecaySchedulerConfig(
                    delay_steps=10000,
                    lr_pre_warmup=1e-4,
                    warmup_steps=3000,
                    lr_final=1e-6,
                    max_steps=50000,
                ),
            ),
            intrinsic_optimizer=IntrinsicOptimizerConfig(
                mode="square_scale",
                optimizer=AdamWOptimizerConfig(lr=1e-4, eps=1e-8, weight_decay=1e-4),
                scheduler=ExponentialDecaySchedulerConfig(
                    delay_steps=35000,
                    warmup_steps=3000,
                    lr_pre_warmup=1e-8,
                    lr_final=1e-5,
                    max_steps=50000,
                ),
            ),
            distortion_optimizer=DistortionOptimizerConfig(
                mode="shift",
                optimizer=AdamWOptimizerConfig(lr=1e-4, eps=1e-8, weight_decay=1e-3),
                scheduler=ExponentialDecaySchedulerConfig(
                    delay_steps=45000,
                    lr_pre_warmup=1e-8,
                    warmup_steps=3000,
                    lr_final=1e-5,
                    max_steps=50000,
                ),
            ),
        ),
        model=DepthZipRefNerfactoModelConfig(
            eval_num_rays_per_chunk=1 << 12,
            hidden_dim=512,
            hidden_dim_color=512,
            near_plane=0.01,
            proposal_weights_anneal_max_num_iters=50000,
            use_bundle_adjust=True,
            use_gradient_scaling=False,
            compute_hash_regularization=True,
            proposal_warmup=60000,
            compute_appearence_regularization=True,
            compute_density_regularization=True,
            num_proposal_samples_per_ray=(256, 256),
            visualize_weights_distribution=True,
            use_depth_ranking_loss=True,
            use_depth_loss=False,
            proposal_net_args_list=[
                {"hidden_dim": 64, "log2_hashmap_size": 17, "num_levels": 5, "max_res": 512, "use_linear": False, "features_per_level": 4},
                {"hidden_dim": 64, "log2_hashmap_size": 17, "num_levels": 7, "max_res": 4096, "use_linear": False, "features_per_level": 4},
            ],
            features_per_level=8,
        ),
    ),
    optimizers={
        "proposal_networks": {
            "optimizer": AdamWOptimizerConfig(lr=6e-3, eps=1e-15, weight_decay=1e-5),
            "scheduler": ExponentialDecaySchedulerConfig(
                delay_steps=20000,
                lr_pre_warmup=6e-3,
                max_steps=120000,
                lr_final=5e-5,
            ),
        },
        "fields": {
            "optimizer": AdamWOptimizerConfig(lr=6e-3, eps=1e-15, weight_decay=1e-5),
            "scheduler": ExponentialDecaySchedulerConfig(
                delay_steps=20000,
                lr_pre_warmup=6e-3,
                max_steps=120000,
                lr_final=5e-5,
            ),
        },
    },
    viewer=ViewerConfig(num_rays_per_chunk=1 << 12),
    vis="viewer",
)

method_configs["volinga"] = TrainerConfig(
    method_name="volinga",
    steps_per_eval_batch=500,
    steps_per_save=2000,
    max_num_iterations=30000,
    mixed_precision=True,
    pipeline=VanillaPipelineConfig(
        datamanager=VanillaDataManagerConfig(
            dataparser=NerfstudioDataParserConfig(),
            train_num_rays_per_batch=4096,
            eval_num_rays_per_batch=4096,
            pose_optimizer=PoseOptimizerConfig(
                mode="SO3xR3", optimizer=AdamOptimizerConfig(lr=6e-4, eps=1e-8, weight_decay=1e-2)
            ),
        ),
        model=NerfactoModelConfig(
            eval_num_rays_per_chunk=1 << 15,
            hidden_dim=32,
            hidden_dim_color=32,
            hidden_dim_transient=32,
            num_nerf_samples_per_ray=24,
            proposal_net_args_list=[
                {"hidden_dim": 16, "log2_hashmap_size": 17, "num_levels": 5, "max_res": 128, "use_linear": True},
                {"hidden_dim": 16, "log2_hashmap_size": 17, "num_levels": 5, "max_res": 256, "use_linear": True},
            ],
        ),
    ),
    optimizers={
        "proposal_networks": {
            "optimizer": AdamOptimizerConfig(lr=1e-2, eps=1e-15),
            "scheduler": None,
        },
        "fields": {
            "optimizer": AdamOptimizerConfig(lr=1e-2, eps=1e-15),
            "scheduler": None,
        },
    },
    viewer=ViewerConfig(num_rays_per_chunk=1 << 15),
    vis="viewer",
)

method_configs["instant-ngp"] = TrainerConfig(
    method_name="instant-ngp",
    steps_per_eval_batch=500,
    steps_per_save=2000,
    max_num_iterations=30000,
    mixed_precision=True,
    pipeline=DynamicBatchPipelineConfig(
        datamanager=VanillaDataManagerConfig(
            dataparser=NerfstudioDataParserConfig(),
            train_num_rays_per_batch=4096,
            eval_num_rays_per_batch=4096,
        ),
        model=InstantNGPModelConfig(eval_num_rays_per_chunk=8192),
    ),
    optimizers={
        "fields": {
            "optimizer": AdamOptimizerConfig(lr=1e-2, eps=1e-15),
            "scheduler": ExponentialDecaySchedulerConfig(lr_final=0.0001, max_steps=200000),
        }
    },
    viewer=ViewerConfig(num_rays_per_chunk=1 << 12),
    vis="viewer",
)


method_configs["instant-ngp-bounded"] = TrainerConfig(
    method_name="instant-ngp-bounded",
    steps_per_eval_batch=500,
    steps_per_save=2000,
    max_num_iterations=30000,
    mixed_precision=True,
    pipeline=DynamicBatchPipelineConfig(
        datamanager=VanillaDataManagerConfig(dataparser=InstantNGPDataParserConfig(), train_num_rays_per_batch=8192),
        model=InstantNGPModelConfig(
            eval_num_rays_per_chunk=8192,
            grid_levels=1,
            alpha_thre=0.0,
            cone_angle=0.0,
            disable_scene_contraction=True,
            near_plane=0.01,
            background_color="black",
        ),
    ),
    optimizers={
        "fields": {
            "optimizer": AdamOptimizerConfig(lr=1e-2, eps=1e-15),
            "scheduler": ExponentialDecaySchedulerConfig(lr_final=0.0001, max_steps=200000),
        }
    },
    viewer=ViewerConfig(num_rays_per_chunk=1 << 12),
    vis="viewer",
)

method_configs["mipnerf"] = TrainerConfig(
    method_name="mipnerf",
    pipeline=VanillaPipelineConfig(
        datamanager=VanillaDataManagerConfig(dataparser=NerfstudioDataParserConfig(), train_num_rays_per_batch=1024),
        model=VanillaModelConfig(
            _target=MipNerfModel,
            loss_coefficients={"rgb_loss_coarse": 0.1, "rgb_loss_fine": 1.0},
            num_coarse_samples=128,
            num_importance_samples=128,
            eval_num_rays_per_chunk=1024,
        ),
    ),
    optimizers={
        "fields": {
            "optimizer": RAdamOptimizerConfig(lr=5e-4, eps=1e-08),
            "scheduler": None,
        }
    },
)

method_configs["semantic-nerfw"] = TrainerConfig(
    method_name="semantic-nerfw",
    steps_per_eval_batch=500,
    steps_per_save=2000,
    max_num_iterations=30000,
    mixed_precision=True,
    pipeline=VanillaPipelineConfig(
        datamanager=VanillaDataManagerConfig(
            _target=VanillaDataManager[SemanticDataset],
            dataparser=Sitcoms3DDataParserConfig(),
            train_num_rays_per_batch=4096,
            eval_num_rays_per_batch=8192,
        ),
        model=SemanticNerfWModelConfig(eval_num_rays_per_chunk=1 << 16),
    ),
    optimizers={
        "proposal_networks": {
            "optimizer": AdamOptimizerConfig(lr=1e-2, eps=1e-15),
            "scheduler": None,
        },
        "fields": {
            "optimizer": AdamOptimizerConfig(lr=1e-2, eps=1e-15),
            "scheduler": None,
        },
    },
    viewer=ViewerConfig(num_rays_per_chunk=1 << 16),
    vis="viewer",
)

method_configs["vanilla-nerf"] = TrainerConfig(
    method_name="vanilla-nerf",
    pipeline=VanillaPipelineConfig(
        datamanager=VanillaDataManagerConfig(
            dataparser=BlenderDataParserConfig(),
        ),
        model=VanillaModelConfig(_target=NeRFModel),
    ),
    optimizers={
        "fields": {
            "optimizer": RAdamOptimizerConfig(lr=5e-4, eps=1e-08),
            "scheduler": None,
        },
        "temporal_distortion": {
            "optimizer": RAdamOptimizerConfig(lr=5e-4, eps=1e-08),
            "scheduler": None,
        },
    },
)

method_configs["tensorf"] = TrainerConfig(
    method_name="tensorf",
    steps_per_eval_batch=500,
    steps_per_save=2000,
    max_num_iterations=30000,
    mixed_precision=False,
    pipeline=VanillaPipelineConfig(
        datamanager=VanillaDataManagerConfig(
            dataparser=BlenderDataParserConfig(),
            train_num_rays_per_batch=4096,
            eval_num_rays_per_batch=4096,
        ),
        model=TensoRFModelConfig(
            regularization="tv",
        ),
    ),
    optimizers={
        "fields": {
            "optimizer": AdamOptimizerConfig(lr=0.001),
            "scheduler": ExponentialDecaySchedulerConfig(lr_final=0.0001, max_steps=30000),
        },
        "encodings": {
            "optimizer": AdamOptimizerConfig(lr=0.02),
            "scheduler": ExponentialDecaySchedulerConfig(lr_final=0.002, max_steps=30000),
        },
    },
    viewer=ViewerConfig(num_rays_per_chunk=1 << 15),
    vis="viewer",
)

method_configs["dnerf"] = TrainerConfig(
    method_name="dnerf",
    pipeline=VanillaPipelineConfig(
        datamanager=VanillaDataManagerConfig(dataparser=DNeRFDataParserConfig()),
        model=VanillaModelConfig(
            _target=NeRFModel,
            enable_temporal_distortion=True,
            temporal_distortion_params={"kind": TemporalDistortionKind.DNERF},
        ),
    ),
    optimizers={
        "fields": {
            "optimizer": RAdamOptimizerConfig(lr=5e-4, eps=1e-08),
            "scheduler": None,
        },
        "temporal_distortion": {
            "optimizer": RAdamOptimizerConfig(lr=5e-4, eps=1e-08),
            "scheduler": None,
        },
    },
)

method_configs["phototourism"] = TrainerConfig(
    method_name="phototourism",
    steps_per_eval_batch=500,
    steps_per_save=2000,
    max_num_iterations=30000,
    mixed_precision=True,
    pipeline=VanillaPipelineConfig(
        datamanager=VanillaDataManagerConfig(
            dataparser=PhototourismDataParserConfig(),  # NOTE: one of the only differences with nerfacto
            train_num_rays_per_batch=4096,
            eval_num_rays_per_batch=4096,
            pose_optimizer=PoseOptimizerConfig(
                mode="SO3xR3", optimizer=AdamOptimizerConfig(lr=6e-4, eps=1e-8, weight_decay=1e-2)
            ),
            # Large dataset, so using prior values from VariableResDataManager.
            train_num_images_to_sample_from=40,
            train_num_times_to_repeat_images=100,
            eval_num_images_to_sample_from=40,
            eval_num_times_to_repeat_images=100,
        ),
        model=NerfactoModelConfig(eval_num_rays_per_chunk=1 << 15),
    ),
    optimizers={
        "proposal_networks": {
            "optimizer": AdamOptimizerConfig(lr=1e-2, eps=1e-15),
            "scheduler": None,
        },
        "fields": {
            "optimizer": AdamOptimizerConfig(lr=1e-2, eps=1e-15),
            "scheduler": None,
        },
    },
    viewer=ViewerConfig(num_rays_per_chunk=1 << 15),
    vis="viewer",
)

method_configs["generfacto"] = TrainerConfig(
    method_name="generfacto",
    experiment_name="",
    steps_per_eval_batch=50,
    steps_per_eval_image=50,
    steps_per_save=200,
    max_num_iterations=30000,
    mixed_precision=True,
    pipeline=VanillaPipelineConfig(
        datamanager=RandomCamerasDataManagerConfig(
            horizontal_rotation_warmup=3000,
        ),
        model=GenerfactoModelConfig(
            eval_num_rays_per_chunk=1 << 15,
            distortion_loss_mult=1.0,
            interlevel_loss_mult=100.0,
            max_res=256,
            sphere_collider=True,
            initialize_density=True,
            taper_range=(0, 2000),
            random_background=True,
            proposal_warmup=2000,
            proposal_update_every=0,
            proposal_weights_anneal_max_num_iters=2000,
            start_lambertian_training=500,
            start_normals_training=2000,
            opacity_loss_mult=0.001,
            positional_prompting="discrete",
            guidance_scale=25,
        ),
    ),
    optimizers={
        "proposal_networks": {
            "optimizer": AdamOptimizerConfig(lr=1e-3, eps=1e-15),
            "scheduler": None,
        },
        "fields": {
            "optimizer": AdamOptimizerConfig(lr=5e-4, eps=1e-15),
            "scheduler": None,
        },
    },
    viewer=ViewerConfig(),
    vis="viewer",
)

method_configs["neus"] = TrainerConfig(
    method_name="neus",
    steps_per_eval_image=500,
    steps_per_eval_batch=5000,
    steps_per_save=20000,
    steps_per_eval_all_images=1000000,  # set to a very large number so we don't eval with all images
    max_num_iterations=100000,
    mixed_precision=False,
    pipeline=VanillaPipelineConfig(
        datamanager=VanillaDataManagerConfig(
            _target=VanillaDataManager[SDFDataset],
            dataparser=SDFStudioDataParserConfig(),
            train_num_rays_per_batch=1024,
            eval_num_rays_per_batch=1024,
            pose_optimizer=PoseOptimizerConfig(
                mode="off", optimizer=AdamOptimizerConfig(lr=6e-4, eps=1e-8, weight_decay=1e-2)
            ),
        ),
        model=NeuSModelConfig(eval_num_rays_per_chunk=1024),
    ),
    optimizers={
        "fields": {
            "optimizer": AdamOptimizerConfig(lr=5e-4, eps=1e-15),
            "scheduler": CosineDecaySchedulerConfig(warmup_steps=5000, learning_rate_alpha=0.05, max_steps=300000),
        },
        "field_background": {
            "optimizer": AdamOptimizerConfig(lr=5e-4, eps=1e-15),
            "scheduler": CosineDecaySchedulerConfig(warmup_steps=5000, learning_rate_alpha=0.05, max_steps=300000),
        },
    },
    viewer=ViewerConfig(num_rays_per_chunk=1 << 15),
    vis="viewer",
)

method_configs["neus-facto"] = TrainerConfig(
    method_name="neus-facto",
    steps_per_eval_image=5000,
    steps_per_eval_batch=5000,
    steps_per_save=2000,
    steps_per_eval_all_images=1000000,  # set to a very large model so we don't eval with all images
    max_num_iterations=20001,
    mixed_precision=False,
    pipeline=VanillaPipelineConfig(
        datamanager=VanillaDataManagerConfig(
            _target=VanillaDataManager[SDFDataset],
            dataparser=SDFStudioDataParserConfig(),
            train_num_rays_per_batch=2048,
            eval_num_rays_per_batch=2048,
            pose_optimizer=PoseOptimizerConfig(
                mode="SO3xR3", optimizer=AdamOptimizerConfig(lr=6e-4, eps=1e-8, weight_decay=1e-2)
            ),
        ),
        model=NeuSFactoModelConfig(
            # proposal network allows for significantly smaller sdf/color network
            sdf_field=SDFFieldConfig(
                use_grid_feature=True,
                num_layers=2,
                num_layers_color=2,
                hidden_dim=256,
                bias=0.5,
                beta_init=0.8,
                use_appearance_embedding=False,
            ),
            background_model="none",
            eval_num_rays_per_chunk=2048,
        ),
    ),
    optimizers={
        "proposal_networks": {
            "optimizer": AdamOptimizerConfig(lr=1e-2, eps=1e-15),
            "scheduler": MultiStepSchedulerConfig(max_steps=20001, milestones=(10000, 1500, 18000)),
        },
        "fields": {
            "optimizer": AdamOptimizerConfig(lr=5e-4, eps=1e-15),
            "scheduler": CosineDecaySchedulerConfig(warmup_steps=500, learning_rate_alpha=0.05, max_steps=20001),
        },
        "field_background": {
            "optimizer": AdamOptimizerConfig(lr=5e-4, eps=1e-15),
            "scheduler": CosineDecaySchedulerConfig(warmup_steps=500, learning_rate_alpha=0.05, max_steps=20001),
        },
    },
    viewer=ViewerConfig(num_rays_per_chunk=1 << 15),
    vis="viewer",
)


def merge_methods(methods, method_descriptions, new_methods, new_descriptions, overwrite=True):
    """Merge new methods and descriptions into existing methods and descriptions.
    Args:
        methods: Existing methods.
        method_descriptions: Existing descriptions.
        new_methods: New methods to merge in.
        new_descriptions: New descriptions to merge in.
    Returns:
        Merged methods and descriptions.
    """
    methods = OrderedDict(**methods)
    method_descriptions = OrderedDict(**method_descriptions)
    for k, v in new_methods.items():
        if overwrite or k not in methods:
            methods[k] = v
            method_descriptions[k] = new_descriptions.get(k, "")
    return methods, method_descriptions


def sort_methods(methods, method_descriptions):
    """Sort methods and descriptions by method name."""
    methods = OrderedDict(sorted(methods.items(), key=lambda x: x[0]))
    method_descriptions = OrderedDict(sorted(method_descriptions.items(), key=lambda x: x[0]))
    return methods, method_descriptions


all_methods, all_descriptions = method_configs, descriptions
# Add discovered external methods
all_methods, all_descriptions = merge_methods(all_methods, all_descriptions, *discover_methods())
all_methods, all_descriptions = sort_methods(all_methods, all_descriptions)

# Register all possible external methods which can be installed with Nerfstudio
all_methods, all_descriptions = merge_methods(
    all_methods, all_descriptions, *sort_methods(*get_external_methods()), overwrite=False
)

AnnotatedBaseConfigUnion = tyro.conf.SuppressFixed[  # Don't show unparseable (fixed) arguments in helptext.
    tyro.conf.FlagConversionOff[
        tyro.extras.subcommand_type_from_defaults(defaults=all_methods, descriptions=all_descriptions)
    ]
]
"""Union[] type over config types, annotated with default instances for use with
tyro.cli(). Allows the user to pick between one of several base configurations, and
then override values in it."""
