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
Encoding functions
"""

import math
import itertools
from abc import abstractmethod
from typing import Literal, Optional, Sequence, Callable, Tuple, List, Any

import numpy as np
import torch
import torch.nn.functional as F
from jaxtyping import Float, Int, Shaped
from torch import Tensor, nn
from torch_scatter import segment_coo

from nerfstudio.field_components.base_field_component import FieldComponent
from nerfstudio.utils.math import (
    components_from_spherical_harmonics,
    expected_sin,
    grid_resolution,
    grid_scale,
    powi,
    next_multiple,
    sph_harm_coeff,
)
from nerfstudio.utils.printing import print_tcnn_speed_warning
from nerfstudio.cameras.bundle_adjustment import HashBundleAdjustment
from nerfstudio.utils.external import tcnn, TCNN_EXISTS


class Encoding(FieldComponent):
    """Encode an input tensor. Intended to be subclassed

    Args:
        in_dim: Input dimension of tensor
    """

    def __init__(self, in_dim: int) -> None:
        if in_dim <= 0:
            raise ValueError("Input dimension should be greater than zero")
        super().__init__(in_dim=in_dim)

    @abstractmethod
    def forward(
        self,
        in_tensor: Shaped[Tensor, "*bs input_dim"],
        *args: Any,
        **kwargs: Any,
    ) -> Shaped[Tensor, "*bs output_dim"]:
        """Call forward and returns and processed tensor

        Args:
            in_tensor: the input tensor to process
        """
        raise NotImplementedError


class Identity(Encoding):
    """Identity encoding (Does not modify input)"""

    def get_out_dim(self) -> int:
        if self.in_dim is None:
            raise ValueError("Input dimension has not been set")
        return self.in_dim

    def forward(self, in_tensor: Shaped[Tensor, "*bs input_dim"]) -> Shaped[Tensor, "*bs output_dim"]:
        return in_tensor


class ScalingAndOffset(Encoding):
    """Simple scaling and offset to input

    Args:
        in_dim: Input dimension of tensor
        scaling: Scaling applied to tensor.
        offset: Offset applied to tensor.
    """

    def __init__(self, in_dim: int, scaling: float = 1.0, offset: float = 0.0) -> None:
        super().__init__(in_dim)

        self.scaling = scaling
        self.offset = offset

    def get_out_dim(self) -> int:
        if self.in_dim is None:
            raise ValueError("Input dimension has not been set")
        return self.in_dim

    def forward(self, in_tensor: Float[Tensor, "*bs input_dim"]) -> Float[Tensor, "*bs output_dim"]:
        return self.scaling * in_tensor + self.offset


class NeRFEncoding(Encoding):
    """Multi-scale sinusoidal encodings. Support ``integrated positional encodings`` if covariances are provided.
    Each axis is encoded with frequencies ranging from 2^min_freq_exp to 2^max_freq_exp.

    Args:
        in_dim: Input dimension of tensor
        num_frequencies: Number of encoded frequencies per axis
        min_freq_exp: Minimum frequency exponent
        max_freq_exp: Maximum frequency exponent
        include_input: Append the input coordinate to the encoding
    """

    def __init__(
        self,
        in_dim: int,
        num_frequencies: int,
        min_freq_exp: float,
        max_freq_exp: float,
        include_input: bool = False,
        implementation: Literal["tcnn", "torch"] = "torch",
    ) -> None:
        super().__init__(in_dim)

        self.num_frequencies = num_frequencies
        self.min_freq = min_freq_exp
        self.max_freq = max_freq_exp
        self.include_input = include_input

        self.tcnn_encoding = None
        if implementation == "tcnn" and not TCNN_EXISTS:
            print_tcnn_speed_warning("NeRFEncoding")
        elif implementation == "tcnn":
            encoding_config = {"otype": "Frequency", "n_frequencies": num_frequencies}
            assert min_freq_exp == 0, "tcnn only supports min_freq_exp = 0"
            assert max_freq_exp == num_frequencies - 1, "tcnn only supports max_freq_exp = num_frequencies - 1"
            self.tcnn_encoding = tcnn.Encoding(
                n_input_dims=in_dim,
                encoding_config=encoding_config,
            )

    def get_out_dim(self) -> int:
        if self.in_dim is None:
            raise ValueError("Input dimension has not been set")
        out_dim = self.in_dim * self.num_frequencies * 2
        if self.include_input:
            out_dim += self.in_dim
        return out_dim

    def pytorch_fwd(
        self,
        in_tensor: Float[Tensor, "*bs input_dim"],
        covs: Optional[Float[Tensor, "*bs input_dim input_dim"]] = None,
    ) -> Float[Tensor, "*bs output_dim"]:
        """Calculates NeRF encoding.
            If covariances are provided the encodings will be integrated
            as proposed in mip-NeRF.

        Args:
            in_tensor: For best performance, the input tensor should be between 0 and 1.
            covs: Covariances of input points.
        Returns:
            Output values will be between -1 and 1
        """
        scaled_in_tensor = 2 * torch.pi * in_tensor  # scale to [0, 2pi]
        freqs = 2 ** torch.linspace(self.min_freq, self.max_freq, self.num_frequencies).to(in_tensor.device)
        scaled_inputs = scaled_in_tensor[..., None] * freqs  # [..., "input_dim", "num_scales"]
        scaled_inputs = scaled_inputs.view(*scaled_inputs.shape[:-2], -1)  # [..., "input_dim" * "num_scales"]

        if covs is None:
            encoded_inputs = torch.sin(torch.cat([scaled_inputs, scaled_inputs + torch.pi / 2.0], dim=-1))
        else:
            input_var = torch.diagonal(covs, dim1=-2, dim2=-1)[..., :, None] * freqs[None, :] ** 2
            input_var = input_var.reshape((*input_var.shape[:-2], -1))
            encoded_inputs = expected_sin(
                torch.cat([scaled_inputs, scaled_inputs + torch.pi / 2.0], dim=-1), torch.cat(2 * [input_var], dim=-1)
            )

        if self.include_input:
            encoded_inputs = torch.cat([encoded_inputs, in_tensor], dim=-1)
        return encoded_inputs

    def forward(
        self,
        in_tensor: Float[Tensor, "*bs input_dim"],
        covs: Optional[Float[Tensor, "*bs input_dim input_dim"]] = None,
    ) -> Float[Tensor, "*bs output_dim"]:
        if self.tcnn_encoding is not None:
            return self.tcnn_encoding(in_tensor)
        return self.pytorch_fwd(in_tensor, covs)


class RFFEncoding(Encoding):
    """Random Fourier Feature encoding. Supports integrated encodings.

    Args:
        in_dim: Input dimension of tensor
        num_frequencies: Number of encoding frequencies
        scale: Std of Gaussian to sample frequencies. Must be greater than zero
        include_input: Append the input coordinate to the encoding
    """

    def __init__(self, in_dim: int, num_frequencies: int, scale: float, include_input: bool = False) -> None:
        super().__init__(in_dim)

        self.num_frequencies = num_frequencies
        if not scale > 0:
            raise ValueError("RFF encoding scale should be greater than zero")
        self.scale = scale
        if self.in_dim is None:
            raise ValueError("Input dimension has not been set")
        b_matrix = torch.normal(mean=0, std=self.scale, size=(self.in_dim, self.num_frequencies))
        self.register_buffer(name="b_matrix", tensor=b_matrix)
        self.include_input = include_input

    def get_out_dim(self) -> int:
        out_dim = self.num_frequencies * 2
        if self.include_input:
            if self.in_dim is None:
                raise ValueError("Input dimension has not been set")
            out_dim += self.in_dim
        return out_dim

    def forward(
        self,
        in_tensor: Float[Tensor, "*bs input_dim"],
        covs: Optional[Float[Tensor, "*bs input_dim input_dim"]] = None,
    ) -> Float[Tensor, "*bs output_dim"]:
        """Calculates RFF encoding. If covariances are provided the encodings will be integrated as proposed
            in mip-NeRF.

        Args:
            in_tensor: For best performance, the input tensor should be between 0 and 1.
            covs: Covariances of input points.

        Returns:
            Output values will be between -1 and 1
        """
        scaled_in_tensor = 2 * torch.pi * in_tensor  # scale to [0, 2pi]
        scaled_inputs = scaled_in_tensor @ self.b_matrix  # [..., "num_frequencies"]

        if covs is None:
            encoded_inputs = torch.sin(torch.cat([scaled_inputs, scaled_inputs + torch.pi / 2.0], dim=-1))
        else:
            input_var = torch.sum((covs @ self.b_matrix) * self.b_matrix, -2)
            encoded_inputs = expected_sin(
                torch.cat([scaled_inputs, scaled_inputs + torch.pi / 2.0], dim=-1), torch.cat(2 * [input_var], dim=-1)
            )

        if self.include_input:
            encoded_inputs = torch.cat([encoded_inputs, in_tensor], dim=-1)

        return encoded_inputs


class HashEncoding(Encoding):
    """Hash encoding

    Args:
        num_levels: Number of feature grids.
        min_res: Resolution of smallest feature grid.
        max_res: Resolution of largest feature grid.
        log2_hashmap_size: Size of hash map is 2^log2_hashmap_size.
        features_per_level: Number of features per level.
        hash_init_scale: Value to initialize hash grid.
        implementation: Implementation of hash encoding. Fallback to torch if tcnn not available.
        interpolation: Interpolation override for tcnn hashgrid. Not supported for torch unless linear.
    """

    bundle_adjustment: HashBundleAdjustment

    def __init__(
        self,
        bundle_adjustment: HashBundleAdjustment,
        num_levels: int = 16,
        min_res: int = 16,
        max_res: int = 1024,
        log2_hashmap_size: int = 19,
        features_per_level: int = 2,
        hash_init_scale: float = 0.001,
        implementation: Literal["tcnn", "torch"] = "tcnn",
        interpolation: Optional[Literal["Nearest", "Linear", "Smoothstep"]] = None,
    ) -> None:
        super().__init__(in_dim=3)

        self.bundle_adjustment = bundle_adjustment
        self.num_levels = num_levels
        self.features_per_level = features_per_level
        self.log2_hashmap_size = log2_hashmap_size
        self.hash_table_size = 2**log2_hashmap_size
        self.min_res = min_res
        self.hash_init_scale = hash_init_scale

        levels = torch.arange(num_levels)
        self.growth_factor = np.exp((np.log(max_res) - np.log(min_res)) / (num_levels - 1)) if num_levels > 1 else 1

        self.register_buffer("hash_values", torch.tensor([1, 2654435761, 805459861]), False)

        self.tcnn_encoding = None
        if implementation == "tcnn" and not TCNN_EXISTS:
            print_tcnn_speed_warning("HashEncoding")
            implementation = "torch"

        if implementation == "tcnn":
            encoding_config = {
                "otype": "HashGrid",
                "n_levels": self.num_levels,
                "n_features_per_level": self.features_per_level,
                "log2_hashmap_size": self.log2_hashmap_size,
                "base_resolution": min_res,
                "per_level_scale": self.growth_factor,
            }
            if interpolation is not None:
                encoding_config["interpolation"] = interpolation

            self.tcnn_encoding = tcnn.Encoding(
                n_input_dims=3,
                encoding_config=encoding_config,
            )
            self.hash_table = self.tcnn_encoding.params
            offsets, scalings = self._create_hash_offset_and_scalings_tcnn()

            self.register_buffer("scalings", scalings, False)
            self.register_buffer("hash_offset", offsets, False)
            
            self.custom_forward = self.tcnn_encoding.forward # type: ignore
        elif implementation == "torch":
            self.hash_table = torch.rand(size=(self.hash_table_size * num_levels, features_per_level)) * 2 - 1
            self.hash_table *= hash_init_scale
            self.hash_table = nn.Parameter(self.hash_table)

            self.register_buffer("scalings", torch.floor(min_res * self.growth_factor**levels).view(-1, 1), False)
            self.register_buffer("hash_offset", levels * self.hash_table_size, False)

            self.custom_forward = self.pytorch_fwd
        if self.tcnn_encoding is None:
            assert (
                interpolation is None or interpolation == "Linear"
            ), f"interpolation '{interpolation}' is not supported for torch encoding backend"

        self.register_buffer("level_indexes", self._create_level_indexes(), False)

    def get_out_dim(self) -> int:
        return self.num_levels * self.features_per_level

    def _create_hash_offset_and_scalings_tcnn(self) -> Tuple[Tensor, Tensor]:
        """Create offset map for each weight."""
        offset: int = 0
        offsets: List[int] = []
        resolutions: List[int] = []
        for i in range(self.num_levels):
            resolution = grid_resolution(grid_scale(i, math.log2(self.growth_factor), self.min_res))
            resolutions.append(resolution)
            params_in_level = powi(resolution, self.in_dim)  # type: ignore
            params_in_level = next_multiple(params_in_level, 8)
            params_in_level = min(params_in_level, (2**self.log2_hashmap_size))
            offsets.append(offset)
            offset += params_in_level

        return torch.tensor(offsets), torch.tensor(resolutions)

    def _create_level_indexes(self) -> Tensor:
        """Create an affiliation of each hash pyramid weight to the levels."""
        indexes_shape = self.hash_table.view(-1, self.features_per_level).shape[0]
        level_indexes = self.hash_table.new_empty(indexes_shape, dtype=torch.long)
        for i in range(self.num_levels - 1):
            level_indexes[self.hash_offset[i] : self.hash_offset[i + 1]] = i  # type: ignore
        level_indexes[self.hash_offset[-1] :] = self.num_levels - 1  # type: ignore

        return level_indexes

    def hash_fn(self, in_tensor: Int[Tensor, "*bs num_levels 3"]) -> Shaped[Tensor, "*bs num_levels"]:
        """Returns hash tensor using method described in Instant-NGP

        Args:
            in_tensor: Tensor to be hashed
        """

        # min_val = torch.min(in_tensor)
        # max_val = torch.max(in_tensor)
        # assert min_val >= 0.0
        # assert max_val <= 1.0

        in_tensor = in_tensor * self.hash_values
        x = torch.bitwise_xor(in_tensor[..., 0], in_tensor[..., 1])
        x = torch.bitwise_xor(x, in_tensor[..., 2])
        x %= self.hash_table_size
        x += self.hash_offset
        return x

    def pytorch_fwd(self, in_tensor: Float[Tensor, "*bs input_dim"]) -> Float[Tensor, "*bs output_dim"]:
        """Forward pass using pytorch. Significantly slower than TCNN implementation."""

        assert in_tensor.shape[-1] == 3
        in_tensor = in_tensor[..., None, :]  # [..., 1, 3]
        scaled = in_tensor * self.scalings  # [..., L, 3]
        scaled_c = torch.ceil(scaled).type(torch.int32)
        scaled_f = torch.floor(scaled).type(torch.int32)

        offset = scaled - scaled_f

        hashed_0 = self.hash_fn(scaled_c)  # [..., num_levels]
        hashed_1 = self.hash_fn(torch.cat([scaled_c[..., 0:1], scaled_f[..., 1:2], scaled_c[..., 2:3]], dim=-1))
        hashed_2 = self.hash_fn(torch.cat([scaled_f[..., 0:1], scaled_f[..., 1:2], scaled_c[..., 2:3]], dim=-1))
        hashed_3 = self.hash_fn(torch.cat([scaled_f[..., 0:1], scaled_c[..., 1:2], scaled_c[..., 2:3]], dim=-1))
        hashed_4 = self.hash_fn(torch.cat([scaled_c[..., 0:1], scaled_c[..., 1:2], scaled_f[..., 2:3]], dim=-1))
        hashed_5 = self.hash_fn(torch.cat([scaled_c[..., 0:1], scaled_f[..., 1:2], scaled_f[..., 2:3]], dim=-1))
        hashed_6 = self.hash_fn(scaled_f)
        hashed_7 = self.hash_fn(torch.cat([scaled_f[..., 0:1], scaled_c[..., 1:2], scaled_f[..., 2:3]], dim=-1))

        f_0 = self.hash_table[hashed_0]  # [..., num_levels, features_per_level]
        f_1 = self.hash_table[hashed_1]
        f_2 = self.hash_table[hashed_2]
        f_3 = self.hash_table[hashed_3]
        f_4 = self.hash_table[hashed_4]
        f_5 = self.hash_table[hashed_5]
        f_6 = self.hash_table[hashed_6]
        f_7 = self.hash_table[hashed_7]

        f_03 = f_0 * offset[..., 0:1] + f_3 * (1 - offset[..., 0:1])
        f_12 = f_1 * offset[..., 0:1] + f_2 * (1 - offset[..., 0:1])
        f_56 = f_5 * offset[..., 0:1] + f_6 * (1 - offset[..., 0:1])
        f_47 = f_4 * offset[..., 0:1] + f_7 * (1 - offset[..., 0:1])

        f0312 = f_03 * offset[..., 1:2] + f_12 * (1 - offset[..., 1:2])
        f4756 = f_47 * offset[..., 1:2] + f_56 * (1 - offset[..., 1:2])

        encoded_value = f0312 * offset[..., 2:3] + f4756 * (
            1 - offset[..., 2:3]
        )  # [..., num_levels, features_per_level]

        return torch.flatten(encoded_value, start_dim=-2, end_dim=-1)  # [..., num_levels * features_per_level]

    def regularize_hash_pyramid(
        self,
        regularize_fn: Callable[[Tensor], Tensor] = torch.abs,
    ) -> Float[Tensor, "0"]:
        """Regularize hash pyramid weights."""
        hash_decay = segment_coo(
            src=regularize_fn(self.hash_table.view(-1, self.features_per_level)),
            index=self.level_indexes,  # type: ignore
            out=self.hash_table.new_zeros(self.num_levels, self.features_per_level),
            reduce="mean",
        ).mean()

        return hash_decay

    def scale_featurization(self) -> Float[Tensor, "*num_levels"]:
        """Compute scale featurization proposed in ZipNeRF paper."""
        scale_feat = segment_coo(
            src=self.hash_table.view(-1, self.features_per_level).pow(2).sum(-1),
            index=self.level_indexes,  # type: ignore
            dim_size=self.num_levels,
            reduce="mean",
        )

        return scale_feat

    def forward(
        self,
        in_tensor: Shaped[Tensor, "*bs input_dim"],
    ) -> Shaped[Tensor, "*bs output_dim"]:

        out = self.custom_forward(in_tensor)
        if self.bundle_adjustment.use_bundle_adjust:
            out = self.bundle_adjustment(
                out.view(-1, self.num_levels, self.features_per_level),
            ).view(*out.shape)

        return out


class TensorCPEncoding(Encoding):
    """Learned CANDECOMP/PARFAC (CP) decomposition encoding used in TensoRF

    Args:
        resolution: Resolution of grid.
        num_components: Number of components per dimension.
        init_scale: Initialization scale.
    """

    def __init__(self, resolution: int = 256, num_components: int = 24, init_scale: float = 0.1) -> None:
        super().__init__(in_dim=3)

        self.resolution = resolution
        self.num_components = num_components

        # TODO Learning rates should be different for these
        self.line_coef = nn.Parameter(init_scale * torch.randn((3, num_components, resolution, 1)))

    def get_out_dim(self) -> int:
        return self.num_components

    def forward(self, in_tensor: Float[Tensor, "*bs input_dim"]) -> Float[Tensor, "*bs output_dim"]:
        line_coord = torch.stack([in_tensor[..., 2], in_tensor[..., 1], in_tensor[..., 0]])  # [3, ...]
        line_coord = torch.stack([torch.zeros_like(line_coord), line_coord], dim=-1)  # [3, ...., 2]

        # Stop gradients from going to sampler
        line_coord = line_coord.view(3, -1, 1, 2).detach()

        line_features = F.grid_sample(self.line_coef, line_coord, align_corners=True)  # [3, Components, -1, 1]

        features = torch.prod(line_features, dim=0)
        features = torch.moveaxis(features.view(self.num_components, *in_tensor.shape[:-1]), 0, -1)

        return features  # [..., Components]

    @torch.no_grad()
    def upsample_grid(self, resolution: int) -> None:
        """Upsamples underyling feature grid

        Args:
            resolution: Target resolution.
        """

        self.line_coef.data = F.interpolate(
            self.line_coef.data, size=(resolution, 1), mode="bilinear", align_corners=True
        )

        self.resolution = resolution


class TensorVMEncoding(Encoding):
    """Learned vector-matrix encoding proposed by TensoRF

    Args:
        resolution: Resolution of grid.
        num_components: Number of components per dimension.
        init_scale: Initialization scale.
    """

    plane_coef: Float[Tensor, "3 num_components resolution resolution"]
    line_coef: Float[Tensor, "3 num_components resolution 1"]

    def __init__(
        self,
        resolution: int = 128,
        num_components: int = 24,
        init_scale: float = 0.1,
    ) -> None:
        super().__init__(in_dim=3)

        self.resolution = resolution
        self.num_components = num_components

        self.plane_coef = nn.Parameter(init_scale * torch.randn((3, num_components, resolution, resolution)))
        self.line_coef = nn.Parameter(init_scale * torch.randn((3, num_components, resolution, 1)))

    def get_out_dim(self) -> int:
        return self.num_components * 3

    def forward(self, in_tensor: Float[Tensor, "*bs input_dim"]) -> Float[Tensor, "*bs output_dim"]:
        """Compute encoding for each position in in_positions

        Args:
            in_tensor: position inside bounds in range [-1,1],

        Returns: Encoded position
        """
        plane_coord = torch.stack([in_tensor[..., [0, 1]], in_tensor[..., [0, 2]], in_tensor[..., [1, 2]]])  # [3,...,2]
        line_coord = torch.stack([in_tensor[..., 2], in_tensor[..., 1], in_tensor[..., 0]])  # [3, ...]
        line_coord = torch.stack([torch.zeros_like(line_coord), line_coord], dim=-1)  # [3, ...., 2]

        # Stop gradients from going to sampler
        plane_coord = plane_coord.view(3, -1, 1, 2).detach()
        line_coord = line_coord.view(3, -1, 1, 2).detach()

        plane_features = F.grid_sample(self.plane_coef, plane_coord, align_corners=True)  # [3, Components, -1, 1]
        line_features = F.grid_sample(self.line_coef, line_coord, align_corners=True)  # [3, Components, -1, 1]

        features = plane_features * line_features  # [3, Components, -1, 1]
        features = torch.moveaxis(features.view(3 * self.num_components, *in_tensor.shape[:-1]), 0, -1)

        return features  # [..., 3 * Components]

    @torch.no_grad()
    def upsample_grid(self, resolution: int) -> None:
        """Upsamples underlying feature grid

        Args:
            resolution: Target resolution.
        """
        plane_coef = F.interpolate(
            self.plane_coef.data, size=(resolution, resolution), mode="bilinear", align_corners=True
        )
        line_coef = F.interpolate(self.line_coef.data, size=(resolution, 1), mode="bilinear", align_corners=True)

        self.plane_coef, self.line_coef = torch.nn.Parameter(plane_coef), torch.nn.Parameter(line_coef)
        self.resolution = resolution


class TriplaneEncoding(Encoding):
    """Learned triplane encoding

    The encoding at [i,j,k] is an n dimensional vector corresponding to the element-wise product of the
    three n dimensional vectors at plane_coeff[i,j], plane_coeff[i,k], and plane_coeff[j,k].

    This allows for marginally more expressivity than the TensorVMEncoding, and each component is self standing
    and symmetrical, unlike with VM decomposition where we needed one component with a vector along all the x, y, z
    directions for symmetry.

    This can be thought of as 3 planes of features perpendicular to the x, y, and z axes, respectively and intersecting
    at the origin, and the encoding being the element-wise product of the element at the projection of [i, j, k] on
    these planes.

    The use for this is in representing a tensor decomp of a 4D embedding tensor: (x, y, z, feature_size)

    This will return a tensor of shape (bs:..., num_components)

    Args:
        resolution: Resolution of grid.
        num_components: The number of scalar triplanes to use (ie: output feature size)
        init_scale: The scale of the initial values of the planes
        product: Whether to use the element-wise product of the planes or the sum
    """

    plane_coef: Float[Tensor, "3 num_components resolution resolution"]

    def __init__(
        self,
        resolution: int = 32,
        num_components: int = 64,
        init_scale: float = 0.1,
        reduce: Literal["sum", "product"] = "sum",
    ) -> None:
        super().__init__(in_dim=3)

        self.resolution = resolution
        self.num_components = num_components
        self.init_scale = init_scale
        self.reduce = reduce

        self.plane_coef = nn.Parameter(
            self.init_scale * torch.randn((3, self.num_components, self.resolution, self.resolution))
        )

    def get_out_dim(self) -> int:
        return self.num_components

    def forward(self, in_tensor: Float[Tensor, "*bs 3"]) -> Float[Tensor, "*bs num_components featuresize"]:
        """Sample features from this encoder. Expects in_tensor to be in range [0, resolution]"""

        original_shape = in_tensor.shape
        in_tensor = in_tensor.reshape(-1, 3)

        plane_coord = torch.stack([in_tensor[..., [0, 1]], in_tensor[..., [0, 2]], in_tensor[..., [1, 2]]], dim=0)

        # Stop gradients from going to sampler
        plane_coord = plane_coord.detach().view(3, -1, 1, 2)
        plane_features = F.grid_sample(
            self.plane_coef, plane_coord, align_corners=True
        )  # [3, num_components, flattened_bs, 1]

        if self.reduce == "product":
            plane_features = plane_features.prod(0).squeeze(-1).T  # [flattened_bs, num_components]
        else:
            plane_features = plane_features.sum(0).squeeze(-1).T

        return plane_features.reshape(*original_shape[:-1], self.num_components)

    @torch.no_grad()
    def upsample_grid(self, resolution: int) -> None:
        """Upsamples underlying feature grid

        Args:
            resolution: Target resolution.
        """
        plane_coef = F.interpolate(
            self.plane_coef.data, size=(resolution, resolution), mode="bilinear", align_corners=True
        )

        self.plane_coef = torch.nn.Parameter(plane_coef)
        self.resolution = resolution


class KPlanesEncoding(Encoding):
    """Learned K-Planes encoding

    A plane encoding supporting both 3D and 4D coordinates. With 3D coordinates this is similar to
    :class:`TriplaneEncoding`. With 4D coordinates, the encoding at point ``[i,j,k,q]`` is
    a n-dimensional vector computed as the elementwise product of 6 n-dimensional vectors at
    ``planes[i,j]``, ``planes[i,k]``, ``planes[i,q]``, ``planes[j,k]``, ``planes[j,q]``,
    ``planes[k,q]``.

    Unlike :class:`TriplaneEncoding` this class supports different resolution along each axis.

    This will return a tensor of shape (bs:..., num_components)

    Args:
        resolution: Resolution of the grid. Can be a sequence of 3 or 4 integers.
        num_components: The number of scalar planes to use (ie: output feature size)
        init_a: The lower-bound of the uniform distribution used to initialize the spatial planes
        init_b: The upper-bound of the uniform distribution used to initialize the spatial planes
        reduce: Whether to use the element-wise product of the planes or the sum
    """

    def __init__(
        self,
        resolution: Sequence[int] = (128, 128, 128),
        num_components: int = 64,
        init_a: float = 0.1,
        init_b: float = 0.5,
        reduce: Literal["sum", "product"] = "product",
    ) -> None:
        super().__init__(in_dim=len(resolution))

        self.resolution = resolution
        self.num_components = num_components
        self.reduce = reduce
        if self.in_dim not in {3, 4}:
            raise ValueError(
                f"The dimension of coordinates must be either 3 (static scenes) "
                f"or 4 (dynamic scenes). Found resolution with {self.in_dim} dimensions."
            )
        has_time_planes = self.in_dim == 4

        self.coo_combs = list(itertools.combinations(range(self.in_dim), 2))
        # Unlike the Triplane encoding, we use a parameter list instead of batching all planes
        # together to support uneven resolutions (especially useful for time).
        # Dynamic models (in_dim == 4) will have 6 planes:
        # (y, x), (z, x), (t, x), (z, y), (t, y), (t, z)
        # static models (in_dim == 3) will only have the 1st, 2nd and 4th planes.
        self.plane_coefs = nn.ParameterList()
        for coo_comb in self.coo_combs:
            new_plane_coef = nn.Parameter(
                torch.empty([self.num_components] + [self.resolution[cc] for cc in coo_comb[::-1]])
            )
            if has_time_planes and 3 in coo_comb:  # Time planes initialized to 1
                nn.init.ones_(new_plane_coef)
            else:
                nn.init.uniform_(new_plane_coef, a=init_a, b=init_b)
            self.plane_coefs.append(new_plane_coef)

    def get_out_dim(self) -> int:
        return self.num_components

    def forward(self, in_tensor: Float[Tensor, "*bs input_dim"]) -> Float[Tensor, "*bs output_dim"]:
        """Sample features from this encoder. Expects ``in_tensor`` to be in range [-1, 1]"""
        original_shape = in_tensor.shape

        assert any(self.coo_combs)
        output = 1.0 if self.reduce == "product" else 0.0  # identity for corresponding op
        for ci, coo_comb in enumerate(self.coo_combs):
            grid = self.plane_coefs[ci].unsqueeze(0)  # [1, feature_dim, reso1, reso2]
            coords = in_tensor[..., coo_comb].view(1, 1, -1, 2)  # [1, 1, flattened_bs, 2]
            interp = F.grid_sample(
                grid, coords, align_corners=True, padding_mode="border"
            )  # [1, output_dim, 1, flattened_bs]
            interp = interp.view(self.num_components, -1).T  # [flattened_bs, output_dim]
            if self.reduce == "product":
                output = output * interp
            else:
                output = output + interp

        # Typing: output gets converted to a tensor after the first iteration of the loop
        assert isinstance(output, Tensor)
        return output.reshape(*original_shape[:-1], self.num_components)


class SHEncoding(Encoding):
    """Spherical harmonic encoding

    Args:
        levels: Number of spherical harmonic levels to encode.
    """

    def __init__(self, levels: int = 4, implementation: Literal["tcnn", "torch"] = "torch") -> None:
        super().__init__(in_dim=3)

        if levels <= 0 or levels > 4:
            raise ValueError(f"Spherical harmonic encoding only supports 1 to 4 levels, requested {levels}")

        self.levels = levels

        self.tcnn_encoding = None
        if implementation == "tcnn" and not TCNN_EXISTS:
            print_tcnn_speed_warning("SHEncoding")
        elif implementation == "tcnn":
            encoding_config = {
                "otype": "SphericalHarmonics",
                "degree": levels,
            }
            self.tcnn_encoding = tcnn.Encoding(
                n_input_dims=3,
                encoding_config=encoding_config,
            )

    def get_out_dim(self) -> int:
        return self.levels**2

    @torch.no_grad()
    def pytorch_fwd(self, in_tensor: Float[Tensor, "*bs input_dim"]) -> Float[Tensor, "*bs output_dim"]:
        """Forward pass using pytorch. Significantly slower than TCNN implementation."""
        return components_from_spherical_harmonics(levels=self.levels, directions=in_tensor)

    def forward(self, in_tensor: Float[Tensor, "*bs input_dim"]) -> Float[Tensor, "*bs output_dim"]:
        if self.tcnn_encoding is not None:
            return self.tcnn_encoding(in_tensor)
        return self.pytorch_fwd(in_tensor)


class IDEncoding(Encoding):
    """Module for integrated directional encoding (IDE).
        from Equations 6-8 of arxiv.org/abs/2112.03907.
    """

    def __init__(self, deg_view: int = 4) -> None:
        """Initialize integrated directional encoding (IDE) module.
        Args:
            deg_view: number of spherical harmonics degrees to use.
        """
        super().__init__(in_dim=3)
        self.deg_view = deg_view

        if deg_view > 4:
            raise ValueError("Only deg_view of at most 4 is numerically stable.")

        ml_array = self._get_ml_array(deg_view)
        l_max = 2 ** (deg_view - 1)

        # Create a matrix corresponding to ml_array holding all coefficients, which,
        # when multiplied (from the right) by the z coordinate Vandermonde matrix,
        # results in the z component of the encoding.
        mat = np.zeros((l_max + 1, ml_array.shape[1]))
        for i, (m, l) in enumerate(ml_array.T):
            for k in range(l - m + 1):
                mat[k, i] = sph_harm_coeff(l, m, k)

        sigma = 0.5 * ml_array[1, :] * (ml_array[1, :] + 1)
        self.register_buffer("mat", torch.from_numpy(mat).to(torch.float32), False)
        self.register_buffer("ml_array", torch.from_numpy(ml_array).to(torch.float32), False)
        self.register_buffer("pow_level", torch.arange(l_max + 1).to(torch.float32), False)
        self.register_buffer("sigma", torch.from_numpy(sigma).to(torch.float32), False)

    def get_out_dim(self) -> int:
        return (2**self.deg_view - 1 + self.deg_view) * 2

    @staticmethod
    def _get_ml_array(deg_view: int) -> np.ndarray:
        """Create a list with all pairs of (l, m) values to use in the encoding."""
        ml_list = [(m, 2**i) for i in range(deg_view) for m in range(2**i + 1)]
        ml_array = np.array(ml_list).T
        return ml_array

    def forward(
        self,
        in_tensor: Float[Tensor, "*bs input_dim"],
        roughness: Float[Tensor, "*bs 1"],
    ) -> Float[Tensor, "*bs output_dim"]:
        """Compute integrated directional encoding (IDE).
        Args:
            in_tensor: [..., 3] array of Cartesian coordinates of directions to evaluate at.
            roughness: [..., 1] reciprocal of the concentration parameter of the von
                Mises-Fisher distribution.
        """
        x = in_tensor[..., 0:1]
        y = in_tensor[..., 1:2]
        z = in_tensor[..., 2:3]

        # avoid 0 + 0j exponentiation
        zero_xy = torch.logical_and(x == 0, y == 0)
        y = y + zero_xy

        vmz = z ** self.pow_level
        vmxy = (x + 1j * y) ** self.ml_array[0, :]
        sph_harms = vmxy * torch.matmul(vmz, self.mat)
        ide = sph_harms * torch.exp(-self.sigma * roughness)

        return torch.cat([torch.real(ide), torch.imag(ide)], dim=-1)
