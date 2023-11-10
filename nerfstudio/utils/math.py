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

""" Math Helper Functions """

from dataclasses import dataclass
from typing import Literal, Tuple

import math
import torch
import numpy as np
from jaxtyping import Bool, Float
from torch import Tensor

from nerfstudio.utils.misc import torch_compile


@torch_compile(dynamic=True, mode="reduce-overhead")
def components_from_spherical_harmonics(
    levels: int,
    directions: Float[Tensor, "*batch 3"],
) -> Float[Tensor, "*batch components"]:
    """
    Returns value for each component of spherical harmonics.
    # Based on https://github.com/NVlabs/tiny-cuda-nn/blob/8575542682cb67cddfc748cc3d3cfc12593799aa/include/include/tiny-cuda-nn/encodings/spherical_harmonics.h#L76

    Args:
        levels: Number of spherical harmonic levels to compute.
        directions: Spherical harmonic coefficients
    """
    # torch.compile change dtype even if the standard method returns int 
    num_components = int(pow(levels, 2))

    # use transpose matrix to inference faster
    # at the end, transpose again
    components = directions.new_zeros((num_components, *directions.shape[:-1]))

    assert 1 <= levels <= 8, f"SH levels must be in [1,8], got {levels}"
    assert directions.shape[-1] == 3, f"Direction input should have three dimensions. Got {directions.shape[-1]}"

    x = directions[..., 0]
    y = directions[..., 1]
    z = directions[..., 2]

    # l0
    components[0] = 0.28209479177387814

    # l1
    if levels > 1:
        components[1] = -0.48860251190291987 * y
        components[2] = 0.48860251190291987 * z
        components[3] = -0.48860251190291987 * x

    # l2
    if levels > 2:
        # precompute 
        xy = x * y
        xz = x * z
        yz = y * z
        x2 = x * x
        y2 = y * y
        z2 = z * z

        components[4] = 1.0925484305920792 * xy
        components[5] = -1.0925484305920792 * yz
        components[6] = 0.94617469575755997 * z2 - 0.31539156525251999
        components[7] = -1.0925484305920792 * xz
        components[8] = 0.54627421529603959 * x2 - 0.54627421529603959 * y2

    # l3
    if levels > 3:
        # precompute
        xyz = xy * z

        components[9] = 0.59004358992664352 * y * (-3.0 * x2 + y2)
        components[10] = 2.8906114426405538 * xyz
        components[11] = 0.45704579946446572 * y * (1.0 - 5.0 * z2)
        components[12] = 0.3731763325901154 * z * (5.0 * z2 - 3.0)
        components[13] = 0.45704579946446572 * x * (1.0 - 5.0 * z2)
        components[14] = 1.4453057213202769 * z * (x2 - y2)
        components[15] = 0.59004358992664352 * x * (-x2 + 3.0 * y2)

    # l4
    if levels > 4:
        # precompute
        x4 = x2 * x2
        y4 = y2 * y2
        z4 = z2 * z2

        components[16] = 2.5033429417967046 * xy * (x2 - y2)
        components[17] = 1.7701307697799304 * yz * (-3.0 * x2 + y2)
        components[18] = 0.94617469575756008 * xy * (7.0 * z2 - 1.0)
        components[19] = 0.66904654355728921 * yz * (3.0 - 7.0 * z2)
        components[20] = (
            -3.1735664074561294 * z2 + 3.7024941420321507 * z4 + 0.31735664074561293
        )
        components[21] = 0.66904654355728921 * xz * (3.0 - 7.0 * z2)
        components[22] = 0.47308734787878004 * (x2 - y2) * (7.0 * z2 - 1.0)
        components[23] = 1.7701307697799304 * xz * (-x2 + 3.0 * y2)
        components[24] = (
            -3.7550144126950569 * x2 * y2
            + 0.62583573544917614 * x4
            + 0.62583573544917614 * y4
        )

    # l5
    if levels > 5:
        components[25] = 0.65638205684017015 * y * (10.0 * x2 * y2 - 5.0 * x4 - y4)
        components[26] = 8.3026492595241645 * xyz * (x2 - y2)
        components[27] = -0.48923829943525038 * y * (3.0 * x2 - y2) * (9.0 * z2 - 1.0)
        components[28] = 4.7935367849733241 * xyz * (3.0 * z2 - 1.0)
        components[29] = 0.45294665119569694 * y * (14.0 * z2 - 21.0 * z4 - 1.0)
        components[30] = 0.1169503224534236 * z * (-70.0 * z2 + 63.0 * z4 + 15.0)
        components[31] = 0.45294665119569694 * x * (14.0 * z2 - 21.0 * z4 - 1.0)
        components[32] = 2.3967683924866621 * z * (x2 - y2) * (3.0 * z2 - 1.0)
        components[33] = -0.48923829943525038 * x * (x2 - 3.0 * y2) * (9.0 * z2 - 1.0)
        components[34] = 2.0756623148810411 * z * (-6.0 * x2 * y2 + x4 + y4)
        components[35] = 0.65638205684017015 * x * (10.0 * x2 * y2 - x4 - 5.0 * y4)

     # l6
    if levels > 6:
        # precompute
        x6 = x4 * x2
        y6 = y4 * y2
        z6 = z4 * z2

        components[36] = 1.3663682103838286 * xy * (-10.0 * x2 * y2 + 3.0 * x4 + 3.0 * y4)
        components[37] = 2.3666191622317521 * yz * (10.0 * x2 * y2 - 5.0 * x4 - y4)
        components[38] = 2.0182596029148963 * xy * (x2 - y2) * (11.0 * z2 - 1.0)
        components[39] = -0.92120525951492349 * yz * (3.0 * x2 - y2) * (11.0 * z2 - 3.0)
        components[40] = 0.92120525951492349 * xy * (-18.0 * z2 + 33.0 * z4 + 1.0)
        components[41] = 0.58262136251873131 * yz * (30.0 * z2 - 33.0 * z4 - 5.0)
        components[42] = (
            6.6747662381009842 * z2
            - 20.024298714302954 * z4
            + 14.684485723822165 * z6
            - 0.31784601133814211
        )
        components[43] = 0.58262136251873131 * xz * (30.0 * z2 - 33.0 * z4 - 5.0)
        components[44] = (
            0.46060262975746175
            * (x2 - y2)
            * (11.0 * z2 * (3.0 * z2 - 1.0) - 7.0 * z2 + 1.0)
        )
        components[45] = -0.92120525951492349 * xz * (x2 - 3.0 * y2) * (11.0 * z2 - 3.0)
        components[46] = 0.50456490072872406 * (11.0 * z2 - 1.0) * (-6.0 * x2 * y2 + x4 + y4)
        components[47] = 2.3666191622317521 * xz * (10.0 * x2 * y2 - x4 - 5.0 * y4)
        components[48] = (
            10.247761577878714 * x2 * y4
            - 10.247761577878714 * x4 * y2
            + 0.6831841051919143 * x6
            - 0.6831841051919143 * y6
        )

     # l7
    if levels > 7:
        components[49] = (
            0.70716273252459627 * y * (-21.0 * x2 * y4 + 35.0 * x4 * y2 - 7.0 * x6 + y6)
        )
        components[50] = 5.2919213236038001 * xyz * (-10.0 * x2 * y2 + 3.0 * x4 + 3.0 * y4)
        components[51] = (
            -0.51891557872026028
            * y
            * (13.0 * z2 - 1.0)
            * (-10.0 * x2 * y2 + 5.0 * x4 + y4)
        )
        components[52] = 4.1513246297620823 * xyz * (x2 - y2) * (13.0 * z2 - 3.0)
        components[53] = (
            -0.15645893386229404
            * y
            * (3.0 * x2 - y2)
            * (13.0 * z2 * (11.0 * z2 - 3.0) - 27.0 * z2 + 3.0)
        )
        components[54] = 0.44253269244498261 * xyz * (-110.0 * z2 + 143.0 * z4 + 15.0)
        components[55] = (
            0.090331607582517306 * y * (-135.0 * z2 + 495.0 * z4 - 429.0 * z6 + 5.0)
        )
        components[56] = (
            0.068284276912004949 * z * (315.0 * z2 - 693.0 * z4 + 429.0 * z6 - 35.0)
        )
        components[57] = (
            0.090331607582517306 * x * (-135.0 * z2 + 495.0 * z4 - 429.0 * z6 + 5.0)
        )
        components[58] = (
            0.07375544874083044
            * z
            * (x2 - y2)
            * (143.0 * z2 * (3.0 * z2 - 1.0) - 187.0 * z2 + 45.0)
        )
        components[59] = (
            -0.15645893386229404
            * x
            * (x2 - 3.0 * y2)
            * (13.0 * z2 * (11.0 * z2 - 3.0) - 27.0 * z2 + 3.0)
        )
        components[60] = (
            1.0378311574405206 * z * (13.0 * z2 - 3.0) * (-6.0 * x2 * y2 + x4 + y4)
        )
        components[61] = (
            -0.51891557872026028
            * x
            * (13.0 * z2 - 1.0)
            * (-10.0 * x2 * y2 + x4 + 5.0 * y4)
        )
        components[62] = 2.6459606618019 * z * (15.0 * x2 * y4 - 15.0 * x4 * y2 + x6 - y6)
        components[63] = (
            0.70716273252459627 * x * (-35.0 * x2 * y4 + 21.0 * x4 * y2 - x6 + 7.0 * y6)
        )

    return components.T


@dataclass
class Gaussians:
    """Stores Gaussians

    Args:
        mean: Mean of multivariate Gaussian
        cov: Covariance of multivariate Gaussian.
    """

    mean: Float[Tensor, "*batch dim"]
    cov: Float[Tensor, "*batch dim dim"]


@torch_compile
def compute_3d_gaussian(
    directions: Float[Tensor, "*batch 3"],
    means: Float[Tensor, "*batch 3"],
    dir_variance: Float[Tensor, "*batch 1"],
    radius_variance: Float[Tensor, "*batch 1"],
) -> Gaussians:
    """Compute gaussian along ray.

    Args:
        directions: Axis of Gaussian.
        means: Mean of Gaussian.
        dir_variance: Variance along direction axis.
        radius_variance: Variance tangent to direction axis.

    Returns:
        Gaussians: Oriented 3D gaussian.
    """

    dir_outer_product = directions[..., :, None] * directions[..., None, :]
    eye = torch.eye(directions.shape[-1], device=directions.device)
    dir_mag_sq = torch.clamp(torch.sum(directions**2, dim=-1, keepdim=True), min=1e-10)
    null_outer_product = eye - directions[..., :, None] * (directions / dir_mag_sq)[..., None, :]
    dir_cov_diag = dir_variance[..., None] * dir_outer_product[..., :, :]
    radius_cov_diag = radius_variance[..., None] * null_outer_product[..., :, :]
    cov = dir_cov_diag + radius_cov_diag
    return Gaussians(mean=means, cov=cov)


@torch_compile
def cylinder_to_gaussian(
    origins: Float[Tensor, "*batch 3"],
    directions: Float[Tensor, "*batch 3"],
    starts: Float[Tensor, "*batch 1"],
    ends: Float[Tensor, "*batch 1"],
    radius: Float[Tensor, "*batch 1"],
) -> Gaussians:
    """Approximates cylinders with a Gaussian distributions.

    Args:
        origins: Origins of cylinders.
        directions: Direction (axis) of cylinders.
        starts: Start of cylinders.
        ends: End of cylinders.
        radius: Radii of cylinders.

    Returns:
        Gaussians: Approximation of cylinders
    """
    means = origins + directions * ((starts + ends) / 2.0)
    dir_variance = (ends - starts) ** 2 / 12
    radius_variance = radius**2 / 4.0
    return compute_3d_gaussian(directions, means, dir_variance, radius_variance)


@torch_compile
def conical_frustum_to_gaussian(
    origins: Float[Tensor, "*batch 3"],
    directions: Float[Tensor, "*batch 3"],
    starts: Float[Tensor, "*batch 1"],
    ends: Float[Tensor, "*batch 1"],
    radius: Float[Tensor, "*batch 1"],
) -> Gaussians:
    """Approximates conical frustums with a Gaussian distributions.

    Uses stable parameterization described in mip-NeRF publication.

    Args:
        origins: Origins of cones.
        directions: Direction (axis) of frustums.
        starts: Start of conical frustums.
        ends: End of conical frustums.
        radius: Radii of cone a distance of 1 from the origin.

    Returns:
        Gaussians: Approximation of conical frustums
    """
    mu = (starts + ends) / 2.0
    hw = (ends - starts) / 2.0
    means = origins + directions * (mu + (2.0 * mu * hw**2.0) / (3.0 * mu**2.0 + hw**2.0))
    dir_variance = (hw**2) / 3 - (4 / 15) * ((hw**4 * (12 * mu**2 - hw**2)) / (3 * mu**2 + hw**2) ** 2)
    radius_variance = radius**2 * ((mu**2) / 4 + (5 / 12) * hw**2 - 4 / 15 * (hw**4) / (3 * mu**2 + hw**2))
    return compute_3d_gaussian(directions, means, dir_variance, radius_variance)


@torch_compile
def multisampled_frustum_to_gaussian(
    origins: Float[Tensor, "*batch num_samples 3"],
    directions: Float[Tensor, "*batch num_samples 3"],
    starts: Float[Tensor, "*batch num_samples 1"],
    ends: Float[Tensor, "*batch num_samples 1"],
    radius: Float[Tensor, "*batch num_samples 1"],
    rand: bool = True,
    cov_scale: float = 0.5,
    eps: float = 1e-8,
) -> Gaussians:
    """Approximates frustums with a Gaussian distributions via multisampling.
    Proposed in ZipNeRF https://arxiv.org/pdf/2304.06706.pdf

    Taken from https://github.com/SuLvXiangXin/zipnerf-pytorch/blob/b1cb42943d244301a013bd53f9cb964f576b0af4/internal/render.py#L92

    Args:
        origins: Origins of cones.
        directions: Direction (axis) of frustums.
        starts: Start of frustums.
        ends: End of frustums.
        radius: Radii of cone a distance of 1 from the origin.
        rand: Whether should add noise to points or not.
        cov_scale: Covariance scale parameter.
        eps: Small number.

    Returns:
        Gaussians: Approximation of frustums via multisampling
    """

    # middle points
    t_m = (starts + ends) / 2.0
    # half of the width
    t_d = (ends - starts) / 2.0

    # prepare 6-point hexagonal pattern for each sample
    j = torch.arange(6, device=starts.device, dtype=starts.dtype)
    t = starts + t_d / (t_d**2 + 3 * t_m**2) * (
        ends**2 + 2 * t_m**2 + 3 / 7**0.5 * (2 * j / 5 - 1) * ((t_d**2 - t_m**2) ** 2 + 4 * t_m**4).sqrt()
    )  # [..., num_samples, 6]

    deg = torch.pi / 3 * starts.new_tensor([0, 2, 4, 3, 5, 1]).expand(t.shape)
    if rand:
        # randomly rotate and flip
        mask = torch.rand_like(starts) > 0.5  # [..., num_samples, 1]
        deg = deg + 2 * torch.pi * torch.rand_like(starts)
        deg = torch.where(mask, deg, 5 * torch.pi / 3.0 - deg)
    else:
        # rotate 30 degree and flip every other pattern
        mask = (
            (
                torch.arange(
                    end=starts.shape[-2],
                    device=starts.device,
                    dtype=starts.dtype,
                )
                % 2
                == 0
            )
            .unsqueeze(-1)
            .expand(starts.shape)
        )  # [..., num_samples, 6]
        deg = torch.where(mask, deg, deg + torch.pi / 6.0)
        deg = torch.where(mask, deg, 5 * torch.pi / 3.0 - deg)

    means = torch.stack(
        [
            radius * t * torch.cos(deg) / 2**0.5,
            radius * t * torch.sin(deg) / 2**0.5,
            t,
        ],
        dim=-1,
    )  # [..., "num_samples", 6, 3]
    stds = cov_scale * radius * t / 2**0.5  # [..., "num_samples", 6]

    # extend stds as diagonal
    # stds = stds.unsqueeze(-1).broadcast_to(*stds.shape, 3).diag_embed() # [..., "num_samples", 6, 3, 3]

    # two basis in parallel to the image plane
    rand_vec = torch.rand(
        list(directions.shape[:-2]) + [1, 3],
        device=directions.device,
        dtype=directions.dtype,
    )  # [..., 1, 3]
    ortho1 = torch.nn.functional.normalize(
        torch.cross(directions, rand_vec, dim=-1), dim=-1, eps=eps
    )  # [..., num_samples, 3]
    ortho2 = torch.nn.functional.normalize(
        torch.cross(directions, ortho1, dim=-1), dim=-1, eps=eps
    )  # [..., num_samples, 3]

    # just use directions to be the third vector of the orthonormal basis,
    # while the cross section of cone is parallel to the image plane
    basis_matrix = torch.stack([ortho1, ortho2, directions], dim=-1)
    means = torch.matmul(means, basis_matrix.transpose(-1, -2))  # [..., "num_samples", 6, 3]
    means = means + origins[..., None, :]

    return Gaussians(mean=means, cov=stds)


@torch_compile(dynamic=True, mode="reduce-overhead")
def expected_sin(x_means: Tensor, x_vars: Tensor) -> Tensor:
    """Computes the expected value of sin(y) where y ~ N(x_means, x_vars)

    Args:
        x_means: Mean values.
        x_vars: Variance of values.

    Returns:
        torch.Tensor: The expected value of sin.
    """

    return torch.exp(-0.5 * x_vars) * torch.sin(x_means)


@torch_compile(dynamic=True, mode="reduce-overhead")
def intersect_aabb(
    origins: Tensor,
    directions: Tensor,
    aabb: Tensor,
    max_bound: float = 1e10,
    invalid_value: float = 1e10,
) -> Tuple[Tensor, Tensor]:
    """
    Implementation of ray intersection with AABB box

    Args:
        origins: [N,3] tensor of 3d positions
        directions: [N,3] tensor of normalized directions
        aabb: [6] array of aabb box in the form of [x_min, y_min, z_min, x_max, y_max, z_max]
        max_bound: Maximum value of t_max
        invalid_value: Value to return in case of no intersection

    Returns:
        t_min, t_max - two tensors of shapes N representing distance of intersection from the origin.
    """

    tx_min = (aabb[:3] - origins) / directions
    tx_max = (aabb[3:] - origins) / directions

    t_min = torch.stack((tx_min, tx_max)).amin(dim=0)
    t_max = torch.stack((tx_min, tx_max)).amax(dim=0)

    t_min = t_min.amax(dim=-1)
    t_max = t_max.amin(dim=-1)

    t_min = torch.clamp(t_min, min=0, max=max_bound)
    t_max = torch.clamp(t_max, min=0, max=max_bound)

    cond = t_max <= t_min
    t_min = torch.where(cond, invalid_value, t_min)
    t_max = torch.where(cond, invalid_value, t_max)

    return t_min, t_max


def safe_normalize(
    vectors: Float[Tensor, "*batch_dim N"],
    eps: float = 1e-10,
) -> Float[Tensor, "*batch_dim N"]:
    """Normalizes vectors.

    Args:
        vectors: Vectors to normalize.
        eps: Epsilon value to avoid division by zero.

    Returns:
        Normalized vectors.
    """
    return vectors / (torch.norm(vectors, dim=-1, keepdim=True) + eps)


def masked_reduction(
    input_tensor: Float[Tensor, "1 32 mult"],
    mask: Bool[Tensor, "1 32 mult"],
    reduction_type: Literal["image", "batch"],
) -> Tensor:
    """
    Whether to consolidate the input_tensor across the batch or across the image
    Args:
        input_tensor: input tensor
        mask: mask tensor
        reduction_type: either "batch" or "image"
    Returns:
        input_tensor: reduced input_tensor
    """
    if reduction_type == "batch":
        # avoid division by 0 (if sum(M) = sum(sum(mask)) = 0: sum(image_loss) = 0)
        divisor = torch.sum(mask)
        if divisor == 0:
            return torch.tensor(0, device=input_tensor.device)
        input_tensor = torch.sum(input_tensor) / divisor
    elif reduction_type == "image":
        # avoid division by 0 (if M = sum(mask) = 0: image_loss = 0)
        valid = mask.nonzero()

        input_tensor[valid] = input_tensor[valid] / mask[valid]
        input_tensor = torch.mean(input_tensor)
    return input_tensor


def normalized_depth_scale_and_shift(
    prediction: Float[Tensor, "1 32 mult"],
    target: Float[Tensor, "1 32 mult"],
    mask: Bool[Tensor, "1 32 mult"],
):
    """
    More info here: https://arxiv.org/pdf/2206.00665.pdf supplementary section A2 Depth Consistency Loss
    This function computes scale/shift required to normalizes predicted depth map,
    to allow for using normalized depth maps as input from monocular depth estimation networks.
    These networks are trained such that they predict normalized depth maps.

    Solves for scale/shift using a least squares approach with a closed form solution:
    Based on:
    https://github.com/autonomousvision/monosdf/blob/d9619e948bf3d85c6adec1a643f679e2e8e84d4b/code/model/loss.py#L7
    Args:
        prediction: predicted depth map
        target: ground truth depth map
        mask: mask of valid pixels
    Returns:
        scale and shift for depth prediction
    """
    # system matrix: A = [[a_00, a_01], [a_10, a_11]]
    a_00 = torch.sum(mask * prediction * prediction, (1, 2))
    a_01 = torch.sum(mask * prediction, (1, 2))
    a_11 = torch.sum(mask, (1, 2))

    # right hand side: b = [b_0, b_1]
    b_0 = torch.sum(mask * prediction * target, (1, 2))
    b_1 = torch.sum(mask * target, (1, 2))

    # solution: x = A^-1 . b = [[a_11, -a_01], [-a_10, a_00]] / (a_00 * a_11 - a_01 * a_10) . b
    scale = torch.zeros_like(b_0)
    shift = torch.zeros_like(b_1)

    det = a_00 * a_11 - a_01 * a_01
    valid = det.nonzero()

    scale[valid] = (a_11[valid] * b_0[valid] - a_01[valid] * b_1[valid]) / det[valid]
    shift[valid] = (-a_01[valid] * b_0[valid] + a_00[valid] * b_1[valid]) / det[valid]

    return scale, shift


@torch_compile(dynamic=True)
def power_fn(
    x: Tensor,
    lam: float = -1.5,
    max_bound: float = 1e10,
) -> Tensor:
    """Power transformation function from Eq. 4 in ZipNeRF paper."""

    if lam == 1:
        return x
    if lam == 0:
        return torch.log1p(x)
    # infinity case
    if lam > max_bound:
        return torch.expm1(x)
    # -infinity case
    if lam < -max_bound:
        return -torch.expm1(-x)

    lam_1 = abs(lam - 1)
    return (lam_1 / lam) * ((x / lam_1 + 1) ** lam - 1)


@torch_compile(dynamic=True)
def inv_power_fn(
    x: Tensor,
    lam: float = -1.5,
    eps: float = 1e-10,
    max_bound: float = 1e10,
) -> Tensor:
    """Inverse power transformation function from Eq. 4 in ZipNeRF paper."""

    if lam == 1:
        return x
    if lam == 0:
        return torch.expm1(x)
    # infinity case
    if lam > max_bound:
        return torch.log1p(x)
    # -infinity case
    if lam < -max_bound:
        return -torch.log(1 - x)

    lam_1 = abs(lam - 1)
    return ((x * lam / lam_1 + 1).clamp_min(eps) ** (1 / lam) - 1) * lam_1


@torch_compile(dynamic=True, mode="reduce-overhead")
def erf_approx(x: Tensor) -> Tensor:
    """Error function approximation proposed in ZipNeRF paper (Eq. 11)."""
    return torch.sign(x) * torch.sqrt(1 - torch.exp(-4 / torch.pi * x**2))


def div_round_up(val: int, divisor: int) -> int:
    return (val + divisor - 1) // divisor


def grid_scale(level: int, log2_per_level_scale: float, base_resolution: int) -> float:
    # The -1 means that `base_resolution` refers to the number of grid _vertices_ rather
    # than the number of cells. This is slightly different from the notation in the paper,
    # but results in nice, power-of-2-scaled parameter grids that fit better into cache lines.
    return np.exp2(level * log2_per_level_scale) * base_resolution - 1.0


def grid_resolution(scale: float) -> int:
    return math.ceil(scale) + 1


def next_multiple(val: int, divisor: int) -> int:
    return div_round_up(val, divisor) * divisor


def generalized_binomial_coeff(a, k):
    """Compute generalized binomial coefficients."""
    return np.prod(a - np.arange(k)) / math.factorial(k)


def assoc_legendre_coeff(l, m, k):
    """Compute associated Legendre polynomial coefficients.
    Returns the coefficient of the cos^k(theta)*sin^m(theta) term in the
    (l, m)th associated Legendre polynomial, P_l^m(cos(theta)).
    """
    return ((-1) ** m * 2**l * math.factorial(l)
        / math.factorial(k) / math.factorial(l - k - m)
        * generalized_binomial_coeff(0.5 * (l + k + m - 1.0), l))


def sph_harm_coeff(l, m, k):
    """Compute spherical harmonic coefficients."""
    return math.sqrt((2.0 * l + 1.0) * math.factorial(l - m)
        / (4.0 * math.pi * math.factorial(l + m))) * assoc_legendre_coeff(l, m, k)


@torch_compile(dynamic=True, mode="reduce-overhead")
def reflect(viewdirs: Tensor, normals: Tensor) -> Tensor:
    """Reflect view directions about normals.

    The reflection of a vector v about a unit vector n is a vector u such that
    dot(v, n) = dot(u, n), and dot(u, u) = dot(v, v). The solution to these two
    equations is u = 2 dot(n, v) n - v.

    Args:
        viewdirs: [..., 3] array of view directions.
        normals: [..., 3] array of normal directions (assumed to be unit vectors).

    Returns:
        [..., 3] array of reflection directions.
    """
    return (
        2.0 * (normals * viewdirs).sum(dim=-1, keepdim=True) * normals - viewdirs
    )


@torch_compile(dynamic=True)
def linear_rgb_to_srgb(
    linear: Tensor,
) -> Tensor:
    """Convert a linear RGB image to sRGB. Used in colorspace conversions.
    See https://en.wikipedia.org/wiki/SRGB. Code taken from
    https://kornia.readthedocs.io/en/latest/_modules/kornia/color/rgb.html#linear_rgb_to_rgb."""
    threshold = 0.0031308
    rgb: torch.Tensor = torch.where(
        linear > threshold,
        1.055 * linear.clamp_min(threshold).pow(5 / 12) - 0.055,
        12.92 * linear,
    )

    return rgb


@torch_compile(dynamic=True)
def srgb_to_linear_rgb(
    rgb: Tensor,
) -> Tensor:
    """Convert an sRGB image to linear RGB. Used in colorspace conversions.
    See https://en.wikipedia.org/wiki/SRGB. Code taken from
    https://kornia.readthedocs.io/en/latest/_modules/kornia/color/rgb.html#rgb_to_linear_rgb."""
    threshold = 0.04045
    linear_rgb: torch.Tensor = torch.where(
        rgb > threshold,
        torch.pow(((rgb + 0.055) / 1.055), 2.4),
        rgb / 12.92,
    )

    return linear_rgb


def blur_stepfun(x: Tensor, y: Tensor, r: float) -> Tuple[Tensor, Tensor]:
    xr, xr_idx = torch.sort(torch.cat([x - r, x + r], dim=-1))
    y1 = (
        torch.cat([y, torch.zeros_like(y[..., :1])], dim=-1) - torch.cat([torch.zeros_like(y[..., :1]), y], dim=-1)
    ) / (2 * r)
    y2 = torch.cat([y1, -y1], dim=-1).take_along_dim(xr_idx[..., :-1], dim=-1)
    yr = torch.cumsum((xr[..., 1:] - xr[..., :-1]) * torch.cumsum(y2, dim=-1), dim=-1).clamp_min(0)
    yr = torch.cat([torch.zeros_like(yr[..., :1]), yr], dim=-1)
    return xr, yr


@torch_compile
def sorted_interp_quad(x: Tensor, xp: Tensor, fpdf: Tensor, fcdf: Tensor) -> Tensor:
    """interp in quadratic"""

    # Identify the location in `xp` that corresponds to each `x`.
    # The final `True` index in `mask` is the start of the matching interval.
    mask = x[..., None, :] >= xp[..., :, None]

    def find_interval(x: Tensor, return_idx=False):
        # Grab the value where `mask` switches from True to False, and vice versa.
        # This approach takes advantage of the fact that `x` is sorted.
        x0, x0_idx = torch.max(torch.where(mask, x[..., None], x[..., :1, None]), -2)
        x1, x1_idx = torch.min(torch.where(~mask, x[..., None], x[..., -1:, None]), -2)
        if return_idx:
            return x0, x1, x0_idx, x1_idx
        return x0, x1

    fcdf0, fcdf1, fcdf0_idx, fcdf1_idx = find_interval(fcdf, return_idx=True)
    fpdf0 = fpdf.take_along_dim(fcdf0_idx, dim=-1)
    fpdf1 = fpdf.take_along_dim(fcdf1_idx, dim=-1)
    xp0, xp1 = find_interval(xp)

    offset = torch.where(
        (xp1 - xp0) == 0,
        0.0,
        (x - xp0) / (xp1 - xp0),
    ).clamp_max_(1.0)

    ret = fcdf0 + (x - xp0) * (fpdf0 + fpdf1 * offset + fpdf0 * (1 - offset)) / 2
    return ret


@torch_compile(dynamic=True)
def leaky_clip(
    x: torch.Tensor,
    min: float = 0.0,
    max: float = 1.0,
) -> torch.Tensor:
    """
    Clip x to the range [0, 1] while still allowing gradients to 
    push it back inside the bounds. If x will be in range (0, 1)
    nothing happend.
    """
    with torch.no_grad():
        delta = x.clip(min=min, max=max).sub(x)
    return x + delta
