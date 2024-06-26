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
Collection of Losses.
"""

from enum import Enum
from typing import Literal, Optional, Tuple, List, Callable

import torch
from jaxtyping import Bool, Float, Bool
from torch import Tensor, nn

from nerfstudio.cameras.rays import RaySamples
from nerfstudio.utils.math import (
    masked_reduction,
    normalized_depth_scale_and_shift,
    sorted_interp_quad,
    blur_stepfun,
)
from nerfstudio.utils.misc import Numeric, torch_compile

L1Loss = nn.L1Loss
MSELoss = nn.MSELoss

EPS = 1.0e-7

# Sigma scale factor from Urban Radiance Fields (Rematas et al., 2022)
URF_SIGMA_SCALE_FACTOR = 3.0

# Depth ranking loss
FORCE_PSEUDODEPTH_LOSS = False


@torch_compile(dynamic=True, mode="reduce-overhead")
def outer(
    t0_starts: Float[Tensor, "*batch num_samples_0"],
    t0_ends: Float[Tensor, "*batch num_samples_0"],
    t1_starts: Float[Tensor, "*batch num_samples_1"],
    t1_ends: Float[Tensor, "*batch num_samples_1"],
    y1: Float[Tensor, "*batch num_samples_1"],
) -> Float[Tensor, "*batch num_samples_0"]:
    """Faster version of

    https://github.com/kakaobrain/NeRF-Factory/blob/f61bb8744a5cb4820a4d968fb3bfbed777550f4a/src/model/mipnerf360/helper.py#L117
    https://github.com/google-research/multinerf/blob/b02228160d3179300c7d499dca28cb9ca3677f32/internal/stepfun.py#L64

    Args:
        t0_starts: start of the interval edges
        t0_ends: end of the interval edges
        t1_starts: start of the interval edges
        t1_ends: end of the interval edges
        y1: weights
    """
    cy1 = torch.cat([y1.new_zeros(*y1.shape[:-1], 1), y1.cumsum(dim=-1)], dim=-1)
    max_idx = y1.shape[-1] - 1

    idx_lo = (
        torch.searchsorted(t1_starts.contiguous(), t0_starts.contiguous(), side="right")
        .sub_(1)
        .clamp_(min=0, max=max_idx)
    )

    idx_hi = (
        torch.searchsorted(t1_ends.contiguous(), t0_ends.contiguous(), side="right")
        .clamp_(min=0, max=max_idx)
    )

    cy1_lo = torch.take_along_dim(cy1[..., :-1], idx_lo, dim=-1)
    cy1_hi = torch.take_along_dim(cy1[..., 1:], idx_hi, dim=-1)
    y0_outer = cy1_hi - cy1_lo

    return y0_outer


@torch_compile(dynamic=True, mode="reduce-overhead")
def lossfun_outer(
    t: Float[Tensor, "*batch num_samples_1"],
    w: Float[Tensor, "*batch num_samples"],
    t_env: Float[Tensor, "*batch num_samples_1"],
    w_env: Float[Tensor, "*batch num_samples"],
):
    """
    https://github.com/kakaobrain/NeRF-Factory/blob/f61bb8744a5cb4820a4d968fb3bfbed777550f4a/src/model/mipnerf360/helper.py#L136
    https://github.com/google-research/multinerf/blob/b02228160d3179300c7d499dca28cb9ca3677f32/internal/stepfun.py#L80

    Args:
        t: interval edges
        w: weights
        t_env: interval edges of the upper bound enveloping histogram
        w_env: weights that should upper bound the inner (t,w) histogram
    """
    w_outer = outer(t[..., :-1], t[..., 1:], t_env[..., :-1], t_env[..., 1:], w_env)
    return torch.clip(w - w_outer, min=0) ** 2 / (w + EPS)


def interlevel_loss(weights_list, ray_samples_list) -> Tensor:
    """Calculates the proposal loss in the MipNeRF-360 paper.

    https://github.com/kakaobrain/NeRF-Factory/blob/f61bb8744a5cb4820a4d968fb3bfbed777550f4a/src/model/mipnerf360/model.py#L515
    https://github.com/google-research/multinerf/blob/b02228160d3179300c7d499dca28cb9ca3677f32/internal/train_utils.py#L133
    """
    c = ray_samples_list[-1].to_sdist().detach()
    w = weights_list[-1][..., 0].detach()
    assert len(ray_samples_list) > 0

    loss_interlevel = 0.0
    for ray_samples, weights in zip(ray_samples_list[:-1], weights_list[:-1]):
        sdist = ray_samples.to_sdist()
        assert sdist is not None
        cp = sdist  # (num_rays, num_samples + 1)
        wp = weights[..., 0]  # (num_rays, num_samples)
        loss_interlevel += torch.mean(lossfun_outer(c, w, cp, wp))

    assert isinstance(loss_interlevel, Tensor)
    return loss_interlevel


def zipnerf_loss(
    weights_list: List[Tensor],
    ray_samples_list: List[RaySamples],
    pulse_widths: Tuple[float, float] = (0.03, 0.003),
) -> Tensor:
    """Anti-aliased interlevel loss proposed in ZipNeRF paper."""
    # ground truth s and w (real nerf samples)
    # This implementation matches ZipNeRF up to the scale.
    # In the paper the loss is computed as the sum over the ray samples.
    # Here we take the mean and the multiplier for this loss should be changed accordingly.
    c = ray_samples_list[-1].to_sdist()
    assert c is not None
    c = c.detach()
    w = weights_list[-1][..., 0].detach()

    w_norm = w / (c[..., 1:] - c[..., :-1])
    loss = 0
    for i, (ray_samples, weights) in enumerate(zip(ray_samples_list[:-1], weights_list[:-1])):
        cp = ray_samples.to_sdist()
        wp = weights[..., 0]  # (num_rays, num_samples)
        c_, w_ = blur_stepfun(c, w_norm, pulse_widths[i])

        # piecewise linear pdf to piecewise quadratic cdf
        area = 0.5 * (w_[..., 1:] + w_[..., :-1]) * (c_[..., 1:] - c_[..., :-1])
        cdf = torch.cat([area.new_zeros(*area.shape[:-1], 1), area.cumsum(dim=-1)], dim=-1)

        # query piecewise quadratic interpolation
        cdf_interp = sorted_interp_quad(cp, c_, w_, cdf)

        # difference between adjacent interpolated values
        w_s = torch.diff(cdf_interp, dim=-1)
        loss += ((w_s.detach() - wp).clamp_min(0) ** 2 / (wp + 1e-5)).mean()

    assert isinstance(loss, Tensor)
    return loss


# Verified
@torch_compile(dynamic=True, mode="reduce-overhead")
def lossfun_distortion(
    t: Float[Tensor, "*batch num_samples+1"],
    w: Float[Tensor, "*batch num_samples"],
) -> Tensor:
    """
    https://github.com/kakaobrain/NeRF-Factory/blob/f61bb8744a5cb4820a4d968fb3bfbed777550f4a/src/model/mipnerf360/helper.py#L142
    https://github.com/google-research/multinerf/blob/b02228160d3179300c7d499dca28cb9ca3677f32/internal/stepfun.py#L266

    Original O(N^2) realization of distortion loss from MipNeRF 360.
    There are B rays each with N sampled points.
    t: Samples in ray of shape [B, N + 1]. Volume rendering weights of each point.
    w: Float tensor in shape [B, N]. Volume rendering weights of each point.
    """
    ut = (t[..., 1:] + t[..., :-1]) / 2  # [B, N] midpoint
    it = t[..., 1:] - t[..., :-1]  # [B, N] interval
    dut = torch.abs(ut[..., :, None] - ut[..., None, :])  # [B, N, N] mm
    loss_inter = torch.sum(w * torch.sum(w[..., None, :] * dut, dim=-1), dim=-1) # ww
    loss_intra = torch.sum(w**2 * it, dim=-1) / 3

    return loss_inter + loss_intra


# Verified
@torch_compile(dynamic=True, mode="reduce-overhead")
def eff_lossfun_distortion(
    t: Float[Tensor, "*batch num_samples+1"],
    w: Float[Tensor, "*batch num_samples"],
) -> Tensor:
    """
    https://github.com/sunset1995/torch_efficient_distloss/blob/03ea697ca261e839cae10e2ccc25cb350c04d226/torch_efficient_distloss/eff_distloss.py#L16

    Efficient O(N) and Faster realization of distortion loss from MipNeRF 360.
    There are B rays each with N sampled points.
    t: Samples in ray of shape [B, N + 1]. Volume rendering weights of each point.
    w: Float tensor in shape [B, N]. Volume rendering weights of each point.
    """
    ut = (t[..., 1:] + t[..., :-1]) / 2  # [B, N] midpoint
    it = t[..., 1:] - t[..., :-1]  # [B, N] interval

    loss_intra = torch.sum(w**2 * it, dim=-1) / 3

    wm = w * ut
    w_cumsum = w.cumsum(dim=-1)
    wm_cumsum = wm.cumsum(dim=-1)
    loss_inter_0 = wm[..., 1:] * w_cumsum[..., :-1]
    loss_inter_1 = w[..., 1:] * wm_cumsum[..., :-1]
    loss_inter = 2 * (loss_inter_0 - loss_inter_1).sum(dim=-1)

    return loss_inter + loss_intra


def distortion_loss(
    weights_list: List[Float[Tensor, "*batch num_samples"]],
    ray_samples_list: List[RaySamples],
) -> Tensor:
    """From mipnerf360"""
    loss = 0
    for weight, ray_samples in zip(weights_list, ray_samples_list):
        c = ray_samples.to_sdist()
        w = weight[..., 0]
        loss += eff_lossfun_distortion(c, w).mean()
    assert isinstance(loss, Tensor)
    return loss


def nerfstudio_distortion_loss(
    ray_samples: RaySamples,
    densities: Optional[Float[Tensor, "*bs num_samples 1"]] = None,
    weights: Optional[Float[Tensor, "*bs num_samples 1"]] = None,
) -> Float[Tensor, "*bs 1"]:
    """Ray based distortion loss proposed in MipNeRF-360. Returns distortion Loss.

    .. math::

        \\mathcal{L}(\\mathbf{s}, \\mathbf{w}) =\\iint\\limits_{-\\infty}^{\\,\\,\\,\\infty}
        \\mathbf{w}_\\mathbf{s}(u)\\mathbf{w}_\\mathbf{s}(v)|u - v|\\,d_{u}\\,d_{v}

    where :math:`\\mathbf{w}_\\mathbf{s}(u)=\\sum_i w_i \\mathbb{1}_{[\\mathbf{s}_i, \\mathbf{s}_{i+1})}(u)`
    is the weight at location :math:`u` between bin locations :math:`s_i` and :math:`s_{i+1}`.

    Args:
        ray_samples: Ray samples to compute loss over
        densities: Predicted sample densities
        weights: Predicted weights from densities and sample locations
    """
    if torch.is_tensor(densities):
        assert not torch.is_tensor(weights), "Cannot use both densities and weights"
        assert densities is not None
        # Compute the weight at each sample location
        weights = ray_samples.get_weights(densities)
    if torch.is_tensor(weights):
        assert not torch.is_tensor(densities), "Cannot use both densities and weights"
    assert weights is not None

    starts = ray_samples.spacing_starts
    ends = ray_samples.spacing_ends

    assert starts is not None and ends is not None, "Ray samples must have spacing starts and ends"
    midpoints = (starts + ends) / 2.0  # (..., num_samples, 1)

    loss = (
        weights * weights[..., None, :, 0] * torch.abs(midpoints - midpoints[..., None, :, 0])
    )  # (..., num_samples, num_samples)
    loss = torch.sum(loss, dim=(-1, -2))[..., None]  # (..., num_samples)
    loss = loss + 1 / 3.0 * torch.sum(weights**2 * (ends - starts), dim=-2)

    return loss


def orientation_loss(
    weights: Float[Tensor, "*bs num_samples 1"],
    normals: Float[Tensor, "*bs num_samples 3"],
    viewdirs: Float[Tensor, "*bs 3"],
):
    """Orientation loss proposed in Ref-NeRF.
    Loss that encourages that all visible normals are facing towards the camera.
    """
    w = weights
    n = normals
    # Negate viewdirs to represent normalized vectors from point to camera.
    v = -viewdirs
    n_dot_v = (n * v[..., None, :]).sum(dim=-1)
    return (w[..., 0] * torch.fmin(torch.zeros_like(n_dot_v), n_dot_v) ** 2).sum(dim=-1)


def pred_normal_loss(
    weights: Float[Tensor, "*bs num_samples 1"],
    normals: Float[Tensor, "*bs num_samples 3"],
    pred_normals: Float[Tensor, "*bs num_samples 3"],
):
    """Loss between normals calculated from density and normals from prediction network."""
    return (weights[..., 0] * (1.0 - torch.sum(normals * pred_normals, dim=-1))).sum(dim=-1)


class iMAPLoss(nn.Module):
    """iMAP implementation of depth loss.
    https://arxiv.org/abs/2103.12352
    
    Args:
        is_euclidean: Whether ground truth depths corresponds to normalized direction vectors.
    """
    def __init__(
        self,
        is_euclidean: bool = False,
    ) -> None:
        super().__init__()
        self.is_euclidean = is_euclidean

    def forward(
        self,
        termination_depth: Float[Tensor, "*batch 1"],
        predicted_depth: Float[Tensor, "*batch 1"],
        directions_norm: Float[Tensor, "*batch 1"],
        *args,
        **kwargs,
    ) -> Float[Tensor, "0"]:
        """Args:
            weights: Weights predicted for each sample.
            ray_samples: Samples along rays corresponding to weights.
            termination_depth: Ground truth depth of rays.
            predicted_depth: Depth prediction from the network.
            sigma: Uncertainty around depth value.
            directions_norm: Norms of ray direction vectors in the camera frame.
        """
        if self.is_euclidean:
            termination_depth = termination_depth * directions_norm

        depth_mask = termination_depth > 0

        # Expected depth loss
        expected_depth_loss = (termination_depth - predicted_depth) ** 2

        return (expected_depth_loss * depth_mask).mean()

    
class DSNeRFLoss(iMAPLoss):
    """Depth loss from Depth-supervised NeRF (Deng et al., 2022)
    https://arxiv.org/abs/2107.02791
    """

    def forward(
        self,
        weights: Float[Tensor, "*batch num_samples 1"],
        ray_samples: RaySamples,
        termination_depth: Float[Tensor, "*batch 1"],
        directions_norm: Float[Tensor, "*batch 1"],
        sigma: Float[Tensor, "0"],
        *args,
        **kwargs,
    ) -> Float[Tensor, "0"]:
        """Args:
            weights: Weights predicted for each sample.
            ray_samples: Samples along rays corresponding to weights.
            termination_depth: Ground truth depth of rays.
            directions_norm: Norms of ray direction vectors in the camera frame.
            sigma: Uncertainty around depth values.
        """
        if self.is_euclidean:
            termination_depth = termination_depth * directions_norm

        steps = ray_samples.frustums.get_steps()
        lengths = ray_samples.frustums.get_length()

        depth_mask = termination_depth > 0
        
        # Line of sight loss
        termination_depth = termination_depth[:, None]
        loss = -(weights + EPS).log() * (-((steps - termination_depth) ** 2) / (2 * sigma)).exp() * lengths
        loss = loss.sum(-2) * depth_mask

        return loss.mean()
        

class URFLoss(iMAPLoss):
    """Lidar losses from Urban Radiance Fields (Rematas et al., 2022)
    with `line_of_sight_near_distribution` as normal distribution.
    https://urban-radiance-fields.github.io/images/go_urf.pdf
    """

    @staticmethod
    def line_of_sight_near_distribution(
        steps: Float[Tensor, "*batch num_samples 1"],
        sigma: Float[Tensor, "0"],
        termination_depth: Float[Tensor, "*batch 1 1"],
    ) -> Float[Tensor, "*batch 1 1"]:
        """Generates target distribution in line of sight near zone.
        Args:
            steps: Middle points in ray samples
            termination_depth: Ground truth depth of rays.
            sigma: Uncertainty around depth value.
        """
        
        target_distribution = torch.distributions.normal.Normal(
            loc=0.0, scale=sigma / URF_SIGMA_SCALE_FACTOR,
        ).log_prob(steps - termination_depth).exp_()

        return target_distribution
    
    def compute_line_of_sight_near_loss(
        self,
        steps: Float[Tensor, "*batch num_samples 1"],
        sigma: Float[Tensor, "0"],
        termination_depth: Float[Tensor, "*batch 1 1"],
        line_of_sight_loss_near_mask: Bool[Tensor, "*batch num_samples 1"],
        weights: Float[Tensor, "*batch num_samples 1"],
    ) -> Float[Tensor, "*batch  1"]:
        """Computes line of sight loss in near zone.
        Args:
            steps: Middle points in ray samples
            termination_depth: Ground truth depth of rays.
            sigma: Uncertainty around depth value.
            line_of_sight_loss_near_mask: Mask indicates where target dist located
            weights: Weights predicted by neural network
        """

        target_distribution = self.line_of_sight_near_distribution(
            steps=steps, sigma=sigma,
            termination_depth=termination_depth,
        )  # [bs num_samples 1]

        # correction for discrete variant
        div_value = (target_distribution * line_of_sight_loss_near_mask).sum(-2, keepdim=True)
        target_distribution = torch.where(
            line_of_sight_loss_near_mask,
            target_distribution / div_value,
            0.,
        )

        # Near step, see eq. 16
        line_of_sight_loss_near = (weights - target_distribution) ** 2
        line_of_sight_loss_near = (line_of_sight_loss_near_mask * line_of_sight_loss_near).sum(-2)

        return line_of_sight_loss_near
        
    def forward(
        self,
        weights: Float[Tensor, "*batch num_samples 1"],
        ray_samples: RaySamples,
        termination_depth: Float[Tensor, "*batch 1"],
        predicted_depth: Float[Tensor, "*batch 1"],
        sigma: Float[Tensor, "0"],
        directions_norm: Float[Tensor, "*batch 1"],
        density: Float[Tensor, "*batch num_samples 1"],
    ) -> Float[Tensor, "0"]:
        """Args:
            weights: Weights predicted for each sample.
            ray_samples: Samples along rays corresponding to weights.
            termination_depth: Ground truth depth of rays.
            predicted_depth: Depth prediction from the network.
            sigma: Uncertainty around depth value.
            directions_norm: Norms of ray direction vectors in the camera frame.
        """

        if self.is_euclidean:
            termination_depth = termination_depth * directions_norm

        steps = ray_samples.frustums.get_steps()

        depth_mask = termination_depth > 0

        # Expected depth loss
        expected_depth_loss = (termination_depth - predicted_depth) ** 2

        # Line of sight losses
        termination_depth = termination_depth[:, None]

        line_of_sight_loss_near_mask = torch.logical_and(
            steps <= termination_depth + sigma,
            steps >= termination_depth - sigma,
        )  # [batch num_samples 1]

        # # Near step, see eq. 16
        line_of_sight_loss_near = self.compute_line_of_sight_near_loss(
            steps=steps,
            sigma=sigma,
            termination_depth=termination_depth,
            line_of_sight_loss_near_mask=line_of_sight_loss_near_mask,
            weights=weights,
        )  # [batch 1]

        # Empty & dist step, see eq. 17 & 18
        line_of_sight_loss_empty_dist_mask = torch.logical_not(line_of_sight_loss_near_mask)
        line_of_sight_loss_empty_dist = (line_of_sight_loss_empty_dist_mask * weights ** 2).sum(-2)

        line_of_sight_loss = line_of_sight_loss_empty_dist + line_of_sight_loss_near

        loss = (expected_depth_loss + line_of_sight_loss) * depth_mask
        return loss.mean()
    

class PowURFLoss(URFLoss):
    """Lidar losses from Urban Radiance Fields (Rematas et al., 2022)
    with modification `line_of_sight_near_distribution` as pascal distribution.
    https://urban-radiance-fields.github.io/images/go_urf.pdf

    Args:
        is_euclidean: Whether ground truth depths corresponds to normalized direction vectors.
        freqs: Min / Max values in frequency
        mask_scale: Min / Max values in scales
        regularize_fn: Regularize function
    """

    def __init__(
        self,
        is_euclidean: bool,
        freqs: Tuple[Numeric, Numeric] = (2, 10),
        mask_scale: Tuple[Numeric, Numeric] = (2, 10),
        regularize_fn: Callable[[Tensor], Tensor] = torch.square,
    ) -> None:
        super().__init__(is_euclidean=is_euclidean)

        # TODO
        self.freqs = freqs
        self.mask_scale = mask_scale
        self.regularize_fn = regularize_fn

        self.current_freq = freqs[0]
        self.current_mask_scale = mask_scale[0]
    
    def forward(
        self,
        weights: Float[Tensor, "*batch num_samples 1"],
        ray_samples: RaySamples,
        termination_depth: Float[Tensor, "*batch 1"],
        predicted_depth: Float[Tensor, "*batch 1"],
        sigma: Float[Tensor, "0"],
        directions_norm: Float[Tensor, "*batch 1"],
        density: Float[Tensor, "*batch num_samples 1"],
    ) -> Float[Tensor, "0"]:
        """Args:
            weights: Weights predicted for each sample.
            ray_samples: Samples along rays corresponding to weights.
            termination_depth: Ground truth depth of rays.
            predicted_depth: Depth prediction from the network.
            sigma: Uncertainty around depth value.
            directions_norm: Norms of ray direction vectors in the camera frame.
        """

        if self.is_euclidean:
            termination_depth = termination_depth * directions_norm

        steps = ray_samples.frustums.get_steps()

        depth_mask = termination_depth > 0

        # Expected depth loss
        expected_depth_loss = (termination_depth - predicted_depth) ** 2

        # Line of sight losses
        termination_depth = termination_depth[:, None]

        focal_mask = self.compute_focal_mask(steps=steps, termination_depth=termination_depth)

        # Empty step
        line_of_sight_loss_empty_mask = steps <= termination_depth - sigma
        line_of_sight_loss_empty = (line_of_sight_loss_empty_mask * focal_mask * self.regularize_fn(density)).sum(-2)

        # Dist step
        line_of_sight_loss_dist_mask = steps >= termination_depth + sigma
        transmittance = ray_samples.get_transmittance(density, transmittance_only=True)
        line_of_sight_loss_dist = (line_of_sight_loss_dist_mask * focal_mask * self.regularize_fn(transmittance)).sum(-2)

        line_of_sight_loss_near_mask = torch.logical_and(
            steps < termination_depth + sigma,
            steps > termination_depth - sigma,
        )  # [batch num_samples 1]

        # Near step, see eq. 16
        line_of_sight_loss_near = self.compute_line_of_sight_near_loss(
            steps=steps,
            sigma=sigma,
            termination_depth=termination_depth,
            line_of_sight_loss_near_mask=line_of_sight_loss_near_mask,
            weights=weights,
        )  # [batch 1]

        line_of_sight_loss = line_of_sight_loss_empty + line_of_sight_loss_dist + line_of_sight_loss_near

        loss = (expected_depth_loss + line_of_sight_loss) * depth_mask
        return loss.mean()
    
    def compute_focal_mask(
        self,
        steps: Float[Tensor, "*batch num_samples 1"],
        termination_depth: Float[Tensor, "*batch 1 1"],
    ) -> Float[Tensor, "*batch num_samples 1"]:
        
        focal_mask = self.current_mask_scale * self.regularize_fn(steps - termination_depth)

        return focal_mask


class DepthLossType(Enum):
    """Types of depth losses for depth supervision."""

    DS_NERF = DSNeRFLoss
    URF = URFLoss
    iMAP = iMAPLoss
    POWURF = PowURFLoss


def decay_loss(
    decay_list: List[Float[Tensor, "*"]],
    regularize_fn: Callable[[Tensor], Tensor] = lambda x: x,
) -> Float[Tensor, "0"]:
    """Implementation of decay loss."""
    loss: Tensor = sum(
        regularize_fn(decay).mean()
        for decay in decay_list
    )  # type: ignore

    return loss


def monosdf_normal_loss(
    normal_pred: Float[Tensor, "num_samples 3"], normal_gt: Float[Tensor, "num_samples 3"]
) -> Float[Tensor, "0"]:
    """
    Normal consistency loss proposed in monosdf - https://niujinshuchong.github.io/monosdf/
    Enforces consistency between the volume rendered normal and the predicted monocular normal.
    With both angluar and L1 loss. Eq 14 https://arxiv.org/pdf/2206.00665.pdf
    Args:
        normal_pred: volume rendered normal
        normal_gt: monocular normal
    """
    normal_gt = torch.nn.functional.normalize(normal_gt, p=2, dim=-1)
    normal_pred = torch.nn.functional.normalize(normal_pred, p=2, dim=-1)
    l1 = (normal_pred - normal_gt).abs().sum(dim=-1).mean()
    cos = (1.0 - torch.sum(normal_pred * normal_gt, dim=-1)).mean()
    return l1 + cos


class MiDaSMSELoss(nn.Module):
    """
    data term from MiDaS paper
    """

    def __init__(self, reduction_type: Literal["image", "batch"] = "batch"):
        super().__init__()

        self.reduction_type: Literal["image", "batch"] = reduction_type
        # reduction here is different from the image/batch-based reduction. This is either "mean" or "sum"
        self.mse_loss = MSELoss(reduction="none")

    def forward(
        self,
        prediction: Float[Tensor, "1 32 mult"],
        target: Float[Tensor, "1 32 mult"],
        mask: Bool[Tensor, "1 32 mult"],
    ) -> Float[Tensor, "0"]:
        """
        Args:
            prediction: predicted depth map
            target: ground truth depth map
            mask: mask of valid pixels
        Returns:
            mse loss based on reduction function
        """
        summed_mask = torch.sum(mask, (1, 2))
        image_loss = torch.sum(self.mse_loss(prediction, target) * mask, (1, 2))
        # multiply by 2 magic number?
        image_loss = masked_reduction(image_loss, 2 * summed_mask, self.reduction_type)

        return image_loss


# losses based on https://github.com/autonomousvision/monosdf/blob/main/code/model/loss.py
class GradientLoss(nn.Module):
    """
    multiscale, scale-invariant gradient matching term to the disparity space.
    This term biases discontinuities to be sharp and to coincide with discontinuities in the ground truth
    More info here https://arxiv.org/pdf/1907.01341.pdf Equation 11
    """

    def __init__(self, scales: int = 4, reduction_type: Literal["image", "batch"] = "batch"):
        """
        Args:
            scales: number of scales to use
            reduction_type: either "batch" or "image"
        """
        super().__init__()
        self.reduction_type: Literal["image", "batch"] = reduction_type
        self.__scales = scales

    def forward(
        self,
        prediction: Float[Tensor, "1 32 mult"],
        target: Float[Tensor, "1 32 mult"],
        mask: Bool[Tensor, "1 32 mult"],
    ) -> Float[Tensor, "0"]:
        """
        Args:
            prediction: predicted depth map
            target: ground truth depth map
            mask: mask of valid pixels
        Returns:
            gradient loss based on reduction function
        """
        assert self.__scales >= 1
        total = 0.0

        for scale in range(self.__scales):
            step = pow(2, scale)

            grad_loss = self.gradient_loss(
                prediction[:, ::step, ::step],
                target[:, ::step, ::step],
                mask[:, ::step, ::step],
            )
            total += grad_loss

        assert isinstance(total, Tensor)
        return total

    def gradient_loss(
        self,
        prediction: Float[Tensor, "1 32 mult"],
        target: Float[Tensor, "1 32 mult"],
        mask: Bool[Tensor, "1 32 mult"],
    ) -> Float[Tensor, "0"]:
        """
        multiscale, scale-invariant gradient matching term to the disparity space.
        This term biases discontinuities to be sharp and to coincide with discontinuities in the ground truth
        More info here https://arxiv.org/pdf/1907.01341.pdf Equation 11
        Args:
            prediction: predicted depth map
            target: ground truth depth map
            reduction: reduction function, either reduction_batch_based or reduction_image_based
        Returns:
            gradient loss based on reduction function
        """
        summed_mask = torch.sum(mask, (1, 2))
        diff = prediction - target
        diff = torch.mul(mask, diff)

        grad_x = torch.abs(diff[:, :, 1:] - diff[:, :, :-1])
        mask_x = torch.mul(mask[:, :, 1:], mask[:, :, :-1])
        grad_x = torch.mul(mask_x, grad_x)

        grad_y = torch.abs(diff[:, 1:, :] - diff[:, :-1, :])
        mask_y = torch.mul(mask[:, 1:, :], mask[:, :-1, :])
        grad_y = torch.mul(mask_y, grad_y)

        image_loss = torch.sum(grad_x, (1, 2)) + torch.sum(grad_y, (1, 2))
        image_loss = masked_reduction(image_loss, summed_mask, self.reduction_type)

        return image_loss


class ScaleAndShiftInvariantLoss(nn.Module):
    """
    Scale and shift invariant loss as described in
    "Towards Robust Monocular Depth Estimation: Mixing Datasets for Zero-shot Cross-dataset Transfer"
    https://arxiv.org/pdf/1907.01341.pdf
    """

    def __init__(self, alpha: float = 0.5, scales: int = 4, reduction_type: Literal["image", "batch"] = "batch"):
        """
        Args:
            alpha: weight of the regularization term
            scales: number of scales to use
            reduction_type: either "batch" or "image"
        """
        super().__init__()
        self.__data_loss = MiDaSMSELoss(reduction_type=reduction_type)
        self.__regularization_loss = GradientLoss(scales=scales, reduction_type=reduction_type)
        self.__alpha = alpha

        self.__prediction_ssi = None

    def forward(
        self,
        prediction: Float[Tensor, "1 32 mult"],
        target: Float[Tensor, "1 32 mult"],
        mask: Bool[Tensor, "1 32 mult"],
    ) -> Float[Tensor, "0"]:
        """
        Args:
            prediction: predicted depth map (unnormalized)
            target: ground truth depth map (normalized)
            mask: mask of valid pixels
        Returns:
            scale and shift invariant loss
        """
        scale, shift = normalized_depth_scale_and_shift(prediction, target, mask)
        self.__prediction_ssi = scale.view(-1, 1, 1) * prediction + shift.view(-1, 1, 1)

        total = self.__data_loss(self.__prediction_ssi, target, mask)
        if self.__alpha > 0:
            total += self.__alpha * self.__regularization_loss(self.__prediction_ssi, target, mask)

        return total

    def __get_prediction_ssi(self):
        """
        scale and shift invariant prediction
        from https://arxiv.org/pdf/1907.01341.pdf equation 1
        """
        return self.__prediction_ssi

    prediction_ssi = property(__get_prediction_ssi)


def tv_loss(grids: Float[Tensor, "grids feature_dim row column"]) -> Float[Tensor, ""]:
    """
    https://github.com/apchenstu/TensoRF/blob/4ec894dc1341a2201fe13ae428631b58458f105d/utils.py#L139

    Args:
        grids: stacks of explicit feature grids (stacked at dim 0)
    Returns:
        average total variation loss for neighbor rows and columns.
    """
    number_of_grids = grids.shape[0]
    h_tv_count = grids[:, :, 1:, :].shape[1] * grids[:, :, 1:, :].shape[2] * grids[:, :, 1:, :].shape[3]
    w_tv_count = grids[:, :, :, 1:].shape[1] * grids[:, :, :, 1:].shape[2] * grids[:, :, :, 1:].shape[3]
    h_tv = torch.pow((grids[:, :, 1:, :] - grids[:, :, :-1, :]), 2).sum()
    w_tv = torch.pow((grids[:, :, :, 1:] - grids[:, :, :, :-1]), 2).sum()
    return 2 * (h_tv / h_tv_count + w_tv / w_tv_count) / number_of_grids


def depth_ranking_loss(
    rendered_depth: Tensor,
    gt_depth: Tensor,
    m: float = 1e-4,
) -> Tensor:
    """
    Depth ranking loss as described in the SparseNeRF paper
    Assumes that the layout of the batch comes from a PairPixelSampler,
    so that adjacent samples in the gt_depth and rendered_depth are from
    pixels with a radius of each other
    """
    dpt_diff = gt_depth[::2, :] - gt_depth[1::2, :]
    out_diff = rendered_depth[::2, :] - rendered_depth[1::2, :] + m
    differing_signs = torch.sign(dpt_diff) != torch.sign(out_diff)
    
    if differing_signs.any():
        return torch.mean((out_diff[differing_signs] * torch.sign(out_diff[differing_signs])))
    
    return rendered_depth.new_zeros(())


class CharbonnierLoss(nn.Module):
    """
    Charbonnier Loss from MipNeRF 360
    https://arxiv.org/pdf/2111.12077.pdf
    """

    __constants__ = ["padding_square"]
    padding_square: float

    def __init__(self, padding: float = 1e-3):
        super().__init__()
        self.padding_square = padding ** 2

    def forward(
        self,
        prediction: Float[Tensor, "*bs 3"],
        target: Float[Tensor, "*bs 3"],
    ) -> Float[Tensor, "1"]:
        """
        Args:
            prediction: predicted rgb values
            target: ground truth rgb values
        Returns:
            loss
        """
        square_loss = (prediction - target) ** 2
        loss = torch.sqrt(square_loss + self.padding_square)

        return torch.mean(loss)


class RawNeRFLoss(nn.Module):
    """
    RawNeRF Loss from RawNeRF paper
    https://arxiv.org/pdf/2111.13679.pdf
    """

    __constants__ = ["padding_square"]
    padding_square: float

    def __init__(self, padding: float = 1e-3):
        super().__init__()
        self.padding_square = padding ** 2

    def forward(
        self,
        prediction: Float[Tensor, "*bs 3"],
        target: Float[Tensor, "*bs 3"],
    ) -> Float[Tensor, "1"]:
        """
        Args:
            prediction: predicted rgb values
            target: ground truth rgb values
        Returns:
            loss
        """
        rgb_render_clip = prediction.clamp_max(1)
        resid_sq_clip = (rgb_render_clip - target) ** 2
        # Scale by gradient of log tonemapping curve.
        scaling_grad = 1.0 / (self.padding_square + rgb_render_clip.detach())
        # Reweighted L2 loss.
        data_loss = resid_sq_clip * scaling_grad**2

        return torch.mean(data_loss)
