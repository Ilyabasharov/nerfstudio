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

""" Matplotlib Helper Functions """

from typing import Optional, List

import torch
from torch import Tensor
from jaxtyping import Float
import matplotlib.pyplot as plt
from matplotlib.figure import Figure


@torch.no_grad()
def plot_weights_distribution(
    weights: Float[Tensor, "num_samples 1"],
    steps: Float[Tensor, "num_samples 1"],
    termination_depth: Optional[Float[Tensor, "1"]] = None,
    fig: Optional[Figure] = None,
    clear: bool = False,
    prop_label: Optional[str] = None,
) -> Figure:
    """Plots weights and optionally ground truth depth on one figure.
    Args:
        weights: Predicted weights distribution
        steps: Sampled steps
        termination_depth: Ground truth depth
        fig: Current figure to use
        clear: Whether to clear current figure
        prop_label: Name of the weight proposal layer from hierarchical sampling
    """

    if prop_label is None:
        prop_label = 'weights'

    prop_index: int = 2 if not prop_label[-1].isdigit() else int(prop_label[-1])

    is_new_figure = True
    if fig is None:
        fig = plt.figure(figsize=(7, 7))
        ax = fig.add_subplot(111)
    else:
        if clear:
            fig.clear()
            ax = fig.add_subplot(111)
        else:
            is_new_figure = False
            ax = fig.get_axes()[0]

    ax.plot(
        steps.numpy(),
        weights.numpy(),
        f"C{prop_index}",
        label=f"{prop_label} distrubution"
    )

    if termination_depth is not None:
        ax.vlines(
            x=termination_depth.numpy(),
            ymin=0, ymax=1,
            label="target_depth",
            colors="r",
        )

    if is_new_figure:
        ax.set_title("Weights distrubution", color="C0")
        ax.set_xlabel("Steps")
        ax.set_xscale("log")
        ax.set_ylabel("Values")
        ax.legend(loc="best")
        ax.grid(visible=True, alpha=0.75)

    return fig


def plot_weights_distribution_multiprop(
    weights: List[Float[Tensor, "num_samples 1"]],
    steps: List[Float[Tensor, "num_samples 1"]],
    i: int,
    termination_depth: Optional[Float[Tensor, "1"]] = None,

) -> Figure:
    """Plots weights and optionally ground truth depth on one figure.
    Args:
        weights: Predicted weights distribution
        steps: Sampled steps
        i: Index to visualize
        termination_depth: Ground truth depth
    """

     # first, show last field predicted weights
    j = len(weights) - 1
    fig_main = plot_weights_distribution(
        weights=weights[j][i],
        steps=steps[j][i],
        prop_label="weights",
        termination_depth=termination_depth,
    )

    # second, superimpose the weights of the previous layers on the main plot
    for k in range(j):
        _ = plot_weights_distribution(
            weights=weights[k][i],
            steps=steps[k][i],
            prop_label=f"prop_depth_{k}",
            fig=fig_main,
            clear=False,
        )

    # update
    ax = fig_main.get_axes()[0]
    ax.legend(loc="best")
    ax.set_title(f"Weights dist for ray {i}", color="C0")

    return fig_main
