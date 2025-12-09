"""Plotting methods for model parameters."""

import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
from torch import Tensor

from ..core.initialization import RunContext
from .common.helpers import (
    architecture_dict,
    color_dict,
    create_centered_array,
    notation_dict,
)
from .common.labels import generate_output_function_labels


def plot_expectation_decay_modes(decay_real: Tensor, decay_imag: Tensor, stationary_mean: Tensor, run: RunContext) -> None:
    """Plot the decay modes of the generator and the stationary mean."""
    theta = "_{\\theta}" if run.PLT_CONFIG["use_theta_in_labels"] else ""
    theta_comma = "\\theta," if run.PLT_CONFIG["use_theta_in_labels"] else ""

    if stationary_mean is not None:
        fig = plt.figure(figsize=(18, 4))
        gs = plt.GridSpec(1, 3)
        stationary_mean = stationary_mean.detach().numpy()
    else:
        fig = plt.figure(figsize=(12, 4))
        gs = plt.GridSpec(1, 2, width_ratios=[2, 2])

    if "spectral" in run.NN_CONFIG["v"]["subnet_architecture"]:  # for spectral decomposition
        decay_real = nn.functional.softplus(decay_real).detach().numpy()
        decay_imag = decay_imag.detach().numpy()
    else:  # for temporal features
        decay_real = decay_real.detach().numpy()
        decay_imag = decay_imag.detach().numpy()

    architecture = architecture_dict[run.NN_CONFIG["v"]["subnet_architecture"]]
    symbol = notation_dict[run.NN_CONFIG["v"]["subnet_architecture"]]

    # Squeeze the decay modes to remove the batch dimension:
    decay_real = np.squeeze(decay_real, axis=0)
    decay_imag = np.squeeze(decay_imag, axis=0)
    # Compute the magnitude of the decay modes:
    magnitude = np.sqrt(decay_real**2 + decay_imag**2)
    sorted_indices = np.argsort(-magnitude)

    # Add the magnitude of the decay modes, sorted by magnitude:
    ax1 = fig.add_subplot(gs[0])
    markerline, stemlines, _ = ax1.stem(magnitude[sorted_indices], basefmt=" ", label="NN (expectation)")
    plt.setp(markerline, "color", color_dict["decay_expectation_dark"])  # set color of markers
    plt.setp(stemlines, "color", color_dict["decay_expectation_light"])  # set color of stems
    plt.setp(stemlines, "alpha", 0.5)  # set transparency of stems
    ax1.set_title(f"Modulus of the decay modes $\\sigma_{{{theta_comma}{symbol}}}$ of the generator")
    ax1.set_xlabel(f"decay mode index ${symbol}$")
    ax1.set_ylabel(f"$|\\sigma_{{{theta_comma}{symbol}}}|$")  # magnitude
    ax1.set_xticks(range(len(magnitude)))
    ax1.set_xticklabels([f"({i})" for i in range(1, len(magnitude) + 1)])
    ax1.set_ylim(bottom=0)

    # Add the coordinate system with dashed lines:
    ax2 = fig.add_subplot(gs[1])
    ax2.axhline(color="lightgray", linestyle="dashed")
    ax2.axvline(color="lightgray", linestyle="dashed")
    # Add real and imaginary parts of the decay modes:
    ax2.scatter(decay_real, decay_imag, color=color_dict["decay_expectation_dark"], label="NN (expectation)", edgecolors="none")
    ax2.scatter(decay_real, -decay_imag, color=color_dict["decay_expectation_dark"], label="NN (expectation)", edgecolors="none", alpha=0.5)  # add the conjugate decay modes
    # Add annotations sorted by magnitude:
    for rank, idx in enumerate(sorted_indices):
        ax2.annotate(f"({rank+1})", (decay_real[idx], decay_imag[idx]), xytext=(5, 0), textcoords="offset points", va="center", fontsize=10)
        if not architecture == "spectral_complex":
            ax2.annotate(f"({rank+1})", (decay_real[idx], -decay_imag[idx]), xytext=(5, 0), textcoords="offset points", va="center", fontsize=10)
    ax2.set_title(f"Decay modes $\\sigma_{{{theta_comma}{symbol}}}$ of the generator")
    ax2.set_xlabel(f"$a_{{{theta_comma}{symbol}}}$")  # real part
    ax2.set_ylabel(f"$b_{{{theta_comma}{symbol}}}$")  # imaginary part
    x_min, x_max = ax2.get_xlim()
    ax2.set_xlim(x_min, x_max + (x_max - x_min) * 0.05)

    if stationary_mean is not None:
        stationary_mean = np.squeeze(stationary_mean, axis=0)  # remove the batch dimension
        output_label = generate_output_function_labels(stationary_mean.shape[0], use_theta=run.PLT_CONFIG["use_theta_in_labels"])
        # Get the number of bars and the width of each bar:
        n_bars = stationary_mean.shape[0]
        width_plus_padding = 0.9 / n_bars
        width = width_plus_padding * 0.9
        bars = [k * (width_plus_padding) for k in create_centered_array(n_bars)]
        # Add all bars (one per output function) for the stationary mean:
        ax3 = fig.add_subplot(gs[2])
        ax3.bar(bars, stationary_mean, width, color=color_dict["decay_expectation_dark"], alpha=0.5)
        ax3.set_title("Stationary mean")
        ax3.set_ylabel(f"$V{theta}(f)$")  # stationary mean
        ax3.set_xticks(bars, output_label)

    fig.tight_layout()
    fig.savefig(f"{run.results_subdir}/{run.timestamp}___B_ExpectationDecayModes.pdf")
    plt.close(fig)
