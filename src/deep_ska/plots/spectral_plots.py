"""Plots related to the spectral decomposition."""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

from ..core.initialization import RunContext
from .common.helpers import (
    architecture_dict,
    color_dict,
    notation_dict,
)
from .common.line_wrappers import temp_plot


def plot_temporal_eigenfunctions(state: np.ndarray, inputs: dict, run: RunContext, index: int, results: dict) -> None:
    """This plot is used to compare the NN output to the NN output combined with SSA simulations."""
    theta = "_{{\\theta}}" if run.PLT_CONFIG["use_theta_in_labels"] else ""
    theta_comma = "\\theta," if run.PLT_CONFIG["use_theta_in_labels"] else ""

    # Real part of NN output:
    lhs_c = results.get("exp_eigenfunction_lhs_c")
    rhs_c = results.get("exp_eigenfunction_rhs_c")
    lhs_c_ci = results.get("exp_eigenfunction_lhs_c_ci")
    # Imaginary part of NN output:
    lhs_d = results.get("exp_eigenfunction_lhs_d")
    rhs_d = results.get("exp_eigenfunction_rhs_d")
    lhs_d_ci = results.get("exp_eigenfunction_lhs_d_ci")

    n_rows = run.NN_CONFIG["v"]["n_spectral_terms"]
    n_cols = 2

    architecture = architecture_dict[run.NN_CONFIG["v"]["subnet_architecture"]]
    symbol = notation_dict[run.NN_CONFIG["v"]["subnet_architecture"]]

    fig = plt.figure(figsize=(6*n_cols, 4*n_rows))
    gs = plt.GridSpec(n_rows, n_cols)

    # Find validation t_stop and training t_stop, as well as the index of the training t_stop:
    t_stop_train = run.RN_CONFIG["t_stop"]
    t_stop_valid = run.VAL_CONFIG["t_stop"]
    train_idx = np.searchsorted(inputs["times"], t_stop_train, side="right")

    for i in range(n_rows):
        ax1_handles, ax1_labels = [], []  # store custom legend handles and labels for ax1
        ax2_handles, ax2_labels = [], []  # store custom legend handles and labels for ax2

        # Add subplot for real component:
        ax1 = fig.add_subplot(gs[i, 0])
        # Add the coordinate system with dashed lines:
        ax1.axhline(color="lightgray", linestyle="dashed")
        ax1.axvline(color="lightgray", linestyle="dashed")

        # Add subplot for imaginary component:
        ax2 = fig.add_subplot(gs[i, 1])
        # Add the coordinate system with dashed lines:
        ax2.axhline(color="lightgray", linestyle="dashed")
        ax2.axvline(color="lightgray", linestyle="dashed")

        # Plot the real component of the SSA trajectories using the NN output:
        ax1.fill_between(inputs["times"], lhs_c[:, i] - lhs_c_ci[:, i], lhs_c[:, i] + lhs_c_ci[:, i], facecolor=color_dict["nn_indirect"], alpha=0.3)
        real_ssa_fill = Patch(facecolor=color_dict["nn_indirect"], alpha=0.3)
        ax1_handles.append((real_ssa_fill, temp_plot(ax1, inputs["times"], lhs_c[:, i], train_idx+1, t_stop_train, t_stop_valid, color_dict["nn_indirect"])))
        ax1_labels.append(f"SSA using {architecture}")
        # Plot the real component of the NN output:
        ax1_handles.append(temp_plot(ax1, inputs["times"], rhs_c[:, i], train_idx+1, t_stop_train, t_stop_valid, color_dict["nn_direct"]))
        ax1_labels.append(architecture)

        # Plot the imaginary component of the SSA trajectories using the NN output:
        ax2.fill_between(inputs["times"], lhs_d[:, i] - lhs_d_ci[:, i], lhs_d[:, i] + lhs_d_ci[:, i], facecolor=color_dict["nn_indirect"], alpha=0.3)
        imag_ssa_fill = Patch(facecolor=color_dict["nn_indirect"], alpha=0.3)
        ax2_handles.append((imag_ssa_fill, temp_plot(ax2, inputs["times"], lhs_d[:, i], train_idx+1, t_stop_train, t_stop_valid, color_dict["nn_indirect"])))
        ax2_labels.append(f"SSA using {architecture}")
        # Plot the imaginary component of the NN output:
        ax2_handles.append(temp_plot(ax2, inputs["times"], rhs_d[:, i], train_idx+1, t_stop_train, t_stop_valid, color_dict["nn_direct"]))
        ax2_labels.append(architecture)

        ax1.set_title(f"$E_x [c_{{{theta_comma}{symbol}}} (X{theta} (t))]$ for ${symbol}={i+1}$ and $x = [{','.join(map(str, state.astype(int)))}]$")
        ax1.set_xlabel("$t$")
        ax1.set_ylabel(f"$E_x [c_{{{theta_comma}{symbol}}} (X{theta} (t))]$")
        ax1.legend(ax1_handles[::-1], ax1_labels[::-1])

        ax2.set_title(f"$E_x [d_{{{theta_comma}{symbol}}} (X{theta} (t))]$ for ${symbol}={i+1}$ and $x = [{','.join(map(str, state.astype(int)))}]$")
        ax2.set_xlabel("$t$")
        ax2.set_ylabel(f"$E_x [d_{{{theta_comma}{symbol}}} (X{theta} (t))]$")
        ax2.legend(ax2_handles[::-1], ax2_labels[::-1])

    fig.tight_layout()

    fig.savefig(f"{run.results_subdir}/{run.timestamp}_{index}_J_TemporalEigenfunctions.pdf")
    plt.close(fig)
