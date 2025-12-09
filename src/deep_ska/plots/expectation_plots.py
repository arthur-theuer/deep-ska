"""Plotting methods related to the expectation estimates."""

import logging

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

from ..core.initialization import RunContext
from .common import fill_wrappers, line_wrappers  # noqa: F401
from .common.dispatch import render_instruction
from .common.helpers import (
    architecture_dict,
    color_dict,
    create_centered_array,
)
from .common.instruction import PlotInstruction
from .common.labels import (
    generate_output_function_labels,
    generate_stationary_output_function_labels,
    generate_trajectory_labels,
    generate_variance_output_function_labels,
)


def compute_repeat_errors(repeat_results: list[dict], method: str, reference_value: float, output_idx: int) -> tuple:
    """Compute squared error arrays for multiple repeats."""
    errors = []
    for result in repeat_results:
        cummean = result.get(f"{method}_cummean")
        if cummean is None:
            continue
        error = np.square(cummean[:, output_idx] - reference_value[output_idx])
        errors.append(error)

    if not errors:
        return None, None

    errors = np.array(errors)
    mean = np.mean(errors, axis=0)
    lower = np.maximum(0, mean - np.percentile(errors, 2.5, axis=0))
    upper = np.maximum(0, np.percentile(errors, 97.5, axis=0) - mean)

    return mean, np.vstack((lower, upper))


def plot_temporal_trajectories(state: np.ndarray, inputs: dict, run: RunContext, index: int, results: dict, names: list, figure_data: dict) -> None:
    """Plot the temporal evolution of different types of trajectories."""
    raw_mean: np.ndarray | None = results.get("raw_mean")

    n_plots = raw_mean.shape[1]  # based on output function size
    output_labels = generate_trajectory_labels(n_plots, use_theta=run.PLT_CONFIG["use_theta_in_labels"])

    t_min = run.PLT_CONFIG["trajectory_plot_minimum"]
    t_max = run.PLT_CONFIG["trajectory_plot_maximum"]
    rescale = run.PLT_CONFIG["rescale_trajectory_plots"]

    for i in range(n_plots):
        fig, ax = plt.subplots(figsize=(6, 4))  # create a new figure for each plot
        elements = []  # store lines and fills for the panel created later
        handles, labels = [], []  # store custom legend handles and labels

        if "SSA" in names and results.get("raw_ssa_estimate") is not None:
            raw_ssa_estimate: np.ndarray | None = results.get("raw_ssa_estimate")[:run.PLT_CONFIG["n_plotted_trajectories"]]

            line_data_list = []

            for j in range(raw_ssa_estimate.shape[0]):  # iterate over the selected trajectories
                x_full = inputs["times"]
                y_full = raw_ssa_estimate[j, :, i]

                if rescale:
                    mask = (x_full >= t_min) & (x_full <= t_max)
                    x_sliced, y_sliced = x_full[mask], y_full[mask]
                else:
                    x_sliced, y_sliced = x_full, y_full

                line_data = PlotInstruction("step", (x_sliced, y_sliced, color_dict["sim"], "post"))
                line = render_instruction(ax, line_data)
                line_data_list.append(line_data)

            elements.extend(line_data_list)
            handles.append(line)
            labels.append(f"SSA ({run.PLT_CONFIG['n_plotted_trajectories']} trajectories)")

        if "SSA+tsubDeepCV" in names and results.get("diff_time_suboptimal_estimate") is not None:
            diff_time_suboptimal_estimate: np.ndarray | None = results.get("diff_time_suboptimal_estimate")[:run.PLT_CONFIG["n_plotted_trajectories"]]

            line_data_list = []

            for j in range(diff_time_suboptimal_estimate.shape[0]):  # iterate over the selected trajectories
                x_full = inputs["times"][1:]
                y_full = diff_time_suboptimal_estimate[j, :, i]

                if rescale:
                    mask = (x_full >= t_min) & (x_full <= t_max)
                    x_sliced, y_sliced = x_full[mask], y_full[mask]
                else:
                    x_sliced, y_sliced = x_full, y_full

                line_data = PlotInstruction("traj", (x_sliced, y_sliced, color_dict["sim_cv"]))
                line = render_instruction(ax, line_data)
                line_data_list.append(line_data)

            elements.extend(line_data_list)
            handles.append(line)
            labels.append(f"SSA with DeepCV ({run.PLT_CONFIG['n_plotted_trajectories']} trajectories)")

        if "SSA+tsubDeepIS" in names and results.get("is_raw_time_suboptimal_ssa_estimate") is not None:
            is_raw_time_sub_ssa_estimate: np.ndarray | None = results.get("is_raw_time_suboptimal_ssa_estimate")[:run.PLT_CONFIG["n_plotted_trajectories"]]

            line_data_list = []

            for j in range(is_raw_time_sub_ssa_estimate.shape[0]):  # iterate over the selected trajectories
                x_full = inputs["times"][1:]
                y_full = is_raw_time_sub_ssa_estimate[j, :, i]

                if rescale:
                    mask = (x_full >= t_min) & (x_full <= t_max)
                    x_sliced, y_sliced = x_full[mask], y_full[mask]
                else:
                    x_sliced, y_sliced = x_full, y_full

                line_data = PlotInstruction("traj", (x_sliced, y_sliced, color_dict["sim_is"]))
                line = render_instruction(ax, line_data)
                line_data_list.append(line_data)

            elements.extend(line_data_list)
            handles.append(line)
            labels.append(f"SSA with DeepIS ({run.PLT_CONFIG['n_plotted_trajectories']} trajectories)")

        dot_legend = False

        if results.get("raw_ssa_estimate") is not None:  # plot the initial state
            x0: np.ndarray = inputs["times"][0]
            y0: np.ndarray | None = results.get("raw_ssa_estimate")[0, 0, i]
            if not rescale or (t_min <= x0 <= t_max):
                line_data = PlotInstruction("rdot", (x0, y0))
                line = render_instruction(ax, line_data)

                elements.append(line_data)

                dot_legend = True

        if raw_mean is not None:  # plot the final expectation
            x0, y0 = inputs["times"][-1], raw_mean[-1, i]
            if not rescale or (t_min <= x0 <= t_max):
                line_data = PlotInstruction("rdot", (inputs["times"][-1], raw_mean[-1, i]))
                line = render_instruction(ax, line_data)

                elements.append(line_data)
                dot_legend = True

        if dot_legend:
            handles.append(line)
            labels.append(f"SSA ({results.get('raw_samples')} samples)")

        title_color = "black" if np.all(state >= 0) else "tab:red"  # highlight negative states in red

        ax.set_title(f"{output_labels[i]} for $x = [{','.join(map(str, state.astype(int)))}]$", color=title_color)
        ax.set_ylabel(output_labels[i])
        ax.set_xlabel("Time $t$")
        ax.legend(handles[::-1], labels[::-1])

        fig.tight_layout()
        fig_name = f"{run.timestamp}_{index}_A_TemporalTrajectories_f{i+1}_{'_'.join(names)}"

        figure_data[fig_name] = {
            "names": names,
            "elements": elements,
            "handles": handles,
            "labels": labels,
            "state": state.copy(),
            "index": index,
            "output_idx": i,
            "output_label": output_labels[i],
            "title": f"$x = [{','.join(map(str, state.copy().astype(int)))}]$",
            "xlabel": ax.get_xlabel(),
            "ylabel": ax.get_ylabel(),
        }

        fig.savefig(f"{run.results_subdir}/{fig_name}.pdf")
        plt.close(fig)


def plot_temporal_expectation(state: np.ndarray, inputs: dict, run: RunContext, index: int, results: dict, names: list, figure_data: dict) -> None:
    """Plot the temporal evolution of the expectation values."""
    full_V = results.get("full_V")
    true_V = results.get("true_V")

    raw_mean = results.get("raw_mean")
    raw_ci = results.get("raw_ci")
    raw_tolerance = results.get("raw_tolerance")
    raw_subset_mean = results.get("raw_subset_mean")
    raw_subset_ci = results.get("raw_subset_ci")
    raw_subset_tolerance = results.get("raw_subset_tolerance")

    diff_mean = results.get("diff_mean")
    diff_ci = results.get("diff_ci")
    diff_out_sub_mean = results.get("diff_out_suboptimal_mean")
    diff_out_sub_ci = results.get("diff_out_suboptimal_ci")
    diff_time_sub_mean = results.get("diff_time_suboptimal_mean")
    diff_time_sub_ci = results.get("diff_time_suboptimal_ci")

    is_raw_mean = results.get("is_raw_mean")
    is_raw_ci = results.get("is_raw_ci")
    is_raw_out_sub_mean = results.get("is_raw_out_suboptimal_mean")
    is_raw_out_sub_ci = results.get("is_raw_out_suboptimal_ci")
    is_raw_time_sub_mean = results.get("is_raw_time_suboptimal_mean")
    is_raw_time_sub_ci = results.get("is_raw_time_suboptimal_ci")

    n_plots = raw_mean.shape[1]  # based on output function size
    output_labels = generate_output_function_labels(n_plots, use_theta=run.PLT_CONFIG["use_theta_in_labels"])

    # Find validation t_stop and training t_stop, as well as the index of the training t_stop:
    t_stop_train = run.RN_CONFIG["t_stop"]
    t_stop_valid = run.VAL_CONFIG["t_stop"]
    train_idx = np.searchsorted(inputs["times"], t_stop_train, side="right")

    # NOTE: While the inputs["times"] array has shape (n_time_samples), the full_V array has shape
    # (n_time_samples-1, out_function_size). Thus, we later select [1:train_idx+1] and
    # [:train_idx, i] to match the shapes of the two arrays. Before, we would have selected [1:] and
    # [:] which results in the same, because the full_V array is one step shorter (and, therefore,
    # has one index less) than the inputs["times"] array.

    for i in range(n_plots):
        fig, ax = plt.subplots(figsize=(6, 4))  # create a new figure for each plot
        elements = []  # store lines and fills for the panel created later
        handles, labels = [], []  # store custom legend handles and labels

        if "Exact" in names and true_V is not None:
            line_data = PlotInstruction("temp", (inputs["times"], true_V, train_idx+1, t_stop_train, t_stop_valid, color_dict["exact"]))

            line = render_instruction(ax, line_data)

            elements.append(line_data)
            handles.append(line)
            labels.append("Exact")

        if "SSAtol" in names and raw_mean is not None and raw_tolerance is not None:
            fill_data = PlotInstruction("fill", (), {"x": inputs["times"][1:], "y1": (raw_mean - raw_tolerance)[:, i], "y2": (raw_mean + raw_tolerance)[:, i], "facecolor": color_dict["sim_tolerance"], "alpha": 0.3})

            render_instruction(ax, fill_data)
            patch = Patch(facecolor=color_dict["sim_tolerance"], alpha=0.3)

            elements.append(fill_data)
            handles.append(patch)
            labels.append(f"SSA ({results.get('raw_samples')} samples, ± 5% tolerance band)")

        if "SSAtol-" in names and raw_subset_mean is not None and raw_subset_tolerance is not None:
            fill_data = PlotInstruction("fill", (), {"x": inputs["times"][1:], "y1": (raw_subset_mean - raw_subset_tolerance)[:, i], "y2": (raw_subset_mean + raw_subset_tolerance)[:, i], "facecolor": color_dict["sim_tolerance"], "alpha": 0.3})

            render_instruction(ax, fill_data)
            patch = Patch(facecolor=color_dict["sim_tolerance"], alpha=0.3)

            elements.append(fill_data)
            handles.append(patch)
            labels.append(f"SSA ({results.get('raw_samples')} samples, ± 5% tolerance band)")

        if "SSA" in names and raw_mean is not None and raw_ci is not None:
            fill_data = PlotInstruction("fill", (), {"x": inputs["times"][1:], "y1": (raw_mean - raw_ci)[:, i], "y2": (raw_mean + raw_ci)[:, i], "facecolor": color_dict["sim"], "alpha": 0.3})
            line_data = PlotInstruction("temp", (inputs["times"][1:], raw_mean[:, i], train_idx, t_stop_train, t_stop_valid, color_dict["sim"]))

            render_instruction(ax, fill_data)
            line = render_instruction(ax, line_data)
            patch = Patch(facecolor=color_dict["sim"], alpha=0.3)

            elements.extend([fill_data, line_data])
            handles.append((patch, line))
            labels.append(f"SSA ({results.get('raw_samples')} samples, 95% CI)")

        if "SSA-" in names and raw_subset_mean is not None and raw_subset_ci is not None:
            fill_data = PlotInstruction("fill", (), {"x": inputs["times"][1:], "y1": (raw_subset_mean - raw_subset_ci)[:, i], "y2": (raw_subset_mean + raw_subset_ci)[:, i], "facecolor": color_dict["sim_small"], "alpha": 0.3})
            line_data = PlotInstruction("temp", (inputs["times"][1:], raw_subset_mean[:, i], train_idx, t_stop_train, t_stop_valid, color_dict["sim_small"]))

            render_instruction(ax, fill_data)
            line = render_instruction(ax, line_data)
            patch = Patch(facecolor=color_dict["sim_small"], alpha=0.3)

            elements.extend([fill_data, line_data])
            handles.append((patch, line))
            labels.append(f"SSA ({results.get('raw_subset_samples')} samples, 95% CI)")

        if "SSA+DeepCV" in names and diff_mean is not None and diff_ci is not None:
            fill_data = PlotInstruction("fill", (), {"x": inputs["times"][1:], "y1": (diff_mean - diff_ci)[:, i], "y2": (diff_mean + diff_ci)[:, i], "facecolor": color_dict["sim_cv"], "alpha": 0.3})
            line_data = PlotInstruction("temp", (inputs["times"][1:], diff_mean[:, i], train_idx, t_stop_train, t_stop_valid, color_dict["sim_cv"]))

            render_instruction(ax, fill_data)
            line = render_instruction(ax, line_data)
            patch = Patch(facecolor=color_dict["sim_cv"], alpha=0.3)

            elements.extend([fill_data, line_data])
            handles.append((patch, line))
            labels.append(f"SSA with DeepCV ({results.get('diff_samples')} samples, 95% CI)")

        if "SSA+osubDeepCV" in names and diff_out_sub_mean is not None and diff_out_sub_ci is not None:
            fill_data = PlotInstruction("fill", (), {"x": inputs["times"][1:], "y1": (diff_out_sub_mean - diff_out_sub_ci)[:, i], "y2": (diff_out_sub_mean + diff_out_sub_ci)[:, i], "facecolor": color_dict["sim_cv"], "alpha": 0.3})
            line_data = PlotInstruction("temp", (inputs["times"][1:], diff_out_sub_mean[:, i], train_idx, t_stop_train, t_stop_valid, color_dict["sim_cv"]))

            render_instruction(ax, fill_data)
            line = render_instruction(ax, line_data)
            patch = Patch(facecolor=color_dict["sim_cv"], alpha=0.3)

            elements.extend([fill_data, line_data])
            handles.append((patch, line))
            labels.append(f"SSA with output-suboptimal DeepCV ({results.get('diff_samples')} samples, 95% CI)")

        if "SSA+tsubDeepCV" in names and diff_time_sub_mean is not None and diff_time_sub_ci is not None:
            fill_data = PlotInstruction("fill", (), {"x": inputs["times"][1:], "y1": (diff_time_sub_mean - diff_time_sub_ci)[:, i], "y2": (diff_time_sub_mean + diff_time_sub_ci)[:, i], "facecolor": color_dict["sim_cv"], "alpha": 0.3})
            line_data = PlotInstruction("temp", (inputs["times"][1:], diff_time_sub_mean[:, i], train_idx, t_stop_train, t_stop_valid, color_dict["sim_cv"]))

            render_instruction(ax, fill_data)
            line = render_instruction(ax, line_data)
            patch = Patch(facecolor=color_dict["sim_cv"], alpha=0.3)

            elements.extend([fill_data, line_data])
            handles.append((patch, line))
            labels.append(f"SSA with time-suboptimal DeepCV ({results.get('diff_samples')} samples, 95% CI)")

        if "SSA+DeepIS" in names and is_raw_mean is not None and is_raw_ci is not None:
            train_idx_is = np.searchsorted(results.get("is_times"), t_stop_train, side="right")  # the index is different for the DeepIS
            fill_data = PlotInstruction("fill", (), {"x": results.get("is_times")[1:], "y1": (is_raw_mean - is_raw_ci)[:, i], "y2": (is_raw_mean + is_raw_ci)[:, i], "facecolor": color_dict["sim_is"], "alpha": 0.3})
            line_data = PlotInstruction("temp", (results.get("is_times")[1:], is_raw_mean[:, i], train_idx_is, t_stop_train, t_stop_valid, color_dict["sim_is"]))

            render_instruction(ax, fill_data)
            line = render_instruction(ax, line_data)
            patch = Patch(facecolor=color_dict["sim_is"], alpha=0.3)

            elements.extend([fill_data, line_data])
            handles.append((patch, line))
            labels.append(f"SSA with DeepIS ({results.get('is_raw_samples')} samples, 95% CI)")

        if "SSA+osubDeepIS" in names and is_raw_out_sub_mean is not None and is_raw_out_sub_ci is not None:
            train_idx_is = np.searchsorted(results.get("is_times"), t_stop_train, side="right")  # the index is different for the DeepIS
            fill_data = PlotInstruction("fill", (), {"x": results.get("is_times")[1:], "y1": (is_raw_out_sub_mean - is_raw_out_sub_ci)[:, i], "y2": (is_raw_out_sub_mean + is_raw_out_sub_ci)[:, i], "facecolor": color_dict["sim_is"], "alpha": 0.3})
            line_data = PlotInstruction("temp", (results.get("is_times")[1:], is_raw_out_sub_mean[:, i], train_idx_is, t_stop_train, t_stop_valid, color_dict["sim_is"]))

            render_instruction(ax, fill_data)
            line = render_instruction(ax, line_data)
            patch = Patch(facecolor=color_dict["sim_is"], alpha=0.3)

            elements.extend([fill_data, line_data])
            handles.append((patch, line))
            labels.append(f"SSA with output-suboptimal DeepIS ({results.get('is_raw_samples')} samples, 95% CI)")

        if "SSA+tsubDeepIS" in names and is_raw_time_sub_mean is not None and is_raw_time_sub_ci is not None:
            fill_data = PlotInstruction("fill", (), {"x": inputs["times"][1:], "y1": (is_raw_time_sub_mean - is_raw_time_sub_ci)[:, i], "y2": (is_raw_time_sub_mean + is_raw_time_sub_ci)[:, i], "facecolor": color_dict["sim_is"], "alpha": 0.3})
            line_data = PlotInstruction("temp", (inputs["times"][1:], is_raw_time_sub_mean[:, i], train_idx, t_stop_train, t_stop_valid, color_dict["sim_is"]))

            render_instruction(ax, fill_data)
            line = render_instruction(ax, line_data)
            patch = Patch(facecolor=color_dict["sim_is"], alpha=0.3)

            elements.extend([fill_data, line_data])
            handles.append((patch, line))
            labels.append(f"SSA with time-suboptimal DeepIS ({results.get('is_raw_samples')} samples, 95% CI)")

        if "NN" in names and full_V is not None:
            line_data = PlotInstruction("temp", (inputs["times"][1:], full_V[:, i], train_idx, t_stop_train, t_stop_valid, color_dict["nn_direct"]))

            line = render_instruction(ax, line_data)

            elements.append(line_data)
            handles.append(line)
            labels.append(architecture_dict[run.NN_CONFIG['v']['subnet_architecture']])

        short_title = f"$x = [{','.join(map(str, state.astype(int)))}]$"
        title_color = "black" if np.all(state >= 0) else "tab:red"  # highlight negative states in red

        ax.set_title(f"{output_labels[i]} for {short_title}", color=title_color)
        ax.set_ylabel(output_labels[i])
        ax.set_xlabel("Time $t$")
        ax.legend(handles[::-1], labels[::-1])

        fig.tight_layout()
        fig_name = f"{run.timestamp}_{index}_B_TemporalExpectation_f{i+1}_{'_'.join(names)}"

        figure_data[fig_name] = {
            "names": names,
            "elements": elements,
            "handles": handles,
            "labels": labels,
            "state": state.copy(),
            "index": index,
            "output_idx": i,
            "output_label": output_labels[i],
            "title": short_title,
            "xlabel": ax.get_xlabel(),
            "ylabel": ax.get_ylabel(),
        }

        fig.savefig(f"{run.results_subdir}/{fig_name}.pdf")
        plt.close(fig)


def plot_expectation_error_convergence(state: np.ndarray, run: RunContext, index: int, results: dict, names: list, figure_data: dict) -> None:
    """Plot the convergence of various estimates with increasing sample sizes."""
    ref_true_mean = results.get("ref_true_mean")
    ref_raw_mean = results.get("ref_raw_mean")
    ref_diff_mean = results.get("ref_diff_mean")
    ref_is_raw_mean = results.get("ref_is_raw_mean")

    raw_cummean = results.get("raw_cummean")
    diff_cummean = results.get("diff_cummean")
    is_raw_cummean = results.get("is_raw_cummean")

    conv_t_stop = run.VAL_CONFIG["conv_t_stop"]

    samples = run.PLT_CONFIG["convergence_plot_samples"]
    maximum = run.PLT_CONFIG["convergence_plot_maximum"]

    idx = np.unique(np.round(np.logspace(0, np.log10(maximum), samples)).astype(int))

    if ref_true_mean is not None:
        n_plots = ref_true_mean.shape[0]  # output function size
    elif ref_raw_mean is not None:
        n_plots = ref_raw_mean.shape[0]
    elif ref_diff_mean is not None:
        n_plots = ref_diff_mean.shape[0]
    elif ref_is_raw_mean is not None:
        n_plots = ref_is_raw_mean.shape[0]
    elif raw_cummean is not None:
        n_plots = raw_cummean.shape[1]
    elif diff_cummean is not None:
        n_plots = diff_cummean.shape[1]
    elif is_raw_cummean is not None:
        n_plots = is_raw_cummean.shape[1]
    else:
        return  # no data available

    output_labels = generate_output_function_labels(n_plots, use_theta=run.PLT_CONFIG["use_theta_in_labels"])

    for i in range(n_plots):
        fig, ax = plt.subplots(figsize=(6, 4))
        elements = []  # store lines and fills for the panel created later
        handles, labels = [], []  # store custom legend handles and labels

        # Compute reference value using all available SSA samples at t_stop_train:
        if ref_true_mean is not None:
            reference_value = ref_true_mean[i]
        elif ref_raw_mean is not None:
            reference_value = np.mean(ref_raw_mean[i])
        elif ref_diff_mean is not None:
            reference_value = np.mean(ref_diff_mean[i])
        elif ref_is_raw_mean is not None:
            reference_value = np.mean(ref_is_raw_mean[i])
        else:
            logging.warning("Skipping expectation error convergence plot (no reference data available).")
            return  # need SSA or DeepCV/DeepIS data as reference

        if "SSA" in names and raw_cummean is not None:
            line_data = PlotInstruction("scat", (idx, np.square(raw_cummean[idx-1, i] - reference_value), color_dict["sim"]))

            line = render_instruction(ax, line_data)

            elements.append(line_data)
            handles.append(line)
            labels.append("SSA")

        if "SSA+DeepCV" in names and diff_cummean is not None:
            line_data = PlotInstruction("scat", (idx, np.square(diff_cummean[idx-1, i] - reference_value), color_dict["sim_cv"]))

            line = render_instruction(ax, line_data)

            elements.append(line_data)
            handles.append(line)
            labels.append("SSA with DeepCV")

        if "SSA+DeepIS" in names and is_raw_cummean is not None:
            line_data = PlotInstruction("scat", (idx, np.square(is_raw_cummean[idx-1, i] - reference_value), color_dict["sim_is"]))

            line = render_instruction(ax, line_data)

            elements.append(line_data)
            handles.append(line)
            labels.append("SSA with DeepIS")

        title_color = "black" if np.all(state >= 0) else "tab:red"  # highlight negative states in red

        ax.set_title(f"Convergence of {output_labels[i]} for $x = [{','.join(map(str, state.astype(int)))}]$ at $t = {conv_t_stop}$", color=title_color)
        ax.set_ylabel(f"Squared error in {output_labels[i]}")
        ax.set_xlabel("Number of samples $n$")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.legend(handles, labels)

        fig.tight_layout()
        fig_name = f"{run.timestamp}_{index}_E_ExpectationErrorConvergence_f{i+1}_{'_'.join(names)}"

        figure_data[fig_name] = {
            "names": names,
            "elements": elements,
            "handles": handles,
            "labels": labels,
            "state": state.copy(),
            "index": index,
            "output_idx": i,
            "output_label": output_labels[i],
            "title": f"$x = [{','.join(map(str, state.copy().astype(int)))}]$",
            "xlabel": ax.get_xlabel(),
            "ylabel": ax.get_ylabel(),
        }

        fig.savefig(f"{run.results_subdir}/{fig_name}.pdf")
        plt.close(fig)


def plot_expectation_error_convergence_repeats(state: np.ndarray, run: RunContext, index: int, results: dict, ssa_results: dict, cv_results: dict, is_results: dict, names: list, figure_data: dict) -> None:
    """Plot the convergence of various estimates with increasing sample sizes using averaged data."""
    reference_keys = ["ref_true_mean", "ref_raw_mean", "ref_diff_mean", "ref_is_raw_mean"]
    reference_value = next((results.get(k) for k in reference_keys if results.get(k) is not None), None)

    if reference_value is None:
        logging.warning("Skipping expectation error convergence CI plot (no reference data available).")
        return  # no data available, need SSA or DeepCV/DeepIS data as reference

    n_plots = reference_value.shape[0]  # output function size
    output_labels = generate_output_function_labels(n_plots, use_theta=run.PLT_CONFIG["use_theta_in_labels"])

    conv_t_stop = run.VAL_CONFIG["conv_t_stop"]
    samples = run.PLT_CONFIG["convergence_plot_samples"]
    maximum = run.PLT_CONFIG["convergence_plot_maximum"]

    idx = np.unique(np.round(np.logspace(0, np.log10(maximum), samples)).astype(int))

    for i in range(n_plots):
        fig, ax = plt.subplots(figsize=(6, 4))
        elements = []  # store lines and fills for the panel created later
        handles, labels = [], []  # store custom legend handles and labels

        raw_error, raw_ci = compute_repeat_errors(ssa_results, "raw", reference_value, i) if ssa_results else (None, None)
        diff_error, diff_ci = compute_repeat_errors(cv_results, "diff", reference_value, i) if cv_results else (None, None)
        is_raw_error, is_raw_ci = compute_repeat_errors(is_results, "is_raw", reference_value, i) if is_results else (None, None)

        if "SSA" in names and raw_error is not None:
            bars_data = PlotInstruction("error", (), {"x": idx, "y": raw_error[idx-1], "yerr": raw_ci[:, idx-1], "fmt": "o", "markerfacecolor": "white", "color": color_dict["sim"], "capsize": 5})
            line_data = PlotInstruction("scat", (idx, raw_error[idx-1], color_dict["sim"]))

            bar = render_instruction(ax, bars_data)
            line = render_instruction(ax, line_data)

            elements.extend([bars_data, line_data])
            handles.append((bar, line))
            labels.append(f"SSA ({run.VAL_CONFIG['conv_n_repeats']} runs, 95% CI)")

        if "SSA+DeepCV" in names and diff_error is not None:
            bars_data = PlotInstruction("error", (), {"x": idx, "y": diff_error[idx-1], "yerr": diff_ci[:, idx-1], "fmt": "o", "markerfacecolor": "white", "color": color_dict["sim_cv"], "capsize": 5})
            line_data = PlotInstruction("scat", (idx, diff_error[idx-1], color_dict["sim_cv"]))

            bar = render_instruction(ax, bars_data)
            line = render_instruction(ax, line_data)

            elements.extend([bars_data, line_data])
            handles.append((bar, line))
            labels.append(f"SSA with DeepCV ({run.VAL_CONFIG['conv_n_repeats']} runs, 95% CI)")

        if "SSA+DeepIS" in names and is_raw_error is not None:
            bars_data = PlotInstruction("error", (), {"x": idx, "y": is_raw_error[idx-1], "yerr": is_raw_ci[:, idx-1], "fmt": "o", "markerfacecolor": "white", "color": color_dict["sim_is"], "capsize": 5})
            line_data = PlotInstruction("scat", (idx, is_raw_error[idx-1], color_dict["sim_is"]))

            bar = render_instruction(ax, bars_data)
            line = render_instruction(ax, line_data)

            elements.extend([bars_data, line_data])
            handles.append((bar, line))
            labels.append(f"SSA with DeepIS ({run.VAL_CONFIG['conv_n_repeats']} runs, 95% CI)")

        title_color = "black" if np.all(state >= 0) else "tab:red"  # highlight negative states in red

        ax.set_title(f"Convergence of {output_labels[i]} for $x = [{','.join(map(str, state.astype(int)))}]$ at $t = {conv_t_stop}$", color=title_color)
        ax.set_ylabel(f"MSE for {output_labels[i]}")
        ax.set_xlabel("Number of samples $n$")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.legend(handles, labels)

        fig.tight_layout()
        fig_name = f"{run.timestamp}_{index}_F_ExpectationErrorConvergenceCI_f{i+1}_{'_'.join(names)}"

        figure_data[fig_name] = {
            "names": names,
            "elements": elements,
            "handles": handles,
            "labels": labels,
            "state": state.copy(),
            "index": index,
            "output_idx": i,
            "output_label": output_labels[i],
            "title": f"$x = [{','.join(map(str, state.copy().astype(int)))}]$",
            "xlabel": ax.get_xlabel(),
            "ylabel": ax.get_ylabel(),
        }

        fig.savefig(f"{run.results_subdir}/{fig_name}.pdf")
        plt.close(fig)


def plot_temporal_expectation_variance(state: np.ndarray, inputs: dict, run: RunContext, index: int, results: dict, names: list, figure_data: dict) -> None:
    """Plot the temporal evolution of the expectation variances."""
    raw_var = results.get("raw_var")
    raw_var_ci_lower = results.get("raw_var_ci_lower")
    raw_var_ci_upper = results.get("raw_var_ci_upper")

    diff_var = results.get("diff_var")
    diff_var_ci_lower = results.get("diff_var_ci_lower")
    diff_var_ci_upper = results.get("diff_var_ci_upper")

    diff_out_sub_var = results.get("diff_out_suboptimal_var")
    diff_out_sub_var_ci_lower = results.get("diff_out_suboptimal_var_ci_lower")
    diff_out_sub_var_ci_upper = results.get("diff_out_suboptimal_var_ci_upper")
    diff_time_sub_var = results.get("diff_time_suboptimal_var")
    diff_time_sub_var_ci_lower = results.get("diff_time_suboptimal_var_ci_lower")
    diff_time_sub_var_ci_upper = results.get("diff_time_suboptimal_var_ci_upper")

    is_raw_var = results.get("is_raw_var")
    is_raw_var_ci_lower = results.get("is_raw_var_ci_lower")
    is_raw_var_ci_upper = results.get("is_raw_var_ci_upper")

    is_raw_out_suboptimal_var = results.get("is_raw_out_suboptimal_var")
    is_raw_out_suboptimal_var_ci_lower = results.get("is_raw_out_suboptimal_var_ci_lower")
    is_raw_out_suboptimal_var_ci_upper = results.get("is_raw_out_suboptimal_var_ci_upper")
    is_raw_time_suboptimal_var = results.get("is_raw_time_suboptimal_var")
    is_raw_time_suboptimal_var_ci_lower = results.get("is_raw_time_suboptimal_var_ci_lower")
    is_raw_time_suboptimal_var_ci_upper = results.get("is_raw_time_suboptimal_var_ci_upper")

    ter_mean = results.get("ter_mean")
    ter_ci = results.get("ter_ci")
    int_mean = results.get("int_mean")
    int_ci = results.get("int_ci")

    n_plots = raw_var.shape[1]  # based on output function size
    if "DeepPVA" in names or "DeepIPA" in names:
        output_labels = generate_variance_output_function_labels(n_plots, use_theta=run.PLT_CONFIG["use_theta_in_labels"])
    else:
        output_labels = ["Var. for " + label for label in generate_output_function_labels(n_plots, use_theta=run.PLT_CONFIG["use_theta_in_labels"])]

    # Find validation t_stop and training t_stop, as well as the index of the training t_stop:
    t_stop_train = run.RN_CONFIG["t_stop"]
    t_stop_valid = run.VAL_CONFIG["t_stop"]
    train_idx = np.searchsorted(inputs["times"], t_stop_train, side="right")

    for i in range(n_plots):
        fig, ax = plt.subplots(figsize=(6, 4))  # create a new figure for each plot
        elements = []  # store lines and fills for the panel created later
        handles, labels = [], []  # store custom legend handles and labels

        if "SSA" in names and raw_var is not None and raw_var_ci_lower is not None and raw_var_ci_upper is not None:
            fill_data = PlotInstruction("fill", (), {"x": inputs["times"][1:], "y1": raw_var_ci_lower[:, i], "y2": raw_var_ci_upper[:, i], "facecolor": color_dict["sim"], "alpha": 0.3})
            line_data = PlotInstruction("temp", (inputs["times"][1:], raw_var[:, i], train_idx, t_stop_train, t_stop_valid, color_dict["sim"]))

            render_instruction(ax, fill_data)
            line = render_instruction(ax, line_data)
            fill = Patch(facecolor=color_dict["sim"], alpha=0.3)

            elements.extend([fill_data, line_data])
            handles.append((fill, line))
            labels.append(f"SSA ({results.get('raw_samples')} samples, 95% CI)")

        if "DeepPVA" in names and ter_mean is not None and ter_ci is not None:
            fill_data = PlotInstruction("fill", (), {"x": inputs["times"][1:], "y1": (ter_mean - ter_ci)[:, i], "y2": (ter_mean + ter_ci)[:, i], "facecolor": color_dict["deep_ipa"], "alpha": 0.3})
            line_data = PlotInstruction("temp", (inputs["times"][1:], ter_mean[:, i], train_idx, t_stop_train, t_stop_valid, color_dict["deep_ipa"]))

            render_instruction(ax, fill_data)
            line = render_instruction(ax, line_data)
            fill = Patch(facecolor=color_dict["deep_ipa"], alpha=0.3)

            elements.extend([fill_data, line_data])
            handles.append((fill, line))
            labels.append(f"DeepPVA ({results.get('ter_samples')} samples, 95% CI)")

        if "DeepIPA" in names and int_mean is not None and int_ci is not None:
            fill_data = PlotInstruction("fill", (), {"x": inputs["times"][1:], "y1": (int_mean - int_ci)[:, i], "y2": (int_mean + int_ci)[:, i], "facecolor": color_dict["deep_ipa"], "alpha": 0.3})
            line_data = PlotInstruction("temp", (inputs["times"][1:], int_mean[:, i], train_idx, t_stop_train, t_stop_valid, color_dict["deep_ipa"]))

            render_instruction(ax, fill_data)
            line = render_instruction(ax, line_data)
            fill = Patch(facecolor=color_dict["deep_ipa"], alpha=0.3)

            elements.extend([fill_data, line_data])
            handles.append((fill, line))
            labels.append(f"DeepIPA ({results.get('int_samples')} samples, 95% CI)")

        if "SSA+DeepCV" in names and diff_var is not None and diff_var_ci_lower is not None and diff_var_ci_upper is not None:
            fill_data = PlotInstruction("fill", (), {"x": inputs["times"][1:], "y1": diff_var_ci_lower[:, i], "y2": diff_var_ci_upper[:, i], "facecolor": color_dict["sim_cv"], "alpha": 0.3})
            line_data = PlotInstruction("temp", (inputs["times"][1:], diff_var[:, i], train_idx, t_stop_train, t_stop_valid, color_dict["sim_cv"]))

            render_instruction(ax, fill_data)
            line = render_instruction(ax, line_data)
            fill = Patch(facecolor=color_dict["sim_cv"], alpha=0.3)

            elements.extend([fill_data, line_data])
            handles.append((fill, line))
            labels.append(f"SSA with DeepCV ({results.get('diff_samples')} samples, 95% CI)")

        if "SSA+osubDeepCV" in names and diff_out_sub_var is not None and diff_out_sub_var_ci_lower is not None and diff_out_sub_var_ci_upper is not None:
            fill_data = PlotInstruction("fill", (), {"x": inputs["times"][1:], "y1": diff_out_sub_var_ci_lower[:, i], "y2": diff_out_sub_var_ci_upper[:, i], "facecolor": color_dict["sim_cv"], "alpha": 0.3})
            line_data = PlotInstruction("temp", (inputs["times"][1:], diff_out_sub_var[:, i], train_idx, t_stop_train, t_stop_valid, color_dict["sim_cv"]))

            render_instruction(ax, fill_data)
            line = render_instruction(ax, line_data)
            fill = Patch(facecolor=color_dict["sim_cv"], alpha=0.3)

            elements.extend([fill_data, line_data])
            handles.append((fill, line))
            labels.append(f"SSA with output-suboptimal DeepCV ({results.get('diff_out_suboptimal_samples')} samples, 95% CI)")

        if "SSA+tsubDeepCV" in names and diff_time_sub_var is not None and diff_time_sub_var_ci_lower is not None and diff_time_sub_var_ci_upper is not None:
            fill_data = PlotInstruction("fill", (), {"x": inputs["times"][1:], "y1": diff_time_sub_var_ci_lower[:, i], "y2": diff_time_sub_var_ci_upper[:, i], "facecolor": color_dict["sim_cv"], "alpha": 0.3})
            line_data = PlotInstruction("temp", (inputs["times"][1:], diff_time_sub_var[:, i], train_idx, t_stop_train, t_stop_valid, color_dict["sim_cv"]))

            render_instruction(ax, fill_data)
            line = render_instruction(ax, line_data)
            fill = Patch(facecolor=color_dict["sim_cv"], alpha=0.3)

            elements.extend([fill_data, line_data])
            handles.append((fill, line))
            labels.append(f"SSA with time-suboptimal DeepCV ({results.get('diff_time_suboptimal_samples')} samples, 95% CI)")

        if "SSA+DeepIS" in names and is_raw_var is not None and is_raw_var_ci_lower is not None and is_raw_var_ci_upper is not None:
            train_idx_is = np.searchsorted(results.get("is_times"), t_stop_train, side="right")  # the index is different for the DeepIS
            fill_data = PlotInstruction("fill", (), {"x": results.get("is_times")[1:], "y1": is_raw_var_ci_lower[:, i], "y2": is_raw_var_ci_upper[:, i], "facecolor": color_dict["sim_is"], "alpha": 0.3})
            line_data = PlotInstruction("temp", (results.get("is_times")[1:], is_raw_var[:, i], train_idx_is, t_stop_train, t_stop_valid, color_dict["sim_is"]))

            render_instruction(ax, fill_data)
            line = render_instruction(ax, line_data)
            fill = Patch(facecolor=color_dict["sim_is"], alpha=0.3)

            elements.extend([fill_data, line_data])
            handles.append((fill, line))
            labels.append(f"SSA with DeepIS ({results.get('is_raw_samples')} samples, 95% CI)")

        if "SSA+osubDeepIS" in names and is_raw_out_suboptimal_var is not None and is_raw_out_suboptimal_var_ci_lower is not None and is_raw_out_suboptimal_var_ci_upper is not None:
            train_idx_is = np.searchsorted(results.get("is_times"), t_stop_train, side="right")  # the index is different for the DeepIS
            fill_data = PlotInstruction("fill", (), {"x": results.get("is_times")[1:], "y1": is_raw_out_suboptimal_var_ci_lower[:, i], "y2": is_raw_out_suboptimal_var_ci_upper[:, i], "facecolor": color_dict["sim_is"], "alpha": 0.3})
            line_data = PlotInstruction("temp", (results.get("is_times")[1:], is_raw_out_suboptimal_var[:, i], train_idx_is, t_stop_train, t_stop_valid, color_dict["sim_is"]))

            render_instruction(ax, fill_data)
            line = render_instruction(ax, line_data)
            fill = Patch(facecolor=color_dict["sim_is"], alpha=0.3)

            elements.extend([fill_data, line_data])
            handles.append((fill, line))
            labels.append(f"SSA with output-suboptimal DeepIS ({results.get('is_raw_out_suboptimal_samples')} samples, 95% CI)")

        if "SSA+tsubDeepIS" in names and is_raw_time_suboptimal_var is not None and is_raw_time_suboptimal_var_ci_lower is not None and is_raw_time_suboptimal_var_ci_upper is not None:
            fill_data = PlotInstruction("fill", (), {"x": inputs["times"][1:], "y1": is_raw_time_suboptimal_var_ci_lower[:, i], "y2": is_raw_time_suboptimal_var_ci_upper[:, i], "facecolor": color_dict["sim_is"], "alpha": 0.3})
            line_data = PlotInstruction("temp", (inputs["times"][1:], is_raw_time_suboptimal_var[:, i], train_idx, t_stop_train, t_stop_valid, color_dict["sim_is"]))

            render_instruction(ax, fill_data)
            line = render_instruction(ax, line_data)
            fill = Patch(facecolor=color_dict["sim_is"], alpha=0.3)

            elements.extend([fill_data, line_data])
            handles.append((fill, line))
            labels.append(f"SSA with time-suboptimal DeepIS ({results.get('is_raw_time_suboptimal_samples')} samples, 95% CI)")

        title_color = "black" if np.all(state >= 0) else "tab:red"  # highlight negative states in red

        ax.set_title(f"{output_labels[i]} for $x = [{','.join(map(str, state.astype(int)))}]$", color=title_color)
        ax.set_ylabel(output_labels[i])
        ax.set_xlabel("Time $t$")
        ax.legend(handles[::-1], labels[::-1])

        fig.tight_layout()

        # Plot with normal axes:
        fig_name = f"{run.timestamp}_{index}_C_TemporalExpectationVariance_f{i+1}_{'_'.join(names)}"

        figure_data[fig_name] = {
            "names": names,
            "elements": elements,
            "handles": handles,
            "labels": labels,
            "state": state.copy(),
            "index": index,
            "output_idx": i,
            "output_label": output_labels[i],
            "title": f"$x = [{','.join(map(str, state.copy().astype(int)))}]$",
            "xlabel": ax.get_xlabel(),
            "ylabel": ax.get_ylabel(),
        }

        fig.savefig(f"{run.results_subdir}/{fig_name}.pdf")
        plt.close(fig)

        # Plot with logarithmic axes:
        fig_name = f"{run.timestamp}_{index}_D_TemporalExpectationVarianceLog_f{i+1}_{'_'.join(names)}"

        ax.set_yscale("log")

        figure_data[fig_name] = {
            "names": names,
            "elements": elements,
            "handles": handles,
            "labels": labels,
            "state": state.copy(),
            "index": index,
            "output_idx": i,
            "output_label": output_labels[i],
            "title": f"$x = [{','.join(map(str, state.copy().astype(int)))}]$",
            "xlabel": ax.get_xlabel(),
            "ylabel": ax.get_ylabel(),
        }

        fig.savefig(f"{run.results_subdir}/{fig_name}.pdf")
        plt.close(fig)


def plot_ergodic_means(state: np.ndarray, run: RunContext, index: int, results: dict) -> None:
    """Creates bar plots comparing the ergodic mean and the ergodic mean with DeepCV."""
    theta = "_{{\\theta}}" if run.PLT_CONFIG["use_theta_in_labels"] else ""

    poiss_em_mean = np.array(results.get("poiss_em_mean"))
    poiss_em_ci = np.array(results.get("poiss_em_ci"))
    raw_small_em_mean = np.array(results.get("raw_small_em_mean"))
    raw_small_em_ci = np.array(results.get("raw_small_em_ci"))
    raw_em_mean = np.array(results.get("raw_em_mean"))
    raw_em_ci = np.array(results.get("raw_em_ci"))

    n_plots = raw_em_mean.shape[0]
    fig = plt.figure(figsize=(4*n_plots, 4))
    gs = plt.GridSpec(1, n_plots)

    output_label = generate_stationary_output_function_labels(n_plots)

    # Get the number of bars and the width and position of each bar:
    n_bars = 3  # number of bars per output function
    width_plus_padding = 0.9 / n_bars
    width = width_plus_padding * 0.9
    bars = np.array([k * (width_plus_padding) for k in create_centered_array(n_bars)])

    # Combine the means and confidence intervals for the raw and Poisson ergodic means:
    comparison_mean = np.vstack((poiss_em_mean, raw_small_em_mean, raw_em_mean))
    comparison_ci = np.vstack((poiss_em_ci, raw_small_em_ci, raw_em_ci))

    for plot in range(n_plots):
        # Plot the ergodic means as grouped bars:
        ax = fig.add_subplot(gs[plot])
        ax.bar(bars[0], comparison_mean[0, plot], width, yerr=comparison_ci[0, plot], color=color_dict["sim_cv"], capsize=5, label=f"EM with DeepCV ({results.get('poiss_em_samples')} samples, $[{results.get('poiss_em_t_min')}, {results.get('poiss_em_t_max')}]$ time interval)")
        ax.bar(bars[1], comparison_mean[1, plot], width, yerr=comparison_ci[1, plot], color=color_dict["sim_small"], capsize=5, label=f"EM ({results.get('raw_small_em_samples')} samples, $[{results.get('raw_small_em_t_min')}, {results.get('raw_small_em_t_max')}]$ time interval)")
        ax.bar(bars[2], comparison_mean[2, plot], width, yerr=comparison_ci[2, plot], color=color_dict["sim"], capsize=5, label=f"EM ({results.get('raw_em_samples')} samples, $[{results.get('raw_em_t_min')}, {results.get('raw_em_t_max')}]$ time interval)")

        # Set the remaining plot properties:
        ax.set_title(f"Stationary mean for $x = [{','.join(map(str, state.astype(int)))}]$")
        if plot == 0:
            ax.set_ylabel(f"$E_{{\\pi{theta}}}(f)$")
        ax.set_xticks([0], [output_label[plot]])

        handles, labels = ax.get_legend_handles_labels()

    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, -0.125), frameon=False)
    fig.tight_layout(rect=[0, 0.05, 1, 1])
    fig.savefig(f"{run.results_subdir}/{run.timestamp}_{index}_G_ErgodicMeans.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_ergodic_means_variance(state: np.ndarray, run: RunContext, index: int, results: dict) -> None:
    """Creates bar plots comparing the ergodic mean and the ergodic mean with DeepCV."""
    theta = "_{{\\theta}}" if run.PLT_CONFIG["use_theta_in_labels"] else ""

    poiss_em_var = np.array(results.get("poiss_em_var"))
    raw_small_em_var = np.array(results.get("raw_small_em_var"))
    raw_em_var = np.array(results.get("raw_em_var"))

    n_plots = raw_em_var.shape[0]
    fig = plt.figure(figsize=(4*n_plots, 4))
    gs = plt.GridSpec(1, n_plots)

    output_label = generate_stationary_output_function_labels(n_plots)

    # Get the number of bars and the width and position of each bar:
    n_bars = 3  # number of bars per output function
    width_plus_padding = 0.9 / n_bars
    width = width_plus_padding * 0.9
    bars = np.array([k * (width_plus_padding) for k in create_centered_array(n_bars)])

    # Combine the variances for the raw and Poisson ergodic means:
    comparison_var = np.vstack((poiss_em_var, raw_small_em_var, raw_em_var))

    for plot in range(n_plots):
        # Plot the ergodic means as grouped bars:
        ax = fig.add_subplot(gs[plot])
        ax.bar(bars[0], comparison_var[0, plot], width, color=color_dict["sim_cv"], capsize=5, label=f"EM with DeepCV ({results.get('poiss_em_samples')} samples, $[{results.get('poiss_em_t_min')}, {results.get('poiss_em_t_max')}]$ time interval)")
        ax.bar(bars[1], comparison_var[1, plot], width, color=color_dict["sim_small"], capsize=5, label=f"EM ({results.get('raw_small_em_samples')} samples, $[{results.get('raw_small_em_t_min')}, {results.get('raw_small_em_t_max')}]$ time interval)")
        ax.bar(bars[2], comparison_var[2, plot], width, color=color_dict["sim"], capsize=5, label=f"EM ({results.get('raw_em_samples')} samples, $[{results.get('raw_em_t_min')}, {results.get('raw_em_t_max')}]$ time interval)")

        # Set the remaining plot properties:
        ax.set_title(f"Stationary mean variance for $x = [{','.join(map(str, state.astype(int)))}]$")
        if plot == 0:
            ax.set_ylabel(f"Var. for $E_{{\\pi{theta}}}(f)$")
        ax.set_xticks([0], [output_label[plot]])

        handles, labels = ax.get_legend_handles_labels()

    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, -0.125), frameon=False)
    fig.tight_layout(rect=[0, 0.05, 1, 1])

    # Plot with normal y-axis:
    fig.savefig(f"{run.results_subdir}/{run.timestamp}_{index}_H_ErgodicMeansVariance.pdf", bbox_inches="tight")
    plt.close(fig)

    # Plot with logarithmic y-axis:
    for ax in fig.get_axes():
        ax.set_yscale("log")

    fig.savefig(f"{run.results_subdir}/{run.timestamp}_{index}_I_ErgodicMeansVarianceLog.pdf", bbox_inches="tight")
    plt.close(fig)
