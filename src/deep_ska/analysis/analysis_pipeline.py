"""Pipeline for analyzing trained models with test states."""

import gzip
import logging
import os
import pickle

import numpy as np
from numpy import ndarray

from ..core.initialization import RunContext
from ..core.simulation import ExpectationGeneratorEM, ExpectationGeneratorSSA
from ..logging.helpers import (
    format_mean_ci_arrays,
    insert_blank_line,
    insert_section_marker,
    insert_subsection_marker,
    log_without_linebreak,
)
from ..plots import (
    expectation_plots,
    panels,
    spectral_plots,
)
from ..plots.common.helpers import architecture_dict
from ..reaction_networks import examples
from ..reaction_networks.definition import ReactionNetwork
from ..timing.test_state_timing import test_state_log_file_analysis
from ..training import ExpectationModelTrainer
from .analyzers import (
    ExpectationGeneratorPoissonEM,
    FinalTimeSubnetAnalyzer,
    SpectralSubnetAnalyzer,
    TemporalSubnetAnalyzer,
)
from .convergence import (
    run_convergence_deep_cv_repeats,
    run_convergence_deep_is_repeats,
    run_convergence_exact_deep_cv_repeats,
    run_convergence_ssa_repeats,
)


def create_test_states(trainer: ExpectationModelTrainer, config_test_states: list, n_random_states: int, seed: int) -> list[np.ndarray]:
    """Create a list of test states for the model evaluation."""
    # Create a new RNG for selecting the test states:
    rng = np.random.default_rng(seed)
    # Don't include the initial state of the training network by default:
    test_states = []
    # Get any additional test states from the config:
    test_states.extend(np.array(state).astype(float) for state in config_test_states)
    # Get all unique states from the validation data set:
    all_states = trainer.valid_data["state_trajectories"].reshape(-1, trainer.valid_data["state_trajectories"].shape[-1])
    unique_states = np.unique(all_states, axis=0)
    # Filter out states already in test_states to avoid duplicates:
    existing_states = {tuple(state) for state in test_states}
    available_states = np.array([state for state in unique_states if tuple(state) not in existing_states])
    # Add n_random_states additional random test states:
    if n_random_states > 0 and available_states.size > 0:
        # Randomly select a few unique states for testing:
        n_random_states = min(n_random_states, available_states.shape[0])
        selected_indices = rng.choice(available_states.shape[0], size=n_random_states, replace=False)
        random_test_states = available_states[selected_indices]
        test_states.extend(random_test_states)
    return test_states


def select_trajectory_subset(inputs: dict, seed: int, subset_fraction: float) -> tuple[dict, np.ndarray]:
    """Select a subset of the input data."""
    if not 0 < subset_fraction <= 1:  # validate the subset fraction
        raise ValueError("The subset fraction must be between 0 and 1.")
    if "times" not in inputs:  # check if the input data contains the necessary times array
        raise KeyError('Input data must include a "times" array.')

    # Create a new RNG for selecting the subset:
    rng = np.random.default_rng(seed)
    # Extract the "times" array because it should not be subsetted:
    times = inputs["times"]
    # Make a new dictionary with all data except the "times" array:
    batched_data = {key: array for key, array in inputs.items() if key != "times"}
    valid_arrays = {key: array for key, array in batched_data.items() if array is not None}
    # Ensure that all arrays have the same number of trajectories:
    batch_sizes = [array.shape[0] for array in valid_arrays.values()]
    if len(set(batch_sizes)) != 1:  # check if all elements in the list are the same
        raise ValueError(f"All arrays must have the same batch size, got batch sizes: {batch_sizes}.")
    # Get the batch size and the number of trajectories needed:
    batch_size = batch_sizes[0]
    n_trajectories_needed = max(int(batch_size * subset_fraction), 1)  # ensure at least one trajectory
    # Select random subset indices using the RNG:
    subset_indices = rng.choice(range(batch_size), n_trajectories_needed, replace=False)
    # Subset each array in the input data:
    subset_data = {"times": times}
    subset_data.update({key: (array[subset_indices] if array is not None else None) for key, array in batched_data.items()})

    return subset_data, subset_indices


def compute_exact_temporal_network_solution(network: ReactionNetwork, test_state: ndarray, results_v: dict, val_config: dict) -> None:
    """Compute the exact temporal expectation for the reaction network if available."""
    # Compute the exact values for the test state:
    true_V = network.compute_exact_values(test_state, val_config["t_stop"], val_config["n_time_samples"])
    # Assign the computed values to the results dictionaries:
    results_v.update({"true_V": true_V})


def compare_nn_expectation_with_ssa_ci(inputs: dict, run: RunContext, results_dict: dict) -> None:
    """Check how much of the NN expectation estimate is within the CI of the SSA estimate."""
    # Unpack the necessary values from the results dictionary:
    full_V = results_dict.get("full_V")
    mean = results_dict.get("raw_mean")
    ci = results_dict.get("raw_ci")
    # Find last index covered by the neural network training time:
    train_idx = np.searchsorted(inputs["times"], run.RN_CONFIG["t_stop"], side="right")
    # Calculate how many predictions are within the confidence interval:
    within_ci_train = ((full_V[:train_idx+1] >= mean[:train_idx+1] - ci[:train_idx+1]) & (full_V[:train_idx+1] <= mean[:train_idx+1] + ci[:train_idx+1])).sum(axis=0)
    within_ci_full = ((full_V >= mean - ci) & (full_V <= mean + ci)).sum(axis=0)
    insert_blank_line()
    logging.info(f"'{architecture_dict[run.NN_CONFIG['v']['subnet_architecture']]}' estimates within the 'SSA' 95% CI:")
    logging.info(f"Training time total: {within_ci_train}, fraction: {within_ci_train / full_V[:train_idx+1].shape[0]}")
    if run.VAL_CONFIG["t_stop"] > run.RN_CONFIG["t_stop"]:
        logging.info(f"Testing time total: {within_ci_full}, fraction: {within_ci_full / full_V.shape[0]}")


def compare_nn_expectation_with_ssa_tolerance(inputs: dict, run: RunContext, results_dict: dict) -> None:
    """Check how much of the NN expectation estimate is within the tolerance band of the SSA estimate."""
    # Unpack the necessary values from the results dictionary:
    full_V = results_dict.get("full_V")
    mean = results_dict.get("raw_mean")
    tolerance = results_dict.get("raw_tolerance")
    # Find last index covered by the neural network training time:
    train_idx = np.searchsorted(inputs["times"], run.RN_CONFIG["t_stop"], side="right")
    # Calculate how many predictions are within the confidence interval:
    within_ci_train = ((full_V[:train_idx+1] >= mean[:train_idx+1] - tolerance[:train_idx+1]) & (full_V[:train_idx+1] <= mean[:train_idx+1] + tolerance[:train_idx+1])).sum(axis=0)
    within_ci_full = ((full_V >= mean - tolerance) & (full_V <= mean + tolerance)).sum(axis=0)
    insert_blank_line()
    logging.info(f"'{architecture_dict[run.NN_CONFIG['v']['subnet_architecture']]}' estimates within the 'SSA' ± 5% tolerance band:")
    logging.info(f"Training time total: {within_ci_train}, fraction: {within_ci_train / full_V[:train_idx+1].shape[0]}")
    if run.VAL_CONFIG["t_stop"] > run.RN_CONFIG["t_stop"]:
        logging.info(f"Testing time total: {within_ci_full}, fraction: {within_ci_full / full_V.shape[0]}")


def compute_max_variance_reduction(base_times: np.ndarray, base_var: np.ndarray, method_times: np.ndarray, method_var: np.ndarray) -> float:
    """Compute the maximum variance reduction factor over overlapping time span."""
    # Interpolate baseline to method's time grid (handle 2D arrays by interpolating each column):
    if base_var.ndim == 1:
        base_var_interp = np.interp(method_times, base_times, base_var)
    else:
        base_var_interp = np.array([np.interp(method_times, base_times, base_var[:, i]) for i in range(base_var.shape[1])]).T

    # Compute element-wise ratio, handling near-zero values:
    with np.errstate(divide="ignore", invalid="ignore"):
        reduction_factor = base_var_interp / np.maximum(method_var, 1e-16)
        # Keep only valid (non-inf, non-nan) values with reduction > 1:
        valid_mask = np.isfinite(reduction_factor) & (reduction_factor > 1)
        max_reduction = np.max(reduction_factor[valid_mask]) if valid_mask.any() else 1.0

    return float(max_reduction)


def compute_max_variance_reduction_single(base_var: np.ndarray, method_var: np.ndarray) -> float:
    """Compute the maximum variance reduction factor for single-value estimates."""
    with np.errstate(divide="ignore", invalid="ignore"):
        reduction_factor = base_var / np.maximum(method_var, 1e-16)
        valid_mask = np.isfinite(reduction_factor) & (reduction_factor > 1)
        max_reduction = np.max(reduction_factor[valid_mask]) if valid_mask.any() else 1.0

    return float(max_reduction)


def save_analysis_results(results_dict: dict, run: RunContext, test_state_index: int) -> None:
    """Save all analysis results for a test state."""
    # Combine results and metadata in a single dictionary:
    combined_data = {
        "metadata": {
            "timestamp": run.timestamp,
            "test_state_index": test_state_index,
            "available_results": list(results_dict.keys()),
            "config": {
                "RN_CONFIG": run.RN_CONFIG,
                "NN_CONFIG": run.NN_CONFIG,
                "VAL_CONFIG": run.VAL_CONFIG,
                "PLT_CONFIG": run.PLT_CONFIG
            }
        },
        "results": results_dict
    }

    filename = f"{run.results_subdir}/{run.timestamp}_{test_state_index}___results.pkl.gz"
    with gzip.open(filename, "wb") as f:
        pickle.dump(combined_data, f)

    return os.path.getsize(filename)


def analyze_test_states(test_states: list, trainer: ExpectationModelTrainer, run: RunContext) -> None:
    """Analyze the model with the given test states."""
    # Create a dictionary to store all test state figures:
    test_state_figures = {}

    for i, test_state in enumerate(test_states):
        insert_section_marker()
        logging.info(f"Running analysis for test state {test_state} ...")

        # Check if the test state is in the convergence states:
        convergence_state = True if any(np.array_equal(test_state, np.array(sub)) for sub in run.VAL_CONFIG["conv_test_states"]) else False

        # Initialize the reaction network with the current test state as initial state:
        network_class = getattr(examples, run.RN_CONFIG["reaction_network"])
        test_network: ReactionNetwork = network_class(rn_config=run.RN_CONFIG, initial_state=test_state)

        analyzer_temporal = TemporalSubnetAnalyzer(test_network, trainer.model, run.VAL_CONFIG)
        analyzer_final = FinalTimeSubnetAnalyzer(test_network, trainer.model, run.VAL_CONFIG)

        # Save the current DeepIS estimate status to reset it later (because there might be a timeout setting it to False):
        original_is_estimate_status = run.VAL_CONFIG["compute_is_estimate"]

        ert = {}  # expectation results temporal
        erf = {}  # expectation results final

        conv_ssa_repeat_data = None
        conv_ssa_repeat_results = None
        conv_cv_repeat_results = None
        conv_is_repeat_results = None

        if run.RN_CONFIG["exact_values_computable"]:
            insert_blank_line()
            # Compute the exact solution if available (depends on the reaction network type):
            logging.info("Computing temporal 'Exact' reaction network solution ...")
            compute_exact_temporal_network_solution(test_network, test_state, ert, run.VAL_CONFIG)
            logging.info(f"Final time 'Exact' results for V: {ert.get('true_V')[-1]}")

            if convergence_state == True:
                if run.VAL_CONFIG["conv_ref_trajectory_type"] == "Exact":
                    insert_blank_line()
                    # Directly use the exact solution as a reference for the convergence plot:
                    logging.info(f"Using 'Exact' solution as reference at t={run.VAL_CONFIG['conv_t_stop']} ...")
                    ert["ref_true_mean"] = test_network.compute_exact_values(test_state, run.VAL_CONFIG["conv_t_stop"], run.VAL_CONFIG["conv_n_time_samples"])
                    erf["ref_true_mean"] = ert.get("ref_true_mean")[-1]
                    logging.info(f"Reference final time 'Exact' solution: {erf.get('ref_true_mean')} (95% CI)")

        if convergence_state == True:
            if run.VAL_CONFIG["conv_ref_trajectory_type"] == "SSA":
                insert_blank_line()
                # Simulate almost infinitely many final time SSA estimates as a reference for the convergence plot:
                log_without_linebreak(f"Generating reference 'SSA' estimate using {run.VAL_CONFIG['conv_ref_n_trajectories']} trajectories at t={run.VAL_CONFIG['conv_t_stop']} ...")
                conv_ref_ssa_generator = ExpectationGeneratorSSA(test_network, run.VAL_CONFIG["analysis_seed"])
                conv_ref_valid_data = conv_ref_ssa_generator.sample_final_rtc_trajectories(run.VAL_CONFIG["conv_t_stop"], run.VAL_CONFIG["conv_ref_n_trajectories"])
                analyzer_final.compute_reference_ssa_estimate(conv_ref_valid_data, erf)
                logging.info(f"Reference final time 'SSA' mean: {format_mean_ci_arrays(erf.get('ref_raw_mean'), erf.get('ref_raw_ci'))} (95% CI)")
                logging.info(f"Reference final time 'SSA' variance: {erf.get('ref_raw_var')}")
                logging.info(f"Reference final time 'SSA' mean squared error: {erf.get('ref_raw_var') / erf.get('ref_raw_samples')}")
                logging.info(f"Reference final time 'SSA' coefficient of variation: {erf.get('ref_raw_cv')}")

            insert_blank_line()
            # Create new (smaller) validation data set for the current test state and compute SSA estimate:
            log_without_linebreak(f"Generating convergence 'SSA' estimate using {run.VAL_CONFIG['conv_n_trajectories']} trajectories at t={run.VAL_CONFIG['conv_t_stop']} ...")
            conv_ssa_generator = ExpectationGeneratorSSA(test_network, run.VAL_CONFIG["analysis_seed"])
            conv_valid_data = conv_ssa_generator.sample_temporal_rtc_trajectories(run.VAL_CONFIG["conv_t_stop"], run.VAL_CONFIG["conv_n_time_samples"], run.VAL_CONFIG["conv_n_trajectories"], run.validation_trajectories)
            analyzer_final.compute_final_ssa_estimate(conv_valid_data, erf)
            logging.info(f"Convergence final time 'SSA' mean: {format_mean_ci_arrays(erf.get('raw_mean'), erf.get('raw_ci'))} (95% CI)")
            logging.info(f"Convergence final time 'SSA' variance: {erf.get('raw_var')}")
            logging.info(f"Convergence final time 'SSA' mean squared error: {erf.get('raw_var') / erf.get('raw_samples')}")
            logging.info(f"Convergence final time 'SSA' coefficient of variation: {erf.get('raw_cv')}")

            if run.VAL_CONFIG["conv_n_repeats"] > 1:
                insert_blank_line()
                # Repeat the convergence SSA estimate several times to get a confidence interval for the convergence plot:
                logging.info(f"Repeating the convergence 'SSA' estimate {run.VAL_CONFIG['conv_n_repeats']} times to get a confidence interval ...")
                conv_ssa_repeat_data, conv_ssa_repeat_results = run_convergence_ssa_repeats(conv_ssa_generator, run.VAL_CONFIG["conv_t_stop"], run.VAL_CONFIG["conv_n_time_samples"], run.VAL_CONFIG["conv_n_trajectories"], run.validation_trajectories, analyzer_final, run.VAL_CONFIG["conv_n_repeats"], first_data=conv_valid_data)

        insert_blank_line()
        # Create new (smaller) validation data set for the current test state and compute SSA estimate:
        log_without_linebreak(f"Generating temporal 'SSA' estimate using {run.VAL_CONFIG['n_trajectories']} trajectories and {run.VAL_CONFIG['n_time_samples']} time points ...")
        ssa_generator = ExpectationGeneratorSSA(test_network, run.VAL_CONFIG["analysis_seed"])
        valid_data = ssa_generator.sample_temporal_rtc_trajectories(run.VAL_CONFIG["t_stop"], run.VAL_CONFIG["n_time_samples"], run.VAL_CONFIG["n_trajectories"], run.validation_trajectories)
        analyzer_temporal.compute_temporal_ssa_estimate(valid_data, ert)
        logging.info(f"Final time 'SSA' mean: {format_mean_ci_arrays(ert.get('raw_mean')[-1], ert.get('raw_ci')[-1])} (95% CI)")
        logging.info(f"Final time 'SSA' variance: {ert.get('raw_var')[-1]}")
        logging.info(f"Final time 'SSA' mean squared error: {ert.get('raw_var')[-1] / ert.get('raw_samples')}")
        logging.info(f"Final time 'SSA' coefficient of variation: {ert.get('raw_cv')[-1]}")
        logging.info(f"Mean 'SSA' variance over time: {np.mean(ert.get('raw_var'), axis=0)}")

        # Initialize results dictionary for saving later:
        analysis_results = {
            "test_state": test_state.copy(),
            "valid_data": {"times": valid_data["times"].copy()}
        }

        insert_subsection_marker()
        # Compute direct expectation estimate coming from the NN model:
        log_without_linebreak(f"Computing temporal '{architecture_dict[run.NN_CONFIG['v']['subnet_architecture']]}' expectation estimate using {run.VAL_CONFIG['n_time_samples']} time points ...")
        analyzer_temporal.compute_temporal_nn_expectation(valid_data, test_state, ert)
        logging.info(f"Final time '{architecture_dict[run.NN_CONFIG['v']['subnet_architecture']]}' estimate: {ert.get('full_V')[-1]}")
        # Plot and save temporal evolution of V:
        expectation_plots.plot_temporal_trajectories(test_state, valid_data, run, i, ert, ["SSA"], test_state_figures)
        expectation_plots.plot_temporal_expectation(test_state, valid_data, run, i, ert, ["SSA", "NN"], test_state_figures)
        expectation_plots.plot_temporal_expectation(test_state, valid_data, run, i, ert, ["SSAtol", "NN"], test_state_figures)

        # Calculate how many expectation predictions are within the SSA tolerance band:
        compare_nn_expectation_with_ssa_tolerance(valid_data, run, ert)

        if run.VAL_CONFIG["compute_cv_estimate"]:
            insert_subsection_marker()
            if convergence_state == True:
                if run.VAL_CONFIG["conv_ref_trajectory_type"] == "SSA+DeepCV":
                    # Simulate almost infinitely many final time SSA estimates to use for the DeepCV estimate for the convergence plot:
                    if run.VAL_CONFIG["conv_use_exact_cv"]:
                        log_without_linebreak(f"Generating reference 'SSA with DeepCV' estimate using {run.VAL_CONFIG['conv_ref_n_trajectories']} trajectories at t={run.VAL_CONFIG['conv_t_stop']} ...")
                        analyzer_final.generate_reference_exact_deep_cv_estimate(run.VAL_CONFIG, erf)
                    else:
                        log_without_linebreak(f"Generating 'SSA' trajectories for the reference 'SSA with DeepCV' estimate using {run.VAL_CONFIG['conv_ref_n_trajectories']} trajectories at t={run.VAL_CONFIG['conv_t_stop']} ...")
                        conv_ref_ssa_generator = ExpectationGeneratorSSA(test_network, run.VAL_CONFIG["analysis_seed"])
                        conv_ref_valid_data = conv_ref_ssa_generator.sample_temporal_rtc_trajectories(run.VAL_CONFIG["conv_t_stop"], run.VAL_CONFIG["conv_n_time_samples"], run.VAL_CONFIG["conv_ref_n_trajectories"], run.validation_trajectories)
                        log_without_linebreak(f"Computing reference 'SSA with DeepCV' estimate using {run.VAL_CONFIG['conv_ref_n_trajectories']} trajectories at t={run.VAL_CONFIG['conv_t_stop']} ...")
                        analyzer_final.compute_final_deep_cv_estimate(conv_ref_valid_data, erf, reference=True)
                    logging.info(f"Reference final time 'SSA with DeepCV' mean: {format_mean_ci_arrays(erf.get('ref_diff_mean'), erf.get('ref_diff_ci'))} (95% CI)")
                    logging.info(f"Reference final time 'SSA with DeepCV' variance: {erf.get('ref_diff_var')}")
                    logging.info(f"Reference final time 'SSA with DeepCV' mean squared error: {erf.get('ref_diff_var') / erf.get('ref_diff_samples')}")
                    logging.info(f"Reference final time 'SSA with DeepCV' coefficient of variation: {erf.get('ref_diff_cv')}")
                    insert_blank_line()

                if run.VAL_CONFIG["conv_use_exact_cv"]:
                    log_without_linebreak(f"Generating convergence 'SSA with DeepCV' expectation estimate for {int(run.VAL_CONFIG['conv_n_trajectories'] * run.VAL_CONFIG['conv_ssa_with_cv_fraction'])} trajectories at t={run.VAL_CONFIG['conv_t_stop']} ...")
                    conv_exact_cv_generator = analyzer_final.generate_final_exact_deep_cv_estimate(run.VAL_CONFIG, erf)
                else:
                    # Compute final time DeepCV expectation estimate with trajectory subset:
                    final_ssa_with_cv_subset, final_ssa_with_cv_indices = select_trajectory_subset(conv_valid_data, run.VAL_CONFIG["analysis_seed"], run.VAL_CONFIG["conv_ssa_with_cv_fraction"])
                    log_without_linebreak(f"Computing convergence 'SSA with DeepCV' expectation estimate for {len(final_ssa_with_cv_indices)} trajectories at t={run.VAL_CONFIG['conv_t_stop']} ...")
                    analyzer_final.compute_final_deep_cv_estimate(final_ssa_with_cv_subset, erf)
                logging.info(f"Convergence final time 'SSA with DeepCV' mean: {format_mean_ci_arrays(erf.get('diff_mean'), erf.get('diff_ci'))} (95% CI)")
                logging.info(f"Convergence final time 'SSA with DeepCV' variance: {erf.get('diff_var')}")
                logging.info(f"Convergence final time 'SSA with DeepCV' mean squared error: {erf.get('diff_var') / erf.get('diff_samples')}")
                logging.info(f"Convergence final time 'SSA with DeepCV' coefficient of variation: {erf.get('diff_cv')}")
                logging.info(f"Convergence final time 'SSA with DeepCV' control variates (should be close to zero): {erf.get('diff_adjustment_mean')}")

                if run.VAL_CONFIG["conv_n_repeats"] > 1:
                    # Repeat the convergence SSA estimate several times to get a confidence interval for the convergence plot:
                    insert_blank_line()
                    logging.info(f"Repeating the convergence 'SSA with DeepCV' estimate {run.VAL_CONFIG['conv_n_repeats']} times to get a confidence interval ...")
                    if run.VAL_CONFIG["conv_use_exact_cv"]:
                        conv_exact_cv_first_results = {"diff_mean": erf.get("diff_mean").copy(), "diff_var": erf.get("diff_var").copy(), "diff_cv": erf.get("diff_cv").copy(), "diff_ci": erf.get("diff_ci").copy(), "diff_tolerance": erf.get("diff_tolerance").copy(), "diff_adjustment_mean": erf.get("diff_adjustment_mean").copy(),"diff_samples": erf.get("diff_samples"), "diff_var_ci_lower": erf.get("diff_var_ci_lower").copy(), "diff_var_ci_upper": erf.get("diff_var_ci_upper").copy(), "diff_cummean": erf.get("diff_cummean").copy()}
                        conv_cv_repeat_results = run_convergence_exact_deep_cv_repeats(analyzer_final, conv_exact_cv_generator, run.VAL_CONFIG, run.VAL_CONFIG["conv_n_repeats"], first_results=conv_exact_cv_first_results)
                    else:
                        conv_cv_repeat_results = run_convergence_deep_cv_repeats(conv_ssa_repeat_data, analyzer_final)

                insert_blank_line()

            # Compute the temporal DeepCV expectation estimate with the trajectory subset:
            ssa_with_cv_subset, ssa_with_cv_indices = select_trajectory_subset(valid_data, run.VAL_CONFIG["analysis_seed"], run.VAL_CONFIG["ssa_with_cv_fraction"])
            log_without_linebreak(f"Computing temporal (and output-suboptimal) 'SSA with DeepCV' expectation estimate for {len(ssa_with_cv_indices)} time courses and {run.VAL_CONFIG['n_time_samples']} time points ...")
            analyzer_temporal.compute_temporal_deep_cv_estimate(ssa_with_cv_subset, ert)  # expectation data subset
            logging.info(f"Final time 'SSA with DeepCV' mean: {format_mean_ci_arrays(ert.get('diff_mean')[-1], ert.get('diff_ci')[-1])} (95% CI)")
            logging.info(f"Final time 'SSA with DeepCV' variance: {ert.get('diff_var')[-1]}")
            logging.info(f"Final time 'SSA with DeepCV' mean squared error: {ert.get('diff_var')[-1] / ert.get('diff_samples')}")
            logging.info(f"Final time 'SSA with DeepCV' coefficient of variation: {ert.get('diff_cv')[-1]}")
            logging.info(f"Final time 'SSA with DeepCV' control variates (should be close to zero): {ert.get('diff_adjustment_mean')[-1]}")
            logging.info(f"Mean optimal 'SSA with DeepCV' variance over time: {np.mean(ert.get('diff_var'), axis=0)}")
            reduction_cv = compute_max_variance_reduction(valid_data["times"][1:], ert.get("raw_var"), valid_data["times"][1:], ert.get("diff_var"))
            logging.info(f"Maximum variance reduction using 'SSA with DeepCV' (compared to 'SSA'): {reduction_cv:.2f}x ({np.log10(reduction_cv):.2f} orders of magnitude)")
            # Plot and save temporal evolution of the DeepCV estimate:
            expectation_plots.plot_temporal_expectation(test_state, valid_data, run, i, ert, ["SSA", "SSA+DeepCV"], test_state_figures)
            expectation_plots.plot_temporal_expectation(test_state, valid_data, run, i, ert, ["SSAtol", "SSA+DeepCV"], test_state_figures)
            expectation_plots.plot_temporal_expectation_variance(test_state, valid_data, run, i, ert, ["SSA", "SSA+DeepCV"], test_state_figures)

            insert_blank_line()
            # Log and plot the computed output-suboptimal temporal DeepCV expectation estimate with the trajectory subset:
            logging.info(f"Final time output-suboptimal 'SSA with DeepCV' mean: {format_mean_ci_arrays(ert.get('diff_out_suboptimal_mean')[-1], ert.get('diff_out_suboptimal_ci')[-1])} (95% CI)")
            logging.info(f"Final time output-suboptimal 'SSA with DeepCV' variance: {ert.get('diff_out_suboptimal_var')[-1]}")
            logging.info(f"Final time output-suboptimal 'SSA with DeepCV' mean squared error: {ert.get('diff_out_suboptimal_var')[-1] / ert.get('diff_out_suboptimal_samples')}")
            logging.info(f"Final time output-suboptimal 'SSA with DeepCV' coefficient of variation: {ert.get('diff_out_suboptimal_cv')[-1]}")
            logging.info(f"Final time output-suboptimal 'SSA with DeepCV' control variates (should be close to zero): {ert.get('diff_out_suboptimal_adjustment_mean')[-1]}")
            logging.info(f"Mean output-suboptimal 'SSA with DeepCV' variance over time: {np.mean(ert.get('diff_out_suboptimal_var'), axis=0)}")
            # Plot and save temporal evolution of the DeepCV estimate:
            expectation_plots.plot_temporal_expectation(test_state, valid_data, run, i, ert, ["SSA", "SSA+osubDeepCV"], test_state_figures)
            expectation_plots.plot_temporal_expectation(test_state, valid_data, run, i, ert, ["SSAtol", "SSA+osubDeepCV"], test_state_figures)
            expectation_plots.plot_temporal_expectation_variance(test_state, valid_data, run, i, ert, ["SSA", "SSA+osubDeepCV"], test_state_figures)

            insert_blank_line()
            # Compute the time-suboptimal temporal DeepCV expectation estimate with the trajectory subset:
            ssa_with_cv_subset, ssa_with_cv_indices = select_trajectory_subset(valid_data, run.VAL_CONFIG["analysis_seed"], run.VAL_CONFIG["ssa_with_cv_fraction"])
            log_without_linebreak(f"Computing time-suboptimal temporal 'SSA with DeepCV' expectation estimate for {len(ssa_with_cv_indices)} trajectories and {run.VAL_CONFIG['n_time_samples']} time points ...")
            analyzer_temporal.compute_time_suboptimal_deep_cv_estimate(ssa_with_cv_subset, run.VAL_CONFIG["t_stop"], ert)  # expectation data subset
            logging.info(f"Final time time-suboptimal 'SSA with DeepCV' mean: {format_mean_ci_arrays(ert.get('diff_time_suboptimal_mean')[-1], ert.get('diff_time_suboptimal_ci')[-1])} (95% CI)")
            logging.info(f"Final time time-suboptimal 'SSA with DeepCV' variance: {ert.get('diff_time_suboptimal_var')[-1]}")
            logging.info(f"Final time time-suboptimal 'SSA with DeepCV' mean squared error: {ert.get('diff_time_suboptimal_var')[-1] / ert.get('diff_time_suboptimal_samples')}")
            logging.info(f"Final time time-suboptimal 'SSA with DeepCV' coefficient of variation: {ert.get('diff_time_suboptimal_cv')[-1]}")
            logging.info(f"Final time time-suboptimal 'SSA with DeepCV' control variates (should be close to zero): {ert.get('diff_time_suboptimal_adjustment_mean')[-1]}")
            logging.info(f"Mean time-suboptimal 'SSA with DeepCV' variance over time: {np.mean(ert.get('diff_time_suboptimal_var'), axis=0)}")
            # Plot and save the temporal evolution of some DeepCV trajectories:
            expectation_plots.plot_temporal_trajectories(test_state, valid_data, run, i, ert, ["SSA+tsubDeepCV"], test_state_figures)
            expectation_plots.plot_temporal_trajectories(test_state, valid_data, run, i, ert, ["SSA", "SSA+tsubDeepCV"], test_state_figures)
            # Plot and save temporal evolution of the DeepCV estimate:
            expectation_plots.plot_temporal_expectation(test_state, valid_data, run, i, ert, ["SSA", "SSA+tsubDeepCV"], test_state_figures)
            expectation_plots.plot_temporal_expectation(test_state, valid_data, run, i, ert, ["SSAtol", "SSA+tsubDeepCV"], test_state_figures)
            expectation_plots.plot_temporal_expectation_variance(test_state, valid_data, run, i, ert, ["SSA", "SSA+tsubDeepCV"], test_state_figures)

        if run.VAL_CONFIG["compute_is_estimate"]:
            insert_subsection_marker()
            if convergence_state == True:
                if run.VAL_CONFIG["conv_ref_trajectory_type"] == "SSA+DeepIS":
                    # Compute reference importance sampling (IS) estimate for the expectation:
                    log_without_linebreak(f"Generating reference 'SSA with DeepIS' expectation estimate using {int(run.VAL_CONFIG['conv_ref_n_trajectories'])} trajectories per output function at t={run.VAL_CONFIG['conv_t_stop']} ...")
                    analyzer_final.generate_reference_deep_is_estimate(run.VAL_CONFIG, run.validation_trajectories, erf)
                    if run.VAL_CONFIG["compute_is_estimate"] == True:
                        logging.info(f"Reference final time 'SSA with DeepIS' mean: {format_mean_ci_arrays(erf.get('ref_is_raw_mean'), erf.get('ref_is_raw_ci'))} (95% CI)")
                        logging.info(f"Reference final time 'SSA with DeepIS' variance: {erf.get('ref_is_raw_var')}")
                        logging.info(f"Reference final time 'SSA with DeepIS' mean squared error: {erf.get('ref_is_raw_var') / erf.get('ref_is_raw_samples')}")
                        logging.info(f"Reference final time 'SSA with DeepIS' coefficient of variation: {erf.get('ref_is_raw_cv')}")
                    insert_blank_line()

                # Compute final time importance sampling (IS) estimate for the expectation:
                log_without_linebreak(f"Generating convergence 'SSA with DeepIS' expectation estimate using {int(run.VAL_CONFIG['conv_n_trajectories'] * run.VAL_CONFIG['conv_ssa_with_is_fraction'])} trajectories per output function at t={run.VAL_CONFIG['conv_t_stop']} ...")
                conv_is_generator = analyzer_final.generate_final_deep_is_estimate(run.VAL_CONFIG, run.validation_trajectories, erf)
                if run.VAL_CONFIG["compute_is_estimate"] == True:
                    logging.info(f"Convergence final time 'SSA with DeepIS' mean: {format_mean_ci_arrays(erf.get('is_raw_mean'), erf.get('is_raw_ci'))} (95% CI)")
                    logging.info(f"Convergence final time 'SSA with DeepIS' variance: {erf.get('is_raw_var')}")
                    logging.info(f"Convergence final time 'SSA with DeepIS' mean squared error: {erf.get('is_raw_var') / erf.get('is_raw_samples')}")
                    logging.info(f"Convergence final time 'SSA with DeepIS' coefficient of variation: {erf.get('is_raw_cv')}")

                    if run.VAL_CONFIG["conv_n_repeats"] > 1:
                        insert_blank_line()
                        # Copy the relevant results from the results dictionary to reuse them in the DeepIS repeats:
                        conv_is_first_results = {"is_raw_mean": erf.get("is_raw_mean").copy(), "is_raw_var": erf.get("is_raw_var").copy(), "is_raw_cv": erf.get("is_raw_cv").copy(), "is_raw_ci": erf.get("is_raw_ci").copy(), "is_raw_samples": erf.get("is_raw_samples"), "is_raw_var_ci_lower": erf.get("is_raw_var_ci_lower").copy(), "is_raw_var_ci_upper": erf.get("is_raw_var_ci_upper").copy(), "is_raw_cummean": erf.get("is_raw_cummean").copy()}
                        # Repeat the convergence DeepIS estimate several times to get a confidence interval for the convergence plot:
                        logging.info(f"Repeating the convergence 'SSA with DeepIS' estimate {run.VAL_CONFIG['conv_n_repeats']} times to get a confidence interval ...")
                        conv_is_repeat_results = run_convergence_deep_is_repeats(analyzer_final, conv_is_generator, run.VAL_CONFIG, run.validation_trajectories, run.VAL_CONFIG["conv_n_repeats"], first_results=conv_is_first_results)

                insert_blank_line()

            # Compute importance sampling (IS) estimate for the expectation:
            log_without_linebreak(f"Generating temporal (and output-suboptimal) 'SSA with DeepIS' expectation estimate using {int(run.VAL_CONFIG['n_trajectories'] * run.VAL_CONFIG['ssa_with_is_fraction'])} time courses per output function and {run.VAL_CONFIG['n_is_samples']} time points ...")
            analyzer_temporal.generate_temporal_deep_is_estimate(run.VAL_CONFIG, ert)
            # Check if the DeepIS estimate took too long to compute (then it would be set to False):
            if run.VAL_CONFIG["compute_is_estimate"] == True:
                logging.info(f"Final time 'SSA with DeepIS' mean: {format_mean_ci_arrays(ert.get('is_raw_mean')[-1], ert.get('is_raw_ci')[-1])} (95% CI)")
                logging.info(f"Final time 'SSA with DeepIS' variance: {ert.get('is_raw_var')[-1]}")
                logging.info(f"Final time 'SSA with DeepIS' mean squared error: {ert.get('is_raw_var')[-1] / ert.get('is_raw_samples')}")
                logging.info(f"Final time 'SSA with DeepIS' coefficient of variation: {ert.get('is_raw_cv')[-1]}")
                logging.info(f"Mean optimal 'SSA with DeepIS' variance over time: {np.mean(ert.get('is_raw_var'), axis=0)}")
                reduction_is = compute_max_variance_reduction(valid_data["times"][1:], ert.get("raw_var"), ert.get("is_times")[1:], ert.get("is_raw_var"))
                logging.info(f"Maximum variance reduction using 'SSA with DeepIS' (compared to 'SSA'): {reduction_is:.2f}x ({np.log10(reduction_is):.2f} orders of magnitude)")
                # Plot and save temporal evolution of the DeepIS estimate:
                expectation_plots.plot_temporal_expectation(test_state, valid_data, run, i, ert, ["SSA", "SSA+DeepIS"], test_state_figures)
                expectation_plots.plot_temporal_expectation(test_state, valid_data, run, i, ert, ["SSAtol", "SSA+DeepIS"], test_state_figures)
                expectation_plots.plot_temporal_expectation_variance(test_state, valid_data, run, i, ert, ["SSA", "SSA+DeepIS"], test_state_figures)

                insert_blank_line()
                # Log and plot the computed output-suboptimal importance sampling (IS) estimate for the expectation:
                logging.info(f"Final time output-suboptimal 'SSA with DeepIS' mean: {format_mean_ci_arrays(ert.get('is_raw_out_suboptimal_mean')[-1], ert.get('is_raw_out_suboptimal_ci')[-1])} (95% CI)")
                logging.info(f"Final time output-suboptimal 'SSA with DeepIS' variance: {ert.get('is_raw_out_suboptimal_var')[-1]}")
                logging.info(f"Final time output-suboptimal 'SSA with DeepIS' mean squared error: {ert.get('is_raw_out_suboptimal_var')[-1] / ert.get('is_raw_out_suboptimal_samples')}")
                logging.info(f"Final time output-suboptimal 'SSA with DeepIS' coefficient of variation: {ert.get('is_raw_out_suboptimal_cv')[-1]}")
                logging.info(f"Mean output-suboptimal 'SSA with DeepIS' variance over time: {np.mean(ert.get('is_raw_out_suboptimal_var'), axis=0)}")
                # Plot and save temporal evolution of the DeepIS estimate:
                expectation_plots.plot_temporal_expectation(test_state, valid_data, run, i, ert, ["SSA", "SSA+osubDeepIS"], test_state_figures)
                expectation_plots.plot_temporal_expectation(test_state, valid_data, run, i, ert, ["SSAtol", "SSA+osubDeepIS"], test_state_figures)
                expectation_plots.plot_temporal_expectation_variance(test_state, valid_data, run, i, ert, ["SSA", "SSA+osubIS"], test_state_figures)

            insert_blank_line()
            # Compute time-suboptimal importance sampling (IS) estimate for the expectation:
            log_without_linebreak(f"Generating time-suboptimal temporal 'SSA with DeepIS' expectation estimate using {int(run.VAL_CONFIG['n_trajectories'] * run.VAL_CONFIG['ssa_with_is_fraction'])} trajectories per output function and {run.VAL_CONFIG['n_time_samples']} time points ...")
            analyzer_temporal.generate_time_suboptimal_deep_is_estimate(run.VAL_CONFIG, run.validation_trajectories, ert)
            # Check if the DeepIS estimate took too long to compute (then it would be set to False):
            if run.VAL_CONFIG["compute_is_estimate"] == True:
                logging.info(f"Final time time-suboptimal 'SSA with DeepIS' mean: {format_mean_ci_arrays(ert.get('is_raw_time_suboptimal_mean')[-1], ert.get('is_raw_time_suboptimal_ci')[-1])} (95% CI)")
                logging.info(f"Final time time-suboptimal 'SSA with DeepIS' variance: {ert.get('is_raw_time_suboptimal_var')[-1]}")
                logging.info(f"Final time time-suboptimal 'SSA with DeepIS' mean squared error: {ert.get('is_raw_time_suboptimal_var')[-1] / ert.get('is_raw_time_suboptimal_samples')}")
                logging.info(f"Final time time-suboptimal 'SSA with DeepIS' coefficient of variation: {ert.get('is_raw_time_suboptimal_cv')[-1]}")
                logging.info(f"Mean time-suboptimal 'SSA with DeepIS' variance over time: {np.mean(ert.get('is_raw_time_suboptimal_var'), axis=0)}")
                # Plot and save temporal evolution of some DeepIS trajectories:
                expectation_plots.plot_temporal_trajectories(test_state, valid_data, run, i, ert, ["SSA+tsubDeepIS"], test_state_figures)
                expectation_plots.plot_temporal_trajectories(test_state, valid_data, run, i, ert, ["SSA", "SSA+tsubDeepIS"], test_state_figures)
                # Plot and save temporal evolution of the DeepIS estimate:
                expectation_plots.plot_temporal_expectation(test_state, valid_data, run, i, ert, ["SSA", "SSA+tsubDeepIS"], test_state_figures)
                expectation_plots.plot_temporal_expectation(test_state, valid_data, run, i, ert, ["SSAtol", "SSA+tsubDeepIS"], test_state_figures)
                expectation_plots.plot_temporal_expectation_variance(test_state, valid_data, run, i, ert, ["SSA", "SSA+tsubDeepIS"], test_state_figures)

        if run.VAL_CONFIG["compute_cv_estimate"] and run.VAL_CONFIG["compute_is_estimate"]:
            expectation_plots.plot_temporal_trajectories(test_state, valid_data, run, i, ert, ["SSA", "SSA+tsubDeepCV", "SSA+tsubDeepIS"], test_state_figures)
            expectation_plots.plot_temporal_expectation(test_state, valid_data, run, i, ert, ["SSA", "SSA+DeepCV", "SSA+DeepIS"], test_state_figures)
            expectation_plots.plot_temporal_expectation(test_state, valid_data, run, i, ert, ["SSAtol", "SSA+DeepCV", "SSA+DeepIS"], test_state_figures)
            expectation_plots.plot_temporal_expectation(test_state, valid_data, run, i, ert, ["SSA+DeepCV", "SSA+DeepIS"], test_state_figures)
            expectation_plots.plot_temporal_expectation_variance(test_state, valid_data, run, i, ert, ["SSA", "SSA+DeepCV", "SSA+DeepIS"], test_state_figures)
            expectation_plots.plot_temporal_expectation_variance(test_state, valid_data, run, i, ert, ["SSA+DeepCV", "SSA+DeepIS"], test_state_figures)

        if convergence_state and run.PLT_CONFIG["plot_estimate_convergence"]:
            # Plot the various estimates for a growing number of samples:
            expectation_plots.plot_expectation_error_convergence(test_state, run, i, erf, ["SSA", "SSA+DeepCV", "SSA+DeepIS"], test_state_figures)
            if run.VAL_CONFIG["conv_n_repeats"] > 1:
                expectation_plots.plot_expectation_error_convergence_repeats(test_state, run, i, erf, conv_ssa_repeat_results, conv_cv_repeat_results, conv_is_repeat_results, ["SSA", "SSA+DeepCV", "SSA+DeepIS"], test_state_figures)

        if run.VAL_CONFIG["compute_deep_var_estimates"]:
            insert_subsection_marker()
            # Compute DeepPVA variance approximation:
            deep_pva_var_subset, deep_pva_var_indices = select_trajectory_subset(valid_data, run.VAL_CONFIG["analysis_seed"], run.VAL_CONFIG["deep_pva_var_fraction"])
            log_without_linebreak(f"Computing temporal 'DeepPVA' variance estimate for {len(deep_pva_var_indices)} trajectories ...")
            analyzer_temporal.compute_temporal_deep_pva_variance(deep_pva_var_subset, ert)
            logging.info(f"Final time 'DeepPVA' mean: {ert.get('ter_mean')[-1]}")
            logging.info(f"Final time 'DeepPVA' variance: {ert.get('ter_var')[-1]}")
            logging.info(f"Final time 'DeepPVA' mean squared error: {ert.get('ter_var')[-1] / ert.get('ter_samples')}")
            logging.info(f"Final time 'DeepPVA' coefficient of variation: {ert.get('ter_cv')[-1]}")

            insert_blank_line()
            # Compute DeepIPA variance approximation:
            deep_ipa_var_subset, deep_ipa_var_indices = select_trajectory_subset(valid_data, run.VAL_CONFIG["analysis_seed"], run.VAL_CONFIG["deep_ipa_var_fraction"])
            log_without_linebreak(f"Computing temporal 'DeepIPA' variance estimate for {len(deep_ipa_var_indices)} trajectories ...")
            analyzer_temporal.compute_temporal_deep_ipa_variance(deep_ipa_var_subset, ert)
            logging.info(f"Final time 'DeepIPA' mean: {ert.get('int_mean')[-1]}")
            logging.info(f"Final time 'DeepIPA' variance: {ert.get('int_var')[-1]}")
            logging.info(f"Final time 'DeepIPA' mean squared error: {ert.get('int_var')[-1] / ert.get('int_samples')}")
            logging.info(f"Final time 'DeepIPA' coefficient of variation: {ert.get('int_cv')[-1]}")

            # Plot and save temporal evolution of the variance estimates:
            expectation_plots.plot_temporal_expectation_variance(test_state, valid_data, run, i, ert, ["SSA", "DeepPVA"], test_state_figures)
            expectation_plots.plot_temporal_expectation_variance(test_state, valid_data, run, i, ert, ["SSA", "DeepIPA"], test_state_figures)

        # Combine the plots for all output functions and the given test state:
        panels.plot_temporal_column(test_state_figures, run, i)

        insert_subsection_marker()
        inf_dict = {}
        # Evaluate the model at a large time point and log the result:
        log_without_linebreak(f"Computing '{architecture_dict[run.NN_CONFIG['v']['subnet_architecture']]}' expectation estimate at a large final time point ...")
        analyzer_temporal.compute_temporal_nn_expectation({"times": np.array([10000, 10000])}, test_state, inf_dict)
        logging.info(f"Final time '{architecture_dict[run.NN_CONFIG['v']['subnet_architecture']]}' estimate at t=10000: {inf_dict.get('full_V')}")

        # Compute and plot additional curves if the refined spectral decomposition was used:
        if run.NN_CONFIG["v"]["subnet_architecture"] in ["spectral_matched_simplified", "spectral_matched_full", "spectral_complex"]:
            sd_dict = {}  # new dictionary for spectral decomposition results

            if run.VAL_CONFIG["em_validation_needed"]:
                insert_blank_line()
                # Compute the ergodic mean:
                log_without_linebreak(f"Computing the 'Ergodic mean' for {run.VAL_CONFIG['erg_n_trajectories']} trajectories over the time interval [{run.NN_CONFIG['v']['erg_t_min']}, {run.NN_CONFIG['v']['erg_t_max']}] ...")
                generator_em = ExpectationGeneratorEM(test_network, run.VAL_CONFIG["analysis_seed"])
                generator_em.get_ergodic_mean_statistics(run.NN_CONFIG["v"]["erg_t_min"], run.NN_CONFIG["v"]["erg_t_max"], run.NN_CONFIG["v"]["erg_n_trajectories"], "raw", sd_dict)
                logging.info(f"Average ergodic mean: {format_mean_ci_arrays(sd_dict.get('raw_em_mean'), sd_dict.get('raw_em_ci'))} (95% CI)")
                # Compute the ergodic mean for a shorter time frame:
                log_without_linebreak(f"Computing the 'Ergodic mean' for {run.VAL_CONFIG['erg_n_trajectories']} trajectories over the time interval [{run.VAL_CONFIG['erg_t_min']}, {run.VAL_CONFIG['erg_t_max']}] ...")
                generator_em = ExpectationGeneratorEM(test_network, run.VAL_CONFIG["analysis_seed"])
                generator_em.get_ergodic_mean_statistics(run.VAL_CONFIG["erg_t_min"], run.VAL_CONFIG["erg_t_max"], run.VAL_CONFIG["erg_n_trajectories"], "raw_small", sd_dict)
                logging.info(f"Average ergodic mean: {format_mean_ci_arrays(sd_dict.get('raw_small_em_mean'), sd_dict.get('raw_small_em_ci'))} (95% CI)")

                insert_blank_line()
                # Compute the ergodic mean using the Poisson equation:
                log_without_linebreak(f"Computing the 'Ergodic mean with DeepCV' for {run.VAL_CONFIG['erg_n_trajectories']} trajectories over the time interval [{run.VAL_CONFIG['erg_t_min']}, {run.VAL_CONFIG['erg_t_max']}] ...")
                generator_poisson = ExpectationGeneratorPoissonEM(test_network, run.VAL_CONFIG["analysis_seed"], trainer.model, run.VAL_CONFIG, run.NN_CONFIG)
                generator_poisson.get_ergodic_mean_statistics(run.VAL_CONFIG["erg_t_min"], run.VAL_CONFIG["erg_t_max"], run.VAL_CONFIG["erg_n_trajectories"], "poiss", sd_dict)
                logging.info(f"Average ergodic mean: {format_mean_ci_arrays(sd_dict.get('poiss_em_mean'), sd_dict.get('poiss_em_ci'))} (95% CI)")
                max_em_reduction = compute_max_variance_reduction_single(np.array(sd_dict.get('raw_small_em_var')), np.array(sd_dict.get('poiss_em_var')))
                logging.info(f"Maximum variance reduction using 'EM with DeepCV' (compared to 'EM'): {max_em_reduction:.2f}x ({np.log10(max_em_reduction):.2f} orders of magnitude)")
                # Plot the ergodic mean comparison:
                expectation_plots.plot_ergodic_means(test_state, run, i, sd_dict)
                expectation_plots.plot_ergodic_means_variance(test_state, run, i, sd_dict)

            insert_subsection_marker()
            analyzer_spectral = SpectralSubnetAnalyzer(test_network, trainer.model, run.VAL_CONFIG, run.NN_CONFIG)

            # Evaluate function coordinate at the test state:
            log_without_linebreak(f"Analyzing the function coordinate learned with the {run.NN_CONFIG['v']['subnet_architecture'].split('_', 1)[1]} spectral decomposition ...")
            analyzer_spectral.evaluate_nn_function_coordinates(test_state, sd_dict)
            logging.info(f"Self-consistency test: {sd_dict.get('function_coordinate_lhs')} = {sd_dict.get('function_coordinate_rhs')}")

            insert_blank_line()
            # Evaluate generator eigenfunctions at the test state:
            log_without_linebreak(f"Analyzing the generator eigenfunctions learned with the {run.NN_CONFIG['v']['subnet_architecture'].split('_', 1)[1]} spectral decomposition ...")
            analyzer_spectral.evaluate_nn_gen_eigenfunctions(test_state, sd_dict)
            logging.info(f"Self-consistency test for c: {sd_dict.get('gen_eigenfunction_lhs_c')} = {sd_dict.get('gen_eigenfunction_rhs_c')}")
            logging.info(f"Self-consistency test for d: {sd_dict.get('gen_eigenfunction_lhs_d')} = {sd_dict.get('gen_eigenfunction_rhs_d')}")

            insert_blank_line()
            # Evaluate expectation eigenfunctions at the test state:
            log_without_linebreak(f"Analyzing the transition semigroup eigenfunctions learned with the {run.NN_CONFIG['v']['subnet_architecture'].split('_', 1)[1]} spectral decomposition ...")
            analyzer_spectral.evaluate_nn_exp_eigenfunctions(valid_data, test_state, sd_dict)
            logging.info(f"Final time self-consistency test for c: {format_mean_ci_arrays(sd_dict.get('exp_eigenfunction_lhs_c')[-1], sd_dict.get('exp_eigenfunction_lhs_c_ci')[-1])} = {sd_dict.get('exp_eigenfunction_rhs_c')[-1]}")
            logging.info(f"Final time self-consistency test for d: {format_mean_ci_arrays(sd_dict.get('exp_eigenfunction_lhs_d')[-1], sd_dict.get('exp_eigenfunction_lhs_d_ci')[-1])} = {sd_dict.get('exp_eigenfunction_rhs_d')[-1]}")

            insert_blank_line()
            # Evaluate the Poisson equation at the test state:
            log_without_linebreak(f"Analyzing the solution to the Poisson equation learned with the {run.NN_CONFIG['v']['subnet_architecture'].split('_', 1)[1]} spectral decomposition ...")
            analyzer_spectral.evaluate_nn_poisson(test_state, sd_dict)
            logging.info(f"Self-consistency test: {sd_dict.get('poisson_lhs')} = {sd_dict.get('poisson_rhs')}")
            spectral_plots.plot_temporal_eigenfunctions(test_state, valid_data, run, i, sd_dict)

            # Copy the dictionary as soon as all SD analyses are finished:
            analysis_results["sd_dict"] = sd_dict.copy()

        insert_blank_line()
        # Save all results dictionaries in analysis_results:
        analysis_results["ert"] = ert.copy()
        analysis_results["erf"] = erf.copy()
        for key, value in [("ssa_results", conv_ssa_repeat_results), ("cv_results", conv_cv_repeat_results), ("is_results", conv_is_repeat_results)]:
            analysis_results[key] = value.copy() if value is not None else None
        # Save all results dictionaries as a pickle file:
        pickle_size = save_analysis_results(analysis_results, run, i)
        logging.info(f"Analysis results for test state {i} saved as compressed pickle file ({(pickle_size / (1024 ** 2)):,.3f} MB on disc).")

        # Analyze and report the log file timings of different methods for the current test state:
        test_state_log_file_analysis(run, str(test_state))

        # Reset the DeepIS estimate status for the next test state (if it was time-outed before):
        run.VAL_CONFIG["compute_is_estimate"] = original_is_estimate_status

    # Plot and save the temporal expectations for all test states in a panel:
    panels.plot_temporal_panel(test_state_figures, run)

