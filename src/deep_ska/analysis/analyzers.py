"""Module for analyzing the trained models and subnets."""

import concurrent.futures
import copy
import logging

import numpy as np
import torch
from scipy.stats import chi2
from torch import Tensor, nn
from tqdm.auto import tqdm

from ..core.model import ExpectationModel
from ..core.simulation import ExpectationGeneratorEM
from ..core.utils import get_available_cpus
from ..logging.helpers import (
    array_memory_logger,
    execution_timer,
    insert_blank_line,
    log_without_linebreak,
)
from ..reaction_networks.definition import ReactionNetwork


class SubnetAnalyzer:
    """Class for analyzing the trained model."""

    def __init__(self, network: ReactionNetwork, model: ExpectationModel, val_config: dict, analysis_mode: bool | None = None) -> None:
        """Initialize with the given network, model, validation configuration, and analysis mode."""
        self.network = network
        self.model = model
        self.val_config = val_config
        self.analysis_mode = analysis_mode
        self.main_net = self.model.main_net
        self.diff_net = self.model.diff_net
        self.quot_net = self.model.quot_net

    def compute_estimate_statistics(self, estimate: np.ndarray) -> tuple:
        """Helper function to compute mean, variance, CV and 95% CI of an estimate."""
        n_samples = estimate.shape[0]
        mean = np.mean(estimate, axis=0)
        var = np.var(estimate, ddof=1, axis=0)
        std = np.std(estimate, ddof=1, axis=0)
        cv = std / mean  # coefficient of variation
        ci = 1.96 * std / np.sqrt(n_samples)  # 95% confidence interval
        var_ci_lower = (n_samples-1) * var / chi2.ppf(0.975, n_samples-1)  # 95% CI variance lower bound
        var_ci_upper = (n_samples-1) * var / chi2.ppf(0.025, n_samples-1)  # 95% CI variance upper bound
        tolerance = self.val_config["tolerance_band"] * mean  # use as ± some % tolerance band
        return mean, var, cv, ci, tolerance, n_samples, var_ci_lower, var_ci_upper

    @torch.no_grad()
    def compute_full_dV(self, state_tensor: Tensor, stop_idx: int, times: Tensor) -> Tensor:
        """Helper function to compute the full dV matrix for all time points given a final time."""
        batch_size = state_tensor.shape[0]  # get the number of trajectories

        if self.model.nn_config["v"]["as_relationship"] == "identity":
            # NOTE: We also need this function during training, where we always want to use the shared subnet, so we check for the analysis mode.
            if self.model.nn_config["v"]["analysis_subnets"] == "shared" or self.analysis_mode is False:
                # Get V values for all time points:
                full_V = self.main_net(state_tensor[:, :stop_idx, :], times[stop_idx], times[:stop_idx], *self.model.main_net_args)
                full_V = full_V.view(batch_size, stop_idx, self.network.out_function_size)
                # Initialize the matrix for computing dV:
                full_dV = torch.zeros(batch_size, stop_idx, self.network.out_function_size, self.network.n_reactions)
                for reaction, row in enumerate(self.model.stoichiometry_matrix):
                    zeta_states = state_tensor[:, :stop_idx, :] + row  # NOTE: This can lead to negative states.
                    zeta_V = self.main_net(zeta_states, times[stop_idx], times[:stop_idx], *self.model.main_net_args)
                    zeta_V = zeta_V.view(batch_size, stop_idx, self.network.out_function_size)
                    full_dV[:, :, :, reaction] = zeta_V - full_V
            elif self.model.nn_config["v"]["analysis_subnets"] == "distinct":
                # Compute dV using a second subnet:
                full_dV = self.diff_net(state_tensor[:, :stop_idx, :], times[stop_idx], times[:stop_idx], *self.model.diff_net_args)
                full_dV = full_dV.view(batch_size, stop_idx, self.network.out_function_size, self.network.n_reactions)
            else:
                raise ValueError(f'Analysis subnet type "{self.model.nn_config["v"]["analysis_subnets"]}" is invalid for V.')

        elif self.model.nn_config["v"]["as_relationship"] == "logarithm":
            if self.model.nn_config["v"]["analysis_subnets"] in ["shared", "distinct"]:
                # Get V values for all time points:
                full_V = torch.exp(self.main_net(state_tensor[:, :stop_idx, :], times[stop_idx], times[:stop_idx], *self.model.main_net_args))
                full_V = full_V.view(batch_size, stop_idx, self.network.out_function_size)
                # Initialize the matrix for computing dV:
                full_dV = torch.zeros(batch_size, stop_idx, self.network.out_function_size, self.network.n_reactions)
                for reaction, row in enumerate(self.model.stoichiometry_matrix):
                    zeta_states = state_tensor[:, :stop_idx, :] + row  # NOTE: This can lead to negative states.
                    zeta_V = torch.exp(self.main_net(zeta_states, times[stop_idx], times[:stop_idx], *self.model.main_net_args))
                    zeta_V = zeta_V.view(batch_size, stop_idx, self.network.out_function_size)
                    full_dV[:, :, :, reaction] = zeta_V - full_V
            else:
                raise ValueError(f'Analysis subnet type "{self.model.nn_config["v"]["analysis_subnets"]}" is invalid for V.')

        else:
            raise ValueError(f'Relationship type "{self.model.nn_config["v"]["as_relationship"]}" is invalid for V.')

        return full_dV


class TemporalSubnetAnalyzer(SubnetAnalyzer):
    """Class for analyzing the trained subnets over a time interval."""

    def __init__(self, network: ReactionNetwork, model: ExpectationModel, val_config: dict, analysis_mode: bool | None = None) -> None:
        """Initialize with the given network, model, validation configuration, and analysis mode."""
        super().__init__(network, model, val_config, analysis_mode)

    def compute_temporal_ssa_estimate(self, inputs: dict, results_dict: dict) -> None:
        """Compute the SSA output function values over time."""
        times = inputs["times"]
        state_trajectories = inputs["state_trajectories"]

        batch_time_samples = times.shape[0]  # get the number of time samples
        batch_size = state_trajectories.shape[0]  # get the number of trajectories

        ssa_estimate = np.zeros([batch_size, batch_time_samples, self.network.out_function_size])

        for t in range(0, batch_time_samples):
            ssa_estimate[:, t, :] = self.network.output_function(state_trajectories[:, t, :])

        mean, var, cv, ci, tolerance, n_samples, var_ci_lower, var_ci_upper = self.compute_estimate_statistics(ssa_estimate[:, 1:, :])
        # NOTE: We return only 100 trajectories because we will likely never plot more than that at once and it saves memory.
        results_dict.update({"raw_mean": mean, "raw_var": var, "raw_cv": cv, "raw_ci": ci, "raw_tolerance": tolerance, "raw_samples": n_samples, "raw_var_ci_lower": var_ci_lower, "raw_var_ci_upper": var_ci_upper, "raw_ssa_estimate": ssa_estimate[:100]})

    @execution_timer
    @torch.no_grad()
    def compute_temporal_nn_expectation(self, inputs: dict, state: np.ndarray, results_dict: dict) -> None:
        """Part of the forward pass necessary for calculating V for a single state."""
        times = inputs["times"]  # load input data
        batch_time_samples = times.shape[0]  # get the number of time samples
        t_stop = times[-1]  # get the final time point
        times = torch.from_numpy(times)
        state_tensor = torch.from_numpy(state).expand(batch_time_samples, -1).unsqueeze(0)  # repeat state over time and add batch dimension

        if self.model.nn_config["v"]["as_relationship"] == "identity":
            # Get V values for all time points:
            full_V = self.main_net(state_tensor[:, :-1, :], t_stop, t_stop-times[:-1], *self.model.main_net_args)
            full_V = full_V.view(batch_time_samples-1, self.network.out_function_size)

        elif self.model.nn_config["v"]["as_relationship"] == "logarithm":
            # Get V values for all time points:
            full_V = torch.exp(self.main_net(state_tensor[:, :-1, :], t_stop, t_stop-times[:-1], *self.model.main_net_args))
            full_V = full_V.view(batch_time_samples-1, self.network.out_function_size)

        else:
            raise ValueError(f'Relationship type "{self.model.nn_config["v"]["as_relationship"]}" is invalid for V.')

        results_dict.update({"full_V": full_V.detach().numpy()})

    @execution_timer
    @torch.no_grad()
    def compute_temporal_deep_cv_estimate(self, inputs: dict, results_dict: dict) -> None:
        """Compute the DeepCV estimate for each final time."""
        times = inputs["times"]
        state_trajectories = inputs["state_trajectories"]
        martingale_trajectories = inputs["martingale_trajectories"]

        batch_time_samples = times.shape[0]  # get the number of time samples
        batch_size = state_trajectories.shape[0]  # get the number of trajectories

        times = torch.from_numpy(times)
        state_tensor = torch.from_numpy(state_trajectories)
        martingale_tensor = torch.from_numpy(martingale_trajectories)

        full_integrals = torch.zeros(batch_size, batch_time_samples-1, self.network.out_function_size, self.network.n_reactions)
        final_output = np.zeros([batch_size, batch_time_samples-1, self.network.out_function_size])  # true (simulated) network output

        for stop_idx in range(1, batch_time_samples):
            # Save the true network output for the current final time:
            final_output[:, stop_idx-1, :] = self.network.output_function(state_trajectories[:, stop_idx, :])
            # Use helper function to compute the full dV matrix for all time points given the final time:
            full_dV = self.compute_full_dV(state_tensor, stop_idx, times)
            # Compute the difference in the martingale and add an output function dimension:
            martingale_increment = (martingale_tensor[:, 1:stop_idx+1, :] - martingale_tensor[:, :stop_idx, :]).unsqueeze(2)
            # Compute the integral for the current t_stop and save it for later:
            full_integrals[:, stop_idx-1, :, :] = torch.sum(full_dV.real * martingale_increment, dim=1)  # NOTE: We take the real part for early stopping, otherwise the complex part is removed anyway.

        # Sum over all reactions to compute the adjustment term:
        adjustment_term = torch.sum(full_integrals, dim=-1).detach().numpy()

        # Compute statistics of the differences between the final output and the adjustment term:
        diff_estimate = final_output - adjustment_term
        mean, var, cv, ci, tolerance, n_samples, var_ci_lower, var_ci_upper = self.compute_estimate_statistics(diff_estimate)
        # Also compute the mean of the adjustment term (should be close to zero):
        adjustment_mean = np.mean(adjustment_term, axis=0)
        # Update the passed results dictionary:
        results_dict.update({"diff_mean": mean, "diff_var": var, "diff_cv": cv, "diff_ci": ci, "diff_tolerance": tolerance, "diff_adjustment_mean": adjustment_mean, "diff_samples": n_samples, "diff_var_ci_lower": var_ci_lower, "diff_var_ci_upper": var_ci_upper})

        # Also compute the output-suboptimal estimate using only the first output function:
        sub_adjustment_term = np.expand_dims(adjustment_term[:, :, 0], axis=2)
        # Also compute the statistics of the output-suboptimal estimate:
        sub_diff_estimate = final_output - sub_adjustment_term
        sub_mean, sub_var, sub_cv, sub_ci, sub_tolerance, sub_n_samples, sub_var_ci_lower, sub_var_ci_upper = self.compute_estimate_statistics(sub_diff_estimate)
        # Also compute the mean of the output-suboptimal adjustment term (should be close to zero):
        sub_adjustment_mean = np.mean(sub_adjustment_term, axis=0)

        results_dict.update({"diff_out_suboptimal_mean": sub_mean, "diff_out_suboptimal_var": sub_var, "diff_out_suboptimal_cv": sub_cv, "diff_out_suboptimal_ci": sub_ci, "diff_out_suboptimal_tolerance": sub_tolerance, "diff_out_suboptimal_adjustment_mean": sub_adjustment_mean, "diff_out_suboptimal_samples": sub_n_samples, "diff_out_suboptimal_var_ci_lower": sub_var_ci_lower, "diff_out_suboptimal_var_ci_upper": sub_var_ci_upper})

    @execution_timer
    @torch.no_grad()
    def compute_time_suboptimal_deep_cv_estimate(self, inputs: dict, t_stop: float, results_dict: dict) -> None:
        """Compute the time-suboptimal DeepCV estimate for a certain final time."""
        times = inputs["times"]
        stop_idx = np.searchsorted(times, t_stop, side="left")
        # Select everything until the stop_idx only:
        times = times[:stop_idx+1]
        state_trajectories = inputs["state_trajectories"][:, :stop_idx+1, :]
        martingale_trajectories = inputs["martingale_trajectories"][:, :stop_idx+1, :]

        batch_time_samples = times.shape[0]  # get the number of time samples
        batch_size = state_trajectories.shape[0]  # get the number of trajectories

        times = torch.from_numpy(times)
        state_tensor = torch.from_numpy(state_trajectories)
        martingale_tensor = torch.from_numpy(martingale_trajectories)

        final_output = np.zeros([batch_size, batch_time_samples-1, self.network.out_function_size])  # true (simulated) network output

        for grid_idx in range(1, batch_time_samples):
            # Save the true network output for the current final time:
            final_output[:, grid_idx-1, :] = self.network.output_function(state_trajectories[:, grid_idx, :])

        # Use helper function to compute the full dV matrix for all time points given the final time:
        full_dV = self.compute_full_dV(state_tensor, stop_idx, times)
        # Compute the difference in the martingale and add an output function dimension:
        martingale_increment = (martingale_tensor[:, 1:, :] - martingale_tensor[:, :-1, :]).unsqueeze(2)
        # Compute the integral for the current t_stop and save it for later:
        full_integrals = torch.cumsum(full_dV.real * martingale_increment, dim=1)  # NOTE: We take the real part for early stopping, otherwise the complex part is removed anyway.

        # Sum over all reactions to compute the adjustment term:
        adjustment_term = torch.sum(full_integrals, dim=-1).detach().numpy()

        # Compute statistics of the differences between the final output and the adjustment term:
        diff_estimate = final_output - adjustment_term
        mean, var, cv, ci, tolerance, n_samples, var_ci_lower, var_ci_upper = self.compute_estimate_statistics(diff_estimate)
        # Also compute the mean of the adjustment term (should be close to zero):
        adjustment_mean = np.mean(adjustment_term, axis=0)

        # NOTE: We only return the first 100 samples of the estimate to save memory.
        results_dict.update({"diff_time_suboptimal_mean": mean, "diff_time_suboptimal_var": var, "diff_time_suboptimal_cv": cv, "diff_time_suboptimal_ci": ci, "diff_time_suboptimal_tolerance": tolerance, "diff_time_suboptimal_adjustment_mean": adjustment_mean, "diff_time_suboptimal_samples": n_samples, "diff_time_suboptimal_var_ci_lower": var_ci_lower, "diff_time_suboptimal_var_ci_upper": var_ci_upper, "diff_time_suboptimal_estimate": diff_estimate[:100]})

    @torch.no_grad()
    def generate_temporal_deep_is_estimate(self, val_config: dict, results_dict: dict) -> None:
        """Compute the SSA output function values over time using the DeepIS trajectories."""
        # Extract validation configuration parameters:
        t_stop = val_config["t_stop"]
        n_time_samples = val_config["n_time_samples"]
        n_is_samples = val_config["n_is_samples"]
        n_trajectories = int(val_config["n_trajectories"] * val_config["ssa_with_is_fraction"])

        # Generate SSA trajectories using importance sampling:
        is_generator = ExpectationGeneratorIS(self.network, self.model, val_config)
        trajectories = is_generator.sample_temporal_deep_is_trajectories(t_stop, n_time_samples, n_is_samples, n_trajectories)

        if self.val_config["compute_is_estimate"] == False:  # changed to False if the simulation took too long
            return

        # Compute mean, variance, CV and 95% CI of the importance sampling estimate:
        mean, var, cv, ci, _, n_samples, var_ci_lower, var_ci_upper = self.compute_estimate_statistics(trajectories["is_trajectories"])
        results_dict.update({"is_times": trajectories["is_times"], "is_raw_mean": mean, "is_raw_var": var, "is_raw_cv": cv, "is_raw_ci": ci, "is_raw_samples": n_samples, "is_raw_var_ci_lower": var_ci_lower, "is_raw_var_ci_upper": var_ci_upper})

        # Also compute the output-suboptimal estimate using only the first output function:
        sub_mean, sub_var, sub_cv, sub_ci, _, sub_n_samples, sub_var_ci_lower, sub_var_ci_upper = self.compute_estimate_statistics(trajectories["sub_is_trajectories"])
        results_dict.update({"is_raw_out_suboptimal_mean": sub_mean, "is_raw_out_suboptimal_var": sub_var, "is_raw_out_suboptimal_cv": sub_cv, "is_raw_out_suboptimal_ci": sub_ci, "is_raw_out_suboptimal_samples": sub_n_samples, "is_raw_out_suboptimal_var_ci_lower": sub_var_ci_lower, "is_raw_out_suboptimal_var_ci_upper": sub_var_ci_upper})

    @torch.no_grad()
    def generate_time_suboptimal_deep_is_estimate(self, val_config: dict, active_trajectories: set[str], results_dict: dict) -> None:
        """Compute the temporal SSA output function values using a single DeepIS trajectory time point."""
        # Extract validation configuration parameters:
        t_stop = val_config["t_stop"]
        n_time_samples = val_config["n_time_samples"]
        n_trajectories = int(val_config["n_trajectories"] * val_config["ssa_with_is_fraction"])

        # Generate SSA trajectories using importance sampling:
        is_generator = ExpectationGeneratorIS(self.network, self.model, val_config)
        trajectories = is_generator.sample_final_deep_is_trajectories(t_stop, n_time_samples, n_trajectories, active_trajectories)

        if self.val_config["compute_is_estimate"] == False:  # changed to False if the simulation took too long
            return

        # Compute mean, variance, CV and 95% CI of the importance sampling estimate:
        mean, var, cv, ci, _, n_samples, var_ci_lower, var_ci_upper = self.compute_estimate_statistics(trajectories["is_trajectories"])
        # Also compute the cumulative means:
        cummean = np.cumsum(trajectories["is_trajectories"], axis=0) / np.arange(1, trajectories["is_trajectories"].shape[0] + 1).reshape((-1,) + (1,) * (trajectories["is_trajectories"].ndim - 1))

        # NOTE: We only return the first 100 samples of the estimate to save memory.
        results_dict.update({"is_raw_time_suboptimal_mean": mean, "is_raw_time_suboptimal_var": var, "is_raw_time_suboptimal_cv": cv, "is_raw_time_suboptimal_ci": ci, "is_raw_time_suboptimal_samples": n_samples, "is_raw_time_suboptimal_var_ci_lower": var_ci_lower, "is_raw_time_suboptimal_var_ci_upper": var_ci_upper, "is_raw_time_suboptimal_cummean": cummean, "is_raw_time_suboptimal_ssa_estimate": trajectories["is_trajectories"][:100]})

    @execution_timer
    @torch.no_grad()
    def compute_temporal_deep_pva_variance(self, inputs: dict, results_dict: dict) -> None:
        """Compute the approximate variance using the DeepPVA relationship."""
        times = inputs["times"]
        state_trajectories = inputs["state_trajectories"]

        batch_time_samples = times.shape[0]  # get the number of time samples
        batch_size = state_trajectories.shape[0]  # get the number of trajectories

        times = torch.from_numpy(times)

        final_output = np.zeros([batch_size, batch_time_samples-1, self.network.out_function_size])  # true (simulated) network output

        for stop_idx in range(1, batch_time_samples):
            # Save the true network output for the current final time:
            final_output[:, stop_idx-1, :] = self.network.output_function(state_trajectories[:, stop_idx, :])

        # Get V values for all time points (only do once when final time is reached):
        full_V = results_dict.get("full_V")

        diff_estimate = (final_output - full_V) ** 2

        # Compute statistics of the differences between the final output and the adjustment term:
        mean, var, cv, ci, _, n_samples, _, _ = self.compute_estimate_statistics(diff_estimate)
        # Update the passed results dictionary:
        results_dict.update({"ter_mean": mean, "ter_var": var, "ter_cv": cv, "ter_ci": ci, "ter_samples": n_samples})

    @execution_timer
    @torch.no_grad()
    def compute_temporal_deep_ipa_variance(self, inputs: dict, results_dict: dict) -> None:
        """Compute the approximate variance using the DeepIPA relationship."""
        times = inputs["times"]
        state_trajectories = inputs["state_trajectories"]
        propensity_trajectories = inputs["propensity_trajectories"]

        batch_time_samples = times.shape[0]  # get the number of time samples
        batch_size = state_trajectories.shape[0]  # get the number of trajectories

        times = torch.from_numpy(times)
        state_tensor = torch.from_numpy(state_trajectories)
        propensity_tensor = torch.from_numpy(propensity_trajectories)

        # Initialize matrices for storing results for each final time:
        deep_ipa = torch.zeros(batch_size, batch_time_samples-1, self.network.out_function_size, self.network.n_reactions)

        for stop_idx in range(1, batch_time_samples):
            # Use helper function to compute the full dV matrix for all time points given the final time:
            full_dV = self.compute_full_dV(state_tensor, stop_idx, times)
            # Prepare inputs for the variance computation:
            propensity_slice = propensity_tensor[:, :stop_idx, :].reshape(batch_size, stop_idx, 1, self.network.n_reactions)
            time_increment = (times[1:stop_idx+1] - times[:stop_idx]).reshape(1, stop_idx, 1, 1)
            # Compute the DeepIPA variance for the current final time:
            deep_ipa[:, stop_idx-1, :, :] = torch.sum(propensity_slice * full_dV**2 * time_increment, dim=1)

        # Compute statistics of the DeepIPA variance:
        deep_ipa = torch.sum(deep_ipa, dim=-1).detach().numpy()
        mean, var, cv, ci, _, n_samples, _, _ = self.compute_estimate_statistics(deep_ipa)

        results_dict.update({"int_mean": mean, "int_var": var, "int_cv": cv, "int_ci": ci, "int_samples": n_samples})


class FinalTimeSubnetAnalyzer(SubnetAnalyzer):
    """Class for analyzing the trained subnets at the final time."""

    def __init__(self, network: ReactionNetwork, model: ExpectationModel, val_config: dict, analysis_mode: bool | None = None) -> None:
        """Initialize with the given network, model, validation configuration, and analysis mode."""
        super().__init__(network, model, val_config, analysis_mode)

    def compute_reference_ssa_estimate(self, inputs: dict, results_dict: dict) -> None:
        """Compute the SSA output function values over time."""
        states = inputs["states"]
        ssa_estimate = self.network.output_function(states)
        # Compute statistics of the SSA estimate:
        mean, var, cv, ci, tolerance, n_samples, var_ci_lower, var_ci_upper = self.compute_estimate_statistics(ssa_estimate)
        # Update the passed results dictionary:
        results_dict.update({"ref_raw_mean": mean, "ref_raw_var": var, "ref_raw_cv": cv, "ref_raw_ci": ci, "ref_raw_tolerance": tolerance, "ref_raw_samples": n_samples, "ref_raw_var_ci_lower": var_ci_lower, "ref_raw_var_ci_upper": var_ci_upper})

    def compute_final_ssa_estimate(self, inputs: dict, results_dict: dict) -> None:
        """Compute the SSA output function values for a final time."""
        state_trajectories = inputs["state_trajectories"]
        ssa_estimate = self.network.output_function(state_trajectories[:, -1, :])
        # Compute statistics of the SSA estimate:
        mean, var, cv, ci, tolerance, n_samples, var_ci_lower, var_ci_upper = self.compute_estimate_statistics(ssa_estimate)
        # Also compute the cumulative means:
        cummean = np.cumsum(ssa_estimate, axis=0) / np.arange(1, ssa_estimate.shape[0] + 1).reshape((-1,) + (1,) * (ssa_estimate.ndim - 1))
        # Update the passed results dictionary:
        results_dict.update({"raw_mean": mean, "raw_var": var, "raw_cv": cv, "raw_ci": ci, "raw_tolerance": tolerance, "raw_samples": n_samples, "raw_var_ci_lower": var_ci_lower, "raw_var_ci_upper": var_ci_upper, "raw_cummean": cummean})

    @execution_timer
    @torch.no_grad()
    def compute_final_deep_cv_estimate(self, inputs: dict, results_dict: dict, reference: bool = False) -> None:
        """Compute the DeepCV estimate for one final time."""
        times = inputs["times"]
        state_trajectories = inputs["state_trajectories"]
        martingale_trajectories = inputs["martingale_trajectories"]

        times = torch.from_numpy(times)
        state_tensor = torch.from_numpy(state_trajectories)
        martingale_tensor = torch.from_numpy(martingale_trajectories)

        # Save the true network output for the current final time:
        final_output = self.network.output_function(state_trajectories[:, -1, :])
        # Use helper function to compute the full dV matrix for all time points given the final time:
        full_dV = self.compute_full_dV(state_tensor, -1, times)
        # Compute the difference in the martingale and add an output function dimension:
        martingale_increment = (martingale_tensor[:, 1:, :] - martingale_tensor[:, :-1, :]).unsqueeze(2)
        # Compute the integral for the current t_stop and save it for later:
        full_integrals = torch.sum(full_dV.real * martingale_increment, dim=1)  # NOTE: We take the real part for early stopping, otherwise the complex part is removed anyway.
        # Sum over all reactions to compute the adjustment term:
        adjustment_term = torch.sum(full_integrals, dim=-1).detach().numpy()

        # Compute statistics of the differences between the final output and the adjustment term:
        diff_estimate = final_output - adjustment_term
        mean, var, cv, ci, tolerance, n_samples, var_ci_lower, var_ci_upper = self.compute_estimate_statistics(diff_estimate)
        # Also compute the cumulative means:
        cummean = np.cumsum(diff_estimate, axis=0) / np.arange(1, diff_estimate.shape[0] + 1).reshape((-1,) + (1,) * (diff_estimate.ndim - 1))
        # Also compute the mean of the adjustment term (should be close to zero):
        adjustment_mean = np.mean(adjustment_term, axis=0)

        if reference:
            results_dict.update({"ref_diff_mean": mean, "ref_diff_var": var, "ref_diff_cv": cv, "ref_diff_ci": ci, "ref_diff_tolerance": tolerance, "ref_diff_adjustment_mean": adjustment_mean, "ref_diff_samples": n_samples, "ref_diff_var_ci_lower": var_ci_lower, "ref_diff_var_ci_upper": var_ci_upper, "ref_diff_cummean": cummean})
        else:
            results_dict.update({"diff_mean": mean, "diff_var": var, "diff_cv": cv, "diff_ci": ci, "diff_tolerance": tolerance, "diff_adjustment_mean": adjustment_mean, "diff_samples": n_samples, "diff_var_ci_lower": var_ci_lower, "diff_var_ci_upper": var_ci_upper, "diff_cummean": cummean})

    @torch.no_grad()
    def generate_reference_exact_deep_cv_estimate(self, val_config: dict, results_dict: dict) -> None:
        """Generate an exact reference SSA with DeepCV estimate."""
        # Extract validation configuration parameters:
        t_stop = val_config["conv_t_stop"]
        n_trajectories = int(val_config["conv_ref_n_trajectories"])

        # Generate SSA trajectories using exact control variates:
        exact_cv_generator = ExpectationGeneratorExactCV(self.network, self.model, val_config)
        trajectories = exact_cv_generator.sample_final_exact_cv_trajectories(t_stop, n_trajectories)

        final_output = trajectories["final_output"]
        adjustment_term = trajectories["adjustment_term"]
        diff_estimate = final_output - adjustment_term
        # Compute statistics of the exact DeepCV estimate:
        mean, var, cv, ci, tolerance, n_samples, var_ci_lower, var_ci_upper = self.compute_estimate_statistics(diff_estimate)
        # Also compute the cumulative means:
        cummean = np.cumsum(diff_estimate, axis=0) / np.arange(1, diff_estimate.shape[0] + 1).reshape((-1,) + (1,) * (diff_estimate.ndim - 1))
        # Also compute the mean of the adjustment term (should be close to zero):
        adjustment_mean = np.mean(adjustment_term, axis=0)

        results_dict.update({"ref_diff_mean": mean, "ref_diff_var": var, "ref_diff_cv": cv, "ref_diff_ci": ci, "ref_diff_tolerance": tolerance, "ref_diff_adjustment_mean": adjustment_mean, "ref_diff_samples": n_samples, "ref_diff_var_ci_lower": var_ci_lower, "ref_diff_var_ci_upper": var_ci_upper, "ref_diff_cummean": cummean})

    @torch.no_grad()
    def generate_final_exact_deep_cv_estimate(self, val_config: dict, results_dict: dict) -> None:
        """Generate an exact SSA with DeepCV estimate for one final time."""
        # Extract validation configuration parameters:
        t_stop = val_config["conv_t_stop"]
        n_trajectories = int(val_config["conv_n_trajectories"] * val_config["conv_ssa_with_cv_fraction"])

        # Generate SSA trajectories using exact control variates:
        exact_cv_generator = ExpectationGeneratorExactCV(self.network, self.model, val_config)
        trajectories = exact_cv_generator.sample_final_exact_cv_trajectories(t_stop, n_trajectories)

        final_output = trajectories["final_output"]
        adjustment_term = trajectories["adjustment_term"]
        diff_estimate = final_output - adjustment_term
        # Compute statistics of the exact DeepCV estimate:
        mean, var, cv, ci, tolerance, n_samples, var_ci_lower, var_ci_upper = self.compute_estimate_statistics(diff_estimate)
        # Also compute the cumulative means:
        cummean = np.cumsum(diff_estimate, axis=0) / np.arange(1, diff_estimate.shape[0] + 1).reshape((-1,) + (1,) * (diff_estimate.ndim - 1))
        # Also compute the mean of the adjustment term (should be close to zero):
        adjustment_mean = np.mean(adjustment_term, axis=0)
        results_dict.update({"diff_mean": mean, "diff_var": var, "diff_cv": cv, "diff_ci": ci, "diff_tolerance": tolerance, "diff_adjustment_mean": adjustment_mean, "diff_samples": n_samples, "diff_var_ci_lower": var_ci_lower, "diff_var_ci_upper": var_ci_upper, "diff_cummean": cummean})
        # Here, the generator is returned for the repeated convergence samples:
        return exact_cv_generator

    @torch.no_grad()
    def generate_reference_deep_is_estimate(self, val_config: dict, active_trajectories: set[str], results_dict: dict) -> None:
        """Compute the reference SSA output function values for a single time point using the DeepIS trajectories."""
        # Extract validation configuration parameters:
        t_stop = val_config["conv_t_stop"]
        n_time_samples = val_config["conv_n_time_samples"]
        n_trajectories = int(val_config["conv_ref_n_trajectories"])

        # Generate SSA trajectories using importance sampling:
        is_generator = ExpectationGeneratorIS(self.network, self.model, val_config)
        trajectories = is_generator.sample_final_deep_is_trajectories(t_stop, n_time_samples, n_trajectories, active_trajectories)

        if self.val_config["compute_is_estimate"] == False:  # changed to False if the simulation took too long
            return

        is_trajectories = trajectories["is_trajectories"][:, -1, :]
        # Compute mean, variance, CV and 95% CI of the importance sampling estimate:
        mean, var, cv, ci, _, n_samples, var_ci_lower, var_ci_upper = self.compute_estimate_statistics(is_trajectories)
        # Also compute the cumulative means:
        cummean = np.cumsum(is_trajectories, axis=0) / np.arange(1, is_trajectories.shape[0] + 1).reshape((-1,) + (1,) * (is_trajectories.ndim - 1))

        results_dict.update({"ref_is_raw_mean": mean, "ref_is_raw_var": var, "ref_is_raw_cv": cv, "ref_is_raw_ci": ci, "ref_is_raw_samples": n_samples, "ref_is_raw_var_ci_lower": var_ci_lower, "ref_is_raw_var_ci_upper": var_ci_upper, "ref_is_raw_cummean": cummean})

    @torch.no_grad()
    def generate_final_deep_is_estimate(self, val_config: dict, active_trajectories: set[str], results_dict: dict) -> None:
        """Compute the SSA output function values for a single time point using the DeepIS trajectories."""
        # Extract validation configuration parameters:
        t_stop = val_config["conv_t_stop"]
        n_time_samples = val_config["conv_n_time_samples"]
        n_trajectories = int(val_config["conv_n_trajectories"] * val_config["conv_ssa_with_is_fraction"])

        # Generate SSA trajectories using importance sampling:
        is_generator = ExpectationGeneratorIS(self.network, self.model, val_config)
        trajectories = is_generator.sample_final_deep_is_trajectories(t_stop, n_time_samples, n_trajectories, active_trajectories)

        if self.val_config["compute_is_estimate"] == False:  # changed to False if the simulation took too long
            return

        is_trajectories = trajectories["is_trajectories"][:, -1, :]
        # Compute mean, variance, CV and 95% CI of the importance sampling estimate:
        mean, var, cv, ci, _, n_samples, var_ci_lower, var_ci_upper = self.compute_estimate_statistics(is_trajectories)
        # Also compute the cumulative means:
        cummean = np.cumsum(is_trajectories, axis=0) / np.arange(1, is_trajectories.shape[0] + 1).reshape((-1,) + (1,) * (is_trajectories.ndim - 1))
        results_dict.update({"is_raw_mean": mean, "is_raw_var": var, "is_raw_cv": cv, "is_raw_ci": ci, "is_raw_samples": n_samples, "is_raw_var_ci_lower": var_ci_lower, "is_raw_var_ci_upper": var_ci_upper, "is_raw_cummean": cummean})
        # Here, the generator is returned for the repeated convergence samples:
        return is_generator


class ExpectationGeneratorExactCV(SubnetAnalyzer):
    """Class for analyzing the trained expectation model with exact control variates."""

    def __init__(self, network: ReactionNetwork, model: ExpectationModel, val_config: dict, analysis_mode: bool | None = None, seed: int = None) -> None:
        """Initialize with the given network, model, validation configuration, and analysis mode."""
        super().__init__(network, model, val_config, analysis_mode)
        self.rng = np.random.default_rng(seed if seed is not None else val_config["analysis_seed"])

    @torch.no_grad()
    def sample_final_exact_cv_trajectory(self, t_stop: float) -> tuple:
        """Sample a single exact DeepCV trajectory."""
        # Initialize arrays for keeping track of time, species and reactions:
        A = torch.zeros([self.network.out_function_size, self.network.n_reactions])  # discontinuous part
        B = torch.zeros([self.network.out_function_size, self.network.n_reactions])  # continuous part

        internal_times = np.zeros([self.network.n_reactions])
        jump_times = -np.log(self.rng.uniform(0, 1, self.network.n_reactions))

        curr_state = self.network.initial_state if self.network.init_type == "deter" else self.network.sample_random_state(self.rng)
        t_curr = 0
        delta_reactions = np.zeros([self.network.n_reactions])

        while 1:
            # Compute the time delta:
            prop = self.network.propensity_vector(curr_state)  # evaluate propensities at every time point
            delta_reactions = np.divide((jump_times - internal_times), prop, out=np.full_like(delta_reactions, np.inf), where=prop > 0)

            if np.any(prop < 0):
                logging.warning(f"Negative propensity detected: {prop}.")

            next_reaction = np.argmin(delta_reactions, axis=0)  # find shortest reaction delta to determine next reaction (returns index)
            delta_time = delta_reactions[next_reaction]  # access time until the found next reaction occurs (returns value)
            delta_time = min(delta_time, t_stop - t_curr)

            if delta_time <= 0:
                logging.warning(f"Non-positive time delta detected: {delta_time}.")

            state_tensor = torch.from_numpy(curr_state).repeat(1, 2, 1)
            times_tensor = torch.tensor([t_curr, t_stop])
            full_dV = torch.squeeze(self.compute_full_dV(state_tensor, -1, times_tensor), dim=(0, 1))

            B += full_dV * prop * delta_time  # update continuous part

            internal_times += prop * delta_time  # update internal times with propensities and time delta
            t_curr += delta_time

            if t_curr >= t_stop: # if the stop time is exceeded, return the final arrays
                final_output = self.network.output_function(curr_state.reshape(1, -1))
                adjustment_term = torch.sum(A - B, axis=-1).numpy()
                return final_output, adjustment_term
            else: # update the state as long as the stop time is not exceeded
                curr_state = self.network.update_state(next_reaction, curr_state)

                if np.any(curr_state < 0):
                    logging.warning(f"Negative state component detected: {curr_state}")

                A[:, next_reaction] += full_dV[:, next_reaction]  # update discontinuous part
                jump_times[next_reaction] += -np.log(self.rng.uniform(0, 1))

    @staticmethod
    def process_final_exact_cv_sample(network: ReactionNetwork, model: ExpectationModel, val_config: dict, t_stop: float, idx: int, seed: int) -> dict:
        """Process a single importance exact DeepCV sample. Needed for multiprocessing."""
        local_network = copy.deepcopy(network)
        # Use a new random seed for each trajectory (which means we need a new simulator):
        local_simulator = ExpectationGeneratorExactCV(local_network, model, val_config, seed=seed)
        final_output, adjustment_term = local_simulator.sample_final_exact_cv_trajectory(t_stop)

        result = {
            "final_output": final_output,
            "adjustment_term": adjustment_term,
            "idx": idx,
        }

        return result

    @execution_timer
    @array_memory_logger
    def sample_final_exact_cv_trajectories(self, t_stop: float, n_trajectories: int, n_jobs: int = None) -> dict:
        """Create several exact DeepCV trajectories until a given final time."""
        trajectories = {"final_output": np.zeros((n_trajectories, self.network.out_function_size)),
                        "adjustment_term": np.zeros((n_trajectories, self.network.out_function_size)),}

        # NOTE: While highly unlikely, nested RNGs mean that duplicate trajectories are possible.
        seeds = self.rng.choice(2**32, size=n_trajectories, replace=False)
        n_jobs = get_available_cpus(n_jobs)

        # Start a subprocess for each trajectory:
        with concurrent.futures.ProcessPoolExecutor(max_workers=n_jobs) as executor:
            futures = []
            for i in range(n_trajectories):
                seed = seeds[i]
                futures.append(executor.submit(self.process_final_exact_cv_sample, self.network, self.model, self.val_config, t_stop, i, seed))

            with tqdm(total=n_trajectories, desc="final_exact_cv_trajectories", dynamic_ncols=True) as pbar:
                for future in concurrent.futures.as_completed(futures):
                    result = future.result()
                    trajectories["final_output"][result["idx"]] = result["final_output"]
                    trajectories["adjustment_term"][result["idx"]] = result["adjustment_term"]
                    pbar.update(1)

        return trajectories


class ExpectationGeneratorIS(SubnetAnalyzer):
    """Class for analyzing the trained expectation model with deep importance sampling."""

    def __init__(self, network: ReactionNetwork, model: ExpectationModel, val_config: dict, analysis_mode: bool | None = None, seed: int = None) -> None:
        """Initialize with the given network, model, validation configuration, and analysis mode."""
        super().__init__(network, model, val_config, analysis_mode)
        self.rng = np.random.default_rng(seed if seed is not None else val_config["analysis_seed"])

    @torch.no_grad()
    def deep_ssa_next_reaction(self, state: np.ndarray, t_curr: float, t_stop: float, out_function_index: int) -> tuple:
        """Compute the next reaction using the DeepSSA method with qV."""
        state_tensor = torch.from_numpy(state).unsqueeze(0).unsqueeze(1)
        t_curr = torch.tensor(t_curr).unsqueeze(0)
        epsilon_lower = 1e-10

        if self.model.nn_config["v"]["as_relationship"] == "identity":
            if self.model.nn_config["v"]["analysis_subnets"] in ["shared", "distinct"]:
                curr_V = self.main_net(state_tensor, t_stop, t_curr, *self.model.main_net_args).real
                qV = torch.zeros(self.network.n_reactions)
                for reaction, row in enumerate(self.model.stoichiometry_matrix):
                    zeta_state = torch.from_numpy(state) + row
                    zeta_V = self.main_net(zeta_state.unsqueeze(0).unsqueeze(1), t_stop, t_curr, *self.model.main_net_args).real
                    qV[reaction] = max(zeta_V[out_function_index], epsilon_lower) / max(curr_V[out_function_index], epsilon_lower)
            else:
                raise ValueError(f'Analysis subnet type "{self.model.nn_config["v"]["analysis_subnets"]}" is invalid for V.')

        elif self.model.nn_config["v"]["as_relationship"] == "logarithm":
            if self.model.nn_config["v"]["analysis_subnets"] == "shared":
                curr_V = torch.exp(self.main_net(state_tensor, t_stop, t_curr, *self.model.main_net_args).real)
                qV = torch.zeros(self.network.n_reactions)
                for reaction, row in enumerate(self.model.stoichiometry_matrix):
                    zeta_state = torch.from_numpy(state) + row
                    zeta_V = torch.exp(self.main_net(zeta_state.unsqueeze(0).unsqueeze(1), t_stop, t_curr, *self.model.main_net_args).real)
                    qV[reaction] = zeta_V[out_function_index] / curr_V[out_function_index]
            elif self.model.nn_config["v"]["analysis_subnets"] == "distinct":
                qV = torch.exp(self.quot_net(state_tensor, t_stop, t_curr, *self.model.quot_net_args).real)
                qV = qV[out_function_index]
            else:
                raise ValueError(f'Analysis subnet type "{self.model.nn_config["v"]["analysis_subnets"]}" is invalid for V.')

        else:
            raise ValueError(f'Relationship type "{self.model.nn_config["v"]["as_relationship"]}" is invalid for V.')

        prop_old = self.network.propensity_vector(state)  # without the quotient
        prop_new = prop_old * qV.detach().numpy()  # with the quotient

        if np.any(prop_new < 0):
            logging.warning(f"Negative propensity detected: {prop_new}.")

        sum_prop_new = np.sum(prop_new)

        if sum_prop_new == 0:
            delta_time = np.inf
            next_reaction = -1
        else:
            cum_prop_new = np.cumsum(prop_new / sum_prop_new)
            delta_time = -np.log(self.rng.uniform(0, 1)) / sum_prop_new
            next_reaction = sum(cum_prop_new < self.rng.uniform(0, 1))

        if delta_time <= 0:
            logging.warning(f"Non-positive time delta detected: {delta_time}.")

        return delta_time, next_reaction, prop_old, prop_new

    def run_deep_ssa(self, curr_state: np.ndarray, t_init: float, t_next: float, t_stop: float, out_function_index: int) -> tuple:
        """Run Gillespie's SSA without storing any values until t_next; start time is t_curr and the curr_state is specified."""
        A = 0
        B = 0
        t_curr = t_init

        curr_reaction_counts = np.zeros([self.network.n_reactions])
        curr_compensators = np.zeros([self.network.n_reactions])

        while 1:
            delta_time, next_reaction, prop_old, prop_new = self.deep_ssa_next_reaction(curr_state, t_init, t_stop, out_function_index)
            delta_time = min(delta_time, t_next - t_curr)
            B += np.sum((prop_old - prop_new) * delta_time)
            curr_compensators += prop_new * delta_time
            t_curr = t_curr + delta_time
            if t_curr >= t_next:
                return curr_state, A, B, curr_reaction_counts, curr_compensators
            else:
                curr_state = self.network.update_state(next_reaction, curr_state)

                if np.any(curr_state < 0):
                    logging.warning(f"Negative state component detected: {curr_state}")

                A += np.log(prop_old[next_reaction] / prop_new[next_reaction]) if next_reaction != -1 else 0
                curr_reaction_counts[next_reaction] += 1

    def sample_temporal_deep_is_trajectory(self, t_stop: float, n_time_samples: int, n_is_samples: int) -> tuple:
        """Create an importance sampling trajectory, looping over the final times."""
        times = np.linspace(0, t_stop, n_time_samples)
        is_trajectory = np.zeros([n_is_samples-1, self.network.out_function_size])
        sub_is_trajectory = np.zeros([n_is_samples-1, self.network.out_function_size])
        # Create indices for the DeepIS samples to speed up the computations:
        stop_indices = np.linspace(1, n_time_samples-1, n_is_samples-1, dtype=int)

        for out_function_index in range(self.network.out_function_size):
            for i, stop_idx in enumerate(stop_indices):
                total_A = 0
                total_B = 0

                curr_times = times[:stop_idx+1]
                curr_t_stop = curr_times[-1]
                curr_state = self.network.initial_state
                for j in range(0, stop_idx):
                    curr_state, A, B, _, _ = self.run_deep_ssa(curr_state, curr_times[j], curr_times[j+1], curr_t_stop, out_function_index)
                    total_A += A
                    total_B += B

                is_trajectory[i, out_function_index] = self.network.output_function(curr_state.reshape(1, -1))[:, out_function_index] * np.exp(total_A - total_B)

                if out_function_index == 0:
                    sub_is_trajectory[i, :] = self.network.output_function(curr_state.reshape(1, -1)) * np.exp(total_A - total_B)

        # NOTE: We directly create the stop indices starting with index 1, which means we have the
        # smallest possible gap on the x-axis of the plot. To account for that, we insert a zero at
        # the beginning of the stop indices.
        return times, times[np.insert(stop_indices, 0, 0)], is_trajectory, sub_is_trajectory

    @staticmethod
    def process_temporal_deep_is_sample(network: ReactionNetwork, model: ExpectationModel, val_config: dict, t_stop: float, n_time_samples: int, n_is_samples: int, idx: int, seed: int) -> dict:
        """Process a single DeepIS sample. Needed for multiprocessing."""
        local_network = copy.deepcopy(network)
        # Use a new random seed for each trajectory (which means we need a new simulator):
        local_simulator = ExpectationGeneratorIS(local_network, model, val_config, seed=seed)
        times, is_times, is_trajectory, sub_is_trajectory = local_simulator.sample_temporal_deep_is_trajectory(t_stop, n_time_samples, n_is_samples)

        return {
            "times": times,
            "is_times": is_times,
            "is_trajectory": is_trajectory,
            "sub_is_trajectory": sub_is_trajectory,
            "idx": idx,
        }

    @execution_timer
    @array_memory_logger
    def sample_temporal_deep_is_trajectories(self, t_stop: float, n_time_samples: int, n_is_samples: int, n_trajectories: int, n_jobs: int = None) -> dict:
        """Create several DeepIS trajectories, looping over the final times."""
        is_trajectories = np.zeros([n_trajectories, n_is_samples-1, self.network.out_function_size])
        sub_is_trajectories = np.zeros([n_trajectories, n_is_samples-1, self.network.out_function_size])
        timeout = self.val_config["is_simulation_timeout"]

        # NOTE: While highly unlikely, nested RNGs mean that duplicate trajectories are possible.
        seeds = self.rng.choice(2**32, size=n_trajectories, replace=False)
        n_jobs = get_available_cpus(n_jobs)

        # Start a subprocess for each trajectory:
        with concurrent.futures.ProcessPoolExecutor(max_workers=n_jobs) as executor:
            futures = [executor.submit(self.process_temporal_deep_is_sample, self.network, self.model, self.val_config, t_stop, n_time_samples, n_is_samples, i, seeds[i]) for i in range(n_trajectories)]
            done, not_done = concurrent.futures.wait(futures, timeout=timeout, return_when=concurrent.futures.FIRST_EXCEPTION)  # wait for jobs to complete or timeout

            if not_done:  # skip importance sampling if any futures did not complete
                insert_blank_line()
                logging.warning(f"Some importance sampling trajectories did not finish within the timeout of {timeout} seconds.")
                log_without_linebreak("Skipping importance sampling analysis after")
                self.val_config["compute_is_estimate"] = False
                executor.shutdown(wait=False, cancel_futures=True)
                return None

            for future in done:  # collect results from completed futures
                result = future.result(timeout=timeout)
                times = result["times"]
                is_times = result["is_times"]
                is_trajectories[result["idx"], :, :] = result["is_trajectory"]
                sub_is_trajectories[result["idx"], :, :] = result["sub_is_trajectory"]

        return {"times": times, "is_times": is_times, "is_trajectories": is_trajectories, "sub_is_trajectories": sub_is_trajectories}

    def sample_deep_is_trajectory_until_t_stop(self, curr_times: np.ndarray, curr_t_stop: float, n_time_samples: int, out_function_index: int) -> np.ndarray:
        """Sample a single DeepIS trajectory for a single output function index until a given t_stop."""
        total_A = 0
        total_B = 0

        states = np.zeros([n_time_samples, self.network.n_species])
        reaction_counts = np.zeros([n_time_samples, self.network.n_reactions])  # reaction counts over time
        compensators = np.zeros([n_time_samples, self.network.n_reactions])  # to correct simulation biases

        total_A = np.zeros([n_time_samples-1])
        total_B = np.zeros([n_time_samples-1])

        curr_state = self.network.initial_state if self.network.init_type == "deter" else self.network.sample_random_state(self.rng).astype(np.float64)
        states[0, :] = curr_state

        for j in range(0, n_time_samples-1):
            curr_state, total_A[j], total_B[j], reaction_counts[j, :], compensators[j, :] = self.run_deep_ssa(curr_state, curr_times[j], curr_times[j+1], curr_t_stop, out_function_index)
            states[j+1] = curr_state

        outputs = self.network.output_function(states)[:, out_function_index]
        is_trajectory = outputs[1:] * np.exp(np.cumsum(total_A - total_B))

        return states, np.cumsum(reaction_counts, axis=0), np.cumsum(compensators, axis=0), outputs, is_trajectory

    def sample_deep_is_trajectory_for_all_output_functions(self, curr_times: np.ndarray, curr_t_stop: float, n_time_samples: int) -> np.ndarray:
        """Sample DeepIS trajectories for all output functions."""
        states = np.zeros([n_time_samples, self.network.n_species, self.network.out_function_size])
        reaction_counts = np.zeros([n_time_samples, self.network.n_reactions, self.network.out_function_size])
        compensators = np.zeros([n_time_samples, self.network.n_reactions, self.network.out_function_size])
        outputs = np.zeros([n_time_samples, self.network.out_function_size])
        is_trajectories = np.zeros([n_time_samples-1, self.network.out_function_size])

        for out_function_index in range(self.network.out_function_size):
            states[:, :, out_function_index], reaction_counts[:, :, out_function_index], compensators[:, :, out_function_index], outputs[:, out_function_index], is_trajectories[:, out_function_index] = self.sample_deep_is_trajectory_until_t_stop(curr_times, curr_t_stop, n_time_samples, out_function_index)
        return states, reaction_counts, compensators, outputs, is_trajectories

    def sample_final_deep_is_trajectory(self, t_stop: float, n_time_samples: int) -> tuple:
        """Sample a single DeepIS trajectory until a given t_stop."""
        times = np.linspace(0, t_stop, n_time_samples)

        states, reaction_counts, compensators, outputs, is_trajectories = self.sample_deep_is_trajectory_for_all_output_functions(times, t_stop, n_time_samples)

        return times, states, reaction_counts, compensators, outputs, is_trajectories

    @staticmethod
    def process_final_deep_is_sample(network: ReactionNetwork, model: ExpectationModel, val_config: dict, t_stop: float, n_time_samples: int, active_trajectories: set[str], idx: int, seed: int) -> dict:
        """Process a single DeepIS sample. Needed for multiprocessing."""
        local_network = copy.deepcopy(network)
        # Use a new random seed for each trajectory (which means we need a new simulator):
        local_simulator = ExpectationGeneratorIS(local_network, model, val_config, seed=seed)
        times, states, reaction_counts, compensators, outputs, is_trajectories = local_simulator.sample_final_deep_is_trajectory(t_stop, n_time_samples)

        result = {
            "times": times,
            "state_trajectories": states if "state_trajectories" in active_trajectories else None,
            "output_trajectories": outputs if "output_trajectories" in active_trajectories else None,
            "is_trajectories": is_trajectories if "is_trajectories" in active_trajectories else None,
            "idx": idx,
        }

        if "martingale_trajectories" in active_trajectories:
            result["martingale_trajectories"] = reaction_counts - compensators

        return result

    @execution_timer
    @array_memory_logger
    def sample_final_deep_is_trajectories(self, t_stop: float, n_time_samples: int, n_trajectories: int, active_trajectories: set[str], n_jobs: int = None) -> dict:
        """Create several DeepIS trajectories until a given final time."""
        # Define shapes of all possible trajectory arrays:
        trajectory_shapes = {
            "state_trajectories": (n_trajectories, n_time_samples, self.network.n_species, self.network.out_function_size),
            "martingale_trajectories": (n_trajectories, n_time_samples, self.network.n_reactions, self.network.out_function_size),
            "output_trajectories": (n_trajectories, n_time_samples, self.network.out_function_size),
            "is_trajectories": (n_trajectories, n_time_samples-1, self.network.out_function_size),
        }

        # Initialize only the active ones:
        trajectories = {name: np.zeros(shape) if name in active_trajectories else None for name, shape in trajectory_shapes.items()}

        timeout = self.val_config["is_simulation_timeout"]

        # NOTE: While highly unlikely, nested RNGs mean that duplicate trajectories are possible.
        seeds = self.rng.choice(2**32, size=n_trajectories, replace=False)
        n_jobs = get_available_cpus(n_jobs)

        # Start a subprocess for each trajectory:
        with concurrent.futures.ProcessPoolExecutor(max_workers=n_jobs) as executor:
            futures = [executor.submit(self.process_final_deep_is_sample, self.network, self.model, self.val_config, t_stop, n_time_samples, active_trajectories, i, seeds[i]) for i in range(n_trajectories)]
            done, not_done = concurrent.futures.wait(futures, timeout=timeout, return_when=concurrent.futures.FIRST_EXCEPTION)  # wait for jobs to complete or timeout

            if not_done:  # skip importance sampling if any futures did not complete
                insert_blank_line()
                logging.warning(f"Some final time importance sampling trajectories did not finish within the timeout of {timeout} seconds.")
                log_without_linebreak("Skipping importance sampling analysis after")
                self.val_config["compute_is_estimate"] = False
                executor.shutdown(wait=False, cancel_futures=True)
                return None

            for future in done:  # collect results from completed futures
                result = future.result(timeout=timeout)

                trajectories["times"] = result["times"]
                for name in active_trajectories:
                    try:
                        trajectories[name][result["idx"]] = result[name]
                    except KeyError:
                        pass

        return trajectories


class ExpectationGeneratorPoissonEM(ExpectationGeneratorEM):
    """Class for computing the ergodic mean of the reaction network expectation using the Poisson LHS."""

    def __init__(self, network: ReactionNetwork, seed: int, model: ExpectationModel, val_config: dict, nn_config: dict) -> None:
        """Initialize with the given network, seed, model, validation configuration, and neural network configuration."""
        super().__init__(network, seed)
        self.model = model
        self.val_config = val_config
        self.nn_config = nn_config
        self.analyzer_spectral = SpectralSubnetAnalyzer(network, model, val_config, nn_config)

    def get_ergodic_mean(self, t_min: float, t_max: float) -> np.ndarray:
        """Compute the ergodic mean of the system using the RTC method and the network's initial state."""
        curr_state = self.run_random_time_change(t_min, self.network.initial_state)
        t_curr = t_min

        internal_times = np.zeros([self.network.n_reactions])
        jump_times = -np.log(self.rng.uniform(0, 1, self.network.n_reactions))
        delta_reactions = np.zeros([self.network.n_reactions])
        ergodic_mean = np.zeros([self.network.out_function_size])

        while 1:
            # Compute the time delta:
            propensities = self.network.propensity_vector(curr_state)  # evaluate propensities at every time point
            for k, prop in enumerate(propensities):  # make time interval between reactions infinite if propensity is 0
                delta_reactions[k] = (jump_times[k] - internal_times[k]) / prop if prop > 0 else np.inf

            next_reaction = np.argmin(delta_reactions, axis=0)  # find shortest reaction delta to determine next reaction (returns index)
            delta_time = min(delta_reactions[next_reaction], t_max)  # access time until the found next reaction occurs (returns value)
            internal_times += propensities * delta_time  # update internal times with propensities and time delta

            t_curr += delta_time
            ergodic_mean += np.squeeze(self.network.output_function(curr_state.reshape(1, -1)) + self.analyzer_spectral.poisson_lhs(curr_state)) * delta_time

            if t_curr < t_max:  # update the state as long as the stop time is not exceeded
                curr_state = self.network.update_state(next_reaction, curr_state)
                jump_times[next_reaction] += -np.log(self.rng.uniform(0, 1))
            else:  # if the stop time is exceeded, return the final state
                return ergodic_mean / (t_max - t_min)

    @staticmethod
    def process_ergodic_mean_sample(network: ReactionNetwork, model: ExpectationModel, val_config: dict, nn_config: dict, t_min: float, t_max: float, idx: int, seed: int) -> np.ndarray:
        """Process an ergodic mean sample. Needs to be a static method for multiprocessing."""
        local_network = copy.deepcopy(network)
        # Use a new random seed for each trajectory (which means we need a new simulator):
        local_simulator = ExpectationGeneratorPoissonEM(local_network, seed, model, val_config, nn_config)
        return local_simulator.get_ergodic_mean(t_min, t_max), idx

    @execution_timer
    def get_ergodic_mean_statistics(self, t_min: float, t_max: float, n_trajectories: int, prefix: str, results_dict: dict, n_jobs: int = None) -> None:
        """Compute the average ergodic mean of the system using the RTC method."""
        ergodic_mean = np.zeros([n_trajectories, self.network.out_function_size])

        # NOTE: While highly unlikely, nested RNGs mean that duplicate trajectories are possible.
        seeds = self.rng.choice(2**32, size=n_trajectories, replace=False)
        n_jobs = get_available_cpus(n_jobs)

        # Start a subprocess for each trajectory:
        with concurrent.futures.ProcessPoolExecutor(max_workers=n_jobs) as executor:
            futures = [executor.submit(self.process_ergodic_mean_sample, self.network, self.model, self.val_config, self.nn_config, t_min, t_max, i, seeds[i]) for i in range(n_trajectories)]

            with tqdm(total=n_trajectories, desc="ergodic_means", dynamic_ncols=True) as pbar:
                for future in concurrent.futures.as_completed(futures):
                    result, idx = future.result()
                    ergodic_mean[idx, :] = result
                    pbar.update(1)

        ergodic_mean_average = np.mean(ergodic_mean, axis=0)
        ergodic_mean_ci = 1.96 * np.std(ergodic_mean, ddof=1, axis=0) / np.sqrt(n_trajectories)
        ergodic_mean_variance = np.var(ergodic_mean, ddof=1, axis=0)

        results_dict.update({
            f"{prefix}_em_mean": ergodic_mean_average,
            f"{prefix}_em_ci": ergodic_mean_ci,
            f"{prefix}_em_var": ergodic_mean_variance,
            f"{prefix}_em_samples": n_trajectories,
            f"{prefix}_em_t_min": t_min,
            f"{prefix}_em_t_max": t_max,
        })


class SpectralSubnetAnalyzer(SubnetAnalyzer):
    """Class for analyzing the trained spectral decomposition subnets."""

    def __init__(self, network: ReactionNetwork, model: ExpectationModel, val_config: dict, nn_config: dict, analysis_mode: bool | None = None) -> None:
        """Initialize with the given network, model, validation configuration, neural network configuration, and analysis mode."""
        super().__init__(network, model, val_config, analysis_mode)
        self.nn_config = nn_config

    @execution_timer
    def evaluate_nn_function_coordinates(self, state: np.ndarray, results_dict: dict) -> None:
        """Evaluate function coordinates using the trained model."""
        state = state.reshape(1, -1)
        state_tensor = torch.from_numpy(state).unsqueeze(0)

        # Get the c_m eigenfunction for the given state and reshape it:
        c_m, d_m = self.main_net.estimate_eigenfunctions(state_tensor)
        c_m, d_m = c_m.squeeze(0).unsqueeze(-1), d_m.squeeze(0).unsqueeze(-1)  # get rid of time dimension and add output_function dimension

        # Compute LHS and RHS of the function coordinate self-consistency test:
        lhs = self.network.output_function(state)
        if self.model.v_subnet_architecture == "spectral_matched_simplified":  # no imaginary function coordinate
            rhs = self.model.v_stationary_mean + 2 * torch.sum(self.model.func_coord_real*c_m, dim=1)
        elif self.model.v_subnet_architecture in ["spectral_matched_full", "spectral_complex"]:  # with imaginary function coordinate
            if self.main_net.pairing_mode == "paired":  # using spectral_matched_full mode
                rhs = self.model.v_stationary_mean + 2 * torch.sum(self.model.func_coord_real*c_m - self.model.func_coord_imag*d_m, dim=1)
            elif self.main_net.pairing_mode == "unconstrained":  # without creating/assuming complex conjugate pairs
                rhs = self.model.v_stationary_mean + torch.sum((self.model.func_coord_real + 1j * self.model.func_coord_imag) * (c_m + 1j * d_m), dim=1).real

        results_dict.update({
            "function_coordinate_lhs": lhs,
            "function_coordinate_rhs": rhs.detach().numpy(),
        })

    @execution_timer
    def evaluate_nn_gen_eigenfunctions(self, state: np.ndarray, results_dict: dict) -> None:
        """Evaluate the generator eigenfunctions using the trained model."""
        state_prop = torch.from_numpy(self.network.propensity_vector(state).reshape(1, -1))
        state_tensor = torch.from_numpy(state.reshape(1, -1)).unsqueeze(0)

        # Get the decay modes of the model:
        decay_real = nn.functional.softplus(self.model.decay_real)
        decay_imag = self.model.decay_imag

        # Get the eigenfunctions for the given state and reshape it:
        c_m, d_m = self.main_net.estimate_eigenfunctions(state_tensor)
        c_m, d_m = c_m.squeeze(0), d_m.squeeze(0)  # get rid of time dimension

        # Get dc_m and dd_m for the given state:
        if self.nn_config["v"]["analysis_subnets"] == "shared":
            dc_m = torch.zeros(1, self.nn_config["v"]["n_spectral_terms"], self.network.n_reactions)
            dd_m = torch.zeros(1, self.nn_config["v"]["n_spectral_terms"], self.network.n_reactions)
            for reaction, row in enumerate(self.model.stoichiometry_matrix):
                zeta_state = state_tensor + row
                zeta_c_m, zeta_d_m = self.main_net.estimate_eigenfunctions(zeta_state)
                zeta_c_m, zeta_d_m = zeta_c_m.squeeze(0), zeta_d_m.squeeze(0)  # get rid of time dimension
                dc_m[:, :, reaction] = zeta_c_m - c_m
                dd_m[:, :, reaction] = zeta_d_m - d_m
        elif self.nn_config["v"]["analysis_subnets"] == "distinct":
            dc_m, dd_m = self.diff_net.estimate_eigenfunctions(state_tensor)
            dc_m, dd_m = dc_m.squeeze(0), dd_m.squeeze(0)  # get rid of time dimension

        # Compute the LHS and RHS of the eigenfunction equation for c:
        lhs_c = torch.sum(state_prop * dc_m, dim=-1)
        rhs_c = - decay_real * c_m + decay_imag * d_m

        # Compute the LHS and RHS of the eigenfunction equation for d:
        lhs_d = torch.sum(state_prop * dd_m, dim=-1)
        rhs_d = - decay_real * d_m - decay_imag * c_m

        results_dict.update({
            "gen_eigenfunction_lhs_c": lhs_c.detach().numpy(),
            "gen_eigenfunction_rhs_c": rhs_c.detach().numpy(),
            "gen_eigenfunction_lhs_d": lhs_d.detach().numpy(),
            "gen_eigenfunction_rhs_d": rhs_d.detach().numpy(),
        })

    @execution_timer
    def evaluate_nn_exp_eigenfunctions(self, inputs: dict, state: np.ndarray, results_dict: dict) -> None:
        """Evaluate the expectation eigenfunctions using the trained model."""
        state = torch.from_numpy(state.reshape(1, -1)).unsqueeze(0)
        times = torch.from_numpy(inputs["times"]).unsqueeze(-1)
        state_trajectories = inputs["state_trajectories"]

        # Get the exp, cos and sin of the decay modes:
        exp_real = torch.exp(-nn.functional.softplus(self.model.decay_real) * times)
        cos_imag = torch.cos(self.model.decay_imag * times)
        sin_imag = torch.sin(self.model.decay_imag * times)

        # Get the eigenfunctions for the trajectories:
        c_m_traj, d_m_traj = self.main_net.estimate_eigenfunctions(torch.from_numpy(state_trajectories))

        # Get the eigenfunctions for the given state and reshape them:
        c_m, d_m = self.main_net.estimate_eigenfunctions(state)
        c_m, d_m = c_m.squeeze(0), d_m.squeeze(0)  # get rid of time dimension

        # Compute the LHS (with CI) and RHS of the eigenfunction equation for c:
        lhs_c = torch.mean(c_m_traj, dim=0)
        rhs_c = exp_real * (c_m * cos_imag + d_m * sin_imag)
        lhs_c_ci = 1.96 * torch.std(c_m_traj, dim=0) / np.sqrt(state_trajectories.shape[0])

        # Compute the LHS (with CI) and RHS of the eigenfunction equation for d:
        lhs_d = torch.mean(d_m_traj, dim=0)
        rhs_d = exp_real * (d_m * cos_imag - c_m * sin_imag)
        lhs_d_ci = 1.96 * torch.std(d_m_traj, dim=0) / np.sqrt(state_trajectories.shape[0])

        results_dict.update({
            "exp_eigenfunction_lhs_c": lhs_c.detach().numpy(),
            "exp_eigenfunction_rhs_c": rhs_c.detach().numpy(),
            "exp_eigenfunction_lhs_c_ci": lhs_c_ci.detach().numpy(),
            "exp_eigenfunction_lhs_d": lhs_d.detach().numpy(),
            "exp_eigenfunction_rhs_d": rhs_d.detach().numpy(),
            "exp_eigenfunction_lhs_d_ci": lhs_d_ci.detach().numpy(),
        })

    def poisson_lhs(self, state: np.ndarray) -> None:
        """Compute the Poisson equation LHS using the trained model."""
        state_prop = torch.from_numpy(self.network.propensity_vector(state).reshape(1, -1))
        state_tensor = torch.from_numpy(state.reshape(1, -1)).unsqueeze(0)

        # Get the decay modes of the model:
        decay_real = nn.functional.softplus(self.model.decay_real).unsqueeze(-1)
        decay_imag = self.model.decay_imag.unsqueeze(-1)

        # Get the eigenfunctions for the given state and reshape it:
        c_m, d_m = self.main_net.estimate_eigenfunctions(state_tensor)
        c_m, d_m = c_m.squeeze(0), d_m.squeeze(0)  # get rid of time dimension

        # Get dc_m and dd_m for the given state:
        if self.nn_config["v"]["analysis_subnets"] == "shared":
            dc_m = torch.zeros(1, self.nn_config["v"]["n_spectral_terms"], self.network.n_reactions)
            dd_m = torch.zeros(1, self.nn_config["v"]["n_spectral_terms"], self.network.n_reactions)
            for reaction, row in enumerate(self.model.stoichiometry_matrix):
                zeta_state = state_tensor + row
                zeta_c_m, zeta_d_m = self.main_net.estimate_eigenfunctions(zeta_state)
                zeta_c_m, zeta_d_m = zeta_c_m.squeeze(0), zeta_d_m.squeeze(0)  # get rid of time dimension
                dc_m[:, :, reaction] = zeta_c_m - c_m
                dd_m[:, :, reaction] = zeta_d_m - d_m
        elif self.nn_config["v"]["analysis_subnets"] == "distinct":
            dc_m, dd_m = self.diff_net.estimate_eigenfunctions(state_tensor)
            dc_m, dd_m = dc_m.squeeze(0), dd_m.squeeze(0)  # get rid of time dimension

        # Compute the LHS of the Poisson equation:
        if self.model.v_subnet_architecture == "spectral_matched_simplified":
            delta_poiss = torch.sum(self.model.func_coord_real.unsqueeze(3) * ((decay_real*dc_m + decay_imag*dd_m) / (decay_real**2 + decay_imag**2)).unsqueeze(2), dim=1)
            poiss = 2 * torch.sum(state_prop * delta_poiss, dim=-1)
        elif self.model.v_subnet_architecture in ["spectral_matched_full", "spectral_complex"]:
            if self.main_net.pairing_mode == "paired":
                delta_poiss = torch.sum((self.model.func_coord_real.unsqueeze(3) * (decay_real*dc_m + decay_imag*dd_m).unsqueeze(2) + self.model.func_coord_imag.unsqueeze(3) * (decay_imag*dc_m - decay_real*dd_m).unsqueeze(2)) / (decay_real**2 + decay_imag**2).unsqueeze(2), dim=1)
                poiss = 2 * torch.sum(state_prop * delta_poiss, dim=-1)
            elif self.main_net.pairing_mode == "unconstrained":
                delta_poiss = torch.sum((self.model.func_coord_real + 1j * self.model.func_coord_imag).unsqueeze(3) * (dc_m + 1j * dd_m).unsqueeze(2) / (decay_real + 1j * decay_imag).unsqueeze(2), dim=1)
                poiss = torch.sum(state_prop * delta_poiss, dim=-1).real

        return poiss.detach().numpy()

    @execution_timer
    def evaluate_nn_poisson(self, state: np.ndarray, results_dict: dict) -> None:
        """Evaluate the Poisson equation using the trained model."""
        state_tensor = torch.from_numpy(state.reshape(1, -1)).unsqueeze(0)
        # Get the c_m eigenfunction for the given state and reshape it:
        c_m, d_m = self.main_net.estimate_eigenfunctions(state_tensor)
        c_m, d_m = c_m.squeeze(0).unsqueeze(-1), d_m.squeeze(0).unsqueeze(-1)  # get rid of time dimension and add output_function dimension

        # Compare the output of the network with model output:
        lhs = self.poisson_lhs(state)
        if self.model.v_subnet_architecture == "spectral_matched_simplified":
            rhs = - 2 * torch.sum(self.model.func_coord_real*c_m, dim=1)
        elif self.model.v_subnet_architecture in ["spectral_matched_full", "spectral_complex"]:
            if self.main_net.pairing_mode == "paired":
                rhs = - 2 * torch.sum(self.model.func_coord_real*c_m - self.model.func_coord_imag*d_m, dim=1)
            elif self.main_net.pairing_mode == "unconstrained":
                rhs = - torch.sum((self.model.func_coord_real + 1j * self.model.func_coord_imag) * (c_m + 1j * d_m), dim=1).real

        results_dict.update({
            "poisson_lhs": lhs,
            "poisson_rhs": rhs.detach().numpy(),
        })
