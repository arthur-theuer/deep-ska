"""Definition of the model class for training and analysis."""

import logging

import torch
from torch import Tensor, nn

from .. import subnets
from ..logging.helpers import format_mean_ci_arrays, insert_blank_line, log_without_linebreak
from ..reaction_networks.definition import ReactionNetwork
from .simulation import ExpectationGeneratorEM


def inverse_softplus(data: list, beta: float = 1.0, threshold: float = 20.0) -> Tensor:
    """Computes the inverse of the nn.functional.softplus function for a list."""
    tensor = torch.tensor(data, dtype=torch.float64)
    # Use the same threshold as in the softplus function and use expm1 to avoid numerical instability:
    return torch.where(beta * tensor > threshold, tensor, (1 / beta) * torch.log(torch.expm1(beta * tensor)))


class Model(nn.Module):
    """General model initialization for training on a reaction network."""

    def __init__(self, network: ReactionNetwork, nn_config: dict) -> None:
        """Initialize the model with the given reaction network and configuration."""
        super().__init__()
        self.network = network
        self.nn_config = nn_config

        # The model needs the stoichiometry matrix as a torch tensor:
        self.stoichiometry_matrix = torch.from_numpy(self.network.stoichiometry_matrix)


class ExpectationModel(Model):
    """Model for training V."""

    def __init__(self, network: ReactionNetwork, torch_rng: torch.Generator, nn_config: dict) -> None:
        """Initialize the model with the given reaction network and configuration."""
        super().__init__(network, nn_config)
        # Initialization for learning V:
        self.v_as_relationship = self.nn_config["v"]["as_relationship"]
        self.v_forward_subnets = self.nn_config["v"]["forward_subnets"]
        self.v_subnet_architecture = self.nn_config["v"]["subnet_architecture"]

        if self.v_as_relationship == "identity":
            self.forward = self.forward_v
        elif self.v_as_relationship == "logarithm":
            self.forward = self.forward_log_v
        else:
            raise ValueError(f'Relationship type "{self.v_as_relationship}" is invalid for V.')

        self.v_stationary_mean = self._initialize_v_stationary_mean(torch_rng)
        self.func_coord_real, self.func_coord_imag = self._initialize_v_function_coordinates()
        self.decay_real, self.decay_imag, self.decay_phas = self._initialize_v_decay_modes(torch_rng)

        self.main_net = self._v_subnet_class()(
            self.network.n_species, self.network.n_reactions, self.network.out_function_size,
            self.nn_config, as_relationship=self.v_as_relationship, int_net=False)

        self.main_net_args = self._v_subnet_args(self.main_net)

        if self.v_subnet_architecture == "spectral_complex":
            self.main_net.pairing_mode = "unconstrained"
        if self.nn_config["v"]["use_exact_eigenfunctions"]:  # only works for certain reaction networks
            self.main_net.forward = self.network.get_moments_exact

        if self.v_forward_subnets == "shared":
            self.diff_net = None
            self.quot_net = None

        elif self.v_forward_subnets == "distinct":
            if self.v_as_relationship == "identity":
                self.diff_net = self._v_subnet_class()(
                    self.network.n_species, self.network.n_reactions, self.network.out_function_size,
                    self.nn_config, as_relationship=self.v_as_relationship, int_net=True)
                self.quot_net = None

                self.diff_net_args = self._v_subnet_args(self.diff_net)

                if self.v_subnet_architecture == "spectral_complex":
                    self.diff_net.pairing_mode = "unconstrained"
                if self.nn_config["v"]["use_exact_eigenfunctions"]:  # only works for certain reaction networks
                    self.diff_net.forward = self.network.get_moments_exact

            elif self.v_as_relationship == "logarithm":
                self.diff_net = None
                self.quot_net = self._v_subnet_class()(
                    self.network.n_species, self.network.n_reactions, self.network.out_function_size,
                    self.nn_config, as_relationship=self.v_as_relationship, int_net=True)

                self.quot_net_args = self._v_subnet_args(self.quot_net)

                if self.v_subnet_architecture == "spectral_complex":
                    self.quot_net.pairing_mode = "unconstrained"
                if self.nn_config["v"]["use_exact_eigenfunctions"]:  # only works for certain reaction networks
                    self.quot_net.forward = self.network.get_moments_exact

            else:
                raise ValueError(f'Relationship type "{self.v_as_relationship}" is invalid for V.')

        else:
            raise ValueError(f'Forward subnet type "{self.v_forward_subnets}" is invalid for V.')

        # Placeholder main subnet for learning the sensitivity:
        self.sens_main_net = subnets.s_subnets.PlaceholderSensitivitySubnet(
            self.network.n_species, self.network.n_reactions, self.network.sel_param_size,
            self.nn_config, int_net=False)

        # Placeholder difference subnet for learning the sensitivity:
        self.sens_diff_net = subnets.s_subnets.PlaceholderSensitivitySubnet(
            self.network.n_species, self.network.n_reactions, self.network.sel_param_size,
            self.nn_config, int_net=True)

    def _compute_ergodic_means(self) -> Tensor:
        """Wrapper for the simulation function to log all necessary values."""
        results = {}

        insert_blank_line()
        log_without_linebreak(f"Computing the 'Ergodic mean' for model initialization using {self.nn_config['v']['erg_n_trajectories']} trajectories ...")
        em_generator = ExpectationGeneratorEM(self.network, self.nn_config["training_seed"])
        em_generator.get_ergodic_mean_statistics(self.nn_config["v"]["erg_t_min"], self.nn_config["v"]["erg_t_max"], self.nn_config["v"]["erg_n_trajectories"], "model", results)
        logging.info(f"Average ergodic mean (model initialization): {format_mean_ci_arrays(results['model_em_mean'], results['model_em_ci'])} (95% CI, {results['model_em_samples']} samples)")

        return torch.tensor(results["model_em_mean"], dtype=torch.float64)

    def _initialize_v_stationary_mean(self, torch_rng: torch.Generator) -> Tensor:
        """Only initialize stationary mean if spectral decomposition or normalization is used."""
        if self.v_subnet_architecture not in ["spectral_matched_lumped", "spectral_matched_simplified", "spectral_matched_full", "spectral_complex"]:
            if self.nn_config["v"]["normalization"] != "ergodic_mean":
                return None

        def try_get_exact() -> Tensor:
            """Try to compute exact stationary mean, fallback to estimate if it fails."""
            try:
                return torch.tensor(self.network.get_stationary_mean_exact(), dtype=torch.float64).unsqueeze(0)
            except ValueError as e:
                logging.warning(f"{e} Falling back to estimated stationary mean.")
                return self._compute_ergodic_means().unsqueeze(0)

        stationary_mean_initializers = {
            "zeros": lambda: nn.Parameter(torch.zeros([1, self.network.out_function_size])),
            "ones": lambda: nn.Parameter(torch.ones([1, self.network.out_function_size])),
            "rand": lambda: nn.Parameter(torch.rand([1, self.network.out_function_size], generator=torch_rng)),
            "estimate_trainable": lambda: nn.Parameter(self._compute_ergodic_means().unsqueeze(0)),
            "exact_trainable": lambda: nn.Parameter(try_get_exact()),
            "estimate": lambda: self._compute_ergodic_means().unsqueeze(0),
            "exact": lambda: try_get_exact()
        }
        return stationary_mean_initializers.get(self.nn_config["v"].get("stationary_mean_initialization"), lambda: None)()

    def _initialize_v_function_coordinates(self) -> tuple:
        """Only initialize the function coordinate if spectral decomposition is used."""
        if self.v_subnet_architecture not in ["spectral_matched_lumped", "spectral_matched_simplified", "spectral_matched_full", "spectral_complex"]:
            func_coord_real = None
            func_coord_imag = None
        elif self.v_subnet_architecture in ["spectral_matched_lumped", "spectral_matched_simplified"]:
            func_coord_real = nn.Parameter(torch.ones([1, self.nn_config["v"]["n_spectral_terms"], self.network.out_function_size]))
            func_coord_imag = None
        elif self.v_subnet_architecture in ["spectral_matched_full", "spectral_complex"]:
            func_coord_real = nn.Parameter(torch.ones([1, self.nn_config["v"]["n_spectral_terms"], self.network.out_function_size]))
            func_coord_imag = nn.Parameter(torch.ones([1, self.nn_config["v"]["n_spectral_terms"], self.network.out_function_size]))

        return func_coord_real, func_coord_imag

    def _initialize_v_decay_modes(self, torch_rng: torch.Generator) -> tuple:
        """Initialize decay modes for training temporal dynamics."""
        if self.v_subnet_architecture in ["spectral_matched_lumped", "spectral_matched_simplified", "spectral_matched_full", "spectral_complex"]:
            if self.nn_config["v"]["decay_modes_initialization"] == "rand":
                decay_real = nn.Parameter(torch.rand([1, self.nn_config["v"]["n_spectral_terms"]], generator=torch_rng))
                decay_imag = nn.Parameter(torch.zeros([1, self.nn_config["v"]["n_spectral_terms"]]))
            elif self.nn_config["v"]["decay_modes_initialization"] == "trainable":  # inverse of the softplus for decay_real with custom function
                decay_real = nn.Parameter(inverse_softplus(self.nn_config["v"]["decay_modes_real"]).unsqueeze(0))
                decay_imag = nn.Parameter(torch.tensor(self.nn_config["v"]["decay_modes_imag"], dtype=torch.float64).unsqueeze(0))
            elif self.nn_config["v"]["decay_modes_initialization"] == "fixed":  # inverse of the softplus for decay_real with custom function
                decay_real = inverse_softplus(self.nn_config["v"]["decay_modes_real"]).unsqueeze(0)
                decay_imag = torch.tensor(self.nn_config["v"]["decay_modes_imag"], dtype=torch.float64).unsqueeze(0)
            decay_phas = None
        else:
            decay_real = nn.Parameter(torch.rand([1, self.nn_config["v"]["n_temporal_features"]], generator=torch_rng))
            decay_imag = nn.Parameter(torch.zeros([1, self.nn_config["v"]["n_temporal_features"]]))
            decay_phas = nn.Parameter(torch.zeros([1, self.nn_config["v"]["n_temporal_features"]]))

        return decay_real, decay_imag, decay_phas

    def _v_subnet_class(self) -> nn.Module:
        """Determine the class to use for all subnetwork types."""
        if self.v_subnet_architecture == "naive":
            return subnets.v_subnets.NaiveExpectationSubnet
        elif self.v_subnet_architecture == "features":
            return subnets.v_subnets.TemporalFeatureExpectationSubnet
        elif self.v_subnet_architecture in ["spectral_matched_lumped", "spectral_matched_simplified", "spectral_matched_full", "spectral_complex"]:
            return subnets.v_subnets.SpectralDecompositionExpectationSubnet
        else:
            raise ValueError(f'Subnet architecture "{self.v_subnet_architecture}" is invalid for V.')

    def _v_subnet_args(self, subnet: nn.Module) -> tuple:
        """Get arguments dynamically based on the arguments required by the given subnet type."""
        if subnet is None:
            return None  # if the subnet is not initialized

        args_dict = {
            "stationary_mean": self.v_stationary_mean,
            "func_coord_real": self.func_coord_real,
            "func_coord_imag": self.func_coord_imag,
            "decay_real": self.decay_real,
            "decay_imag": self.decay_imag,
            "decay_phas": self.decay_phas,
            "network": self.network,
            "v_main_net": self.main_net,
        }
        return tuple(args_dict[arg] for arg in subnet.required_subnet_args())

    def forward_v(self, inputs: dict) -> tuple:
        """Vectorized forward pass through the network."""
        times = inputs["times"]
        state_trajectories = inputs["state_trajectories"]
        martingale_trajectories = inputs["martingale_trajectories"]
        propensity_trajectories = inputs["propensity_trajectories"]

        batch_time_samples = times.shape[0]  # get the number of time samples
        batch_size = state_trajectories.shape[0]  # get the number of trajectories
        t_stop = times[-1]  # get the final time point

        # Convert necessary data to torch tensors:
        times = torch.from_numpy(times)
        state_trajectories = torch.from_numpy(state_trajectories)
        martingale_trajectories = torch.from_numpy(martingale_trajectories)

        # Get V values for all time points:
        full_V = self.main_net(state_trajectories[:, :-1, :], t_stop, times[:-1], *self.main_net_args)
        full_V = full_V.view(batch_size, batch_time_samples-1, self.network.out_function_size)

        if self.nn_config["v"]["forward_subnets"] == "shared":
            # Initialize the matrix for computing dV:
            full_dV = torch.zeros(batch_size, batch_time_samples-1, self.network.out_function_size, self.network.n_reactions)
            for reaction, row in enumerate(self.stoichiometry_matrix):
                zeta_states = state_trajectories[:, :-1, :] + row  # NOTE: This can lead to negative states.
                zeta_V = self.main_net(zeta_states, t_stop, times[:-1], *self.main_net_args)
                zeta_V = zeta_V.view(batch_size, batch_time_samples-1, self.network.out_function_size)
                full_dV[:, :, :, reaction] = zeta_V - full_V
        elif self.nn_config["v"]["forward_subnets"] == "distinct":
            # Compute dV using a second subnet:
            full_dV = self.diff_net(state_trajectories[:, :-1, :], t_stop, times[:-1], *self.diff_net_args)
            full_dV = full_dV.view(batch_size, batch_time_samples-1, self.network.out_function_size, self.network.n_reactions)

        # Compute the difference in the martingale between current and next time point and preserve dimensions:
        martingale_increment = (martingale_trajectories[:, 1:, :] - martingale_trajectories[:, :-1, :]).unsqueeze(2)
        # Save the currently approximated integral and current V:
        full_intervals = full_dV * martingale_increment

        # Compute reverse cumulative sum of intervals over time (flip, cumsum, flip back):
        cumulative_intervals = torch.cumsum(full_intervals.flip(dims=[1]), dim=1).flip(dims=[1])
        if self.nn_config["v"]["single_pred_term"]:
            id_pred = full_V[:, 0, :] + torch.sum(cumulative_intervals[:, 0, :, :], dim=-1).unsqueeze(1)  # compute only first term
        else:
            id_pred = full_V + torch.sum(cumulative_intervals, dim=-1)  # compute all terms

        if self.nn_config["v"]["loss_function"] in ["pinn_loss", "id_pinn_loss"]:
            propensity_trajectories = torch.from_numpy(propensity_trajectories)
            final_output = torch.from_numpy(self.network.output_function(state_trajectories[:, -1, :]))

            der_shape = (batch_size, batch_time_samples-1, self.network.out_function_size)
            d_times = (times[1:] - times[:-1]).reshape(1, batch_time_samples-1, 1).expand(*der_shape)

            if self.nn_config["v"]["subnet_architecture"] in ["spectral_matched_lumped", "spectral_matched_simplified", "spectral_matched_full", "spectral_complex"]:
                Vdot = self.main_net.estimate_time_derivative(state_trajectories[:, :-1, :], t_stop, times[:-1], *self.main_net_args)
                Vdot = Vdot.view(*der_shape) * d_times
            else:
                if self.nn_config["v"]["exact_time_derivative"]:  # compute the derivative of the subnet with respect to time
                    Vdot = torch.zeros(*der_shape)
                    def main_net_output(t: float) -> Tensor:
                        """Function for calling the subnet to compute the Jacobian."""
                        return self.main_net(curr_states, t_stop, t.unsqueeze(0), *self.main_net_args)
                    for t in range(batch_time_samples-1):
                        curr_states = state_trajectories[:, t, :].unsqueeze(1).detach()
                        Vdot[:, t, :] = torch.autograd.functional.jacobian(func=main_net_output, inputs=times[t], vectorize=True, strategy="forward-mode")
                    Vdot = Vdot * d_times
                else:  # approximate the derivative of the subnet with respect to time
                    next_V = self.main_net(state_trajectories[:, :-1, :], t_stop, times[1:], *self.main_net_args)
                    next_V = next_V.view(*der_shape)
                    Vdot = (next_V - full_V)

            # Get the precomputed propensities for all time points and preserve dimensions:
            full_propensities = propensity_trajectories[:, :-1, :].reshape(batch_size, batch_time_samples-1, 1, self.network.n_reactions).expand(*der_shape, self.network.n_reactions)
            # Compute the CTMC generator matrix for all time points:
            genm_V = torch.sum(full_propensities * full_dV, dim=-1) * d_times

            # Update pinn_pred using the CTMC generator matrix and the derivative of the subnet:
            if self.nn_config["v"]["square_pinn_once"]:
                pinn_pred = torch.abs(torch.sum(Vdot + genm_V, dim=1))**2  # update pinn_pred without squaring each term and only square in the end
            else:
                pinn_pred = torch.sum(torch.abs(Vdot + genm_V)**2, dim=1)  # update pinn_pred with squaring each term

            # Add constraint to pinn_pred before returning it for loss calculation:
            final_states = state_trajectories[:, -1, :].unsqueeze(1)
            final_V = self.main_net(final_states, t_stop, times[-1].unsqueeze(0), *self.main_net_args)
            final_V = final_V.view(batch_size, self.network.out_function_size)
            pinn_pred = pinn_pred + torch.abs(final_V - final_output)**2
        else:
            pinn_pred = torch.tensor([0.])

        return id_pred, pinn_pred

    def forward_deep_is_resampling(self, inputs: dict) -> tuple:
        """Vectorized forward pass through the network used with DeepIS resampling."""
        times = inputs["times"]
        state_trajectories = inputs["state_trajectories"]
        martingale_trajectories = inputs["martingale_trajectories"]

        batch_time_samples = times.shape[0]  # get the number of time samples
        batch_size = state_trajectories.shape[0]  # get the number of trajectories
        t_stop = times[-1]  # get the final time point

        # Convert necessary data to torch tensors:
        times = torch.from_numpy(times)
        state_trajectories = torch.from_numpy(state_trajectories)
        martingale_trajectories = torch.from_numpy(martingale_trajectories)

        outer_full_V = torch.zeros(batch_size, batch_time_samples-1, self.network.out_function_size, dtype=torch.complex64)
        full_intervals = torch.zeros(batch_size, batch_time_samples-1, self.network.out_function_size, self.network.n_reactions, dtype=torch.complex64)

        for i in range(self.network.out_function_size):
            # Get V values for all time points:
            full_V = self.main_net(state_trajectories[:, :-1, :, i], t_stop, times[:-1], *self.main_net_args)
            outer_full_V[:, :, i] = full_V.view(batch_size, batch_time_samples-1, self.network.out_function_size)[:, :, i]

            if self.nn_config["v"]["forward_subnets"] == "shared":
                # Initialize the matrix for computing dV:
                full_dV = torch.zeros(batch_size, batch_time_samples-1, self.network.n_reactions)
                for reaction, row in enumerate(self.stoichiometry_matrix):
                    zeta_states = state_trajectories[:, :-1, :, i] + row  # NOTE: This can lead to negative states.
                    zeta_V = self.main_net(zeta_states, t_stop, times[:-1], *self.main_net_args)
                    zeta_V = zeta_V.view(batch_size, batch_time_samples-1, self.network.out_function_size)[:, :, i]
                    full_dV[:, :, reaction] = zeta_V - full_V[:, :, i]
            elif self.nn_config["v"]["forward_subnets"] == "distinct":
                # Compute dV using a second subnet:
                full_dV = self.diff_net(state_trajectories[:, :-1, :, i], t_stop, times[:-1], *self.diff_net_args)
                full_dV = full_dV.view(batch_size, batch_time_samples-1, self.network.out_function_size, self.network.n_reactions)[:, :, i, :]

            # Compute the difference in the martingale between current and next time point and preserve dimensions:
            martingale_increment = (martingale_trajectories[:, 1:, :, i] - martingale_trajectories[:, :-1, :, i])
            # Save the currently approximated integral and current V:
            full_intervals[:, :, i, :] = full_dV * martingale_increment

        # Compute reverse cumulative sum of intervals over time (flip, cumsum, flip back):
        cumulative_intervals = torch.cumsum(full_intervals.flip(dims=[1]), dim=1).flip(dims=[1])
        if self.nn_config["v"]["single_pred_term"]:
            id_pred = outer_full_V[:, 0, :] + torch.sum(cumulative_intervals[:, 0, :, :], dim=-1).unsqueeze(1)  # compute only first term
        else:
            id_pred = outer_full_V + torch.sum(cumulative_intervals, dim=-1)  # compute all terms

        pinn_pred = torch.tensor([0.])

        return id_pred, pinn_pred

    def forward_log_v(self, inputs: dict) -> Tensor:
        """Vectorized forward pass through the network for the logarithmic relationship."""
        times = inputs["times"]
        state_trajectories = inputs["state_trajectories"]
        reaction_count_trajectories = inputs["reaction_count_trajectories"]
        propensity_trajectories = inputs["propensity_trajectories"]

        batch_time_samples = times.shape[0]  # get the number of time samples
        batch_size = state_trajectories.shape[0]  # get the number of trajectories
        t_stop = times[-1]  # get the final time point

        # Convert necessary data to torch tensors:
        times = torch.from_numpy(times)
        state_trajectories = torch.from_numpy(state_trajectories)
        reaction_count_trajectories = torch.from_numpy(reaction_count_trajectories)
        propensity_trajectories = torch.from_numpy(propensity_trajectories)

        # Get log(V) values for all time points:
        log_full_V = self.main_net(state_trajectories[:, :-1, :], t_stop, times[:-1], *self.main_net_args)
        log_full_V = log_full_V.view(batch_size, batch_time_samples-1, self.network.out_function_size)

        if self.nn_config["v"]["forward_subnets"] == "shared":
            # Initialize the matrix for computing log(qV):
            log_full_qV = torch.zeros(batch_size, batch_time_samples-1, self.network.out_function_size, self.network.n_reactions)
            for reaction, row in enumerate(self.stoichiometry_matrix):
                zeta_states = state_trajectories[:, :-1, :] + row  # NOTE: This can lead to negative states.
                log_zeta_V = self.main_net(zeta_states, t_stop, times[:-1], *self.main_net_args)
                log_zeta_V = log_zeta_V.view(batch_size, batch_time_samples-1, self.network.out_function_size)
                log_full_qV[:, :, :, reaction] = log_zeta_V - log_full_V
        elif self.nn_config["v"]["forward_subnets"] == "distinct":
            # Compute log(qV) using a second subnet:
            log_full_qV = self.quot_net(state_trajectories[:, :-1, :], t_stop, times[:-1], *self.quot_net_args)
            log_full_qV = log_full_qV.view(batch_size, batch_time_samples-1, self.network.out_function_size, self.network.n_reactions)

        # Compute the difference in the reaction count between current and next time point and preserve dimensions:
        reaction_count_increment = (reaction_count_trajectories[:, 1:, :] - reaction_count_trajectories[:, :-1, :]).unsqueeze(2)
        # Save the currently approximated integral and current V:
        full_qV_intervals = log_full_qV * reaction_count_increment

        # Get the precomputed propensities for all time points and preserve dimensions:
        full_propensities = propensity_trajectories[:, :-1, :].unsqueeze(2)
        # Compute second type of intervals for the log(pred) term:
        second_qV_intervals = (1 - torch.exp(log_full_qV)) * full_propensities * (times[1:] - times[:-1]).view(1, batch_time_samples-1, 1, 1)

        # Compute reverse cumulative sum of intervals over time (flip, cumsum, flip back):
        cumulative_qV_intervals = torch.cumsum(full_qV_intervals.flip(dims=[1]), dim=1).flip(dims=[1])
        cumsecond_qV_intervals = torch.cumsum(second_qV_intervals.flip(dims=[1]), dim=1).flip(dims=[1])

        if self.nn_config["v"]["single_pred_term"]:
            log_pred = log_full_V[:, 0, :] + torch.sum(cumulative_qV_intervals[:, 0, :, :] + cumsecond_qV_intervals[:, 0, :, :], dim=-1).unsqueeze(1)  # compute only first term
        else:
            log_pred = log_full_V + torch.sum(cumulative_qV_intervals + cumsecond_qV_intervals, dim=-1)  # compute all terms
        return log_pred
