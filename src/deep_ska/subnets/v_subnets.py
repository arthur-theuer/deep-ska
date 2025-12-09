"""Implements all subnet architectures for learning V."""

import torch
from torch import Tensor, nn


# IDENTITY RELATIONSHIP:          # LOGARITHM RELATIONSHIP:
# V:  no output layer activation  # log(V):  no output layer activation
# ΔV: no output layer activation  # log(ΠV): no output layer activation
class NaiveExpectationSubnet(nn.Module):
    """Naive expectation subnet to compute different quantities related to V.

    This subnet can compute V, ΔV, log(V) and log(ΠV).
    """

    def __init__(
        self,
        n_species: int,
        n_reactions: int,
        out_function_size: int,
        nn_config: dict,
        as_relationship: str = "identity",
        int_net: bool = False
    ) -> None:
        """Initialize the naive expectation subnet."""
        self._validate_subnet_type(as_relationship, int_net)
        super().__init__()
        self.as_relationship = as_relationship
        self.int_net = int_net
        # Get network size from configuration file:
        n_nodes = nn_config["n_nodes_per_layer"]
        n_hidden_layers = nn_config["n_hidden_layers"]
        # Create feedforward layers:
        layers = [nn.Linear(n_species+1, n_nodes), nn.ReLU()]  # input layer with activation
        layers.extend([layer for _ in range(n_hidden_layers) for layer in (nn.Linear(n_nodes, n_nodes), nn.ReLU())])  # hidden layers with activation
        layers.append(nn.Linear(n_nodes, out_function_size) if not int_net else nn.Linear(n_nodes, out_function_size*n_reactions))  # output layer size based on "int_net" flag
        # Create the feedforward network from the layers:
        self.feedforward = nn.Sequential(*layers)  # to be used in the forward pass

    def _validate_subnet_type(self, as_relationship: str, int_net: bool) -> None:
        """Check if combination of as_relationship and int_net values is valid."""
        allowed_combinations = {"identity": {True, False},  # both True and False for int_net are allowed when using "identity" as_relationship
                                "logarithm": {True, False}}  # both True and False for int_net are allowed when using "logarithm" as_relationship
        if as_relationship not in allowed_combinations or int_net not in allowed_combinations[as_relationship]:
            raise ValueError(f'Invalid combination of as_relationship="{as_relationship}" and int_net={int_net}.')

    def required_subnet_args(self) -> list[str]:
        """Return the required subnet arguments for the naive expectation subnet."""
        return []

    def forward(self, states: Tensor, t_stop: float, t: Tensor) -> Tensor:
        """Forward pass through the naive expectation subnet to compute V, ΔV, log(V) or log(ΠV)."""
        t_left = t_stop - t  # compute T-t
        # Create inputs for the subnetwork from the current states and the time left:
        batch_time_samples = t.shape[0]  # get the number of time samples
        batch_size = states.shape[0]  # get the number of trajectories
        t_left = t_left.reshape(1, batch_time_samples, 1).expand(batch_size, batch_time_samples, 1)

        x = torch.cat((states, t_left), dim=2)
        x = x.view(batch_size*batch_time_samples, -1)
        x = self.feedforward(x)  # pass the input matrix through the feedforward network

        return x.view(-1)


# IDENTITY RELATIONSHIP:          # LOGARITHM RELATIONSHIP:
# V:  no output layer activation  # log(V):  no output layer activation
# ΔV: no output layer activation  # log(ΠV): no output layer activation
class TemporalFeatureExpectationSubnet(nn.Module):
    """Temporal feature expectation subnet to compute different quantities related to V.

    This subnet can compute V, ΔV, log(V) and log(ΠV).
    """

    def __init__(
        self,
        n_species: int,
        n_reactions: int,
        out_function_size: int,
        nn_config: dict,
        as_relationship: str = "identity",
        int_net: bool = False
    ) -> None:
        """Initialize the temporal feature expectation subnet."""
        self._validate_subnet_type(as_relationship, int_net)  # throws an error if the combination of arguments is invalid
        super().__init__()
        self.as_relationship = as_relationship  # flag for the almost sure relationship used
        self.int_net = int_net  # flag for integral network
        # Get network size from configuration file:
        n_nodes = nn_config["n_nodes_per_layer"]
        self.n_temporal_features = nn_config["v"]["n_temporal_features"]
        n_hidden_layers = nn_config["n_hidden_layers"]
        # Create feedforward layers:
        layers = [nn.Linear(n_species+2*self.n_temporal_features, n_nodes), nn.ReLU()]  # input layer with activation
        layers.extend([layer for _ in range(n_hidden_layers) for layer in (nn.Linear(n_nodes, n_nodes), nn.ReLU())])  # hidden layers with activation
        layers.append(nn.Linear(n_nodes, out_function_size) if not int_net else nn.Linear(n_nodes, out_function_size*n_reactions))  # output layer size based on "int_net" flag
        # Create the feedforward network from the layers:
        self.feedforward = nn.Sequential(*layers)  # to be used in the forward pass

    def _validate_subnet_type(self, as_relationship: str, int_net: bool) -> None:
        """Check if combination of as_relationship and int_net values is valid."""
        allowed_combinations = {"identity": {True, False},  # both True and False for int_net are allowed when using "identity" as_relationship
                                "logarithm": {True, False}}  # both True and False for int_net are allowed when using "logarithm" as_relationship
        if as_relationship not in allowed_combinations or int_net not in allowed_combinations[as_relationship]:
            raise ValueError(f'Invalid combination of as_relationship="{as_relationship}" and int_net={int_net}.')

    def required_subnet_args(self) -> list[str]:
        """Return the required subnet arguments for the temporal feature expectation subnet."""
        return ["decay_real", "decay_imag", "decay_phas"]

    def forward(
        self, states: Tensor, t_stop: float, t: Tensor,
        decay_real: nn.Parameter, decay_imag: nn.Parameter, decay_phas: nn.Parameter
    ) -> Tensor:
        """Forward pass through the temporal feature expectation subnet to compute V, ΔV, log(V) or log(ΠV)."""
        t_left = t_stop - t  # compute T-t
        # Create inputs for the subnetwork from the current states and the exponential features:
        batch_time_samples = t.shape[0]  # get the number of time samples
        batch_size = states.shape[0]  # get the number of trajectories
        # Expand decay modes and t_left to match shape required for the temporal features:
        feat_shape = (batch_size, batch_time_samples, self.n_temporal_features)
        decay_real = decay_real.reshape(1, 1, self.n_temporal_features).expand(*feat_shape)
        decay_imag = decay_imag.reshape(1, 1, self.n_temporal_features).expand(*feat_shape)
        decay_phas = decay_phas.reshape(1, 1, self.n_temporal_features).expand(*feat_shape)
        t_left = t_left.reshape(1, batch_time_samples, 1).expand(*feat_shape)
        # Define the real and imaginary part of the exponential features using their decay modes:
        features_real = torch.exp(-decay_real * t_left)
        features_imag = torch.sin(decay_imag * t_left + decay_phas)
        # Add exponential features to the current states to create the input matrix for the subnetwork:
        x = torch.cat((states, features_real, features_imag), dim=2)
        x = x.view(batch_size*batch_time_samples, -1)
        x = self.feedforward(x)  # pass the input matrix through the feedforward network

        return x.view(-1)


# IDENTITY RELATIONSHIP:          # LOGARITHM RELATIONSHIP:
# V:  no output layer activation  # log(V):  -
# ΔV: no output layer activation  # log(ΠV): -
class SpectralDecompositionExpectationSubnet(nn.Module):
    """Spectral decomposition expectation subnet to compute different quantities related to V.

    This subnet can compute V and ΔV using spectral decomposition.
    """

    def __init__(
        self,
        n_species: int,
        n_reactions: int,
        out_function_size: int,
        nn_config: dict,
        as_relationship: str = "identity",
        int_net: bool = False
    ) -> None:
        """Initialize the spectral decomposition expectation subnet."""
        self._validate_subnet_type(as_relationship, int_net)
        super().__init__()
        self.as_relationship = as_relationship
        self.int_net = int_net
        self.nn_config = nn_config
        # Set attributes relevant for complex spectral decomposition:
        self.pairing_mode = "paired"  # standard for [spectral_matched_lumped, spectral_matched_simplified, spectral_matched_full], set to "unconstrained" for spectral_complex
        self.allow_complex = True  # only relevant for spectral_complex, set to False for analysis
        # Get network size from configuration file:
        n_nodes = nn_config["n_nodes_per_layer"]
        n_hidden_layers = nn_config["n_hidden_layers"]
        n_spectral_terms = nn_config["v"]["n_spectral_terms"]
        # Save dimensions for use in the forward pass:
        self.n_reactions = n_reactions
        self.out_function_size = out_function_size
        self.n_spectral_terms = n_spectral_terms
        # Create feedforward layers:
        layers = [nn.Linear(n_species, n_nodes), nn.ReLU()]  # input layer with activation
        layers.extend([layer for _ in range(n_hidden_layers) for layer in (nn.Linear(n_nodes, n_nodes), nn.ReLU())])  # hidden layers with activation
        if self.nn_config["v"]["subnet_architecture"] == "spectral_matched_lumped":
            layers.append(nn.Linear(n_nodes, 2*n_spectral_terms*out_function_size) if not int_net else nn.Linear(n_nodes, 2*n_spectral_terms*out_function_size*n_reactions))  # output layer size based on "int_net" flag
        elif self.nn_config["v"]["subnet_architecture"] in ["spectral_matched_simplified", "spectral_matched_full", "spectral_complex"]:
            layers.append(nn.Linear(n_nodes, 2*n_spectral_terms) if not int_net else nn.Linear(n_nodes, 2*n_spectral_terms*n_reactions))  # output layer size based on "int_net" flag
        # Create the feedforward network from the layers:
        self.feedforward = nn.Sequential(*layers)  # to be used in the forward pass

    def _validate_subnet_type(self, as_relationship: str, int_net: bool) -> None:
        """Check if combination of as_relationship and int_net values is valid for this subnet architecture."""
        allowed_combinations = {"identity": {True, False}}  # both True and False for int_net are allowed when using "identity" as_relationship
        if as_relationship not in allowed_combinations or int_net not in allowed_combinations[as_relationship]:
            raise ValueError(f'Invalid combination of as_relationship="{as_relationship}" and int_net={int_net}.')

    def required_subnet_args(self) -> list[str]:
        """Return the required subnet arguments for the spectral decomposition expectation subnet."""
        return ["stationary_mean", "func_coord_real", "func_coord_imag", "decay_real", "decay_imag"]

    def forward(
        self, states: Tensor, t_stop: float, t: Tensor, stationary_mean: nn.Parameter,
        func_coord_real: nn.Parameter, func_coord_imag: nn.Parameter,
        decay_real: nn.Parameter, decay_imag: nn.Parameter
    ) -> Tensor:
        """Forward pass through the spectral decomposition expectation subnet to compute V or ΔV."""
        t_left = t_stop - t  # compute T-t
        # Create inputs for the subnetwork from the current states and the exponential features:
        batch_time_samples = t.shape[0]  # get the number of time samples
        batch_size = states.shape[0]  # get the number of trajectories

        # View the states without the batch_time_samples dimension:
        x = states.reshape(batch_size*batch_time_samples, -1)
        x = self.feedforward(x)  # pass the input matrix through the feedforward network

        # NOTE: These general shapes are "unsqueezed" in some places to account for "spectral_matched_simplified" and "int_net" configurations.
        out_shape = (batch_size, batch_time_samples, self.out_function_size)
        sum_shape = (batch_size, batch_time_samples, self.n_spectral_terms, self.out_function_size)
        # Reshape the stationary mean to match the outer shape after taking the sum:
        stationary_mean = stationary_mean.reshape(1, 1, self.out_function_size).expand(*out_shape)
        # Reshape and expand all parameters and t_left to match the required shape for the sum:
        decay_real = nn.functional.softplus(decay_real).reshape(1, 1, self.n_spectral_terms, 1).expand(*sum_shape)  # softplus to ensure positivity
        decay_imag = decay_imag.reshape(1, 1, self.n_spectral_terms, 1).expand(*sum_shape)
        t_left = t_left.reshape(1, batch_time_samples, 1, 1).expand(*sum_shape)

        if self.pairing_mode == "paired":
            exp_real = torch.exp(-decay_real * t_left)
            cos_imag = torch.cos(decay_imag * t_left)
            sin_imag = torch.sin(decay_imag * t_left)
        elif self.pairing_mode == "unconstrained":
            exp_complex = torch.exp(-(decay_real + 1j * decay_imag) * t_left)

        if self.nn_config["v"]["subnet_architecture"] == "spectral_matched_lumped":
            if self.int_net is False:
                kap_m = x[:, :self.n_spectral_terms*self.out_function_size].view(batch_size, batch_time_samples, self.n_spectral_terms, self.out_function_size)
                chi_m = x[:, self.n_spectral_terms*self.out_function_size:].view(batch_size, batch_time_samples, self.n_spectral_terms, self.out_function_size)
                # We need one prediction per output function: (shape: batch_size*batch_time_samples, out_function_size)
                pred = stationary_mean + 2 * torch.sum(exp_real * (kap_m*cos_imag + chi_m*sin_imag), dim=2)

            else:
                dkap_m = x[:, :self.n_spectral_terms*self.out_function_size*self.n_reactions].view(batch_size, batch_time_samples, self.n_spectral_terms, self.out_function_size, self.n_reactions)
                dchi_m = x[:, self.n_spectral_terms*self.out_function_size*self.n_reactions:].view(batch_size, batch_time_samples, self.n_spectral_terms, self.out_function_size, self.n_reactions)
                # We need one prediction per output function: (shape: batch_size*batch_time_samples, out_function_size*n_reactions)
                pred = 2 * torch.sum(exp_real.unsqueeze(-1) * (dkap_m*cos_imag.unsqueeze(-1) + dchi_m*sin_imag.unsqueeze(-1)), dim=2)

        elif self.nn_config["v"]["subnet_architecture"] == "spectral_matched_simplified":
            func_coord_real = func_coord_real.expand(batch_size, batch_time_samples, -1, -1)  # only used for simplified decomposition
            if self.int_net is False:
                c_m = x[:, :self.n_spectral_terms].view(batch_size, batch_time_samples, self.n_spectral_terms, 1)
                d_m = x[:, self.n_spectral_terms:].view(batch_size, batch_time_samples, self.n_spectral_terms, 1)
                # We need one prediction per output function: (shape: batch_size*batch_time_samples, out_function_size)
                pred = stationary_mean + 2 * torch.sum(func_coord_real * exp_real * (c_m*cos_imag + d_m*sin_imag), dim=2)

            else:
                dc_m = x[:, :self.n_spectral_terms*self.n_reactions].view(batch_size, batch_time_samples, self.n_spectral_terms, 1, self.n_reactions)
                dd_m = x[:, self.n_spectral_terms*self.n_reactions:].view(batch_size, batch_time_samples, self.n_spectral_terms, 1, self.n_reactions)
                # We need one prediction per output function: (shape: batch_size*batch_time_samples, out_function_size*n_reactions)
                pred = 2 * torch.sum(func_coord_real.unsqueeze(-1) * exp_real.unsqueeze(-1) * (dc_m*cos_imag.unsqueeze(-1) + dd_m*sin_imag.unsqueeze(-1)), dim=2)

        elif self.nn_config["v"]["subnet_architecture"] in ["spectral_matched_full", "spectral_complex"]:
            func_coord_real = func_coord_real.expand(batch_size, batch_time_samples, -1, -1)
            func_coord_imag = func_coord_imag.expand(batch_size, batch_time_samples, -1, -1)  # only used for full or complex decomposition
            if self.int_net is False:
                c_m = x[:, :self.n_spectral_terms].view(batch_size, batch_time_samples, self.n_spectral_terms, 1)
                d_m = x[:, self.n_spectral_terms:].view(batch_size, batch_time_samples, self.n_spectral_terms, 1)
                # We need one prediction per output function: (shape: batch_size*batch_time_samples, out_function_size)
                if self.pairing_mode == "paired":  # for spectral_matched_full (and sometimes the analysis of spectral_complex)
                    pred = stationary_mean + 2 * torch.sum(exp_real * (func_coord_real * (c_m*cos_imag + d_m*sin_imag) +
                                                                       func_coord_imag * (c_m*sin_imag - d_m*cos_imag)), dim=2)
                elif self.pairing_mode == "unconstrained":  # for spectral_complex only
                    pred = stationary_mean + torch.sum(exp_complex * (func_coord_real + 1j * func_coord_imag) * (c_m + 1j * d_m), dim=2)
                    pred = pred.real if not self.allow_complex else pred

            else:
                dc_m = x[:, :self.n_spectral_terms*self.n_reactions].view(batch_size, batch_time_samples, self.n_spectral_terms, 1, self.n_reactions)
                dd_m = x[:, self.n_spectral_terms*self.n_reactions:].view(batch_size, batch_time_samples, self.n_spectral_terms, 1, self.n_reactions)
                # We need one prediction per output function: (shape: batch_size*batch_time_samples, out_function_size*n_reactions)
                if self.pairing_mode == "paired":  # for spectral_matched_full (and sometimes the analysis of spectral_complex)
                    pred = 2 * torch.sum(exp_real.unsqueeze(-1) * (func_coord_real.unsqueeze(-1) * (dc_m*cos_imag.unsqueeze(-1) + dd_m*sin_imag.unsqueeze(-1)) +
                                                                   func_coord_imag.unsqueeze(-1) * (dc_m*sin_imag.unsqueeze(-1) - dd_m*cos_imag.unsqueeze(-1))), dim=2)
                elif self.pairing_mode == "unconstrained":  # for spectral_complex only
                    pred = torch.sum(exp_complex.unsqueeze(-1) * (func_coord_real + 1j * func_coord_imag).unsqueeze(-1) * (dc_m + 1j * dd_m), dim=2)
                    pred = pred.real if not self.allow_complex else pred

        return pred.view(-1)

    @torch.no_grad()
    def estimate_eigenfunctions(self, states: Tensor) -> tuple[Tensor, Tensor]:
        """Forward pass through the fully connected subnet, but only returning the eigenfunctions.

        The eigenfunctions are κ_m and χ_m or c_m and d_m.
        """
        # Create inputs for the subnetwork from the current states and the exponential features:
        batch_size = states.shape[0]  # get the number of trajectories
        batch_time_samples = states.shape[1]  # we support multiple time samples without passing the time grid

        # View the states without the batch_time_samples dimension:
        x = states.reshape(batch_size*batch_time_samples, -1)
        x = self.feedforward(x)  # pass the input matrix through the feedforward network

        if self.nn_config["v"]["subnet_architecture"] == "spectral_matched_lumped":
            if self.int_net is False:
                kap_m = x[:, :self.n_spectral_terms*self.out_function_size].view(batch_size, batch_time_samples, self.n_spectral_terms, self.out_function_size)
                chi_m = x[:, self.n_spectral_terms*self.out_function_size:].view(batch_size, batch_time_samples, self.n_spectral_terms, self.out_function_size)
                return kap_m, chi_m

            else:
                dkap_m = x[:, :self.n_spectral_terms*self.out_function_size*self.n_reactions].view(batch_size, batch_time_samples, self.n_spectral_terms, self.out_function_size, self.n_reactions)
                dchi_m = x[:, self.n_spectral_terms*self.out_function_size*self.n_reactions:].view(batch_size, batch_time_samples, self.n_spectral_terms, self.out_function_size, self.n_reactions)
                return dkap_m, dchi_m

        elif self.nn_config["v"]["subnet_architecture"] in ["spectral_matched_simplified", "spectral_matched_full", "spectral_complex"]:
            if self.int_net is False:
                c_m = x[:, :self.n_spectral_terms].view(batch_size, batch_time_samples, self.n_spectral_terms)
                d_m = x[:, self.n_spectral_terms:].view(batch_size, batch_time_samples, self.n_spectral_terms)
                return c_m, d_m

            else:
                dc_m = x[:, :self.n_spectral_terms*self.n_reactions].view(batch_size, batch_time_samples, self.n_spectral_terms, self.n_reactions)
                dd_m = x[:, self.n_spectral_terms*self.n_reactions:].view(batch_size, batch_time_samples, self.n_spectral_terms, self.n_reactions)
                return dc_m, dd_m

    @torch.no_grad()
    def estimate_time_derivative(
            self, states: Tensor, t_stop: float, t: Tensor, stationary_mean: nn.Parameter,
            func_coord_real: nn.Parameter, func_coord_imag: nn.Parameter,
            decay_real: nn.Parameter, decay_imag: nn.Parameter
        ) -> Tensor:
        """Return the time derivative of the eigenfunctions.

        Forward pass through the fully connected subnet, but only returning the time derivative of
        the eigenfunctions.
        """
        t_left = t_stop - t  # compute T-t
        # Create inputs for the subnetwork from the current states and the exponential features:
        batch_time_samples = t.shape[0]  # get the number of time samples
        batch_size = states.shape[0]  # get the number of trajectories

        # View the states without the batch_time_samples dimension:
        x = states.reshape(batch_size*batch_time_samples, -1)
        x = self.feedforward(x)  # pass the input matrix through the feedforward network

        # NOTE: These general shapes are "unsqueezed" in some places to account for "spectral_matched_simplified" and "int_net" configurations.
        out_shape = (batch_size, batch_time_samples, self.out_function_size)
        sum_shape = (batch_size, batch_time_samples, self.n_spectral_terms, self.out_function_size)
        # Reshape the stationary mean to match the outer shape after taking the sum:
        stationary_mean = stationary_mean.reshape(1, 1, self.out_function_size).expand(*out_shape)
        # Reshape and expand all parameters and t_left to match the required shape for the sum:
        decay_real = nn.functional.softplus(decay_real).reshape(1, 1, self.n_spectral_terms, 1).expand(*sum_shape)  # softplus to ensure positivity
        decay_imag = decay_imag.reshape(1, 1, self.n_spectral_terms, 1).expand(*sum_shape)
        t_left = t_left.reshape(1, batch_time_samples, 1, 1).expand(*sum_shape)

        if self.pairing_mode == "paired":
            exp_real = torch.exp(-decay_real * t_left)
            cos_imag = torch.cos(decay_imag * t_left)
            sin_imag = torch.sin(decay_imag * t_left)
        elif self.pairing_mode == "unconstrained":
            exp_complex = torch.exp(-(decay_real + 1j * decay_imag) * t_left)

        if self.nn_config["v"]["subnet_architecture"] == "spectral_matched_lumped":
            if self.int_net is False:
                kap_m = x[:, :self.n_spectral_terms*self.out_function_size].view(batch_size, batch_time_samples, self.n_spectral_terms, self.out_function_size)
                chi_m = x[:, self.n_spectral_terms*self.out_function_size:].view(batch_size, batch_time_samples, self.n_spectral_terms, self.out_function_size)
                # We need one prediction per output function: (shape: batch_size*batch_time_samples, out_function_size)
                pred = 2 * torch.sum(decay_real * exp_real * (kap_m*cos_imag + chi_m*sin_imag) + decay_imag * exp_real * (kap_m*sin_imag - chi_m*cos_imag), dim=2)

            else:
                raise ValueError("Only main_nets should be used for the time derivative!")

        elif self.nn_config["v"]["subnet_architecture"] == "spectral_matched_simplified":
            func_coord_real = func_coord_real.expand(batch_size, batch_time_samples, -1, -1)  # only used for simplified decomposition
            if self.int_net is False:
                c_m = x[:, :self.n_spectral_terms].view(batch_size, batch_time_samples, self.n_spectral_terms, 1)
                d_m = x[:, self.n_spectral_terms:].view(batch_size, batch_time_samples, self.n_spectral_terms, 1)
                # We need one prediction per output function: (shape: batch_size*batch_time_samples, out_function_size)
                pred = 2 * torch.sum(func_coord_real * exp_real * (decay_real * (c_m*cos_imag + d_m*sin_imag) + decay_imag * (c_m*sin_imag - d_m*cos_imag)), dim=2)

            else:
                raise ValueError("Only main_nets should be used for the time derivative!")

        elif self.nn_config["v"]["subnet_architecture"] in ["spectral_matched_full", "spectral_complex"]:
            func_coord_real = func_coord_real.expand(batch_size, batch_time_samples, -1, -1)
            func_coord_imag = func_coord_imag.expand(batch_size, batch_time_samples, -1, -1)  # only used for full or complex decomposition
            if self.int_net is False:
                c_m = x[:, :self.n_spectral_terms].view(batch_size, batch_time_samples, self.n_spectral_terms, 1)
                d_m = x[:, self.n_spectral_terms:].view(batch_size, batch_time_samples, self.n_spectral_terms, 1)
                # We need one prediction per output function: (shape: batch_size*batch_time_samples, out_function_size)
                if self.pairing_mode == "paired":  # for spectral_matched_full (and sometimes the analysis of spectral_complex)
                    pred = 2 * torch.sum(exp_real * (decay_real * (func_coord_real * (c_m*cos_imag + d_m*sin_imag) + func_coord_imag * (c_m*sin_imag - d_m*cos_imag)) +
                                                     decay_imag * (func_coord_real * (c_m*sin_imag - d_m*cos_imag) - func_coord_imag * (c_m*cos_imag + d_m*sin_imag))), dim=2)
                elif self.pairing_mode == "unconstrained":  # for spectral_complex only
                    pred = torch.sum((decay_real + 1j * decay_imag) * exp_complex * (func_coord_real + 1j * func_coord_imag) * (c_m + 1j * d_m), dim=2)
                    pred = pred.real if not self.allow_complex else pred

            else:
                raise ValueError("Only main_nets should be used for the time derivative!")

        return pred.view(-1)
