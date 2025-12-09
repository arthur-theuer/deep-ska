"""Reaction network examples for training and analysis."""

import logging

import numpy as np
import torch
import torch.nn as nn
from scipy.integrate import odeint
from torch import Tensor

from .definition import ReactionNetwork


class ConstitutiveGeneExpression(ReactionNetwork):
    """Constitutive gene expression reaction network example."""

    def __init__(self, rn_config: dict, init_type: str = "deter", initial_state: np.ndarray = None) -> None:
        """Initialize the constitutive gene expression reaction network."""
        super().__init__(rn_config, init_type)

        self.n_species = 2
        self.n_reactions = 4  # one birth and one death reaction per species

        # To store stoichiometric coefficients of species (cols) in reactions (rows):
        self.reactant_matrix = np.zeros([self.n_reactions, self.n_species], dtype=int)  # beginning of reaction
        self.product_matrix = np.zeros([self.n_reactions, self.n_species], dtype=int)  # result of reaction

        # 1. 0 -> M
        self.product_matrix[0, 0] = 1
        # 2. M -> M + P
        self.reactant_matrix[1, 0] = 1
        self.product_matrix[1, 0] = 1
        self.product_matrix[1, 1] = 1
        # 3. M -> 0
        self.reactant_matrix[2, 0] = 1
        # 4. P -> 0
        self.reactant_matrix[3, 1] = 1

        # Define reactions using the parameter dictionary:
        self.reaction_dict = {0: ["mass action", "transcription rate"],
                              1: ["mass action", "translation rate"],
                              2: ["mass action", "mRNA degradation rate"],
                              3: ["mass action", "protein degradation rate"]}

        # Define the labels and indices of the input and output species:
        self.species_labels = ["mRNA", "protein"]

        # Get additional reaction network parameters:
        self._initialize_network_parameters()

        # Set initial state to 0 by default (there are no species in the beginning), but allow for custom initial states:
        self.initial_state = initial_state if initial_state is not None else np.zeros(self.n_species)  # set initial state if given

    def get_stationary_mean_exact(self) -> list:
        """Compute the stationary mean of the output species in the CGE reaction network."""
        k_r = self.parameter_dict["transcription rate"]
        k_p = self.parameter_dict["translation rate"]
        gamma_r = self.parameter_dict["mRNA degradation rate"]
        gamma_p = self.parameter_dict["protein degradation rate"]
        # First moment of the output species:
        x_1 = k_r / gamma_r
        x_2 = (k_r * k_p) / (gamma_r * gamma_p)
        # Variance of the output species:
        sigma_11 = (k_r * (k_r + gamma_r)) / (gamma_r**2)
        sigma_22 = (k_p * k_r * (gamma_r + gamma_p + k_p)) / (gamma_r * gamma_p * (gamma_r + gamma_p))
        # Covariance between the output species:
        sigma_21 = (k_p * k_r) / (gamma_r * (gamma_r + gamma_p))
        # Second moment of the output species:
        second_1 = x_1**2 + sigma_11
        second_2 = x_2**2 + sigma_22
        # Cross moments:
        cross_moment = x_1 * x_2 + sigma_21

        output_map = {("monomials", (1,), ("mRNA",)): [x_1],
                      ("monomials", (1,), ("protein",)): [x_2],
                      ("monomials", (2,), ("mRNA",)): [second_1],
                      ("monomials", (2,), ("protein",)): [second_2],
                      ("monomials", (1, 2), ("mRNA",)): [x_1, second_1],
                      ("monomials", (1, 2), ("protein",)): [x_2, second_2],
                      ("monomials", (1, 2), ("mRNA", "protein")): [x_1, x_2, second_1, second_2],
                      ("monomials_cross", (1,), ("mRNA",)): [x_1],
                      ("monomials_cross", (1,), ("protein",)): [x_2],
                      ("monomials_cross", (2,), ("mRNA",)): [second_1],
                      ("monomials_cross", (2,), ("protein",)): [second_2],
                      ("monomials_cross", (1, 2), ("mRNA",)): [x_1, second_1],
                      ("monomials_cross", (1, 2), ("protein",)): [x_2, second_2],
                      ("monomials_cross", (1, 2), ("mRNA", "protein")): [x_1, x_2, second_1, cross_moment, second_2]}

        key = (self.rn_config["output_function"], tuple(self.rn_config["output_function_arguments"]["monomial_orders"]), tuple(self.out_species_labels))

        if key not in output_map:
            raise ValueError(f"Invalid combination of output function '{self.rn_config['output_function']}', monomial orders '{self.rn_config['output_function_arguments']['monomial_orders']}' and species labels '{self.out_species_labels}'.")
        logging.info(f"Using exact stationary mean for output function '{self.rn_config['output_function']}' with monomial orders '{self.rn_config['output_function_arguments']['monomial_orders']}' and species labels '{self.out_species_labels}': {output_map[key]}")
        return output_map[key]

    def get_moments_exact(self, states: Tensor, t_stop: float, t: Tensor, stationary_mean: nn.Parameter, func_coord_real: nn.Parameter, func_coord_imag: nn.Parameter, decay_real: nn.Parameter, decay_imag: nn.Parameter) -> Tensor:
        """Compute the moment estimate for the protein species in the CGE reaction network."""
        _ = func_coord_real, func_coord_imag  # unused arguments

        t_left = t_stop - t  # compute T-t
        # Create inputs for the subnetwork from the current states and the exponential features:
        batch_time_samples = t.shape[0]  # get the number of time samples
        batch_size = states.shape[0]  # get the number of trajectories

        x_0_r = states[:, :, 0]  # initial mRNA state (batch_size, batch_time_samples)
        x_0_p = states[:, :, 1]  # initial protein state (batch_size, batch_time_samples)

        k_r = self.parameter_dict["transcription rate"]
        k_p = self.parameter_dict["translation rate"]
        gamma_r = self.parameter_dict["mRNA degradation rate"]
        gamma_p = self.parameter_dict["protein degradation rate"]

        decay_real = nn.functional.softplus(decay_real).expand(batch_size, batch_time_samples, -1)  # softplus to ensure positivity
        decay_imag = decay_imag.expand(batch_size, batch_time_samples, -1)
        t_left = t_left.expand(batch_size, -1).unsqueeze(-1)

        exp_real = torch.exp(-decay_real * t_left).unsqueeze(-1)
        cos_imag = torch.cos(decay_imag * t_left).unsqueeze(-1)

        # Select the first moment:
        exp_r = exp_real[:, :, 0, 0]
        exp_p = exp_real[:, :, 1, 0]
        cos_r = cos_imag[:, :, 0, 0]
        cos_p = cos_imag[:, :, 1, 0]

        if self.out_species_labels == ["mRNA"]:
            # Calculate the coefficient for the mRNA moment estimate:
            alpha_1 = x_0_r - k_r / gamma_r
            # Moment estimate for mRNA species:
            moment_estimate = stationary_mean + alpha_1 * cos_r * exp_r

        if self.out_species_labels == ["protein"]:
            # Calculate the coefficients for the protein moment estimate:
            alpha_1 = k_p / (gamma_p - gamma_r) * (x_0_r - k_r / gamma_r)
            alpha_2 = k_p * (k_r / (gamma_r * (gamma_p - gamma_r)) - x_0_r / (gamma_p - gamma_r) - k_r / (gamma_r * gamma_p)) + x_0_p

            # Moment estimate for protein species (combination of both exponential terms):
            moment_estimate = stationary_mean + alpha_1 * cos_r * exp_r + alpha_2 * cos_p * exp_p

        return moment_estimate.view(-1)

    def moment_equations(self, y: np.ndarray, t: float) -> np.ndarray:  # noqa: ARG002
        """Define ODEs for numerically computing the exact solution of the system."""
        k_r = self.parameter_dict["transcription rate"]
        k_p = self.parameter_dict["translation rate"]
        g_r = self.parameter_dict["mRNA degradation rate"]
        g_p = self.parameter_dict["protein degradation rate"]

        E_R, E_P, E_R2, E_RP, E_P2 = y  # unpack moments

        # Initialize the derivatives array with zeros:
        dydt = np.zeros(5, dtype=np.float64)

        # First moments:
        dydt[0] = k_r - g_r * E_R  # dE[R]/dt
        dydt[1] = k_p * E_R - g_p * E_P  # dE[P]/dt

        # Second and cross moments:
        dydt[2] = k_r + (2 * k_r + g_r) * E_R - 2 * g_r * E_R2  # dE[R^2]/dt
        dydt[3] = k_r * E_P + k_p * E_R2 - (g_r + g_p) * E_RP  # dE[RP]/dt
        dydt[4] = k_p * E_R + g_p * E_P - 2 * g_p * E_P2 + 2 * k_p * E_RP  # dE[P^2]/dt

        return dydt

    def compute_exact_values(self, state: np.ndarray, t_stop: float, n_time_samples: int) -> np.ndarray:
        """Compute the exact values of the first and second moments by numerically solving the moment equations."""
        # The initial state y0 is given by the function's input state:
        y0 = np.array([state[0], state[1], state[0]**2, state[0]*state[1], state[1]**2], dtype=np.float64)
        t = t_stop - np.linspace(0, t_stop, n_time_samples)[::-1]  # compute t_stop - reverse_time_grid in case the time intervals are not constant

        # Solve the moment equations:
        sol = odeint(self.moment_equations, y0, t)

        # Map the solution to the output format:
        x_1 = sol[:, 0]  # first moment of mRNA
        x_2 = sol[:, 1]  # first moment of protein
        second_1 = sol[:, 2]  # second moment of mRNA
        second_2 = sol[:, 4]  # second moment of protein

        output_map = {
            ("monomials", (1,), ("mRNA",)): lambda: x_1,
            ("monomials", (1,), ("protein",)): lambda: x_2,
            ("monomials", (2,), ("mRNA",)): lambda: second_1,
            ("monomials", (2,), ("protein",)): lambda: second_2,
            ("monomials", (1, 2), ("mRNA",)): lambda: np.column_stack([x_1, second_1]),
            ("monomials", (1, 2), ("protein",)): lambda: np.column_stack([x_2, second_2]),
            ("monomials", (1, 2), ("mRNA", "protein")): lambda: np.column_stack([x_1, x_2, second_1, second_2]),
        }

        key = (self.rn_config["output_function"], tuple(self.rn_config["output_function_arguments"]["monomial_orders"]), tuple(self.out_species_labels))

        if key not in output_map:
            logging.warning(f"Invalid combination of output function '{self.rn_config['output_function']}', monomial orders '{self.rn_config['output_function_arguments']['monomial_orders']}' and species labels '{self.out_species_labels}' for exact solution. Returning NaNs and setting 'exact_values_computable' to False.")
            self.rn_config['exact_values_computable'] = False
        logging.info(f"Using exact temporal solution for output function '{self.rn_config['output_function']}' with monomial orders '{self.rn_config['output_function_arguments']['monomial_orders']}' and species labels '{self.out_species_labels}'.")
        return output_map[key]()


class SelfRegulatoryGeneExpression(ReactionNetwork):
    """Self-regulatory gene expression reaction network example."""

    def __init__(self, rn_config: dict, init_type: str = "deter", initial_state: np.ndarray = None) -> None:
        """Initialize the self-regulatory gene expression reaction network."""
        super().__init__(rn_config, init_type)

        self.n_species = 1
        self.n_reactions = 2  # one birth and one death reaction

        # To store stoichiometric coefficients of species (cols) in reactions (rows):
        self.reactant_matrix = np.zeros([self.n_reactions, self.n_species], dtype=int)  # beginning of reaction
        self.product_matrix = np.zeros([self.n_reactions, self.n_species], dtype=int)  # result of reaction

        # 1. 0 -> X
        self.product_matrix[0, 0] = 1
        # 2. X -> 0
        self.reactant_matrix[1, 0] = 1

        # Define reactions using the parameter dictionary:
        self.reaction_dict = {0: ["Hill repression", 0, "production rate", "Hill constant den", "Hill coefficient", "basal rate"],
                              1: ["mass action", "degradation rate"]}

        # Define the labels and indices of the input and output species:
        self.species_labels = ["protein"]

        # Get additional reaction network parameters:
        self._initialize_network_parameters()

        # Set initial state to 0 by default (there are no species in the beginning), but allow for custom initial states:
        self.initial_state = initial_state if initial_state is not None else np.zeros(self.n_species)  # set initial state if given


class GeneticToggleSwitch(ReactionNetwork):
    """Genetic toggle-switch reaction network example."""

    def __init__(self, rn_config: dict, init_type: str = "deter", initial_state: np.ndarray = None) -> None:
        """Initialize the genetic toggle-switch reaction network."""
        super().__init__(rn_config, init_type)

        self.n_species = 2
        self.n_reactions = 4

        # To store stoichiometric coefficients of species (cols) in reactions (rows):
        self.reactant_matrix = np.zeros([self.n_reactions, self.n_species], dtype=int)  # beginning of reaction
        self.product_matrix = np.zeros([self.n_reactions, self.n_species], dtype=int)  # result of reaction

        # 1. 0 -> S_1
        self.product_matrix[0, 0] = 1
        # 2. S_1 -> 0
        self.reactant_matrix[1, 0] = 1
        # 3. 0 -> S_2
        self.product_matrix[2, 1] = 1
        # 4. S_2 -> 0
        self.reactant_matrix[3, 1] = 1

        # Define reactions using the parameter dictionary:
        self.reaction_dict = {0: ["Hill repression", 1, "alpha_1", "k_1", "beta", "b_1"],
                              1: ["mass action", "mak_1"],
                              2: ["Hill repression", 0, "alpha_2", "k_2", "gamma", "b_2"],
                              3: ["mass action", "mak_2"]}

        # Define the labels and indices of the input and output species:
        self.species_labels = ["species_1", "species_2"]

        # Get additional reaction network parameters:
        self._initialize_network_parameters()

        # Set initial state to 0 by default (there are no species in the beginning), but allow for custom initial states:
        self.initial_state = initial_state if initial_state is not None else np.zeros(self.n_species)  # set initial state if given


class SusceptibleInfectedRecoveredNetwork(ReactionNetwork):
    """Susceptible-infected-recovered reaction network example from epidemiology."""

    def __init__(self, rn_config: dict, init_type: str = "deter", initial_state: np.ndarray = None) -> None:
        """Initialize the SIR reaction network."""
        super().__init__(rn_config, init_type)

        self.n_species = 3
        self.n_reactions = 6

        # To store stoichiometric coefficients of species (cols) in reactions (rows):
        self.reactant_matrix = np.zeros([self.n_reactions, self.n_species], dtype=int)  # beginning of reaction
        self.product_matrix = np.zeros([self.n_reactions, self.n_species], dtype=int)  # result of reaction

        # 1. S + I -> 2I
        self.reactant_matrix[0, 0] = 1
        self.reactant_matrix[0, 1] = 1
        self.product_matrix[0, 1] = 2
        # 2. I -> R
        self.reactant_matrix[1, 1] = 1
        self.product_matrix[1, 2] = 1
        # 3. R -> S
        self.reactant_matrix[2, 2] = 1
        self.product_matrix[2, 0] = 1
        # 4. S -> 0
        self.reactant_matrix[3, 0] = 1
        # 5. I -> 0
        self.reactant_matrix[4, 1] = 1
        # 6. R -> 0
        self.reactant_matrix[5, 2] = 1

        # Define reactions using the parameter dictionary:
        self.reaction_dict = {0: ["mass action", "infection rate"],
                              1: ["mass action", "recovery rate"],
                              2: ["mass action", "reversion rate"],
                              3: ["mass action", "susceptible death rate"],
                              4: ["mass action", "infected death rate"],
                              5: ["mass action", "recovered death rate"]}

        # Define the labels and indices of the input and output species:
        self.species_labels = ["susceptible", "infected", "recovered"]

        # Get additional reaction network parameters:
        self._initialize_network_parameters()

        # Set initial state to literature value by default, but allow for custom initial states:
        self.initial_state = initial_state if initial_state is not None else np.array([51, 11, 0])  # set initial state if given


class ReferenceBasedAntitheticIntegralController(ReactionNetwork):
    """Reference-based antithetic integral control of gene expression reaction network example."""

    def __init__(self, rn_config: dict, init_type: str = "deter", initial_state: np.ndarray = None) -> None:
        """Initialize the reference-based AIC reaction network."""
        super().__init__(rn_config, init_type)

        self.n_species = 4
        self.n_reactions = 7

        # To store stoichiometric coefficients of species (cols) in reactions (rows):
        self.reactant_matrix = np.zeros([self.n_reactions, self.n_species], dtype=int)  # beginning of reaction
        self.product_matrix = np.zeros([self.n_reactions, self.n_species], dtype=int)  # result of reaction

        # 1. Z_1 -> Z_1 + M
        self.product_matrix[0, 2] = 1
        self.product_matrix[0, 0] = 1
        self.reactant_matrix[0, 2] = 1
        # 2. M -> M + P
        self.reactant_matrix[1, 0] = 1
        self.product_matrix[1, 0] = 1
        self.product_matrix[1, 1] = 1
        # 3. M -> 0
        self.reactant_matrix[2, 0] = 1
        # 4. P -> 0
        self.reactant_matrix[3, 1] = 1
        # 5. P -> P + Z_2
        self.reactant_matrix[4, 1] = 1
        self.product_matrix[4, 1] = 1
        self.product_matrix[4, 3] = 1
        # 6. Z_1 + Z_2 -> 0
        self.reactant_matrix[5, 2] = 1
        self.reactant_matrix[5, 3] = 1
        # 7. 0 -> Z_1
        self.product_matrix[6, 2] = 1

        # Define reactions using the parameter dictionary:
        self.reaction_dict = {0: ["mass action", "activation rate"],
                              1: ["mass action", "translation rate"],
                              2: ["mass action", "mRNA degradation rate"],
                              3: ["mass action", "protein degradation rate"],
                              4: ["mass action", "theta"],
                              5: ["mass action", "eta"],
                              6: ["mass action", "mu"]}

        # Define the labels and indices of the input and output species:
        self.species_labels = ["mRNA", "protein", "Z1", "Z2"]

        # Get additional reaction network parameters:
        self._initialize_network_parameters()

        # Set initial state to 0 by default (there are no species in the beginning), but allow for custom initial states:
        self.initial_state = initial_state if initial_state is not None else np.zeros(self.n_species)  # set initial state if given

    def get_stationary_mean_exact(self) -> list:
        """Compute the stationary mean of the output species in the reference-based AIC reaction network."""
        theta = self.parameter_dict["theta"]
        mu = self.parameter_dict["mu"]
        # First moment of the output species:
        x_2 = mu / theta

        output_map = {("monomials", (1,), ("protein",)): [x_2],
                      ("monomials_cross", (1,), ("protein",)): [x_2]}

        key = (self.rn_config["output_function"], tuple(self.rn_config["output_function_arguments"]["monomial_orders"]), tuple(self.out_species_labels))

        if key not in output_map:
            raise ValueError(f"Invalid combination of output function '{self.rn_config['output_function']}', monomial orders '{self.rn_config['output_function_arguments']['monomial_orders']}' and species labels '{self.out_species_labels}'.")
        logging.info(f"Using exact stationary mean for output function '{self.rn_config['output_function']}' with monomial orders '{self.rn_config['output_function_arguments']['monomial_orders']}' and species labels '{self.out_species_labels}': {output_map[key]}")
        return output_map[key]


class SensorBasedAntitheticIntegralController(ReactionNetwork):
    """Sensor-based antithetic integral control of gene expression reaction network example."""

    def __init__(self, rn_config: dict, init_type: str = "deter", initial_state: np.ndarray = None) -> None:
        """Initialize the sensor-based AIC reaction network."""
        super().__init__(rn_config, init_type)

        self.n_species = 4
        self.n_reactions = 7

        # To store stoichiometric coefficients of species (cols) in reactions (rows):
        self.reactant_matrix = np.zeros([self.n_reactions, self.n_species], dtype=int)  # beginning of reaction
        self.product_matrix = np.zeros([self.n_reactions, self.n_species], dtype=int)  # result of reaction

        # 1. 0 -> M
        self.product_matrix[0, 0] = 1
        # 2. M -> M + P
        self.reactant_matrix[1, 0] = 1
        self.product_matrix[1, 0] = 1
        self.product_matrix[1, 1] = 1
        # 3. M -> 0
        self.reactant_matrix[2, 0] = 1
        # 4. P -> 0
        self.reactant_matrix[3, 1] = 1
        # 5. P -> P + Z_2
        self.reactant_matrix[4, 1] = 1
        self.product_matrix[4, 1] = 1
        self.product_matrix[4, 3] = 1
        # 6. Z_1 + Z_2 -> 0
        self.reactant_matrix[5, 2] = 1
        self.reactant_matrix[5, 3] = 1
        # 7. 0 -> Z_1
        self.product_matrix[6, 2] = 1

        # Define reactions using the parameter dictionary:
        self.reaction_dict = {0: ["Hill repression", 3, "activation rate", "Hill constant den", "Hill coefficient", "basal rate"],
                              1: ["mass action", "translation rate"],
                              2: ["mass action", "mRNA degradation rate"],
                              3: ["mass action", "protein degradation rate"],
                              4: ["mass action", "theta"],
                              5: ["mass action", "eta"],
                              6: ["mass action", "mu"]}

        # Define the labels and indices of the input and output species:
        self.species_labels = ["mRNA", "protein", "Z1", "Z2"]

        # Get additional reaction network parameters:
        self._initialize_network_parameters()

        # Set initial state to 0 by default (there are no species in the beginning), but allow for custom initial states:
        self.initial_state = initial_state if initial_state is not None else np.zeros(self.n_species)  # set initial state if given

    def get_stationary_mean_exact(self) -> list:
        """Compute the stationary mean of the output species in the sensor-based AIC reaction network."""
        theta = self.parameter_dict["theta"]
        mu = self.parameter_dict["mu"]
        # First moment of the output species:
        x_2 = mu / theta

        output_map = {("monomials", (1,), ("protein",)): [x_2],
                      ("monomials_cross", (1,), ("protein",)): [x_2]}

        key = (self.rn_config["output_function"], tuple(self.rn_config["output_function_arguments"]["monomial_orders"]), tuple(self.out_species_labels))

        if key not in output_map:
            raise ValueError(f"Invalid combination of output function '{self.rn_config['output_function']}', monomial orders '{self.rn_config['output_function_arguments']['monomial_orders']}' and species labels '{self.out_species_labels}'.")
        logging.info(f"Using exact stationary mean for output function '{self.rn_config['output_function']}' with monomial orders '{self.rn_config['output_function_arguments']['monomial_orders']}' and species labels '{self.out_species_labels}': {output_map[key]}")
        return output_map[key]


class Repressilator(ReactionNetwork):
    """Repressilator reaction network example."""

    def __init__(self, rn_config: dict, init_type: str = "deter", initial_state: np.ndarray = None) -> None:
        """Initialize the repressilator reaction network."""
        super().__init__(rn_config, init_type)

        self.n_species = 6
        self.n_reactions = 12

        # To store stoichiometric coefficients of species (cols) in reactions (rows):
        self.reactant_matrix = np.zeros([self.n_reactions, self.n_species], dtype=int)  # beginning of reaction
        self.product_matrix = np.zeros([self.n_reactions, self.n_species], dtype=int)  # result of reaction

        # 1. 0 -> M_1
        self.product_matrix[0, 0] = 1
        # 2. 0 -> M_2
        self.product_matrix[1, 1] = 1
        # 3. 0 -> M_3
        self.product_matrix[2, 2] = 1
        # 4. M_1 -> 0
        self.reactant_matrix[3, 0] = 1
        # 5. M_2 -> 0
        self.reactant_matrix[4, 1] = 1
        # 6. M_3 -> 0
        self.reactant_matrix[5, 2] = 1
        # 7. M_1 -> M_1 + P_1
        self.reactant_matrix[6, 0] = 1
        self.product_matrix[6, 0] = 1
        self.product_matrix[6, 3] = 1
        # 8. M_2 -> M_2 + P_2
        self.reactant_matrix[7, 1] = 1
        self.product_matrix[7, 1] = 1
        self.product_matrix[7, 4] = 1
        # 9. M_3 -> M_3 + P_3
        self.product_matrix[8, 2] = 1
        self.reactant_matrix[8, 2] = 1
        self.product_matrix[8, 5] = 1
        # 10. P_1 -> 0
        self.reactant_matrix[9, 3] = 1
        # 11. P_2 -> 0
        self.reactant_matrix[10, 4] = 1
        # 12. P_3 -> 0
        self.reactant_matrix[11, 5] = 1

        # Define reactions using the parameter dictionary:
        self.reaction_dict = {0: ["Hill repression", 4, "a_1", "k_1", "alpha_1", "b_1"],
                              1: ["Hill repression", 5, "a_2", "k_2", "alpha_2", "b_2"],
                              2: ["Hill repression", 3, "a_3", "k_3", "alpha_3", "b_3"],
                              3: ["mass action", "mak_1"],
                              4: ["mass action", "mak_2"],
                              5: ["mass action", "mak_3"],
                              6: ["mass action", "mak_4"],
                              7: ["mass action", "mak_5"],
                              8: ["mass action", "mak_6"],
                              9: ["mass action", "mak_7"],
                              10: ["mass action", "mak_8"],
                              11: ["mass action", "mak_9"]}

        # Define the labels and indices of the input and output species:
        self.species_labels = ["M_1", "M_2", "M_3", "P_1", "P_2", "P_3"]

        # Get additional reaction network parameters:
        self._initialize_network_parameters()

        # Set initial state to 0 by default (there are no species in the beginning), but allow for custom initial states:
        self.initial_state = initial_state if initial_state is not None else np.zeros(self.n_species)  # set initial state if given


class NonlinearConversionCascade(ReactionNetwork):
    """Nonlinear conversion cascade reaction network example."""

    def __init__(self, rn_config: dict, init_type: str = "deter", initial_state: np.ndarray = None) -> None:
        """Initialize the nonlinear conversion cascade reaction network."""
        super().__init__(rn_config, init_type)

        self.n_species = rn_config["n_species"]
        self.n_reactions = 2 * self.n_species

        # To store stoichiometric coefficients of species (cols) in reactions (rows):
        self.reactant_matrix = np.zeros([self.n_reactions, self.n_species], dtype=int)  # beginning of reaction
        self.product_matrix = np.zeros([self.n_reactions, self.n_species], dtype=int)  # result of reaction

        self.product_matrix[0, 0] = 1  # the first species is created constitutively (base production rate)
        for i in range(self.n_species - 1):  # other species are produced catalytically (max translation rate)
            # Reaction i ...
            self.reactant_matrix[i + 1, i] = 1  # ... degrades species i
            self.product_matrix[i + 1, i + 1] = 1  # ... produces species i+1 -> next in cascade
        for i in range(self.n_species):  # all species are diluted (dilution rate)
            # Reaction n_species+i ...
            self.reactant_matrix[self.n_species + i, i] = 1  # ... degrades species i

        # Define reaction dictionary:
        self.reaction_dict = {0: ["mass action", "base production rate"]}
        # Associate each reaction with a type and a reaction rate:
        self.reaction_dict.update({i + 1: ["Hill conversion", i, "max translation rate", "Hill constant den", "Hill coefficient", "basal rate"] for i in range(self.n_species - 1)})
        self.reaction_dict.update({i + self.n_species: ["mass action", "dilution rate"] for i in range(self.n_species)})

        # Define the labels and indices of the input and output species:
        self.species_labels = [f"X{i}" for i in range(self.n_species)]  # X0, X1, X2, ...

        # Get additional reaction network parameters:
        self._initialize_network_parameters()

        # Set initial state to 0 by default (there are no species in the beginning), but allow for custom initial states:
        self.initial_state = initial_state if initial_state is not None else np.zeros(self.n_species)  # set initial state if given


class LinearConversionCascadeWithFeedback(ReactionNetwork):
    """Linear conversion cascade with feedback reaction network example."""

    def __init__(self, rn_config: dict, init_type: str = "deter", initial_state: np.ndarray = None) -> None:
        """Initialize the linear conversion cascade with feedback reaction network."""
        super().__init__(rn_config, init_type)

        self.n_species = rn_config["n_species"]
        self.n_reactions = 2 * self.n_species

        # To store stoichiometric coefficients of species (cols) in reactions (rows):
        self.reactant_matrix = np.zeros([self.n_reactions, self.n_species], dtype=int)  # beginning of reaction
        self.product_matrix = np.zeros([self.n_reactions, self.n_species], dtype=int)  # result of reaction

        self.product_matrix[0, 0] = 1  # the first species is created constitutively (basal rate)
        for i in range(self.n_species - 1):  # other species are produced catalytically (max translation rate)
            # Reaction i ...
            self.reactant_matrix[i + 1, i] = 1  # ... degrades species i
            self.product_matrix[i + 1, i + 1] = 1  # ... produces species i+1 -> next in cascade
        for i in range(self.n_species):  # all species are diluted (dilution rate)
            # Reaction n_species+i ...
            self.reactant_matrix[self.n_species + i, i] = 1  # ... degrades species i

        # Define reaction dictionary:
        self.reaction_dict = {0: ["Hill repression", self.n_species - 1, "max translation rate", "Hill constant den", "Hill coefficient", "basal rate"]}
        # Associate each reaction with a type and a reaction rate:
        self.reaction_dict.update({i + 1: ["mass action", "translation rate"] for i in range(self.n_species - 1)})
        self.reaction_dict.update({i + self.n_species: ["mass action", "dilution rate"] for i in range(self.n_species)})

        # Define the labels and indices of the input and output species:
        self.species_labels = [f"X{i}" for i in range(self.n_species)]  # X0, X1, X2, ...

        # Get additional reaction network parameters:
        self._initialize_network_parameters()

        # Set initial state to 0 by default (there are no species in the beginning), but allow for custom initial states:
        self.initial_state = initial_state if initial_state is not None else np.zeros(self.n_species)  # set initial state if given


if __name__ == "__main__":
    pass
