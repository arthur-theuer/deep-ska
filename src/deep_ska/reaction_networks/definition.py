"""Definition of the reaction network class."""

import numpy as np
from scipy.special import factorial


class ReactionNetwork:
    """Parent class defining a reaction network. Specific examples inherit from this."""

    def __init__(self, rn_config: dict, init_type: str = "deter") -> None:
        """Initialize the reaction network with a configuration file and random number generator."""
        # Create placeholder attributes to overwrite when using specific examples:
        self.rn_config = rn_config  # configuration file for the reaction network
        self.init_type = init_type  # determines trajectory generation

        self.n_species = None  # number of species in the network
        self.n_reactions = None  # number of reactions in the network

        # Matrices with n_reactions rows and n_species cols:
        self.reactant_matrix = None  # rows = number of species consumed in that reaction
        self.product_matrix = None  # rows = number of species produced in that reaction

        # Store information about the parameters:
        self.parameter_dict = rn_config.get("parameter_dict", None)
        self.reaction_dict = None
        self.species_labels = None
        self.out_species_labels = rn_config.get("out_species_labels", None)

        self.initial_state = None  # needs to be created in the inheriting class
        self.propensity_vector = None  # defined through a method called at initialization

    def _initialize_network_parameters(self) -> None:
        """Initialize further reaction network parameters based on the configuration file."""
        self.out_species_indices = [self.species_labels.index(i) for i in self.out_species_labels]  # get input indices for output species
        self.out_species_size = len(self.out_species_indices)

        self.param_dict_size = len(self.parameter_dict)
        self.param_labels = list(self.parameter_dict.keys())

        self.sel_param_labels = self.rn_config["sel_param_labels"]
        self.sel_param_size = len(self.sel_param_labels)
        self.sel_param_indices = [list(self.parameter_dict).index(key) for key in self.sel_param_labels]

        # Now set reaction network properties:
        self.set_propensity_vector()  # propensities are calculated according to method specified in the reaction_dict
        self.set_output_function(self.rn_config["output_function"])

        self.stoichiometry_matrix = self.product_matrix - self.reactant_matrix  # to update the state of the network

    def mass_action_propensity(self, state: np.ndarray, reaction_index: int, rate_constant: str) -> np.ndarray:
        """Calculate the propensity of a state using the law of mass action."""
        propensity = self.parameter_dict[rate_constant]  # use rate constant as initial propensity
        reactants = self.reactant_matrix[reaction_index]  # get reactants of current reaction (specific row in reaction_matrix)

        for j in range(self.n_species):  # loop over species to calculate final propensity
            for k in range(reactants[j]):  # check order of indices
                propensity *= state[j] - k
            if reactants[j] > 1:  # make code faster by skipping division by factorial if it would be 1 anyways
                propensity /= factorial(reactants[j], exact=True)

        return max(propensity, 0)

    def hill_propensity_repression(self, state: np.ndarray, species_no: int, key_1: str, key_2: str, key_3: str, key_4: str) -> np.ndarray:
        """Implements the propensity for a given state using Hill repression."""
        a = self.parameter_dict[key_1]
        k = self.parameter_dict[key_2]
        h = self.parameter_dict[key_3]
        b = self.parameter_dict[key_4] if key_4 is not None else 0
        xp = state[species_no]
        return max(b + a / (k + (xp**h)), 0)

    def hill_propensity_conversion(self, state: np.ndarray, species_no: int, key_1: str, key_2: str, key_3: str, key_4: str) -> np.ndarray:
        """Implements the propensity for a given state using Hill conversion."""
        a = self.parameter_dict[key_1]
        k = self.parameter_dict[key_2]
        h = self.parameter_dict[key_3]
        b = self.parameter_dict[key_4] if key_4 is not None else 0
        xp = state[species_no]
        if xp > 0:
            return max(b + a * (xp**h) / (k + (xp**h)), 0)
        else:
            return 0

    def _calculate_propensity(self, current_state: np.ndarray, k: int) -> np.ndarray:
        """Calculates reaction propensity using the defined reaction function."""
        reaction_type, *args = self.reaction_dict[k]
        if reaction_type == "mass action":
            return self.mass_action_propensity(current_state, k, *args)
        elif reaction_type == "Hill repression":
            return self.hill_propensity_repression(current_state, *args)
        elif reaction_type == "Hill conversion":
            return self.hill_propensity_conversion(current_state, *args)
        else:
            raise NotImplementedError(f"Reaction type '{reaction_type}' not implemented.")

    def _calculate_all_propensities(self, current_state: np.ndarray) -> np.ndarray:
        """Calculates propensities for all reactions and returns them in an array."""
        return np.array([self._calculate_propensity(current_state, k) for k in range(self.n_reactions)])

    def set_propensity_vector(self) -> None:
        """Generate the vector of reaction rates in the system."""
        self.propensity_vector = self._calculate_all_propensities  # set the propensity vector for the current state

    def update_state(self, next_reaction: int, state: np.ndarray) -> np.ndarray:
        """Update the state of the network according to the stoichiometry matrix."""
        if next_reaction != -1:
            state = state + self.stoichiometry_matrix[next_reaction, :]
        return state

    def sample_random_state(self, rng: np.random.Generator) -> np.ndarray:
        """Sample a random state from a uniform distribution based on config boundaries."""
        lower_bound = self.rn_config["lower_bound"]
        upper_bound = self.rn_config["upper_bound"]

        # Ensure lower_bound and upper_bound are lists (a single species value is broadcasted to all species):
        lower_bound = [lower_bound] * self.n_species if not isinstance(lower_bound, list) else lower_bound
        upper_bound = [upper_bound] * self.n_species if not isinstance(upper_bound, list) else upper_bound

        # Sample random state for each species
        return rng.integers(lower_bound, upper_bound)

    def set_output_function(self, output_function_name: str) -> None:
        """Set the output function and compute the output size."""
        self.output_function_name = output_function_name

        if output_function_name == "monomials":
            self.output_function = self.monomials_output_function
            self.out_function_size = self.out_species_size * len(self.rn_config["output_function_arguments"]["monomial_orders"])
        else:
            raise ValueError("Please configure a supported output function type.")

    def monomials_output_function(self, state: np.ndarray, _unused_array: np.ndarray = None, _unused_value: float = None) -> np.ndarray:
        """Compute moments based on the specified monomial orders, without cross moments."""
        output_species = state[:, self.out_species_indices]

        powers = np.array(self.rn_config["output_function_arguments"]["monomial_orders"])
        output_features = np.concatenate([output_species ** power for power in powers], axis=1)

        return output_features

    def compute_exact_values(self) -> None:
        """Compute the exact values of the reaction network."""
        raise NotImplementedError("Exact value computation is not available for this reaction network.")

    def get_stationary_mean_exact(self) -> None:
        """Calculate the exact stationary mean of the network."""
        raise NotImplementedError("Exact stationary mean calculation is not available for this network. Use the ergodic estimate instead.")

    def get_moments_exact(self) -> None:
        """Calculate the moment estimate of the network."""
        raise NotImplementedError("Moment estimation is not implemented for this network.")
