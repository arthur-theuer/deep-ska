"""Trajectory generation and ergodic mean computation for reaction networks."""

import concurrent.futures
import copy
import logging

import numpy as np
from tqdm.auto import tqdm

from ..logging.helpers import array_memory_logger, execution_timer
from ..reaction_networks.definition import ReactionNetwork
from .utils import create_state_space, get_available_cpus


class ExpectationGeneratorSSA:
    """Class for generating trajectories using Gillespie's Stochastic Simulation Algorithm (SSA)."""

    def __init__(self, network: ReactionNetwork, seed: int) -> None:
        """Initialize the RTC trajectory generator with a reaction network and a random seed."""
        self.network = network
        self.rng = np.random.default_rng(seed)
        self.noise_rng = np.random.default_rng(seed+1)

    def sample_temporal_rtc_trajectory(self, t_stop: float, n_time_samples: int) -> tuple:
        """Sample a single random time change trajectory from a uniform distribution."""
        # Initialize arrays for keeping track of time, species and reactions:
        sampling_times = np.linspace(0, t_stop, n_time_samples)  # sampling at constant intervals
        states = np.zeros([n_time_samples, self.network.n_species])  # states of the species over time
        reaction_counts = np.zeros([n_time_samples, self.network.n_reactions])  # reaction counts over time
        compensators = np.zeros([n_time_samples, self.network.n_reactions])  # to correct simulation biases

        curr_reaction_counts = np.zeros([self.network.n_reactions])
        internal_times = np.zeros([self.network.n_reactions])
        jump_times = -np.log(self.rng.uniform(0, 1, self.network.n_reactions))

        curr_state = self.network.initial_state if self.network.init_type == "deter" else self.network.sample_random_state(self.rng)
        t_curr = 0
        delta_reactions = np.zeros([self.network.n_reactions])
        count = 0

        while 1:
            # Compute the time delta:
            prop = self.network.propensity_vector(curr_state)  # evaluate propensities at every time point
            delta_reactions = np.divide((jump_times - internal_times), prop, out=np.full_like(delta_reactions, np.inf), where=prop > 0)

            if np.any(prop < 0):
                logging.warning(f"Negative propensity detected: {prop}.")

            next_reaction = np.argmin(delta_reactions, axis=0)  # find shortest reaction delta to determine next reaction (returns index)
            delta_time = delta_reactions[next_reaction]  # access time until the found next reaction occurs (returns value)

            if delta_time <= 0:
                logging.warning(f"Non-positive time delta detected: {delta_time}.")

            # Update the arrays:
            while count < n_time_samples and t_curr <= sampling_times[count] < (t_curr + delta_time):
                states[count, :] = curr_state
                reaction_counts[count, :] = curr_reaction_counts
                if count > 0:
                    compensators[count, :] = compensators[count-1, :] + prop * (sampling_times[count] - sampling_times[count-1])
                count += 1

            t_curr += delta_time

            if t_curr <= t_stop:  # update the state as long as the stop time is not exceeded
                internal_times += prop * delta_time  # update internal times with propensities and time delta
                curr_state = self.network.update_state(next_reaction, curr_state)

                if np.any(curr_state < 0):
                    logging.warning(f"Negative state component detected: {curr_state}")

                curr_reaction_counts[next_reaction] += 1
                jump_times[next_reaction] += -np.log(self.rng.uniform(0, 1))
            else:  # if the stop time is exceeded, return the final arrays
                return sampling_times, states, reaction_counts, compensators

    @staticmethod
    def process_temporal_rtc_trajectory(network: ReactionNetwork, t_stop: float, n_time_samples: int, active_trajectories: set[str], idx: int, seed: int) -> dict[list]:
        """Process an RTC trajectory. Needs to be a static method for multiprocessing."""
        local_network = copy.deepcopy(network)
        # Use a new random seed for each trajectory (which means we need a new simulator):
        local_simulator = ExpectationGeneratorSSA(local_network, seed)
        times, states, reaction_counts, compensators = local_simulator.sample_temporal_rtc_trajectory(t_stop, n_time_samples)
        # Save results in a dictionary:
        result = {
            "idx": idx,
            "times": times,
            "state_trajectories": states if "state_trajectories" in active_trajectories else None,
            "reaction_count_trajectories": reaction_counts if "reaction_count_trajectories" in active_trajectories else None,
        }

        # Only calculate values for the active trajectories:
        if "martingale_trajectories" in active_trajectories:
            result["martingale_trajectories"] = reaction_counts - compensators
        if "propensity_trajectories" in active_trajectories:
            result["propensity_trajectories"] = np.apply_along_axis(local_network.propensity_vector, axis=1, arr=states)

        return result

    @execution_timer
    @array_memory_logger
    def sample_temporal_rtc_trajectories(self, t_stop: float, n_time_samples: int, n_trajectories: int, active_trajectories: set[str], n_jobs: int = None) -> dict:
        """Sample multiple random time change trajectories."""
        # Define shapes of all possible trajectory arrays:
        trajectory_shapes = {
            "state_trajectories": (n_trajectories, n_time_samples, self.network.n_species),
            "reaction_count_trajectories": (n_trajectories, n_time_samples, self.network.n_reactions),
            "martingale_trajectories": (n_trajectories, n_time_samples, self.network.n_reactions),
            "propensity_trajectories": (n_trajectories, n_time_samples, self.network.n_reactions),
        }

        # Initialize only the active ones:
        trajectories = {name: np.zeros(shape) if name in active_trajectories else None for name, shape in trajectory_shapes.items()}

        # Precompute a list of random seeds (one per trajectory):
        # NOTE: While highly unlikely, nested RNGs mean that duplicate trajectories are possible.
        seeds = self.rng.choice(2**32, size=n_trajectories, replace=False)

        # Determine the number of parallel jobs:
        n_jobs = get_available_cpus(n_jobs)  # set dynamically in case SLURM is used

        # Start a subprocess for each trajectory:
        with concurrent.futures.ProcessPoolExecutor(max_workers=n_jobs) as executor:
            futures = []
            for i in range(n_trajectories):
                seed = seeds[i]
                futures.append(executor.submit(self.process_temporal_rtc_trajectory, self.network, t_stop, n_time_samples, active_trajectories, i, seed))

            with tqdm(total=n_trajectories, desc="rtc_trajectories", dynamic_ncols=True) as pbar:
                for future in concurrent.futures.as_completed(futures):
                    result = future.result()

                    trajectories["times"] = result["times"]
                    for name in active_trajectories:
                        try:
                            trajectories[name][result["idx"]] = result[name]
                        except KeyError:
                            pass

                    pbar.update(1)

        return trajectories

    def sample_final_rtc_trajectory(self, t_stop: float) -> tuple:
        """Sample a single random time change trajectory from a uniform distribution."""
        # Initialize arrays for keeping track of time, species and reactions:
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

            if delta_time <= 0:
                logging.warning(f"Non-positive time delta detected: {delta_time}.")

            t_curr += delta_time

            if t_curr <= t_stop:  # update the state as long as the stop time is not exceeded
                internal_times += prop * delta_time  # update internal times with propensities and time delta
                curr_state = self.network.update_state(next_reaction, curr_state)

                if np.any(curr_state < 0):
                    logging.warning(f"Negative state component detected: {curr_state}")

                jump_times[next_reaction] += -np.log(self.rng.uniform(0, 1))
            else:  # if the stop time is exceeded, return the final arrays
                return curr_state

    @staticmethod
    def process_final_rtc_trajectory(network: ReactionNetwork, t_stop: float, idx: int, seed: int) -> dict[list]:
        """Process an RTC trajectory. Needs to be a static method for multiprocessing."""
        local_network = copy.deepcopy(network)
        # Use a new random seed for each trajectory (which means we need a new simulator):
        local_simulator = ExpectationGeneratorSSA(local_network, seed)
        state = local_simulator.sample_final_rtc_trajectory(t_stop)
        # Save results in a dictionary:
        result = {
            "idx": idx,
            "state": state,
        }

        return result

    @execution_timer
    @array_memory_logger
    def sample_final_rtc_trajectories(self, t_stop: float, n_trajectories: int, n_jobs: int = None) -> dict:
        """Sample multiple random time change trajectories with only the final state being returned."""
        trajectories = {"states": np.zeros((n_trajectories, self.network.n_species))}

        # NOTE: While highly unlikely, nested RNGs mean that duplicate trajectories are possible.
        seeds = self.rng.choice(2**32, size=n_trajectories, replace=False)

        # Determine the number of parallel jobs:
        n_jobs = get_available_cpus(n_jobs)  # set dynamically in case SLURM is used

        # Start a subprocess for each trajectory:
        with concurrent.futures.ProcessPoolExecutor(max_workers=n_jobs) as executor:
            futures = []
            for i in range(n_trajectories):
                seed = seeds[i]
                futures.append(executor.submit(self.process_final_rtc_trajectory, self.network, t_stop, i, seed))

            with tqdm(total=n_trajectories, desc="final_rtc_trajectories", dynamic_ncols=True) as pbar:
                for future in concurrent.futures.as_completed(futures):
                    result = future.result()
                    trajectories["states"][result["idx"]] = result["state"]
                    pbar.update(1)

        return trajectories

    def sample_constant_trajectory(self, t_stop: float, n_time_samples: int, state: np.array) -> tuple:
        """Sample a single constant trajectory (no reactions occur)."""
        times = np.linspace(0, t_stop, n_time_samples)  # sampling at constant intervals
        states = np.tile(state, (n_time_samples, 1))
        propensities = self.network.propensity_vector(state)
        reaction_counts = np.zeros([n_time_samples, self.network.n_reactions])
        compensators = np.zeros([n_time_samples, self.network.n_reactions])
        dt = np.diff(times)
        compensators[1:, :] = propensities * dt[:, np.newaxis]
        return times, states, reaction_counts, compensators

    @staticmethod
    def process_constant_trajectory(network: ReactionNetwork, t_stop: float, n_time_samples: int, active_trajectories: set[str], idx: int, seed: int, state: np.array) -> dict[list]:
        """Process a constant trajectory. Needs to be a static method for multiprocessing."""
        local_network = copy.deepcopy(network)
        # Use a new random seed for each trajectory (only needed for noisy trajectories):
        local_simulator = ExpectationGeneratorSSA(local_network, seed)
        times, states, reaction_counts, compensators = local_simulator.sample_constant_trajectory(t_stop, n_time_samples, state)
        # Save results in a dictionary:
        result = {
            "idx": idx,
            "times": times,
            "state_trajectories": states if "state_trajectories" in active_trajectories else None,
            "reaction_count_trajectories": reaction_counts if "reaction_count_trajectories" in active_trajectories else None,
        }

        # Only calculate values for the active trajectories:
        if "martingale_trajectories" in active_trajectories:
            result["martingale_trajectories"] = reaction_counts - compensators
        if "propensity_trajectories" in active_trajectories:
            result["propensity_trajectories"] = np.apply_along_axis(local_network.propensity_vector, axis=1, arr=states)

        return result

    @execution_timer
    @array_memory_logger
    def sample_constant_trajectories(self, t_stop: float, n_time_samples: int, n_trajectories: int, active_trajectories: set[str], n_jobs: int = None) -> dict:
        """Sample multiple constant trajectories."""
        # Define shapes of all possible trajectory arrays:
        trajectory_shapes = {
            "state_trajectories": (n_trajectories, n_time_samples, self.network.n_species),
            "reaction_count_trajectories": (n_trajectories, n_time_samples, self.network.n_reactions),
            "martingale_trajectories": (n_trajectories, n_time_samples, self.network.n_reactions),
            "propensity_trajectories": (n_trajectories, n_time_samples, self.network.n_reactions),
        }

        # Initialize only the active ones:
        trajectories = {name: np.zeros(shape) if name in active_trajectories else None for name, shape in trajectory_shapes.items()}

        # Generate random initial states array (without replacement if possible):
        state_space = create_state_space(self.network.n_species, self.network.rn_config["lower_bound"], self.network.rn_config["upper_bound"])
        replace = n_trajectories > len(state_space)
        sampled_indices = self.rng.choice(len(state_space), size=n_trajectories, replace=replace)
        state_space = state_space[sampled_indices]

        # NOTE: While highly unlikely, nested RNGs mean that duplicate trajectories are possible.
        seeds = self.rng.choice(2**32, size=n_trajectories, replace=False)

        # Determine the number of parallel jobs:
        n_jobs = get_available_cpus(n_jobs)  # set dynamically in case SLURM is used

        # Start a subprocess for each trajectory:
        with concurrent.futures.ProcessPoolExecutor(max_workers=n_jobs) as executor:
            futures = []
            for i in range(n_trajectories):
                seed = seeds[i]
                state = state_space[i]
                futures.append(executor.submit(self.process_constant_trajectory, self.network, t_stop, n_time_samples, active_trajectories, i, seed, state))

            with tqdm(total=n_trajectories, desc="rtc_trajectories", dynamic_ncols=True) as pbar:
                for future in concurrent.futures.as_completed(futures):
                    result = future.result()

                    trajectories["times"] = result["times"]
                    for name in active_trajectories:
                        try:
                            trajectories[name][result["idx"]] = result[name]
                        except KeyError:
                            pass

                    pbar.update(1)

        return trajectories


class ExpectationGeneratorEM:
    """Class for computing an ergodic mean of the reaction network expectation."""

    def __init__(self, network: ReactionNetwork, seed: int) -> None:
        """Initialize the ergodic mean generator with a reaction network and a random seed."""
        self.network = network
        self.rng = np.random.default_rng(seed)

    def run_random_time_change(self, t_stop: float, initial_state: np.ndarray) -> np.ndarray:
        """Random time change without generating trajectories, needed for the validation routine."""
        internal_times = np.zeros([self.network.n_reactions])
        jump_times = -np.log(self.rng.uniform(0, 1, self.network.n_reactions))

        curr_state = initial_state
        t_curr = 0
        delta_reactions = np.zeros([self.network.n_reactions])

        while 1:
            # Compute the time delta:
            prop = self.network.propensity_vector(curr_state)  # evaluate propensities at every time point
            delta_reactions = np.divide((jump_times - internal_times), prop, out=np.full_like(delta_reactions, np.inf), where=prop > 0)  # make time interval between reactions infinite if propensity is 0

            next_reaction = np.argmin(delta_reactions, axis=0)  # find shortest reaction delta to determine next reaction (returns index)
            delta_time = delta_reactions[next_reaction]  # access time until the found next reaction occurs (returns value)
            internal_times += prop * delta_time  # update internal times with propensities and time delta

            t_curr += delta_time

            if t_curr <= t_stop:  # update the state as long as the stop time is not exceeded
                curr_state = self.network.update_state(next_reaction, curr_state)
                jump_times[next_reaction] += -np.log(self.rng.uniform(0, 1))
            else:  # if the stop time is exceeded, return the final state
                return curr_state

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

            t_curr += delta_time
            ergodic_mean += np.squeeze(self.network.output_function(np.expand_dims(curr_state, axis=0))) * delta_time / (t_max - t_min)

            if t_curr < t_max:  # update the state as long as the stop time is not exceeded
                internal_times += propensities * delta_time  # update internal times with propensities and time delta
                curr_state = self.network.update_state(next_reaction, curr_state)
                jump_times[next_reaction] += -np.log(self.rng.uniform(0, 1))
            else:  # if the stop time is exceeded, return the final state
                return ergodic_mean

    @staticmethod
    def process_ergodic_mean_sample(network: ReactionNetwork, t_min: float, t_max: float, idx: int, seed: int) -> np.ndarray:
        """Process an ergodic mean sample. Needs to be a static method for multiprocessing."""
        local_network = copy.deepcopy(network)
        # Use a new random seed for each trajectory (which means we need a new simulator):
        local_simulator = ExpectationGeneratorEM(local_network, seed)
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
            futures = [executor.submit(self.process_ergodic_mean_sample, self.network, t_min, t_max, i, seeds[i]) for i in range(n_trajectories)]

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
