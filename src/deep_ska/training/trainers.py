"""Class implementing the training logic for the ExpectationModel."""

import logging

import numpy as np
import torch
from torch import Tensor, optim
from tqdm import tqdm

from ..analysis.analyzers import ExpectationGeneratorIS, TemporalSubnetAnalyzer
from ..core.initialization import RunContext
from ..core.model import ExpectationModel
from ..core.simulation import ExpectationGeneratorSSA
from ..logging.helpers import insert_blank_line, log_without_linebreak, training_timer
from ..reaction_networks.definition import ReactionNetwork


def inner_loss(x: Tensor, step: int, loss_type: str | list[str], loss_change_index: int) -> Tensor:
    """Calculate the inner loss based on the specified type."""
    if isinstance(loss_type, list):
        # If loss_type is a list, use the loss_change_index to select the appropriate loss type:
        loss_type = loss_type[0] if step < loss_change_index else loss_type[1]

    if loss_type == "manhattan":
        # Use the L1 norm (mean absolute error):
        return torch.mean(torch.abs(x), dim=0)
    elif loss_type == "euclidean":
        # Use the L2 norm (mean squared error):
        return torch.mean(torch.square(torch.abs(x)), dim=0)
    elif loss_type == "huber":
        # Deltas are squared if they are < 1, else (2 * their absolute value - 1) is used:
        return torch.mean(torch.where(torch.abs(x) < 1, torch.square(torch.abs(x)), 2 * torch.abs(x) - 1), dim=0)
    elif loss_type == "inverse_huber":
        # Deltas are squared if they are > 1, else (2 * their absolute value - 1) is used:
        return torch.mean(torch.where(torch.abs(x) < 1, torch.abs(x), 2 * torch.square(torch.abs(x)) - 1), dim=0)


class ModelTrainer:
    """General class for training a model."""

    def __init__(self, network: ReactionNetwork, run: RunContext) -> None:
        """Initialize the model trainer with the reaction network and run context."""
        # Initialize network and configuration data:
        self.network = network
        self.np_rng = run.np_rng
        self.rn_config = run.RN_CONFIG
        self.nn_config = run.NN_CONFIG
        self.active_trajectories = run.active_trajectories

        # Calculate total number of simulated trajectories:
        self.n_simulated_trajectories = self.nn_config["train_size"] + self.nn_config["valid_size"]
        self.n_steps = self.nn_config["n_steps"]

        if self.nn_config["train_samples_needed"] or self.nn_config["valid_samples_needed"]:
            self.ssa_generator = ExpectationGeneratorSSA(network, self.nn_config["training_seed"])

        # Create or load training and validation data:
        self._create_or_load_trajectory_data()

    def _create_or_load_trajectory_data(self) -> None:
        """Create or load the training and validation trajectory data."""
        # Create or load training data:
        if self.nn_config["train_samples_needed"]:
            if self.nn_config["use_ssa_samples"]:
                log_without_linebreak(f"Generating {self.nn_config['train_size']} training samples ...")
                self.train_data = self.ssa_generator.sample_temporal_rtc_trajectories(self.rn_config["t_stop"], self.rn_config["n_time_samples"], self.nn_config["train_size"], self.active_trajectories)
                if self.nn_config["save_train_data"]:
                    self.save_trajectory_data(self.train_data, "train", self.nn_config["train_size"])
            elif not self.nn_config["use_ssa_samples"] and self.nn_config["resample_during_training"]:
                logging.info("Not simulating any training trajectories right now, they will be generated on the fly.")
                self.train_data = None
            else:
                log_without_linebreak(f"Generating {self.nn_config['train_size']} training samples ...")
                self.train_data = self.ssa_generator.sample_constant_trajectories(self.rn_config["t_stop"], self.rn_config["n_time_samples"], self.nn_config["train_size"], self.active_trajectories)
                if self.nn_config["save_train_data"]:
                    self.save_trajectory_data(self.train_data, "train", self.nn_config["train_size"])
        else:
            self.train_data = self.load_trajectory_data("train", self.nn_config["train_size"])

        # Create or load validation data:
        if self.nn_config["valid_samples_needed"]:
            if self.nn_config["use_ssa_samples"]:
                log_without_linebreak(f"Generating {self.nn_config['valid_size']} validation samples ...")
                self.valid_data = self.ssa_generator.sample_temporal_rtc_trajectories(self.rn_config["t_stop"], self.rn_config["n_time_samples"], self.nn_config["valid_size"], self.active_trajectories)
                if self.nn_config["save_valid_data"]:
                    self.save_trajectory_data(self.valid_data, "valid", self.nn_config["valid_size"])
            else:
                log_without_linebreak(f"Generating {self.nn_config['valid_size']} validation samples ...")
                self.valid_data = self.ssa_generator.sample_constant_trajectories(self.rn_config["t_stop"], self.rn_config["n_time_samples"], self.nn_config["valid_size"], self.active_trajectories)
                if self.nn_config["save_valid_data"]:
                    self.save_trajectory_data(self.valid_data, "valid", self.nn_config["valid_size"])
        else:
            self.valid_data = self.load_trajectory_data("valid", self.nn_config["valid_size"])

    def save_trajectory_data(self, inputs: dict, input_type: str, size: int) -> None:
        """Save the trajectory data to a .npz file."""
        trajectories_path = f"{self.rn_config['trajectories_dir']}/{self.rn_config['reaction_network_abbreviation']}_{input_type}_{self.network.n_species}_{size}.npz"
        np.savez(trajectories_path, **inputs)
        logging.info(f"Saved {input_type} trajectory data to {trajectories_path}.")

    def load_trajectory_data(self, input_type: str, size: int) -> dict:
        """Load the trajectory data from a .npz file."""
        trajectories_path = f"{self.rn_config['trajectories_dir']}/{self.rn_config['reaction_network_abbreviation']}_{input_type}_{self.network.n_species}_{size}.npz"
        logging.info(f"Loading {input_type} trajectory data from {trajectories_path} ...")
        trajectories = np.load(trajectories_path)
        return {key: trajectories[key] for key in trajectories}

    def sample_mini_batch(self, inputs: dict, sample_size: int, early_stopping_mode: bool = False, history: dict = None) -> dict:
        """Sample a mini-batch of the training data for training the neural network."""
        times = inputs["times"]
        state_trajectories = inputs["state_trajectories"]
        n_time_samples = times.shape[0]
        # Randomly select a time window length:
        n_time_points = self.np_rng.integers(2, n_time_samples+1)
        if self.nn_config["use_full_length_trajectories"] or early_stopping_mode:
            n_time_points = n_time_samples  # overwrite window size to use full trajectories
        # Randomly select a start and end time:
        start_time = self.np_rng.integers(0, n_time_samples-n_time_points+1)
        end_time = start_time + n_time_points
        # Calculate the number of samples needed for the mini-batch and randomly select them:
        n_trajectories_needed = min(sample_size // n_time_points, state_trajectories.shape[0])
        batch_indices = self.np_rng.choice(range(state_trajectories.shape[0]), n_trajectories_needed, replace=False)

        # Update the history dictionary with the current mini-batch information:
        if history is not None:
            history["mini_batch_window_size"].append(int(n_time_points))
            history["mini_batch_start_time"].append(int(start_time))
            history["mini_batch_end_time"].append(int(end_time))
            history["mini_batch_n_trajectories"].append(int(n_trajectories_needed))

        # Initialize a dictionary to store the selected time intervals and trajectories:
        mini_batch = {}
        # Loop through each input and select the corresponding time intervals and trajectories:
        for key, data in inputs.items():
            if data is None:
                mini_batch[key] = None
            elif data.ndim == 1:  # e.g., "times", with shape (n_time_samples,)
                mini_batch[key] = data[start_time:end_time]
            elif data.ndim == 2:
                if data.shape[1] == n_time_samples:  # e.g., (n_trajectories, n_time_samples)
                    mini_batch[key] = data[batch_indices, start_time:end_time]
                else:
                    mini_batch[key] = data[batch_indices, :]  # e.g., (n_trajectories, some_feature_dim)
            elif data.ndim == 3:  # e.g., (n_trajectories, n_time_samples, some_feature_dim)
                mini_batch[key] = data[batch_indices, start_time:end_time, :]
            elif data.ndim == 4:  # e.g., (n_trajectories, n_time_samples, some_feature_dim_1, some_feature_dim_2)
                mini_batch[key] = data[batch_indices, start_time:end_time, :, :]
            else:
                raise ValueError(f"Unexpected data shape for {key}: {data.shape}")

        return mini_batch

    def simulate_new_batch(self) -> dict:
        """Simulate a new (mini) batch of trajectories for training."""
        if self.nn_config["use_mini_batches"]:
            # Create the times array to find the correct final time later:
            times = np.linspace(0, self.rn_config["t_stop"], self.rn_config["n_time_samples"])
            n_time_samples = times.shape[0]
            # Randomly select a time window length:
            n_time_points = self.np_rng.integers(2, n_time_samples+1)
            if self.nn_config["use_full_length_trajectories"]:
                n_time_points = n_time_samples  # overwrite window size to use full trajectories
            # Set stop time and calculate the number of trajectories:
            t_stop = self.rn_config["t_stop"]
            n_trajectories_needed = min(self.nn_config["mini_batch_size"] // n_time_points, self.nn_config["train_size"])
        else:
            # Use simulation parameters from the configuration:
            n_time_points = self.nn_config["n_time_samples"]
            t_stop = self.rn_config["t_stop"]
            n_trajectories_needed = self.nn_config["train_size"]

        # Simulate the trajectories using the (randomly determined) parameters:
        log_without_linebreak(f"Generating a new batch of {n_trajectories_needed} training samples ...")
        new_batch = self.ssa_generator.sample_temporal_rtc_trajectories(t_stop, n_time_samples, n_trajectories_needed, self.active_trajectories)

        # Randomly select a start and end time:
        start_time = self.np_rng.integers(0, n_time_samples-n_time_points+1)
        end_time = start_time + n_time_points

        # Initialize a dictionary to store the selected time intervals and trajectories:
        mini_batch = {}
        # Loop through each input and select the corresponding time intervals and trajectories:
        for key, data in new_batch.items():
            if data is None:
                mini_batch[key] = None
            elif data.ndim == 1:  # e.g., "times", with shape (n_time_samples,)
                mini_batch[key] = data[start_time:end_time]
            elif data.ndim == 2:
                if data.shape[1] == n_time_samples:  # e.g., (n_trajectories, n_time_samples)
                    mini_batch[key] = data[:, start_time:end_time]
                else:
                    mini_batch[key] = data[:, :]  # e.g., (n_trajectories, some_feature_dim)
            elif data.ndim == 3:  # e.g., (n_trajectories, n_time_samples, some_feature_dim)
                mini_batch[key] = data[:, start_time:end_time, :]
            elif data.ndim == 4:  # e.g., (n_trajectories, n_time_samples, some_feature_dim_1, some_feature_dim_2)
                mini_batch[key] = data[:, start_time:end_time, :, :]
            else:
                raise ValueError(f"Unexpected data shape for {key}: {data.shape}")

        return mini_batch


class ExpectationModelTrainer(ModelTrainer):
    """Class holding the logic for training and loss calculation of V."""

    def __init__(self, network: ReactionNetwork, run: RunContext) -> None:
        """Initialize the expectation model trainer with the reaction network and run context."""
        super().__init__(network, run)
        self.val_config = run.VAL_CONFIG
        # Initialize model, loss function, optimizer and scheduler for training:
        self.model = ExpectationModel(network, run.torch_rng, self.nn_config)

        # Training initialization for learning V:
        if self.model.v_as_relationship == "identity":
            self.loss_function = getattr(self, self.nn_config["v"]["loss_function"])
        elif self.model.v_as_relationship == "logarithm":
            self.loss_function = self.logarithm_loss
        else:
            raise ValueError(f'Relationship type "{self.model.v_as_relationship}" is invalid for V.')

        self.loss_component_weights = np.array(self.nn_config["v"]["loss_component_weights"])  # NOTE: Only used if multiple loss components are present.
        self.update_rate = self.nn_config["v"]["update_rate"]  # NOTE: Only used if adaptive weights are used.

        self.model_params = []
        self.frozen_params = []

        if self.nn_config["v"]["only_train_func_dependence"]:  # track which parameters can be frozen for faster training
            self.frozen_params.extend(list(self.model.main_net.parameters()) if self.model.main_net else [])
            self.frozen_params.extend(list(self.model.diff_net.parameters()) if self.model.diff_net else [])
            self.frozen_params.extend(list(self.model.quot_net.parameters()) if self.model.quot_net else [])
        else:
            self.model_params.extend(list(self.model.main_net.parameters()) if self.model.main_net else [])
            self.model_params.extend(list(self.model.diff_net.parameters()) if self.model.diff_net else [])
            self.model_params.extend(list(self.model.quot_net.parameters()) if self.model.quot_net else [])

        if self.model.v_subnet_architecture == "naive":
            pass
        elif self.model.v_subnet_architecture == "features":
            self.model_params.extend([self.model.decay_real, self.model.decay_imag, self.model.decay_phas])
        elif self.model.v_subnet_architecture in ["spectral_matched_lumped", "spectral_matched_simplified", "spectral_matched_full", "spectral_complex"]:
            if self.nn_config["v"]["stationary_mean_initialization"] in ["zeros", "ones", "rand", "estimate_trainable", "exact_trainable"]:
                self.model_params.append(self.model.v_stationary_mean)
            if self.model.v_subnet_architecture in ["spectral_matched_simplified", "spectral_matched_full", "spectral_complex"]:
                self.model_params.append(self.model.func_coord_real)
            if self.model.v_subnet_architecture in ["spectral_matched_full", "spectral_complex"]:
                self.model_params.append(self.model.func_coord_imag)
            if self.model.nn_config["v"]["decay_modes_initialization"] in ["rand", "trainable"] and not self.nn_config["v"]["only_train_func_dependence"]:
                self.model_params.extend([self.model.decay_real, self.model.decay_imag])
        else:
            raise ValueError(f'Subnet architecture "{self.model.v_subnet_architecture}" is invalid for V.')

        # Freeze model parameters that we don't want to update in the second training stage:
        for param in self.frozen_params:
            param.requires_grad_(False)

        self.optimizer = optim.Adam(self.model_params, lr=self.nn_config["lr"])
        if self.nn_config["lr_scheduler"] == "constant":
            self.scheduler = optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lambda _: 1)
        elif self.nn_config["lr_scheduler"] == "cosine":
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, self.n_steps, eta_min=0, last_epoch=-1)

        if self.nn_config["v"]["use_early_stopping"]:
            self.analyzer = TemporalSubnetAnalyzer(self.network, self.model, self.val_config, analysis_mode=self.nn_config["v"]["use_analysis_subnet_for_early_stopping"])
            self.v_threshold_count = 0

        if self.nn_config["use_is_resampling"]:
            self.is_generator = ExpectationGeneratorIS(self.network, self.model, self.val_config, seed=self.nn_config["training_seed"])

    @training_timer
    def train(self) -> dict:
        """Training loop with regular validation loss calculation."""
        history = {"train_step": [], "train_loss": [], "train_comp": [], "lr": [],
                   "valid_step": [], "valid_loss": [], "valid_comp": [], "train_mean": [],
                   "mini_batch_window_size": [], "mini_batch_start_time": [], "mini_batch_end_time": [], "mini_batch_n_trajectories": []}

        backward_frequency = self.nn_config["backward_frequency"]
        logging_frequency = self.nn_config["logging_frequency"]
        total_params = sum(p.numel() for group in self.optimizer.param_groups for p in group['params'] if p.requires_grad)

        # Go through the training steps with updating progress bar for training and validation loss:
        progress = tqdm(range(self.n_steps), desc=f"train: {0:.5f}, valid: {0:.5f}, lr: {0:.5f}", dynamic_ncols=True)
        for self.step in progress:
            if not self.nn_config["use_ssa_samples"] and self.nn_config["resample_during_training"] and self.step % self.nn_config["resampling_frequency"] == 0:
                insert_blank_line()
                log_without_linebreak(f"Generating {self.nn_config['train_size']} training samples ...")
                self.train_data = self.ssa_generator.sample_constant_trajectories(self.rn_config["t_stop"], self.rn_config["n_time_samples"], self.nn_config["train_size"], self.active_trajectories)
                log_without_linebreak(f"Generating {self.nn_config['valid_size']} validation samples ...")
                self.valid_data = self.ssa_generator.sample_constant_trajectories(self.rn_config["t_stop"], self.rn_config["n_time_samples"], self.nn_config["train_size"], self.active_trajectories)
                insert_blank_line()
            # Change the model once (when we reach the resampling start):
            if self.nn_config["use_is_resampling"] and self.step == self.nn_config["is_resampling_start"]:
                insert_blank_line()
                logging.info("Switching training to DeepIS resampling mode with an adjusted forward pass and loss function ...")
                self.model.forward = self.model.forward_deep_is_resampling  # set the specific forward pass
                self.loss_function = self.is_identity_loss
            # Check if we should generate new training data when we use the DeepIS resampling:
            if self.nn_config["use_is_resampling"] and self.step >= self.nn_config["is_resampling_start"] and (self.step - self.nn_config["is_resampling_start"]) % self.nn_config["is_resampling_frequency"] == 0:
                insert_blank_line()
                log_without_linebreak(f"Generating {self.nn_config['is_resampling_train_size']} DeepIS training samples per output function ...")
                self.train_data = self.is_generator.sample_final_deep_is_trajectories(self.rn_config["t_stop"], self.rn_config["n_time_samples"], self.nn_config["is_resampling_train_size"], self.active_trajectories)
                log_without_linebreak(f"Generating {self.nn_config['is_resampling_valid_size']} DeepIS validation samples per output function ...")
                self.valid_data = self.is_generator.sample_final_deep_is_trajectories(self.rn_config["t_stop"], self.rn_config["n_time_samples"], self.nn_config["is_resampling_valid_size"], self.active_trajectories)
                insert_blank_line()

            self.optimizer.zero_grad()  # reset the gradient
            total_train_components_loss = []
            total_train_loss = torch.tensor(0.0, requires_grad=True)  # for loss accumulation
            total_train_components = None

            for _ in range(backward_frequency):
                if self.train_data is not None:
                    # Select a mini-batch if "use_mini_batches" = True, else use the full training data:
                    curr_batch = self.sample_mini_batch(self.train_data, self.nn_config["mini_batch_size"], history=history) if self.nn_config["use_mini_batches"] else self.train_data
                else:
                    # If the training data was not generated, simulate a fresh mini-batch:
                    curr_batch = self.simulate_new_batch()

                if self.nn_config["v"]["adaptive_weights"]:
                    id_pred, pinn_pred = self.model(curr_batch)
                    train_loss, train_components = self.loss_function(curr_batch, self.train_data, id_pred, pinn_pred)
                    total_train_components_loss.append(train_components / backward_frequency)
                else:
                    # Forward pass through the model and calculate the training loss:
                    if self.model.v_as_relationship == "logarithm":
                        log_pred = self.model(curr_batch)
                        train_loss, train_components = self.loss_function(curr_batch, self.train_data, log_pred)
                    else:
                        id_pred, pinn_pred = self.model(curr_batch)
                        train_loss, train_components = self.loss_function(curr_batch, self.train_data, id_pred, pinn_pred)
                    total_train_loss = total_train_loss + train_loss / backward_frequency
                    train_components = train_components.detach().reshape(-1).tolist()
                    total_train_components = [total + train / backward_frequency  for total, train in zip(total_train_components, train_components, strict=True)] if total_train_components is not None else train_components

            if self.nn_config["v"]["adaptive_weights"]:
                train_components_loss = torch.sum(torch.stack(total_train_components_loss, dim=0), dim=0) * torch.tensor(self.loss_component_weights, requires_grad=False)
                train_components_grad = torch.zeros((self.loss_component_weights.shape[0], total_params), requires_grad=False)
                train_components_std = torch.zeros(self.loss_component_weights.shape[0], requires_grad=False)
                for i in range(len(self.loss_component_weights)):
                    train_components_loss[i].backward(retain_graph=True)
                    all_grads = torch.cat([p.grad.view(-1) for group in self.optimizer.param_groups for p in group['params'] if p.grad is not None])
                    if i == 0:
                        train_components_grad[i, :] = all_grads
                    else:
                        train_components_grad[i, :] = all_grads - torch.sum(train_components_grad, dim=0)
                    train_components_std[i] = torch.std(train_components_grad[i])
                eps = 1e-10
                intermediate_weights = max(train_components_std) / (train_components_std + eps)
                total_train_loss = torch.sum(train_components_loss).item()
                total_train_components = torch.sum(torch.stack(total_train_components_loss, dim=0), dim=0).detach().tolist()
            else:
                total_train_loss.backward()
                total_train_loss = total_train_loss.item()

            self.optimizer.step()
            self.scheduler.step()

            history["train_step"].append(self.step)
            history["train_loss"].append(total_train_loss)
            history["train_comp"].append([component for component in total_train_components])
            history["lr"].append(self.scheduler.get_last_lr()[0])

            # Evaluate model and update output every logging_frequency steps:
            with torch.no_grad():
                if self.step % logging_frequency == 0:
                    progress.desc = ""
                    logging_message = f"Progress: {(self.step+1)/self.n_steps:4.0%}, step: {self.step+1:{len(str(self.n_steps))}}/{self.n_steps}"
                    total_valid_loss = 0  # for loss accumulation
                    total_valid_components = None

                    for _ in range(backward_frequency):
                        # Select a mini-batch if "use_mini_batches" = True, else use the full validation data:
                        curr_batch = self.sample_mini_batch(self.valid_data, self.nn_config["mini_batch_size"]) if self.nn_config["use_mini_batches"] else self.valid_data
                        # Forward pass through the model and calculate the validation loss:
                        if self.model.v_as_relationship == "logarithm":
                            log_pred = self.model(curr_batch)
                            valid_loss, valid_components = self.loss_function(curr_batch, self.valid_data, log_pred)
                        else:
                            id_pred, pinn_pred = self.model(curr_batch)
                            valid_loss, valid_components = self.loss_function(curr_batch, self.valid_data, id_pred, pinn_pred)
                        total_valid_loss = total_valid_loss + valid_loss.item() / backward_frequency
                        valid_components = valid_components.detach().reshape(-1).tolist()
                        total_valid_components = [total + valid / backward_frequency for total, valid in zip(total_valid_components, valid_components, strict=True)] if total_valid_components is not None else valid_components

                    history["valid_step"].append(self.step)
                    history["train_mean"].append(np.mean(history["train_loss"][-logging_frequency:]))
                    history["valid_loss"].append(total_valid_loss)
                    history["valid_comp"].append([component for component in total_valid_components])

                    # Update the description of the progress bar:
                    progress.desc += f"| exp_train: {history['train_mean'][-1]:.5f}, exp_valid: {history['valid_loss'][-1]:.5f}, exp_lr: {history['lr'][-1]:.5f}"
                    logging_message += f" | exp_lr: {history['lr'][-1]:.5f}, exp_train: {history['train_mean'][-1]:.5f}, exp_valid: {history['valid_loss'][-1]:.5f}"

                    # Log the step and the learning rate, as well as the training and validation loss:
                    logging.info(logging_message)

                    if self.step % (logging_frequency*1) == 0:
                        logging.debug(history["train_comp"][-1])
                        logging.debug(history["valid_comp"][-1])
                        logging.debug(self.loss_component_weights[1])

                    if self.nn_config["v"]["use_early_stopping"]:
                        log_without_linebreak("Computing temporal 'SSA with DeepCV' estimate for early stopping ...")
                        valid_subset = self.sample_mini_batch(self.valid_data, self.nn_config["v"]["early_stopping_batch_size"], early_stopping_mode=True)
                        self.analyzer.compute_temporal_deep_cv_estimate(valid_subset, temp_dict := {})
                        curr_fraction = np.mean(np.abs(temp_dict["diff_cv"]) < self.nn_config["v"]["early_stopping_threshold"])
                        self.v_threshold_count = self.v_threshold_count + 1 if curr_fraction >= self.nn_config["v"]["early_stopping_fraction"] else 0
                        logging.info(f"Current DeepCV fraction: {curr_fraction:.5f}, current count: {self.v_threshold_count}\n")
                        insert_blank_line()

                        if self.v_threshold_count >= self.nn_config["v"]["early_stopping_count"]:
                            logging.info(f"Stopping training at step {self.step+1} due to DeepCV threshold of {self.nn_config['v']['early_stopping_threshold']} being reached {self.v_threshold_count} time(s) in a row.")
                            insert_blank_line()
                            break

            if self.nn_config["v"]["adaptive_weights"]:
                self.loss_component_weights = self.update_rate * self.loss_component_weights + (1 - self.update_rate) * intermediate_weights.numpy()

        # Also always log the final step:
        logging.info(f"Progress: {(self.step+1)/self.n_steps:4.0%}, step: {self.step+1:{len(str(self.n_steps))}}/{self.n_steps}")

        return history

    def identity_loss(self, inputs: dict, full_inputs: dict, id_pred: Tensor, pinn_pred: Tensor) -> tuple[Tensor, list]:
        """Loss function using the identity constraint."""
        _ = pinn_pred  # unused variable

        state_trajectories = inputs["state_trajectories"]
        # Get true network output at the last time point and compute clipping value:
        id_true = torch.from_numpy(self.network.output_function(state_trajectories[:, -1, :])).unsqueeze(1)

        if "full" in self.nn_config["v"]["normalization"]:
            # Find the index of the last batch time point in the full inputs:
            full_state_trajectories = full_inputs["state_trajectories"]
            time_index = np.where(full_inputs["times"] == inputs["times"][-1])[0].item()
            # Get the simulated reaction network output for all trajectories at the last batch time point:
            full_id_true = torch.from_numpy(self.network.output_function(full_state_trajectories[:, time_index, :])).unsqueeze(1)

        # Get delta between predicted and true identity:
        if self.nn_config["v"]["normalization"] == "none":
            delta = id_pred - id_true
        elif self.nn_config["v"]["normalization"] == "ergodic_mean":
            output_mean = self.model.v_stationary_mean.detach()
            delta_clip = 1 + output_mean
            delta = (id_pred - id_true) / delta_clip
        elif self.nn_config["v"]["normalization"] == "mean_full":
            output_mean = torch.mean(full_id_true, axis=0)
            delta_clip = 1 + output_mean
            delta = (id_pred - id_true) / delta_clip
        elif self.nn_config["v"]["normalization"] == "mean_batch":
            output_mean = torch.mean(id_true, axis=0)
            delta_clip = 1 + output_mean
            delta = (id_pred - id_true) / delta_clip
        elif self.nn_config["v"]["normalization"] == "mean_std_full":
            output_mean = torch.mean(full_id_true, axis=0)
            output_std = torch.std(full_id_true, axis=0)
            delta_clip = 1 + output_mean + 2 * output_std
            delta = (id_pred - id_true) / delta_clip
        elif self.nn_config["v"]["normalization"] == "mean_std_batch":
            output_mean = torch.mean(id_true, axis=0)
            output_std = torch.std(id_true, axis=0)
            delta_clip = 1 + output_mean + 2 * output_std
            delta = (id_pred - id_true) / delta_clip
        else:
            raise ValueError(f'Invalid normalization type "{self.nn_config["v"]["normalization"]}".')

        # Use the specified inner loss function to compute the loss:
        id_loss = torch.sum(inner_loss(delta, self.step, self.nn_config["v"]["inner_loss"], self.nn_config["v"]["inner_loss_change_index"]))  # sum of losses
        return id_loss, id_loss # loss and loss components

    def is_identity_loss(self, inputs: dict, full_inputs: dict, id_pred: Tensor, pinn_pred: Tensor) -> tuple[Tensor, list]:
        """Loss function using the identity constraint for the DeepIS resampling forward pass."""
        _ = pinn_pred  # unused variable

        # Get true network output at the last time point and compute clipping value:
        id_true = torch.from_numpy(inputs["output_trajectories"][:, -1, :]).unsqueeze(1)

        if "full" in self.nn_config["v"]["normalization"]:
            # Find the index of the last batch time point in the full inputs:
            time_index = np.where(full_inputs["times"] == inputs["times"][-1])[0].item()
            # Get the simulated reaction network output for all trajectories at the last batch time point:
            full_id_true = torch.from_numpy(full_inputs["output_trajectories"][:, time_index, :]).unsqueeze(1)

        # Get delta between predicted and true identity:
        if self.nn_config["v"]["normalization"] == "none":
            delta = id_pred - id_true
        elif self.nn_config["v"]["normalization"] == "mean_full":
            output_mean = torch.mean(full_id_true, axis=0)
            delta_clip = 1 + output_mean
            delta = (id_pred - id_true) / delta_clip
        elif self.nn_config["v"]["normalization"] == "mean_batch":
            output_mean = torch.mean(id_true, axis=0)
            delta_clip = 1 + output_mean
            delta = (id_pred - id_true) / delta_clip
        elif self.nn_config["v"]["normalization"] == "mean_std_full":
            output_mean = torch.mean(full_id_true, axis=0)
            output_std = torch.std(full_id_true, axis=0)
            delta_clip = 1 + output_mean + 2 * output_std
            delta = (id_pred - id_true) / delta_clip
        elif self.nn_config["v"]["normalization"] == "mean_std_batch":
            output_mean = torch.mean(id_true, axis=0)
            output_std = torch.std(id_true, axis=0)
            delta_clip = 1 + output_mean + 2 * output_std
            delta = (id_pred - id_true) / delta_clip
        else:
            raise ValueError(f'Invalid normalization type "{self.nn_config["v"]["normalization"]}".')

        # Use the specified inner loss function to compute the loss:
        id_loss = torch.sum(inner_loss(delta, self.step, self.nn_config["v"]["inner_loss"], self.nn_config["v"]["inner_loss_change_index"]))  # sum of losses
        return id_loss, id_loss  # loss and loss components

    def pinn_loss(self, inputs: dict, full_inputs: dict, id_pred: Tensor, pinn_pred: Tensor) -> tuple[Tensor, list]:
        """Loss function using the PINN constraint."""
        _ = inputs, full_inputs, id_pred  # unused variables

        # Take the mean over all samples:
        pinn_loss = torch.sum(torch.mean(pinn_pred, dim=0))  # sum of losses
        return pinn_loss, pinn_loss  # loss and loss components

    def id_pinn_loss(self, inputs: dict, full_inputs: dict, id_pred: Tensor, pinn_pred: Tensor) -> tuple[Tensor, list]:
        """Loss function using both the identity and PINN constraints."""
        id_loss, _ = self.identity_loss(inputs, full_inputs, id_pred, pinn_pred)
        pinn_loss, _ = self.pinn_loss(inputs, full_inputs, id_pred, pinn_pred)
        # Include combined id_loss and pinn_loss terms in the total loss calculation:
        loss = id_loss * self.loss_component_weights[0] + pinn_loss * self.loss_component_weights[1]
        return loss, torch.stack([id_loss, pinn_loss])  # loss and loss components

    def logarithm_loss(self, inputs: dict, full_inputs: dict, log_pred: Tensor) -> tuple[Tensor, list]:
        """Loss function for the logarithm relationship."""
        _ = full_inputs  # unused variable

        # NOTE: Current normalization corresponds to mean_std_batch.

        eps= 1e-16
        state_trajectories = inputs["state_trajectories"]
        # Get true network output at the last time point and compute clipping value:
        log_true = torch.log(torch.from_numpy(self.network.output_function(state_trajectories[:, -1, :])) + eps).unsqueeze(1)
        output_mean = torch.mean(log_true, axis=0)  # compute the mean output as the model input
        output_std = torch.std(log_true, axis=0)  # to set reasonable clipping values
        delta_clip = 1 + output_mean + 2 * output_std  # clipping value to avoid destabilization
        # Get delta between predicted and true identity normalized by delta_clip (unsqueeze if all terms are computed):
        delta = (log_pred - log_true) / delta_clip
        # Use the specified inner loss function to compute the loss:
        log_loss = torch.sum(inner_loss(delta, self.step, self.nn_config["v"]["inner_loss"], self.nn_config["v"]["inner_loss_change_index"]))  # sum of losses
        return log_loss, log_loss  # loss and loss components
