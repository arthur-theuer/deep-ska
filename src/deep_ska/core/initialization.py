"""Contains the RunContext class to initialize the current model run."""

import argparse
import json
import logging
import os
import shutil
from datetime import datetime

import numpy as np
import torch
import yaml

from ..logging.helpers import get_git_commit_short_hash, insert_blank_line, insert_section_marker


class RunContext:
    """Class to initialize the run context for the current model run."""

    def __init__(self, args: argparse.Namespace) -> None:
        """Initialize the run context for the current model run."""
        self.timestamp = self.set_timestamp()  # set timestamp for the current run
        torch.set_default_dtype(torch.float64)  # set the default data type for torch tensors

        # Parse command line arguments:
        self.config_path = os.path.join(args.config_dir, args.config_file)
        self.config_file_name = os.path.splitext(args.config_file)[0]

        # Load configuration file:
        self.RN_CONFIG, self.NN_CONFIG, self.VAL_CONFIG, self.PLT_CONFIG = self.read_config_file(
            self.config_path
        )

        # Initialize random number generators with seeds to make results reproducable:
        self.np_rng = np.random.default_rng(self.NN_CONFIG["training_seed"])
        self.torch_rng = torch.Generator().manual_seed(self.NN_CONFIG["torch_seed"])
        torch.manual_seed(self.NN_CONFIG["torch_seed"])  # where the RNG can't be used directly

        # Set output directory (and create it if it does not exist):
        self.set_directory_path(dir_path=self.RN_CONFIG["trajectories_dir"])
        self.results_dir = self.set_directory_path(dir_path=self.RN_CONFIG["results_dir"])
        self.results_subdir = self.set_directory_path(
            dir_path=os.path.join(self.results_dir, f"{self.timestamp}_{self.config_file_name}")
        )

        # Set up output logging messages for user information and debugging:
        self.VAL_CONFIG["log_file_path"] = os.path.join(
            self.results_subdir,
            f"{self.timestamp}_{self.RN_CONFIG['reaction_network_abbreviation']}_logging_output.log"
        )
        self.set_logging_config(file_path=self.VAL_CONFIG["log_file_path"])
        insert_section_marker()
        # Log run infos and the current short git commit hash:
        logging.info(f"Starting DeepSKA run using git commit {get_git_commit_short_hash()} ...")
        logging.info(f"The timestamp for this run is '{self.timestamp}'.")
        logging.info(f"You are using the '{args.config_file}' configuration file.")

        # Copy config file to results subdir:
        shutil.copy2(self.config_path, self.results_subdir)
        logging.info(f"Configuration file copied to '{self.results_subdir}/'.")

        # Create the list of active trajectories (trajectories to be saved during simulation):
        self.active_trajectories = self.define_active_trajectories()
        self.validation_trajectories = self.define_validation_trajectories(self.active_trajectories)

        # Log the current configuration file content:
        insert_section_marker()
        logging.info("CONFIGURATION FILE CONTENT:")
        insert_blank_line()
        self.log_config_file_content(self.config_path)
        insert_section_marker()

    def set_timestamp(self) -> str:
        """Create a timestamp for the current run."""
        return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    def set_directory_path(self, dir_path: str) -> str:
        """Set a directory path and create it if it does not exist."""
        os.makedirs(dir_path, exist_ok=True) if not os.path.exists(dir_path) else None
        return dir_path

    def set_logging_config(self, file_path: str) -> None:
        """General logging configuration to print and save all output messages."""
        logging.basicConfig(
            format="%(asctime)s - %(levelname)s - %(message)s",
            level=logging.INFO,
            handlers=[logging.FileHandler(file_path), logging.StreamHandler()],
            force=True,
        )

    def read_config_file(self, file_path: str) -> tuple[dict, dict, dict, dict]:
        """Open the configuration file for the run and unpack the dictionary."""
        with open(file_path) as f:
            if file_path.endswith(".yaml"):
                config = yaml.safe_load(f)  # parse YAML file
            elif file_path.endswith(".json"):
                config = json.load(f)  # parse JSON file
            else:
                raise ValueError("Unsupported file format. Please provide a .JSON or .YAML file.")

        # Unpack sub-dictionaries:
        rn_config = config.get("reaction_network_config", {})
        nn_config = config.get("neural_network_config", {})
        val_config = config.get("validation_config", {})
        plt_config = config.get("plotting_config", {})

        return rn_config, nn_config, val_config, plt_config

    def log_config_file_content(self, config_path: str) -> None:
        """Write the content of the configuration file to the log file without logging prefixes."""
        with open(config_path) as file:
            config_content = file.read()

        logger = logging.getLogger()

        for handler in logger.handlers:
            if isinstance(handler, (logging.FileHandler | logging.StreamHandler)):
                handler.stream.write(config_content)
                handler.flush()

    def define_active_trajectories(self) -> set[str]:
        """Define the active trajectories based on the configuration file."""
        active_trajectories = {"state_trajectories"}
        if self.NN_CONFIG["v"]["as_relationship"] == "identity":
            active_trajectories.add("martingale_trajectories")
        elif self.NN_CONFIG["v"]["as_relationship"] == "logarithm":
            active_trajectories.add("reaction_count_trajectories")
            active_trajectories.add("propensity_trajectories")
            if self.NN_CONFIG["v"]["use_early_stopping"]:
                active_trajectories.add("martingale_trajectories")

        if self.NN_CONFIG["v"]["loss_function"] in ["pinn_loss", "id_pinn_loss"]:
            active_trajectories.add("propensity_trajectories")

        if self.NN_CONFIG["use_is_resampling"]:
            active_trajectories.add("output_trajectories")

        return active_trajectories

    def define_validation_trajectories(self, active_trajectories: set) -> set[str]:
        """Define the active validation trajectories based on the configuration file."""
        additional_validation_trajectories = {"propensity_trajectories", "martingale_trajectories"}

        if self.VAL_CONFIG["compute_is_estimate"]:
            additional_validation_trajectories.add("is_trajectories")

        return (active_trajectories | additional_validation_trajectories)
