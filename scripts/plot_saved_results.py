"""Replot the saved results from a directory using the .pkl.gz files generated for each test state.

USAGE EXAMPLES:

Single file processing:
  python scripts/plot_saved_results.py path/to/results_file.pkl.gz
  python scripts/plot_saved_results.py path/to/results_file.pkl.gz --output_dir <custom_output>

Batch processing (all test states in a directory):
  python scripts/plot_saved_results.py path/to/results_directory/
  python scripts/plot_saved_results.py path/to/results_directory/ --output_dir <custom_output>

Config override examples:
  python scripts/plot_saved_results.py path/to/results_file.pkl.gz --config_override
  python scripts/plot_saved_results.py path/to/results_directory/ --config_override

Combined options:
  python scripts/plot_saved_results.py path/to/results_directory/ --output_dir <custom_output> --config_override

FILE FORMAT:
  Expects pickle files with combined metadata and results structure:
  {
    "metadata": {"timestamp", "test_state_index", "config", ...},
    "results": {"test_state", "valid_data", "ert", "sd_dict", ...}
  }
"""

import argparse
import gzip
import logging
import pickle
from pathlib import Path

import numpy as np
import yaml

from deep_ska import RunContext
from deep_ska.logging.helpers import insert_blank_line, insert_section_marker
from deep_ska.plots import expectation_plots, panels, spectral_plots

yaml_to_config_mapping = {
    "reaction_network_config": "RN_CONFIG",
    "neural_network_config": "NN_CONFIG",
    "validation_config": "VAL_CONFIG",
    "plotting_config": "PLT_CONFIG"
}


def load_analysis_results(filepath: str) -> tuple[dict, dict]:
    """Load saved analysis results and metadata from the combined .pkl or .pkl.gz file."""
    path = Path(filepath)
    opener = gzip.open if path.name.endswith(".pkl.gz") else open
    with opener(filepath, "rb") as f:
        combined_data = pickle.load(f)
    return combined_data["results"], combined_data["metadata"]


def load_config_override(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path) as f:
        return yaml.safe_load(f)


def find_ska_yaml_file(directory: Path) -> Path:
    """Find the single .ska.yaml file in the directory."""
    ska_files = list(directory.glob("*.ska.yaml"))
    if len(ska_files) == 0:
        raise FileNotFoundError(f"No .ska.yaml files found in {directory}. Please provide a config override file.")
    elif len(ska_files) > 1:
        raise ValueError(f"Multiple .ska.yaml files found in {directory}: {[f.name for f in ska_files]}. Please only keep the one to use.")
    return ska_files[0]


def apply_config_override(config: dict, results_dir: Path, enable_override: bool) -> dict:
    """Return config, optionally overridden from a single .ska.yaml in results_dir.

    If enable_override is False, returns config unchanged.
    Raises only when multiple .ska.yaml files exist.
    """
    if not enable_override:
        return config

    try:
        yaml_file = find_ska_yaml_file(results_dir)
    except FileNotFoundError:
        logging.warning(f"No .ska.yaml files found in {results_dir}. Skipping override.")
        return config

    if not yaml_file.exists():
        logging.warning(f"Config override file not found: {yaml_file}")
        return config

    try:
        yaml_config = load_config_override(str(yaml_file))
        insert_blank_line()
        logging.info(f"Overriding config with: {yaml_file.name}")

        # Replace the entire config sections with YAML content:
        for yaml_section, config_section in yaml_to_config_mapping.items():
            if yaml_section in yaml_config:
                config[config_section] = yaml_config[yaml_section]
                logging.info(f"• Replaced {config_section} with {yaml_section}")
            elif config_section in yaml_config:
                # Also support direct config section names in YAML:
                config[config_section] = yaml_config[config_section]
                logging.info(f"• Replaced {config_section}")
    except Exception as e:
        logging.warning(f"Failed to load config override {yaml_file}: {e}")

    return config


def create_run_context_from_saved(config: dict, metadata: dict, output_dir: str = None) -> RunContext:
    """Create a minimal run context from saved configuration."""

    class SavedRunContext:
        def __init__(self, saved_config: dict, metadata: dict, output_dir: str) -> None:
            """Initialize the run context from a saved configuration."""
            self.RN_CONFIG = saved_config["RN_CONFIG"]
            self.NN_CONFIG = saved_config["NN_CONFIG"]
            self.VAL_CONFIG = saved_config["VAL_CONFIG"]
            self.PLT_CONFIG = saved_config["PLT_CONFIG"]
            self.timestamp = metadata.get("timestamp", "replotted")

            if output_dir:
                self.results_subdir = Path(output_dir)
            else:
                self.results_subdir = Path.cwd() / "replotted_results"
            self.results_subdir.mkdir(exist_ok=True)

    return SavedRunContext(config, metadata, output_dir)


def run_plotting_from_saved_results(results_file: str, output_dir: str = None, config_override_file: str = None) -> dict:
    """Run plotting routines using saved analysis results."""
    # Load results and metadata from the combined pickle file:
    results, metadata = load_analysis_results(results_file)

    if "config" not in metadata:
        raise KeyError(f"Missing 'config' in metadata of {results_file}")
    if "test_state_index" not in metadata:
        raise KeyError(f"Missing 'test_state_index' in metadata of {results_file}")

    config = metadata["config"].copy()

    # Apply YAML override if requested:
    if config_override_file:
        # Auto-detect .ska.yaml in the same directory as results file:
        results_dir = Path(results_file).parent
        config = apply_config_override(config, results_dir, enable_override=True)

    # Create run context from saved config and metadata:
    run = create_run_context_from_saved(config, metadata, output_dir)

    test_state_index = metadata["test_state_index"]
    test_state = results["test_state"]
    valid_data = results["valid_data"]

    convergence_state = True if any(np.array_equal(test_state, np.array(sub)) for sub in run.VAL_CONFIG["conv_test_states"]) else False

    insert_blank_line()
    logging.info(f"Regenerating plots for test state {test_state_index} ...")
    insert_blank_line()

    test_state_figures = {}

    if "ert" in results:
        logging.info("Regenerating temporal expectation plots ...")

        # Basic temporal expectation plots:
        expectation_plots.plot_temporal_trajectories(test_state, valid_data, run, test_state_index, results["ert"], ["SSA"], test_state_figures)
        expectation_plots.plot_temporal_expectation(test_state, valid_data, run, test_state_index, results["ert"], ["SSA", "NN"], test_state_figures)
        expectation_plots.plot_temporal_expectation(test_state, valid_data, run, test_state_index, results["ert"], ["SSAtol", "NN"], test_state_figures)

        # DeepCV estimates plots if available:
        if results["ert"].get("diff_mean") is not None:
            expectation_plots.plot_temporal_expectation(test_state, valid_data, run, test_state_index, results["ert"], ["SSA", "SSA+DeepCV"], test_state_figures)
            expectation_plots.plot_temporal_expectation(test_state, valid_data, run, test_state_index, results["ert"], ["SSAtol", "SSA+DeepCV"], test_state_figures)
            expectation_plots.plot_temporal_expectation_variance(test_state, valid_data, run, test_state_index, results["ert"], ["SSA", "SSA+DeepCV"], test_state_figures)

        # Output-suboptimal DeepCV estimates plots if available:
        if results["ert"].get("diff_out_suboptimal_mean") is not None:
            expectation_plots.plot_temporal_expectation(test_state, valid_data, run, test_state_index, results["ert"], ["SSA", "SSA+osubDeepCV"], test_state_figures)
            expectation_plots.plot_temporal_expectation(test_state, valid_data, run, test_state_index, results["ert"], ["SSAtol", "SSA+osubDeepCV"], test_state_figures)
            expectation_plots.plot_temporal_expectation_variance(test_state, valid_data, run, test_state_index, results["ert"], ["SSA", "SSA+osubDeepCV"], test_state_figures)

        # Time-suboptimal DeepCV estimates plots if available:
        if results["ert"].get("diff_time_suboptimal_mean") is not None:
            expectation_plots.plot_temporal_trajectories(test_state, valid_data, run, test_state_index, results["ert"], ["SSA+tsubDeepCV"], test_state_figures)
            expectation_plots.plot_temporal_trajectories(test_state, valid_data, run, test_state_index, results["ert"], ["SSA", "SSA+tsubDeepCV"], test_state_figures)
            expectation_plots.plot_temporal_expectation(test_state, valid_data, run, test_state_index, results["ert"], ["SSA", "SSA+tsubDeepCV"], test_state_figures)
            expectation_plots.plot_temporal_expectation(test_state, valid_data, run, test_state_index, results["ert"], ["SSAtol", "SSA+tsubDeepCV"], test_state_figures)
            expectation_plots.plot_temporal_expectation_variance(test_state, valid_data, run, test_state_index, results["ert"], ["SSA", "SSA+tsubDeepCV"], test_state_figures)

        # DeepIS estimates plots if available:
        if results["ert"].get("is_raw_mean") is not None:
            expectation_plots.plot_temporal_expectation(test_state, valid_data, run, test_state_index, results["ert"], ["SSA", "SSA+DeepIS"], test_state_figures)
            expectation_plots.plot_temporal_expectation(test_state, valid_data, run, test_state_index, results["ert"], ["SSAtol", "SSA+DeepIS"], test_state_figures)
            expectation_plots.plot_temporal_expectation_variance(test_state, valid_data, run, test_state_index, results["ert"], ["SSA", "SSA+DeepIS"], test_state_figures)

        # Output-suboptimal DeepIS estimates plots if available:
        if results["ert"].get("is_raw_out_suboptimal_mean") is not None:
            expectation_plots.plot_temporal_expectation(test_state, valid_data, run, test_state_index, results["ert"], ["SSA", "SSA+osubDeepIS"], test_state_figures)
            expectation_plots.plot_temporal_expectation(test_state, valid_data, run, test_state_index, results["ert"], ["SSAtol", "SSA+osubDeepIS"], test_state_figures)
            expectation_plots.plot_temporal_expectation_variance(test_state, valid_data, run, test_state_index, results["ert"], ["SSA", "SSA+osubDeepIS"], test_state_figures)

        # Time-suboptimal DeepIS estimates plots if available:
        if results["ert"].get("is_raw_time_suboptimal_mean") is not None:
            expectation_plots.plot_temporal_trajectories(test_state, valid_data, run, test_state_index, results["ert"], ["SSA+tsubDeepIS"], test_state_figures)
            expectation_plots.plot_temporal_trajectories(test_state, valid_data, run, test_state_index, results["ert"], ["SSA", "SSA+tsubDeepIS"], test_state_figures)
            expectation_plots.plot_temporal_expectation(test_state, valid_data, run, test_state_index, results["ert"], ["SSA", "SSA+tsubDeepIS"], test_state_figures)
            expectation_plots.plot_temporal_expectation(test_state, valid_data, run, test_state_index, results["ert"], ["SSAtol", "SSA+tsubDeepIS"], test_state_figures)
            expectation_plots.plot_temporal_expectation_variance(test_state, valid_data, run, test_state_index, results["ert"], ["SSA", "SSA+tsubDeepIS"], test_state_figures)

        # Combined DeepCV and DeepIS plots if both available:
        if (results["ert"].get("diff_mean") is not None and
            results["ert"].get("is_raw_mean") is not None):
            expectation_plots.plot_temporal_trajectories(test_state, valid_data, run, test_state_index, results["ert"], ["SSA", "SSA+tsubDeepCV", "SSA+tsubDeepIS"], test_state_figures)
            expectation_plots.plot_temporal_expectation(test_state, valid_data, run, test_state_index, results["ert"], ["SSA", "SSA+DeepCV", "SSA+DeepIS"], test_state_figures)
            expectation_plots.plot_temporal_expectation(test_state, valid_data, run, test_state_index, results["ert"], ["SSAtol", "SSA+DeepCV", "SSA+DeepIS"], test_state_figures)
            expectation_plots.plot_temporal_expectation(test_state, valid_data, run, test_state_index, results["ert"], ["SSA+DeepCV", "SSA+DeepIS"], test_state_figures)
            expectation_plots.plot_temporal_expectation_variance(test_state, valid_data, run, test_state_index, results["ert"], ["SSA", "SSA+DeepCV", "SSA+DeepIS"], test_state_figures)
            expectation_plots.plot_temporal_expectation_variance(test_state, valid_data, run, test_state_index, results["ert"], ["SSA+DeepCV", "SSA+DeepIS"], test_state_figures)

        # Approximation variance plots if available:
        if results["ert"].get("ter_mean") is not None:
            expectation_plots.plot_temporal_expectation_variance(test_state, valid_data, run, test_state_index, results["ert"], ["SSA", "DeepPVA"], test_state_figures)
        if results["ert"].get("int_mean") is not None:
            expectation_plots.plot_temporal_expectation_variance(test_state, valid_data, run, test_state_index, results["ert"], ["SSA", "DeepIPA"], test_state_figures)

        # Convergence plots:
        if convergence_state and run.PLT_CONFIG.get("plot_estimate_convergence", False):
            # Note: Convergence plots in main.py use "erf" (final results) and different parameters
            # These would need convergence data which may not be available in saved results
            expectation_plots.plot_expectation_error_convergence(test_state, run, test_state_index, results["erf"], ["SSA", "SSA+DeepCV", "SSA+DeepIS"], test_state_figures)
            expectation_plots.plot_expectation_error_convergence_repeats(test_state, run, test_state_index, results["erf"], results["ssa_results"], results["cv_results"], results["is_results"], ["SSA", "SSA+DeepCV", "SSA+DeepIS"], test_state_figures)

        # Column plots (combining multiple output functions):
        panels.plot_temporal_column(test_state_figures, run, test_state_index)

    # Spectral decomposition plots if available:
    if "sd_dict" in results and results["sd_dict"]:
        logging.info("Regenerating spectral decomposition plots ...")
        sd_dict = results["sd_dict"]

        # Basic spectral plots that don't need the model:
        spectral_plots.plot_temporal_eigenfunctions(test_state, valid_data, run, test_state_index, sd_dict)

        # Ergodic mean plots if available:
        if sd_dict.get("raw_em_mean") is not None:
            expectation_plots.plot_ergodic_means(test_state, run, test_state_index, sd_dict)
            expectation_plots.plot_ergodic_means_variance(test_state, run, test_state_index, sd_dict)

    return test_state_figures


def process_all_test_states(results_dir: str, output_dir: str = None, config_override_file: str = None) -> None:
    """Process all test states in a results directory."""
    results_path = Path(results_dir)

    # Find all results files in the directory:
    results_files = list(results_path.glob("*___results.pkl.gz"))
    results_files.sort()  # sort to process in order

    if not results_files:
        logging.info(f"No results files found in {results_dir}.")
        return

    insert_blank_line()
    logging.info(f"Found {len(results_files)} test states to process:")
    for f in results_files:
        logging.info(f"• {f.name}")

    # Initialize combined figures dict for panel plots:
    all_test_state_figures = {}

    for results_file in results_files:
        insert_section_marker()
        logging.info(f"Processing {results_file.name} ...")
        try:
            # Process each test state and collect figures:
            test_state_figures = run_plotting_from_saved_results(str(results_file), output_dir, config_override_file)
            all_test_state_figures.update(test_state_figures)
        except Exception as e:
            logging.info(f"Error processing {results_file.name}: {e}")
            continue

    # Create combined panel plots if we have figures from multiple test states:
    if len(all_test_state_figures) > 0:
        try:
            # Create a mock run context for panel plotting:
            _, first_metadata = load_analysis_results(str(results_files[0]))

            config = first_metadata["config"].copy()

            # Handle YAML config override file for panel plots:
            if config_override_file:
                config = apply_config_override(config, results_path, enable_override=True)

            run = create_run_context_from_saved(config, first_metadata, output_dir)

            insert_section_marker()
            logging.info("Creating combined panel plots ...")
            panels.plot_temporal_panel(all_test_state_figures, run)

        except Exception as e:
            logging.warning(f"Could not create combined panel plots: {e}")


def main(args: argparse.Namespace) -> None:
    """Main function for plotting from saved results."""
    # Determine default output directory if not specified:
    input_path = Path(args.results_file)
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        if input_path.is_file():
            # For a single file, use the parent folder and prefix with "replotted_":
            output_dir = input_path.parent / f"replotted_{input_path.stem}"
        else:
            # For a directory, create "replotted_results" subfolder:
            output_dir = input_path / "replotted_results"

    output_dir.mkdir(parents=True, exist_ok=True)

    insert_section_marker()
    logging.info("Starting replotting routine ...")

    # Check if we should process all test states or just one:
    if input_path.is_dir():
        # Treat results_file as directory path:
        process_all_test_states(args.results_file, output_dir, args.config_override)
    elif input_path.is_file():
        # Process single file:
        run_plotting_from_saved_results(args.results_file, output_dir, args.config_override)

    insert_section_marker()
    logging.info("Replotting routine completed successfully.")
    insert_section_marker()


def cli() -> None:
    """Command-line entrypoint for the replotting script."""
    # Set up the command line argument parser:
    parser = argparse.ArgumentParser(
        usage="%(prog)s <file_or_directory> [--output_dir <output_dir>] [--config_override]",
        description="Generate plots from saved analysis results.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "results_file",
        type=str,
        help="path to saved analysis results (.pkl.gz) file or directory containing multiple results files"
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        help="directory for output plots (default: <input_dir>/replotted_results)"
    )

    parser.add_argument(
        "--config_override",
        action="store_true",
        help="auto-detects .ska.yaml file in results directory to override config saved in the pickle file"
    )

    # Run the main function:
    main(parser.parse_args())


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    cli()
