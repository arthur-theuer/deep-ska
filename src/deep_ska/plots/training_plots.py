"""Training-related plotting routines."""

import matplotlib.pyplot as plt

from ..core.initialization import RunContext
from .common.helpers import color_dict


def plot_training_history(history: dict, run: RunContext) -> None:
    """Plot the training and validation loss over time and the individual components of the loss."""
    train_comp = history.get("train_comp", [])
    n_components = len(train_comp[0]) if train_comp else 0

    if n_components <= 1:
        fig, axs = plt.subplots(1, 2, figsize=(12, 4))
        ax1, ax3 = axs
        ax2 = None
    else:
        fig, axs = plt.subplots(1, 3, figsize=(18, 4))
        ax1, ax2, ax3 = axs

    if history["train_loss"] != []:
        # Plot the overall training and validation loss:
        ax1.plot(history["train_step"], history["train_loss"], color=color_dict["loss_expectation_light"], label="train loss", alpha=0.5)
        ax1.plot(history["valid_step"], history["train_mean"], color=color_dict["loss_expectation_dark"], label="train loss (mean)")
        ax1.plot(history["valid_step"], history["valid_loss"], color="black", label="valid loss")
        ax1.set_title("Training and validation loss")
        ax1.set_xlabel("steps")
        ax1.set_ylabel("loss")
        ax1.set_yscale("log")
        ax1.legend()

    if ax2 is not None:
        # Plot the individual components of training and validation loss:
        colors_train = ["red", "blue"]
        colors_valid = ["darkred", "darkblue"]
        for i in range(len(history["train_comp"][0])):  # plot the individual components of train and valid loss on the second axis
            ax2.plot(history["train_step"], [comp[i] for comp in history["train_comp"]], color=colors_train[i], label=f"train component {i+1}", alpha=0.5)
            ax2.plot(history["valid_step"], [comp[i] for comp in history["valid_comp"]], color=colors_valid[i], label=f"valid component {i+1}")
        ax2.set_title("Components of training and validation loss")
        ax2.set_xlabel("steps")
        ax2.set_ylabel("loss components")
        ax2.set_yscale("log")
        ax2.legend()

    if history["lr"] != []:
        # Plot the learning rate schedule:
        ax3.plot(history["train_step"], history["lr"], color="#BF00BF", label="learning rate")
        ax3.set_title("Change in learning rate")
        ax3.set_xlabel("steps")
        ax3.set_ylabel("learning rate")
        ax3.legend()

    fig.tight_layout()
    fig.savefig(f"{run.results_subdir}/{run.timestamp}___A_TrainingHistory.pdf")
    plt.close(fig)
