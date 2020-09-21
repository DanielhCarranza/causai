#!/usr/bin/env python
"""Script to run an experiment."""
import argparse
import json
import importlib
from typing import Dict
import os

from training.util import train_model

DEFAULT_TRAIN_ARGS = {"batch_size": 64, "epochs": 16}


def run_experiment(experiment_config: Dict, save_weights: bool, gpu_ind: int, use_experiment_manager: bool = True):
    """
    Run a training experiment.

    Parameters
    ----------
    experiment_config (dict)
        Of the form
        {
            "dataset": "Dataset",
            "dataset_args": {
                "data_arg": 0.4,
            },
            "model": "Model",
            "network": "neural_net",
            "network_args": {
                "hidden_size": 256,
            },
            "train_args": {
                "batch_size": 128,
                "epochs": 10
            }
        }
    save_weights (bool)
        If True, will save the final model weights to a canonical location (see Model in models/base.py)
    gpu_ind (int)
        specifies which gpu to use 
    use_experiment_manager (bool)
        sync training run to wandb, tensorboard, etc.
    """
    print(f"Running experiment with config {experiment_config} on GPU {gpu_ind}")

def _parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--save",
        default=False,
        dest="save",
        action="store_true",
        help="If true, then final weights will be saved to canonical, version-controlled location",
    )
    parser.add_argument(
        "experiment_config",
        type=str,
        help='Experimenet JSON (\'{"dataset": "Dataset", "model": "Model", "network": "mlp"}\'',
    )
    args = parser.parse_args()
    return args


def main():
    """Run experiment."""
    args = _parse_args()
    experiment_config = json.loads(args.experiment_config)
    os.environ["CUDA_VISIBLE_DEVICES"] = f"{args.gpu}"
    run_experiment(experiment_config, args.save, args.gpu)

if __name__ == "__main__":
    main()
