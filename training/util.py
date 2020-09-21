"""Function to train a model."""
from time import time

from causai.datasets.dataset import Dataset
from causai.models.base import Model

def train_model(model: Model, dataset: Dataset, epochs: int, batch_size: int, use_experiment_manager: bool = False) -> Model:
    """Write your Function to Train model. """
    return model
