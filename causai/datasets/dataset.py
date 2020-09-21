"""Dataset class to be extended by dataset-specific classes."""
from pathlib import Path
import argparse
import os
import numpy as np
import pandas as pd

from causai import util


class Dataset:
    """Simple abstract class for datasets."""

    @classmethod
    def data_dirname(cls):
        return Path(__file__).resolve().parents[2] / "data"

    def load_or_generate_data(self):
        pass

# def _download_raw_dataset(metadata):
#     if os.path.exists(metadata["filename"]):
#         return
#     print(f"Downloading raw dataset from {metadata['url']}...")
#     util.download_url(metadata["url"], metadata["filename"])

def _parse_args():
    parser = argparse.ArgumentParser()
    # arguments
    return parser.parse_args()
