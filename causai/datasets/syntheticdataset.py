from pathlib import Path
import argparse
import os
import numpy as np
import pandas as pd

from causai.datasets.dataset import Dataset

class SyntheticDataset(Dataset):
    @classmethod
    def generate_data(cls, n=1000, seed=0, beta1=1.05, alpha1=0.4, alpha2=0.3, binary_treatment=True, binary_cutoff=3.5):
        np.random.seed(seed)
        age = np.random.normal(65, 5, n)
        sodium = age / 18 + np.random.normal(size=n)
        if binary_treatment:
            if binary_cutoff is None:
                binary_cutoff = sodium.mean()
            sodium = (sodium > binary_cutoff).astype(int)
        blood_pressure = beta1 * sodium + 2 * age + np.random.normal(size=n)
        proteinuria = alpha1 * sodium + alpha2 * blood_pressure + np.random.normal(size=n)
        hypertension = (blood_pressure >= 140).astype(int)  # not used, but could be used for binary outcomes
        return pd.DataFrame({'blood_pressure': blood_pressure, 'sodium': sodium,
                            'age': age, 'proteinuria': proteinuria})

