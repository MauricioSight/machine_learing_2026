import numpy as np
import pandas as pd
from typing import Tuple

from sklearn.datasets import fetch_openml

from data_loader.base import DataLoader


class IonosphereLoader(DataLoader):
    def load(self) -> Tuple[np.ndarray, pd.DataFrame]:
        # Load dataset from OpenML
        dataset = fetch_openml(name="ionosphere", version=1, as_frame=True)

        # Features
        X = dataset.data.to_numpy(dtype=np.float32)

        # Labels
        y = dataset.target

        # Convert labels to binary integers if needed
        # 'g' = good, 'b' = bad
        y = y.map({"g": 1, "b": 0})
        y = y.to_frame(name='label')

        return X, y
