import os
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from typing import Tuple

from sklearn.datasets import fetch_openml
import torch

from data_loader.base import DataLoader
from modeling.structure.factory import ModelingStructureFactory
from utils.config_handle import load_config
from utils.experiment_io import get_run_id


class IonosphereKCMkGHLoader(DataLoader):
    def load_raw(self) -> Tuple[np.ndarray, pd.DataFrame]:
        # Load dataset from OpenML
        dataset = fetch_openml(name="ionosphere", version=1, as_frame=True)

        # Features
        X = dataset.data.to_numpy(dtype=np.float32)

        # Labels
        y = dataset.target

        # Convert labels to binary integers if needed
        # 'g' = good, 'b' = bad
        y = y.map({"g": 1, "b": 0})
        y = y.to_frame(name="label")

        return X, y

    def __create_dataset(self):
        X, y = self.load_raw()

        # pre_processing:
        stds = X.std(axis=0)
        valid_features = stds > 1e-8
        X = X[:, valid_features]

        # Modelling
        config = load_config(default_file_name="KCM_K_GH")
        run_id = get_run_id(
            config,
            [config["modeling"]["structure"]["name"], config["data_loader"]["name"]],
        )
        config["run_id"] = run_id
        model = ModelingStructureFactory().get(config, self.logger, self.device)
        model.compile()

        model.fit(X, y, verbose=False)
        labels = model.predict_proba(X)

        return X, labels

    def load(self) -> Tuple[np.ndarray, pd.DataFrame]:
        path = self.get_output_path()
        if path.exists():
            return self.load_cash(path)

        X, labels = self.__create_dataset()

        y = pd.DataFrame({"label": labels.cpu().numpy()})
        y = y["label"].to_frame(name="label")
        y["label"] = y["label"].astype("category")

        self.save_cash(X, y, path)

        return X, y

    def save_cash(self, X: np.ndarray, y: pd.DataFrame, path):
        self.logger.info("Saving data in cash file...")
        torch.save({"X": X, "y": y}, path, pickle_protocol=pickle.HIGHEST_PROTOCOL)
        self.logger.info(f"Data saved to {path}")

    def load_cash(self, path: str) -> tuple[np.ndarray, pd.DataFrame]:
        self.logger.info(f"Loading cached data from: {path}")
        cache = torch.load(path, weights_only=False)
        X, y = cache["X"], cache["y"]
        return X, y

    def get_output_path(self) -> str:
        """
        Get the output path for the processed data.
        """
        base_path = "data"
        dataset_name = self.config.get("data_loader", {}).get("name")

        os.makedirs(base_path, exist_ok=True)
        return Path(f"{base_path}/{dataset_name}.pt")
