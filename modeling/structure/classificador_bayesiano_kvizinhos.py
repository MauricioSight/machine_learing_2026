import numpy as np
import torch

from torch import device as TorchDevice

from utils.experiment_io import get_run_dir


class KNNBayesian:
    def __init__(self, config, device=TorchDevice):
        self.device = device
        self.config = config

        self.phase = self.config.get("phase")
        self.model_dir = (
            get_run_dir(self.config.get("run_id")) / f"{self.phase}_model.pt"
        )

        self.k = config.get("modeling", {}).get("structure", {}).get("k")
        self.dist_metric = (
            config.get("modeling", {}).get("structure", {}).get("dist_metric")
        )

        self.X_train = None
        self.y_train = None
        self.classes = None

    def reset_weights(self):
        self.X_train = None
        self.y_train = None
        self.classes = None

    def compile(self):
        pass

    def save(
        self,
    ):
        torch.save(
            {
                "k": self.k,
                "dist_metric": self.dist_metric,
                "X_train": self.X_train,
                "y_train": self.y_train,
                "classes": self.classes,
            },
            self.model_dir,
        )

    def load(
        self,
    ):
        checkpoint = torch.load(self.model_dir, map_location=self.device)

        self.k = checkpoint["k"]
        self.dist_metric = checkpoint["dist_metric"]
        self.X_train = checkpoint["X_train"].to(self.device)
        self.y_train = checkpoint["y_train"].to(self.device)
        self.classes = checkpoint["classes"].to(self.device)

    def fit(self, X, y):
        self.X_train = torch.from_numpy(X).to(self.device)
        self.y_train = torch.tensor(y.codes).to(self.device)
        self.classes = torch.unique(self.y_train)

    def _compute_distances(self, X_test):
        if self.dist_metric == "euclidean":
            return torch.cdist(X_test, self.X_train, p=2)

        elif self.dist_metric == "city_block":
            return torch.cdist(X_test, self.X_train, p=1)

        elif self.dist_metric == "chebyshev":
            diff = X_test.unsqueeze(1) - self.X_train.unsqueeze(0)
            return torch.max(torch.abs(diff), dim=2)[0]

    def predict_proba(self, X: np.ndarray) -> torch.Tensor:
        X = torch.from_numpy(X).to(self.device)
        distances = self._compute_distances(X)

        # Busca os k vizinhos com menor distância
        _, indices = torch.topk(distances, self.k, largest=False, dim=1)
        neighbor_labels = self.y_train[indices]

        # Votação majoritária (Regra de decisão Bayesiana para KNN)
        preds = []
        for i in range(X.shape[0]):
            counts = torch.bincount(neighbor_labels[i], minlength=len(self.classes))
            preds.append(torch.argmax(counts))

        return torch.stack(preds)

    def predict(self, X: np.ndarray):
        X = torch.from_numpy(X).to(self.device)
        distances = self._compute_distances(X)

        # Busca os k vizinhos com menor distância
        _, indices = torch.topk(distances, self.k, largest=False, dim=1)
        neighbor_labels = self.y_train[indices]

        # Votação majoritária (Regra de decisão Bayesiana para KNN)
        preds = []
        for i in range(X.shape[0]):
            counts = torch.bincount(neighbor_labels[i], minlength=len(self.classes))
            preds.append(torch.argmax(counts))

        return torch.stack(preds)
