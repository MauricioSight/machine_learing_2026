import torch
import numpy as np

from torch import device as TorchDevice

from utils.experiment_io import get_run_dir


class ParzenWindowBayesian:
    def __init__(self, config, device=TorchDevice):
        self.device = device
        self.config = config

        self.phase = self.config.get("phase")
        self.model_dir = (
            get_run_dir(self.config.get("run_id")) / f"{self.phase}_model.pt"
        )

        self.h = config.get("modeling", {}).get("structure", {}).get("h")

        self.X_train_by_class = {}
        self.priors = {}
        self.classes = None

    def reset_weights(self):
        self.X_train_by_class = {}
        self.priors = {}
        self.classes = None

    def compile(self):
        pass

    def save(self):
        torch.save(
            {
                "h": self.h,
                "X_train_by_class": self.X_train_by_class,
                "priors": self.priors,
                "classes": self.classes,
            },
            self.model_dir,
        )

    def load(self):
        checkpoint = torch.load(self.model_dir, map_location=self.device)

        self.h = checkpoint["h"]

        self.X_train_by_class = {
            k: v.to(self.device) for k, v in checkpoint["X_train_by_class"].items()
        }
        self.priors = checkpoint["priors"]
        self.classes = checkpoint["classes"].to(self.device)

    def fit(self, X, y):
        X = torch.from_numpy(X).to(self.device)
        y = torch.tensor(y.codes).to(self.device)

        self.classes = torch.unique(y)
        n_total = X.shape[0]

        for c in self.classes:
            X_c = X[y == c]
            self.X_train_by_class[c.item()] = X_c
            # Prior P(wi)
            self.priors[c.item()] = X_c.shape[0] / n_total

    def _gaussian_kernel_product(self, X_test, X_class):
        # X_test: [M, D], X_class: [Ni, D]
        # Expandindo dimensões para broadcast: [M, 1, D] e [1, Ni, D]
        diff = (X_test.unsqueeze(1) - X_class.unsqueeze(0)) / self.h

        # Kernel Gaussiano Univariado: (1/sqrt(2pi)) * exp(-0.5 * u^2)
        constants = 1.0 / (np.sqrt(2 * np.pi) * self.h)
        kernels = constants * torch.exp(-0.5 * diff**2)

        # Produto dos kernels nas D dimensões (Kernel Multivariado Produto)
        product_kernel = torch.prod(kernels, dim=2)  # [M, Ni]

        # Média sobre as Ni amostras da classe
        return torch.mean(product_kernel, dim=1)  # [M]

    def predict_proba(self, X: np.ndarray) -> torch.Tensor:
        X = torch.from_numpy(X).to(self.device)
        m_samples = X.shape[0]
        posteriors = torch.zeros((m_samples, len(self.classes)), device=self.device)

        for i, c in enumerate(self.classes):
            c_val = c.item()
            # p(x|wi) estimado pela Janela de Parzen
            likelihood = self._gaussian_kernel_product(X, self.X_train_by_class[c_val])
            # Numerador de Bayes: p(x|wi) * P(wi)
            posteriors[:, i] = likelihood * self.priors[c_val]

        return self.classes[torch.argmax(posteriors, dim=1)]

    def predict(self, X: np.ndarray):
        X = torch.from_numpy(X).to(self.device)
        m_samples = X.shape[0]
        posteriors = torch.zeros((m_samples, len(self.classes)), device=self.device)

        for i, c in enumerate(self.classes):
            c_val = c.item()
            # p(x|wi) estimado pela Janela de Parzen
            likelihood = self._gaussian_kernel_product(X, self.X_train_by_class[c_val])
            # Numerador de Bayes: p(x|wi) * P(wi)
            posteriors[:, i] = likelihood * self.priors[c_val]

        return self.classes[torch.argmax(posteriors, dim=1)]
