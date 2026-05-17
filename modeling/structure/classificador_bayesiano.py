import numpy as np
import torch

from torch import device as TorchDevice

from utils.experiment_io import get_run_dir


class BayesClassifier:
    def __init__(self, config, device: TorchDevice):
        self.device = device
        self.config = config

        self.phase = self.config.get("phase")
        model_name = self.config.get("modeling", {}).get("structure", {}).get("name")
        self.model_dir = (
            get_run_dir(self.config.get("run_id")) / f"{self.phase}_{model_name}.pt"
        )

        self.means = None
        self.covs_inv = None
        self.priors = None
        self.log_det_cov = None
        self.classes = None

    def reset_weights(self):
        self.means = None
        self.covs_inv = None
        self.priors = None
        self.log_det_cov = None
        self.classes = None

    def compile(self):
        pass

    def save(self):
        torch.save(
            {
                "means": self.means,
                "covs_inv": self.covs_inv,
                "priors": self.priors,
                "log_det_cov": self.log_det_cov,
                "classes": self.classes,
            },
            self.model_dir,
        )

    def load(self):
        checkpoint = torch.load(self.model_dir, map_location=self.device)

        self.means = checkpoint["means"].to(self.device)
        self.covs_inv = checkpoint["covs_inv"].to(self.device)
        self.priors = checkpoint["priors"].to(self.device)
        self.log_det_cov = checkpoint["log_det_cov"].to(self.device)
        self.classes = checkpoint["classes"].to(self.device)

    def fit(
        self,
        X,
        y,
        X_val: np.ndarray = None,
        y_val=None,
        verbose=True,
    ):
        X = torch.from_numpy(X).to(self.device)
        y = torch.tensor(y.codes).to(self.device)

        n_samples, n_features = X.shape
        self.classes = torch.unique(y)
        num_classes = len(self.classes)

        self.means = torch.zeros((num_classes, n_features), device=self.device)
        self.covs_inv = torch.zeros(
            (num_classes, n_features, n_features), device=self.device
        )
        self.log_det_cov = torch.zeros(num_classes, device=self.device)
        self.priors = torch.zeros(num_classes, device=self.device)

        for i, c in enumerate(self.classes):
            X_c = X[y == c]

            # a) Estimativa de máxima verossimilhança para P(wi)
            self.priors[i] = X_c.shape[0] / n_samples

            # b) Estimativa de MV para média (mu_i)
            self.means[i] = torch.mean(X_c, dim=0)

            # b) Estimativa de MV para covariância (Sigma_i)
            diff = X_c - self.means[i]
            cov = (diff.T @ diff) / X_c.shape[0]

            # Regularização para evitar matriz singular
            cov += torch.eye(n_features, device=self.device) * 1e-5

            self.covs_inv[i] = torch.inverse(cov)
            self.log_det_cov[i] = torch.logdet(cov)

    def predict_proba(self, X: np.ndarray) -> torch.Tensor:
        X = torch.from_numpy(X).to(self.device)
        n_samples = X.shape[0]
        log_posteriors = torch.zeros((n_samples, len(self.classes)), device=self.device)

        for i in range(len(self.classes)):
            diff = X - self.means[i]
            # Cálculo vetorizado da distância de Mahalanobis na GPU
            exponent = -0.5 * torch.sum((diff @ self.covs_inv[i]) * diff, dim=1)

            # Log-verossimilhança + Log-Prior
            log_likelihood = exponent - 0.5 * self.log_det_cov[i]
            log_posteriors[:, i] = log_likelihood + torch.log(self.priors[i])

        return self.classes[torch.argmax(log_posteriors, dim=1)]

    def predict(self, X: np.ndarray):
        X = torch.from_numpy(X).to(self.device)
        n_samples = X.shape[0]
        log_posteriors = torch.zeros((n_samples, len(self.classes)), device=self.device)

        for i in range(len(self.classes)):
            diff = X - self.means[i]
            # Cálculo vetorizado da distância de Mahalanobis na GPU
            exponent = -0.5 * torch.sum((diff @ self.covs_inv[i]) * diff, dim=1)

            # Log-verossimilhança + Log-Prior
            log_likelihood = exponent - 0.5 * self.log_det_cov[i]
            log_posteriors[:, i] = log_likelihood + torch.log(self.priors[i])

        return self.classes[torch.argmax(log_posteriors, dim=1)]
