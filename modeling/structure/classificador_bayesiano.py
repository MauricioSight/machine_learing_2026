import torch
import pandas as pd
import numpy as np

class BayesClassifier:
    def __init__(self, device='cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.means = None
        self.covs_inv = None
        self.priors = None
        self.log_det_cov = None
        self.classes = None

    def fit(self, X, y):

        X = X.to(self.device)
        y = y.to(self.device)
        
        n_samples, n_features = X.shape
        self.classes = torch.unique(y)
        num_classes = len(self.classes)
        
        self.means = torch.zeros((num_classes, n_features), device=self.device)
        self.covs_inv = torch.zeros((num_classes, n_features, n_features), device=self.device)
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

    def predict(self, X):
        X = X.to(self.device)
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
