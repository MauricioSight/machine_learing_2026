import torch
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold

class ParzenWindowBayesian:
    def __init__(self, h=1.0, device='cuda'):
        self.h = h
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.X_train_by_class = {}
        self.priors = {}
        self.classes = None

    def fit(self, X, y):
        X = X.to(self.device)
        y = y.to(self.device)
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
        product_kernel = torch.prod(kernels, dim=2) # [M, Ni]
        
        # Média sobre as Ni amostras da classe
        return torch.mean(product_kernel, dim=1) # [M]

    def predict(self, X_test):
        X_test = X_test.to(self.device)
        m_samples = X_test.shape[0]
        posteriors = torch.zeros((m_samples, len(self.classes)), device=self.device)

        for i, c in enumerate(self.classes):
            c_val = c.item()
            # p(x|wi) estimado pela Janela de Parzen
            likelihood = self._gaussian_kernel_product(X_test, self.X_train_by_class[c_val])
            # Numerador de Bayes: p(x|wi) * P(wi)
            posteriors[:, i] = likelihood * self.priors[c_val]

        return self.classes[torch.argmax(posteriors, dim=1)]
