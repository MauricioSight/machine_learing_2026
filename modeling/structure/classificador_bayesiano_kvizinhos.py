import torch
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold

class KNNBayesian:
    def __init__(self, k=5, dist_metric='euclidean', device='cuda'):
        self.k = k
        self.dist_metric = dist_metric
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.X_train = None
        self.y_train = None
        self.classes = None

    def fit(self, X, y):
        self.X_train = X.to(self.device)
        self.y_train = y.to(self.device)
        self.classes = torch.unique(y)

    def _compute_distances(self, X_test):
        if self.dist_metric == 'euclidean':
            return torch.cdist(X_test, self.X_train, p=2)
        
        elif self.dist_metric == 'city_block':
            return torch.cdist(X_test, self.X_train, p=1)
        
        elif self.dist_metric == 'chebyshev':
            diff = X_test.unsqueeze(1) - self.X_train.unsqueeze(0) 
            return torch.max(torch.abs(diff), dim=2)[0]

    def predict(self, X_test):
        X_test = X_test.to(self.device)
        distances = self._compute_distances(X_test)
        
        # Busca os k vizinhos com menor distância
        _, indices = torch.topk(distances, self.k, largest=False, dim=1)
        neighbor_labels = self.y_train[indices]
        
        # Votação majoritária (Regra de decisão Bayesiana para KNN)
        preds = []
        for i in range(X_test.shape[0]):
            counts = torch.bincount(neighbor_labels[i], minlength=len(self.classes))
            preds.append(torch.argmax(counts))
            
        return torch.stack(preds)
