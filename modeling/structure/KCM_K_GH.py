import torch
import pandas as pd
import numpy as np
from sklearn.metrics import adjusted_rand_score 

from torch import device as TorchDevice
from torch.utils.data import TensorDataset, DataLoader

from tracker.base_tracker import BaseTracker
from utils.experiment_io import get_run_dir

class KCM_K_GH:
    def __init__(self, config, logger, device: TorchDevice):
        """
        Variante KCM-K-GH do algoritmo KCM-K-H.
        :param n_clusters: Número de clusters (c)
        :param max_iter: Número máximo de iterações
        :param tol: Tolerância para critério de parada (opcional)
        :param device: 'cuda' para rodar na placa de vídeo, ou 'cpu'
        """
        self.config = config
        self.logger = logger
        self.device = device

        self.phase = self.config.get("phase")
        self.model_dir = (
            get_run_dir(self.config.get("run_id")) / f"{self.phase}_model.pt"
        )
        self.c = self.config.get('modeling',{}).get('structure',{}).get('n_clusters')
        self.max_iter = self.config.get('modeling',{}).get('structure',{}).get('max_iter')
        
        self.G = None       # Protótipos dos clusters (c, p)
        self.inv_s_sq = None # 1 / s_j^2 (vetor global de hiperparâmetros) (p,)
        self.labels = None  # Partição atual (P)

    def save(self):
        torch.save(
            {
                "G": self.G,
                "inv_s_sq": self.inv_s_sq,
                "labels": self.labels,
            },
            self.model_dir,
        )

    def load(self):
        checkpoint = torch.load(self.model_dir, map_location=self.device)

        self.G = checkpoint["G"]
        self.inv_s_sq = checkpoint["inv_s_sq"]
        self.labels = checkpoint["labels"]


    def reset_weights(self):
        self.G = None       # Protótipos dos clusters (c, p)
        self.inv_s_sq = None # 1 / s_j^2 (vetor global de hiperparâmetros) (p,)
        self.labels = None  # Partição atual (P)

    def compile(self):
        pass

    def _compute_kernel(self, X, G, inv_s_sq):
        """
        Calcula o Kernel Gaussiano (Equação 9) de forma vetorizada.
        K(x_k, g_i) = exp( -0.5 * sum( (1/s_j^2) * (x_kj - g_ij)^2 ) )
        Retorna matriz (n, c)
        """
        # X: (n, 1, p) | G: (1, c, p) -> diff: (n, c, p)
        diff = X.unsqueeze(1) - G.unsqueeze(0)
        
        # D: (n, c)
        D = torch.sum((diff ** 2) * inv_s_sq, dim=2)
        return torch.exp(-0.5 * D)

    def fit(self, X, y):
        """
        Treina o modelo e retorna as labels dos clusters.
        :param X: Tensor PyTorch ou Array NumPy com os dados (n, p)
        """
        if not isinstance(X, torch.Tensor):
            X = torch.tensor(X, dtype=torch.float32)
        X = X.to(self.device)
        
        n, p = X.shape
        
        indices = torch.randperm(n)[:self.c]
        self.G = X[indices].clone()
        # Etapa 2 instrui gamma = 1. Na inicialização: 1/s_j^2 = (gamma)^(1/p) = 1
        self.inv_s_sq = torch.ones(p, dtype=X.dtype, device=self.device)
        
        K = self._compute_kernel(X, self.G, self.inv_s_sq)
        self.labels = torch.argmax(K, dim=1)
        
        for iteration in range(self.max_iter):
            # ========================================================
            # ETAPA 1: Representação (Equação 14)
            # ========================================================
            G_new = torch.zeros_like(self.G)
            for i in range(self.c):
                mask = (self.labels == i)
            
                X_i = X[mask]
                K_i = K[mask, i].unsqueeze(1) # (n_i, 1)
                print(K_i)
                # Numerador e Denominador da Eq (14)
                num = torch.sum(K_i * X_i, dim=0)
                den = torch.sum(K_i)


                G_new[i] = num / den
                
            self.G = G_new

            # ========================================================
            # ETAPA 2: Hiperparâmetros de largura (Equação 16 com gamma=1)
            # ========================================================
            # Atualiza o Kernel com o novo G (P e S são mantidos fixos)
            K = self._compute_kernel(X, self.G, self.inv_s_sq)
            
            E = torch.zeros(p, device=self.device)
            for i in range(self.c):
                #mask = (self.labels == i)
                X_i = X[mask]
                K_i = K[mask, i].unsqueeze(1)
                diff_sq = (X_i - self.G[i]) ** 2
                
                # Somatório duplo: interno ao cluster e entre os clusters
                E += torch.sum(K_i * diff_sq, dim=0)
                
            
            prod_term = torch.exp(torch.mean(torch.log(E)))
            
            # Eq (16) final considerando gamma=1
            self.inv_s_sq = prod_term / E
            #print(self.inv_s_sq)
            #rint(E)

            # ========================================================
            # ETAPA 3: Alocação (Equação 18)
            # ========================================================
            # Atualiza Kernel com o novo S
            K = self._compute_kernel(X, self.G, self.inv_s_sq)
            
            # Regra de alocação: min 2(1-K) -> max K
            labels_new = torch.argmax(K, dim=1)
            
            # Checagem de convergência
            if torch.all(self.labels == labels_new):
                print(f"Convergiu na iteração {iteration}.")
                break
                
            self.labels = labels_new
            
        return self.labels.cpu().numpy()
    
    def predict(self, X):
        predict = self.fit(X)
        return predict