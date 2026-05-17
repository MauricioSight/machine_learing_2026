import torch

from torch import device as TorchDevice
from utils.experiment_io import get_run_dir


class KCM_K_GH:
    def __init__(self, config, logger, device=TorchDevice):

        self.config = config
        self.logger = logger
        self.device = device

        self.phase = self.config.get("phase")
        self.model_dir = (
            get_run_dir(self.config.get("run_id")) / f"{self.phase}_model.pt"
        )

        self.c = self.config.get("modeling", {}).get("structure", {}).get("n_clusters")
        self.gamma = self.config.get("modeling", {}).get("structure", {}).get("gamma")
        self.max_iter = (
            self.config.get("modeling", {}).get("structure", {}).get("max_iter")
        )

        self.device = device

        self.prototypes = None
        self.s2 = None
        self.labels = None

        self.objective_value = float("inf")
        self.loss_history = []
        self.last_valid_D_j = None

    def save(self):
        torch.save(
            {
                "prototypes": self.prototypes,
                "s2": self.s2,
                "labels": self.labels,
            },
            self.model_dir,
        )

    def load(self):
        checkpoint = torch.load(self.model_dir, map_location=self.device)

        self.G = checkpoint["prototypes"]
        self.inv_s_sq = checkpoint["s2"]
        self.labels = checkpoint["labels"]

    def reset_weights(self):
        self.prototypes = None
        self.s2 = None
        self.labels = None

    def compile(self):
        pass

    def _gaussian_kernel(self, X, G):
        squared_diff = (X.unsqueeze(1) - G.unsqueeze(0)) ** 2
        weighted_diff = squared_diff / self.s2.unsqueeze(0).unsqueeze(0)
        sum_weighted = torch.sum(weighted_diff, dim=-1)
        return torch.exp(-0.5 * sum_weighted)

    def compute_objective(self, X, labels):
        with torch.no_grad():
            K = self._gaussian_kernel(X, self.prototypes)
            K_assigned = K[torch.arange(X.shape[0]), labels]
            return torch.sum(2 * (1.0 - K_assigned)).item()

    def fit(self, X, y, verbose=True):
        if not isinstance(X, torch.Tensor):
            X = torch.tensor(X, dtype=torch.float32)
        X = X.to(self.device)
        n, p = X.shape

        self.loss_history = []
        self.last_valid_D_j = None  # Reseta para a nova rodada

        indices = torch.randperm(n)[: self.c]
        self.prototypes = X[indices].clone()

        inv_s2_val = self.gamma ** (1.0 / p)
        self.s2 = torch.full((p,), 1.0 / inv_s2_val, device=self.device)

        labels = torch.zeros(n, dtype=torch.long, device=self.device)

        for iteration in range(self.max_iter):
            old_labels = labels.clone()
            K = self._gaussian_kernel(X, self.prototypes)
            labels = torch.argmin(2 * (1.0 - K), dim=-1)

            current_j = self.compute_objective(X, labels)
            self.loss_history.append(current_j)

            if iteration > 0 and torch.equal(labels, old_labels):
                self.logger.info("Convergiu na iteração " + str(iteration))
                break

            # Etapa 1: Representação
            for i in range(self.c):
                mask = labels == i
                if torch.sum(mask) == 0:
                    continue
                K_i = K[mask, i].unsqueeze(1)
                X_i = X[mask]
                self.prototypes[i] = torch.sum(K_i * X_i, dim=0) / torch.sum(K_i)

            # Etapa 2: Computação dos hiperparâmetros de largura
            squared_diff = (X.unsqueeze(1) - self.prototypes.unsqueeze(0)) ** 2
            K_assigned = K[torch.arange(n), labels].unsqueeze(1).unsqueeze(2)
            squared_diff_assigned = squared_diff[torch.arange(n), labels]

            # Cálculo do denominador original da Equação (16)
            D_j = torch.sum(K_assigned.squeeze(-1) * squared_diff_assigned, dim=0)

            # Verifica se alguma variável j colapsou para zero (ou quase zero devido a precisão de float)
            if torch.any(D_j <= 1e-7):
                if self.last_valid_D_j is not None:
                    # Se tivermos um histórico válido, usamos ele para estabilizar a iteração atual
                    D_j = self.last_valid_D_j.clone()
            else:
                # Se o D_j atual for perfeitamente válido, nós o salvamos para o futuro
                self.last_valid_D_j = D_j.clone()
            # --------------------------------

            prod_term = torch.prod(D_j) ** (1.0 / p)
            inv_s2 = prod_term / D_j
            self.s2 = 1.0 / inv_s2

        self.objective_value = self.compute_objective(X, labels)
        return self

    def predict(self, X):
        if not isinstance(X, torch.Tensor):
            X = torch.tensor(X, dtype=torch.float32)
        X = X.to(self.device)
        K = self._gaussian_kernel(X, self.prototypes)

        self.labels = torch.argmin(2 * (1.0 - K), dim=-1)
        return self.labels
