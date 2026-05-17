from matplotlib import pyplot as plt
import torch

from tracker.base_tracker import BaseTracker

from torch import device as TorchDevice
from utils.experiment_io import get_run_dir


class KCM_K_GH:
    def __init__(self, config, logger, device: TorchDevice, tracker: BaseTracker):
        self.config = config
        self.logger = logger
        self.device = device
        self.tracker = tracker

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

    def _save_matrix_image(self, tensor, name, iteration):

        try:

            tensor_cpu = tensor.detach().cpu()

            if tensor_cpu.ndim == 1:

                tensor_cpu = tensor_cpu.unsqueeze(0)

            plt.figure(figsize=(8, 6))

            plt.imshow(tensor_cpu, aspect="auto")

            plt.colorbar()

            plt.title(
                f"{name} | iter={iteration} | "
                f"min={tensor_cpu.min():.6f} "
                f"max={tensor_cpu.max():.6f}"
            )

            save_path = f"{iteration:04d}_{name}.png"

            plt.savefig(save_path, bbox_inches="tight")

            plt.close()

        except Exception as e:

            self.logger.error(f"Erro salvando imagem {name}: {e}")

    def _debug_tensor(self, tensor, name, iteration):

        if torch.isnan(tensor).any():

            self.logger.error(f"[NaN] {name} na iteração {iteration}")

        if torch.isinf(tensor).any():

            self.logger.error(f"[INF] {name} na iteração {iteration}")

        self.logger.debug(
            f"{name} | "
            f"shape={tuple(tensor.shape)} | "
            f"min={tensor.min().item():.6f} | "
            f"max={tensor.max().item():.6f} | "
            f"mean={tensor.mean().item():.6f}"
        )

        # self._save_matrix_image(tensor, name, iteration) # TODO: add params

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
            self._debug_tensor(K, "K", iteration)

            labels = torch.argmin(2 * (1.0 - K), dim=-1)

            current_j = self.compute_objective(X, labels)

            if self.tracker is not None:
                self.tracker.log_metrics(
                    {"iteration": iteration, "train_loss": current_j}
                )

            self.logger.info(f"current_j: {current_j}")
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

            self._debug_tensor(
                self.prototypes,
                "prototypes",
                iteration,
            )

            # Etapa 2: Computação dos hiperparâmetros de largura
            squared_diff = (X.unsqueeze(1) - self.prototypes.unsqueeze(0)) ** 2
            K_assigned = K[torch.arange(n), labels].unsqueeze(1).unsqueeze(2)
            squared_diff_assigned = squared_diff[torch.arange(n), labels]

            self._debug_tensor(
                squared_diff.mean(dim=0),
                "squared_diff_mean",
                iteration,
            )

            # Cálculo do denominador original da Equação (16)
            D_j = torch.sum(K_assigned.squeeze(-1) * squared_diff_assigned, dim=0)

            self._debug_tensor(
                D_j,
                "D_j_before",
                iteration,
            )

            # Verifica se alguma variável j colapsou para zero (ou quase zero devido a precisão de float)
            invalid = D_j <= 1e-7

            if self.last_valid_D_j is not None:
                D_j[invalid] = self.last_valid_D_j[invalid]
            else:
                # Se o D_j atual for perfeitamente válido, nós o salvamos para o futuro
                self.last_valid_D_j = D_j.clone()
            # --------------------------------

            prod_term = torch.prod(D_j) ** (1.0 / p)
            inv_s2 = prod_term / D_j
            self.s2 = 1.0 / inv_s2

            self._debug_tensor(
                D_j,
                "D_j_after",
                iteration,
            )

            self._debug_tensor(
                self.s2,
                "s2",
                iteration,
            )

        self.objective_value = self.compute_objective(X, labels)
        return self

    def predict(self, X):
        if not isinstance(X, torch.Tensor):
            X = torch.tensor(X, dtype=torch.float32)
        X = X.to(self.device)
        K = self._gaussian_kernel(X, self.prototypes)

        self.labels = torch.argmin(2 * (1.0 - K), dim=-1)
        return self.labels

    @torch.no_grad()
    def predict_proba(self, X):
        if not isinstance(X, torch.Tensor):
            X = torch.tensor(X, dtype=torch.float32)
        X = X.to(self.device)
        K = self._gaussian_kernel(X, self.prototypes)

        self.labels = torch.argmin(2 * (1.0 - K), dim=-1)
        return self.labels
