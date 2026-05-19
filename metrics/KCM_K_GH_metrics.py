import numpy as np
import pandas as pd
from scipy.stats import mode

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    silhouette_score,
)

from sklearn.metrics import adjusted_rand_score


class KCMKGHMetrics:
    def __init__(self, config, logger, device):
        self.config = config
        self.device = device
        self.logger = logger

    def map_clusters_to_classes(self, y_true, y_cluster):
        mapping = {}

        clusters = np.unique(y_cluster)

        for cluster_id in clusters:

            mask = y_cluster == cluster_id

            majority_class = mode(y_true[mask], keepdims=False).mode

            mapping[cluster_id] = majority_class

        return mapping

    def relabel_clusters(self, y_cluster, mapping):
        return np.array([mapping[int(c)] for c in y_cluster])

    def get_run_metrics(self, y_true: pd.DataFrame, y_scores: np.ndarray, verbose=True):
        y_true = np.array(y_true)
        X, labels = y_scores

        n_clusters = len(np.unique(labels.cpu().numpy()))
        if n_clusters < 2:
            self.logger.warning(f"n_clusters = {n_clusters}")
            return {
                "silhouette_score": -1,
                "adjusted_rand_score": -1,
                "accuracy": -1,
                "precision": -1,
                "recall": -1,
                "f_measure": -1,
            }

        sil = silhouette_score(
            X,
            labels.cpu(),
        )

        ari = adjusted_rand_score(
            y_true,
            labels.cpu(),
        )

        mapping = self.map_clusters_to_classes(y_true, labels.cpu())
        y_pred = self.relabel_clusters(labels.cpu(), mapping)

        acc = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, average="macro")
        recall = recall_score(y_true, y_pred, average="macro")
        f1 = f1_score(y_true, y_pred, average="macro")

        if verbose:
            self.logger.info(
                (
                    f"Silhouette score: {sil:.6f}"
                    f" | Rand score: {ari:.6f}"
                    f" | Accuracy: {acc:.6f}"
                    f" | Precision: {precision:.6f}"
                    f" | Recall: {recall:.6f}"
                    f" | F1: {f1:.6f}"
                )
            )

        results = {
            "silhouette_score": sil,
            "adjusted_rand_score": ari,
            "accuracy": acc,
            "precision": precision,
            "recall": recall,
            "f_measure": f1,
        }

        return results

    def get_fold_metrics(
        self, y_true: list[pd.DataFrame], y_scores: list[np.ndarray], verbose=True
    ) -> dict:
        for _, (y_t, y_pred) in enumerate(zip(y_true, y_scores)):
            y_t = np.array(y_t)

            # metrics
            results = self.get_run_metrics(y_t, y_pred, verbose=verbose)

        return results

    def get_threshold(self, *args):
        return None
