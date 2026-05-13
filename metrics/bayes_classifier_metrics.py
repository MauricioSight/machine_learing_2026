import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)


class BayesClassifierMetrics:
    def __init__(self, config, logger, device):
        self.config = config
        self.device = device
        self.logger = logger

    def get_overall_metrics(
        self, y_true: list[pd.DataFrame], y_scores: list[np.ndarray], verbose=True
    ) -> dict:
        """
        Compute metrics over all folds.

        Returns:
        - point estimate (mean)
        - confidence interval (95%)
        """

        error_rates = []
        precisions = []
        recalls = []
        f_measures = []

        for fold_idx, (y_t, y_pred) in enumerate(zip(y_true, y_scores)):
            y_t = np.array(y_t)

            # metrics
            y_pred = y_pred.cpu()
            acc = accuracy_score(y_t, y_pred)

            error_rate = 1.0 - acc

            precision = precision_score(y_t, y_pred, average="binary", zero_division=0)

            recall = recall_score(y_t, y_pred, average="binary", zero_division=0)

            f_measure = f1_score(y_t, y_pred, average="binary", zero_division=0)

            error_rates.append(error_rate)
            precisions.append(precision)
            recalls.append(recall)
            f_measures.append(f_measure)

            if verbose:
                self.logger.info(
                    f"Fold {fold_idx + 1}"
                    f" | Error: {error_rate:.6f}"
                    f" | Precision: {precision:.6f}"
                    f" | Recall: {recall:.6f}"
                    f" | F1: {f_measure:.6f}"
                )

        def summarize(metric_values):

            metric_values = np.array(metric_values)

            mean = np.mean(metric_values)

            if len(metric_values) > 1:
                std = np.std(metric_values, ddof=1)
                ci95 = 1.96 * std / np.sqrt(len(metric_values))
            else:
                std = 0.0
                ci95 = 0.0

            return {
                "mean": mean,
                "std": std,
                "ci95_lower": mean - ci95,
                "ci95_upper": mean + ci95,
            }

        results = {
            "error_rate": summarize(error_rates),
            "precision": summarize(precisions),
            "recall": summarize(recalls),
            "f_measure": summarize(f_measures),
        }

        if verbose:
            self.logger.info("\n===== FINAL RESULTS =====")
            for metric_name, values in results.items():
                self.logger.info(
                    f"{metric_name}"
                    f" | Mean: {values['mean']:.6f}"
                    f" | Std: {values['std']:.6f}"
                    f" | CI95%: "
                    f"[{values['ci95_lower']:.6f}, "
                    f"{values['ci95_upper']:.6f}]"
                )

        return results

    def get_threshold(self, *args):
        """
        Logistic regression does not require threshold tuning.
        """
        return None
