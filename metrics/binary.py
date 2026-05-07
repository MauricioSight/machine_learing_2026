import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay
)

from metrics.base import InferenceMetrics
from utils.experiment_io import get_run_dir


class Binary(InferenceMetrics):

    def get_overall_metrics(
        self,
        y_true: list[pd.DataFrame],
        y_scores: list[np.ndarray]
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

        for fold_idx, (y_t, y_s) in enumerate(
            zip(y_true, y_scores)
        ):
            y_t = np.array(y_t)

            # logits -> predicted class
            y_pred = np.argmax(y_s, axis=1)

            # metrics
            acc = accuracy_score(y_t, y_pred)

            error_rate = 1.0 - acc

            precision = precision_score(
                y_t,
                y_pred,
                average='binary',
                zero_division=0
            )

            recall = recall_score(
                y_t,
                y_pred,
                average='binary',
                zero_division=0
            )

            f_measure = f1_score(
                y_t,
                y_pred,
                average='binary',
                zero_division=0
            )

            error_rates.append(error_rate)
            precisions.append(precision)
            recalls.append(recall)
            f_measures.append(f_measure)

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

            std = np.std(
                metric_values,
                ddof=1
            )

            ci95 = (
                1.96
                * std
                / np.sqrt(len(metric_values))
            )

            return {
                'mean': mean,
                'std': std,
                'ci95_lower': mean - ci95,
                'ci95_upper': mean + ci95
            }

        results = {
            'error_rate': summarize(error_rates),
            'precision': summarize(precisions),
            'recall': summarize(recalls),
            'f_measure': summarize(f_measures)
        }

        self.logger.info(
            "\n===== FINAL RESULTS ====="
        )

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