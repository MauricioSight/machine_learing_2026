import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    multilabel_confusion_matrix,
)


class BayesMultiClassifierMetrics:
    """
    Multilabel classification metrics.

    Expected:
    ----------
    y_true shape:
        (n_samples, n_labels)

    y_scores shape:
        (n_samples, n_labels)

    Supports:
    ----------
    - probabilities/logits
    - binary predictions
    """

    def __init__(self, config, logger, device):
        self.config = config
        self.device = device
        self.logger = logger

        self.threshold = self.config.get("metrics", {}).get("threshold", 0.5)

    def _prepare_targets(self, y_true):
        """
        Convert targets to numpy int array.
        """

        if isinstance(y_true, pd.DataFrame):
            y_true = y_true.values

        y_true = np.asarray(y_true).astype(int)

        return y_true

    def _prepare_predictions(self, y_scores):
        """
        Convert predictions to binary matrix.
        """

        if hasattr(y_scores, "cpu"):
            y_scores = y_scores.cpu().numpy()

        y_scores = np.asarray(y_scores)

        # probabilities/logits
        if np.issubdtype(y_scores.dtype, np.floating):
            y_pred = (y_scores >= self.threshold).astype(int)
        else:
            y_pred = y_scores.astype(int)

        return y_pred

    def _compute_metrics(self, y_true, y_pred):
        """
        Compute multilabel metrics.
        """

        # exact match ratio
        subset_accuracy = accuracy_score(y_true, y_pred)

        error_rate = 1.0 - subset_accuracy

        classification_results = classification_report(
            y_true,
            y_pred,
            output_dict=True,
            zero_division=0,
        )

        confusion_matrices = multilabel_confusion_matrix(
            y_true,
            y_pred,
        )

        results = {
            "subset_accuracy": subset_accuracy,
            "error_rate": error_rate,
            "macro_precision": classification_results["macro avg"]["precision"],
            "macro_recall": classification_results["macro avg"]["recall"],
            "macro_f1": classification_results["macro avg"]["f1-score"],
            "weighted_precision": classification_results["weighted avg"]["precision"],
            "weighted_recall": classification_results["weighted avg"]["recall"],
            "weighted_f1": classification_results["weighted avg"]["f1-score"],
            "classification_report": classification_results,
            "confusion_matrices": confusion_matrices,
        }

        return results

    def get_run_metrics(
        self,
        y_true: pd.DataFrame,
        y_scores,
        verbose=True,
    ):
        """
        Compute metrics for one run.
        """

        y_true = self._prepare_targets(y_true)

        y_pred = self._prepare_predictions(y_scores)

        results = self._compute_metrics(y_true, y_pred)

        if verbose:

            self.logger.info("\n===== RUN RESULTS =====")

            self.logger.info(
                f"Subset Accuracy: {results['subset_accuracy']:.6f}"
                f" | Error Rate: {results['error_rate']:.6f}"
            )

            self.logger.info(
                f"Macro  -> "
                f"P: {results['macro_precision']:.6f}"
                f" | R: {results['macro_recall']:.6f}"
                f" | F1: {results['macro_f1']:.6f}"
            )

            self.logger.info(
                f"Weighted -> "
                f"P: {results['weighted_precision']:.6f}"
                f" | R: {results['weighted_recall']:.6f}"
                f" | F1: {results['weighted_f1']:.6f}"
            )

            self.logger.info("\n===== PER LABEL METRICS =====")

            report = results["classification_report"]

            n_labels = y_true.shape[1]

            for label_idx in range(n_labels):

                label_key = str(label_idx)

                if label_key not in report:
                    continue

                self.logger.info(
                    f"Label {label_idx}"
                    f" | Precision: {report[label_key]['precision']:.6f}"
                    f" | Recall: {report[label_key]['recall']:.6f}"
                    f" | F1: {report[label_key]['f1-score']:.6f}"
                    f" | Support: {report[label_key]['support']}"
                )

                cm = results["confusion_matrices"][label_idx]

                self.logger.info(f"Confusion Matrix:\n{cm}")

        return results

    def get_fold_metrics(
        self,
        y_true: list[pd.DataFrame],
        y_scores: list,
        verbose=True,
    ) -> dict:
        """
        Compute statistics across folds.
        """

        metric_history = {
            "subset_accuracy": [],
            "error_rate": [],
            "macro_precision": [],
            "macro_recall": [],
            "macro_f1": [],
            "weighted_precision": [],
            "weighted_recall": [],
            "weighted_f1": [],
        }

        for fold_idx, (y_t, y_s) in enumerate(zip(y_true, y_scores)):

            y_t = self._prepare_targets(y_t)

            y_pred = self._prepare_predictions(y_s)

            metrics = self._compute_metrics(y_t, y_pred)

            for metric_name in metric_history:
                metric_history[metric_name].append(metrics[metric_name])

            if verbose:

                self.logger.info(f"\n===== Fold {fold_idx + 1} =====")

                self.logger.info(f"Subset Accuracy: {metrics['subset_accuracy']:.6f}")

                self.logger.info(
                    f"Macro F1: {metrics['macro_f1']:.6f}"
                    f" | Weighted F1: {metrics['weighted_f1']:.6f}"
                )

        def summarize(values):

            values = np.asarray(values)

            mean = np.mean(values)

            if len(values) > 1:
                std = np.std(values, ddof=1)
                ci95 = 1.96 * std / np.sqrt(len(values))
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
            metric_name: summarize(metric_values)
            for metric_name, metric_values in metric_history.items()
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
        Return prediction threshold.
        """

        return self.threshold
