import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    multilabel_confusion_matrix,
)


class LogisticMultiClassifierMetrics:
    """
    Multilabel metrics for logistic models.

    Expected:
    ----------
    y_true shape:
        (n_samples, n_labels)

    logits shape:
        (n_samples, n_labels)

    Notes:
    ------
    - Uses BCEWithLogitsLoss for multilabel classification
    - Applies sigmoid + threshold
    """

    def __init__(self, config, logger, device):

        self.config = config
        self.device = device
        self.logger = logger

        self.threshold = self.config.get("metrics", {}).get("threshold", 0.5)

        # correct loss for multilabel
        self.criterion = nn.CrossEntropyLoss()

    def _prepare_targets(self, y_true):
        """
        Convert targets to tensor + numpy.
        """

        if isinstance(y_true, pd.DataFrame):
            y_true = y_true.values

        y_true = np.asarray(y_true).astype(np.float32)

        y_tensor = torch.tensor(
            y_true,
            dtype=torch.float32,
            device=self.device,
        )

        return y_true, y_tensor

    def _prepare_logits(self, logits):
        """
        Convert logits to tensor.
        """

        if not isinstance(logits, torch.Tensor):

            logits = torch.tensor(
                logits,
                dtype=torch.float32,
                device=self.device,
            )

        return logits

    def _logits_to_predictions(self, logits):
        """
        logits -> probabilities -> binary predictions
        """

        if isinstance(logits, torch.Tensor):

            probs = torch.sigmoid(logits)

            probs = probs.detach().cpu().numpy()

        else:

            probs = 1.0 / (1.0 + np.exp(-logits))

        y_pred = (probs >= self.threshold).astype(int)

        return y_pred

    def _compute_metrics(self, y_true, y_pred):
        """
        Compute multilabel metrics.
        """

        subset_accuracy = accuracy_score(
            y_true,
            y_pred,
        )

        error_rate = 1.0 - subset_accuracy

        report = classification_report(
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
            "macro_precision": report["macro avg"]["precision"],
            "macro_recall": report["macro avg"]["recall"],
            "macro_f1": report["macro avg"]["f1-score"],
            "weighted_precision": report["weighted avg"]["precision"],
            "weighted_recall": report["weighted avg"]["recall"],
            "weighted_f1": report["weighted avg"]["f1-score"],
            "classification_report": report,
            "confusion_matrices": confusion_matrices,
        }

        return results

    def get_run_metrics(
        self,
        y_true: pd.DataFrame,
        y_prob,
        verbose=True,
    ) -> dict:
        """
        Metrics for one run.
        """

        y_true_np, y_true_tensor = self._prepare_targets(y_true)

        logits_tensor = self._prepare_logits(y_prob)

        # multilabel loss

        y_true_tensor = y_true_tensor.long()

        loss = self.criterion(
            logits_tensor,
            y_true_tensor,
        ).item()

        # predictions
        y_pred = self._logits_to_predictions(logits_tensor)

        if len(y_pred.shape) > 1 and y_pred.shape[1] > 1:
            y_pred = y_pred.argmax(axis=1) # Use axis=1 se já for NumPy, ou dim=1 se ainda for Tensor do PyTorch

        results = self._compute_metrics(
            y_true_np,
            y_pred,
        )

        # overwrite error with BCE loss
        results["loss"] = loss

        if verbose:

            self.logger.info("\n===== RUN RESULTS =====")

            self.logger.info(
                f"Loss: {loss:.6f}"
                f" | Subset Accuracy: {results['subset_accuracy']:.6f}"
            )

            self.logger.info(
                f"Macro -> "
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

            self.logger.info(
                f"Samples -> "
                f"P: {results['samples_precision']:.6f}"
                f" | R: {results['samples_recall']:.6f}"
                f" | F1: {results['samples_f1']:.6f}"
            )

            self.logger.info("\n===== PER LABEL METRICS =====")

            report = results["classification_report"]

            n_labels = y_true_np.shape[1]

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
        y_prob: list,
        verbose=True,
    ) -> dict:
        """
        Metrics across folds.
        """

        metric_history = {
            "loss": [],
            "subset_accuracy": [],
            "error_rate": [],
            "macro_precision": [],
            "macro_recall": [],
            "macro_f1": [],
            "weighted_precision": [],
            "weighted_recall": [],
            "weighted_f1": [],
            "samples_precision": [],
            "samples_recall": [],
            "samples_f1": [],
        }

        for fold_idx, (y_t, logits) in enumerate(zip(y_true, y_prob)):

            y_true_np, y_true_tensor = self._prepare_targets(y_t)

            logits_tensor = self._prepare_logits(logits)

            loss = self.criterion(
                logits_tensor,
                y_true_tensor,
            ).item()

            y_pred = self._logits_to_predictions(logits_tensor)

            metrics = self._compute_metrics(
                y_true_np,
                y_pred,
            )

            metrics["loss"] = loss

            for metric_name in metric_history:

                metric_history[metric_name].append(metrics[metric_name])

            if verbose:

                self.logger.info(f"\n===== Fold {fold_idx + 1} =====")

                self.logger.info(
                    f"Loss: {loss:.6f}"
                    f" | Subset Accuracy: {metrics['subset_accuracy']:.6f}"
                    f" | Hamming Loss: {metrics['hamming_loss']:.6f}"
                )

                self.logger.info(
                    f"Macro F1: {metrics['macro_f1']:.6f}"
                    f" | Weighted F1: {metrics['weighted_f1']:.6f}"
                )

        def summarize(values):

            values = np.asarray(values)

            mean = np.mean(values)

            if len(values) > 1:

                std = np.std(
                    values,
                    ddof=1,
                )

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
