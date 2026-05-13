from logging import Logger

import numpy as np
import pandas as pd

from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold
import torch
import pickle

from utils.experiment_io import get_run_dir
from utils.seed_all import DEFAULT_SEED


class StratifiedKFoldTrain:

    def __init__(self, config: dict, logger: Logger):
        self.config = config
        self.logger = logger

        self.run_dir = get_run_dir(self.config.get("run_id"))

    def fit(self, model, X_train, y_train):
        """
        Ajusta o modelo Bayesiano.
        """

        model.fit(X_train, y_train)

    def test(self, model, X_val, y_val):
        """
        Executa inferência.
        """

        y_pred = model.predict_proba(X_val)

        return (y_val, y_pred)

    def train_epoch(
        self, fold_id, X: np.ndarray, y: pd.DataFrame, train_idx, test_idx, model
    ):

        self.logger.info(f"Starting Fold {fold_id + 1}")

        # ------------------------------------------------------
        # split
        # ------------------------------------------------------

        X_train = X[train_idx]
        X_test = X[test_idx]

        y_train = y.iloc[train_idx]["label"].values

        y_test = y.iloc[test_idx]["label"].values

        # ------------------------------------------------------
        # reset
        # ------------------------------------------------------

        model.reset_weights()

        # ------------------------------------------------------
        # fit
        # ------------------------------------------------------

        self.fit(model, X_train, y_train)

        # ------------------------------------------------------
        # validation
        # ------------------------------------------------------

        train_out = self.test(model, X_train, y_train)
        val_out = self.test(model, X_test, y_test)

        return train_out, val_out

    def save_best_model(
        self,
        phase,
        metrics_handler,
        best_model_loss,
        model,
        train_idx,
        test_idx,
        train_out,
        test_out,
    ):
        metric = metrics_handler.get_overall_metrics(
            [train_out[0]], [train_out[1]], verbose=False
        )["error_rate"]["mean"]
        if metric < best_model_loss:
            self.logger.info("Saving model...")
            best_model_loss = metric
            model.save()
            torch.save(
                {
                    "train_y_true": train_out[0],
                    "train_y_scores": train_out[1],
                    "test_y_true": test_out[0],
                    "test_y_scores": test_out[1],
                    "train_idx": train_idx,
                    "test_idx": test_idx,
                },
                self.run_dir / f"{phase}_predictions.pt",
                pickle_protocol=pickle.HIGHEST_PROTOCOL,
            )

    def train(self, model, X: np.ndarray, y: pd.DataFrame, metrics_handler):
        n_repeats = (
            self.config.get("modeling", {}).get("training", {}).get("n_repeats", 300)
        )
        phase = self.config.get("phase")
        has_tunning = len(self.config.get("modeling", {}).get("structure")) > 1
        best_model_loss = float("inf")

        y_labels = y["label"].values

        splitter = RepeatedStratifiedKFold(
            n_splits=10, n_repeats=n_repeats, random_state=DEFAULT_SEED
        )

        fold_train_outs = []
        fold_test_outs = []

        for fold_id, (train_idx, test_idx) in enumerate(splitter.split(X, y_labels)):
            if phase == "tunning" and has_tunning:
                inner_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

                X_train_outer = X[train_idx]
                y_train_outer = y.iloc[train_idx].reset_index(drop=True)

                for inner_fold_id, (inner_train_idx, inner_test_idx) in enumerate(
                    inner_cv.split(X_train_outer, y_train_outer["label"])
                ):
                    train_out, test_out = self.train_epoch(
                        inner_fold_id, X, y, inner_train_idx, inner_test_idx, model
                    )

                    self.save_best_model(
                        phase,
                        metrics_handler,
                        best_model_loss,
                        model,
                        inner_train_idx,
                        inner_test_idx,
                        train_out,
                        test_out,
                    )

                    fold_train_outs.append(train_out)
                    fold_test_outs.append(test_out)

                return fold_train_outs, fold_test_outs

            train_out, test_out = self.train_epoch(
                fold_id, X, y, train_idx, test_idx, model
            )

            self.save_best_model(
                phase,
                metrics_handler,
                best_model_loss,
                model,
                train_idx,
                test_idx,
                train_out,
                test_out,
            )

            fold_train_outs.append(train_out)
            fold_test_outs.append(test_out)

        return fold_train_outs, fold_test_outs
