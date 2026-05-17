import json
from logging import Logger

import numpy as np
import pandas as pd

from torch import device as TorchDevice
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold
import torch
import pickle

from modeling.structure.factory import ModelingStructureFactory
from optimizer.factory import OptimizerFactory
from tracker.base_tracker import BaseTracker
from tracker.wandb_tracker import WandBTracker
from tunning_models_automated import apply_trial_to_config
from utils.config_handle import flatten_dict, load_config
from utils.experiment_io import get_run_dir, get_run_id
from utils.seed_all import DEFAULT_SEED


class StratifiedKFoldTunningTrain:

    def __init__(
        self, config: dict, logger: Logger, device: TorchDevice, tracker: BaseTracker
    ):
        self.config = config
        self.logger = logger
        self.device = device
        self.tracker = tracker

        self.run_dir = get_run_dir(self.config.get("run_id"))

    def fit(self, model, X_train, y_train, verbose=True):
        """
        Ajusta o modelo Bayesiano.
        """

        model.fit(X_train, y_train, verbose=verbose)

    def test(self, model, X_val, y_val):
        """
        Executa inferência.
        """

        y_pred = model.predict_proba(X_val)

        return (y_val, y_pred)

    def train_epoch(
        self,
        fold_id,
        X: np.ndarray,
        y: pd.DataFrame,
        train_idx,
        test_idx,
        model,
        verbose=True,
    ):
        if verbose:
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

        self.fit(model, X_train, y_train, verbose=verbose)

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
        best_model_metric,
        model,
        train_idx,
        test_idx,
        train_out,
        test_out,
        objective_metric,
        direction,
    ):
        tunning_automated_id = self.config.get("tunning_automated_id", None)
        target_metric = (
            self.config.get("modeling", {}).get("training", {}).get("target_metric")
        )

        metrics = metrics_handler.get_run_metrics(
            train_out[0], train_out[1], verbose=False
        )

        if tunning_automated_id is None:
            target_metric = metrics[objective_metric]
            if (
                best_model_metric is None
                or (direction == "minimize" and best_model_loss < target_metric)
                or (direction == "maximize" and best_model_loss > target_metric)
            ):
                self.logger.info("Saving model...")
                best_model_loss = objective_metric
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

        return metrics

    def train(self, model, X: np.ndarray, y: pd.DataFrame, metrics_handler):
        n_repeats = (
            self.config.get("modeling", {}).get("training", {}).get("n_repeats", 300)
        )
        phase = self.config.get("phase")
        has_tunning = len(self.config.get("modeling", {}).get("structure")) > 1
        best_model_metric = None

        tune_config_name = (
            self.config.get("modeling", {}).get("training", {}).get("tunning_config")
        )
        tune_config = load_config(default_file_name=tune_config_name)
        direction = tune_config.get("direction")
        objective_metric = tune_config.get("objective_metric")

        completed_folds_file = self.run_dir / "completed_folds.json"
        if completed_folds_file.exists():
            with open(completed_folds_file, "r") as f:
                completed_folds = set(json.load(f))
        else:
            completed_folds = set()

        y_labels = y["label"].values

        total_folds = 10 * n_repeats
        splitter = RepeatedStratifiedKFold(
            n_splits=10, n_repeats=n_repeats, random_state=DEFAULT_SEED
        )

        fold_train_outs = []
        fold_test_outs = []

        for fold_id, (train_idx, test_idx) in enumerate(splitter.split(X, y_labels)):
            if fold_id in completed_folds:
                self.logger.info(f"Skipping completed fold {fold_id}/{total_folds}")
                continue

            self.logger.info(f"Running fold {fold_id}/{total_folds}")

            best_fold_params = None
            best_fold_value = None
            if phase == "tunning" and has_tunning:
                inner_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

                X_train_outer = X[train_idx]
                y_train_outer = y.iloc[train_idx].reset_index(drop=True)

                for inner_fold_id, (inner_train_idx, inner_test_idx) in enumerate(
                    inner_cv.split(X_train_outer, y_train_outer["label"])
                ):
                    self.logger.info(
                        f"Find best params [{fold_id}/{total_folds}] [{inner_fold_id+1}/5]..."
                    )

                    def objective_fn(params):
                        updated_config = apply_trial_to_config(self.config, params)

                        inner_model = ModelingStructureFactory().get(
                            updated_config, self.logger, self.device
                        )
                        inner_model.compile()

                        _, test_out = self.train_epoch(
                            inner_fold_id,
                            X,
                            y,
                            inner_train_idx,
                            inner_test_idx,
                            inner_model,
                            verbose=False,
                        )

                        metrics = metrics_handler.get_run_metrics(
                            test_out[0], test_out[1], verbose=False
                        )

                        return metrics[objective_metric]

                    optimizer_factory = OptimizerFactory(tune_config)
                    optimizer = optimizer_factory.get_optimizer()
                    best_trial = optimizer.optimize(objective_fn)

                    self.logger.info("Best trial:")
                    self.logger.info(json.dumps(best_trial.params, indent=4))

                    if (
                        best_fold_value is None
                        or (
                            direction == "minimize"
                            and best_fold_value < best_trial.value
                        )
                        or (
                            direction == "maximize"
                            and best_fold_value > best_trial.value
                        )
                    ):
                        best_fold_value = best_trial.value
                        best_fold_params = best_trial.params

            if best_fold_params is not None:
                updated_config = apply_trial_to_config(self.config, best_fold_params)
                self.config = updated_config

                self.logger.debug("Initializing tracker...")
                flat_config = flatten_dict(self.config)
                self.tracker = WandBTracker(config=flat_config)

                model = ModelingStructureFactory().get(
                    self.config, self.logger, self.device, self.tracker
                )
                model.compile()

            self.tracker.start_run(model, fold_id=fold_id)

            train_out, test_out = self.train_epoch(
                fold_id, X, y, train_idx, test_idx, model
            )

            metrics = self.save_best_model(
                phase,
                metrics_handler,
                best_model_metric,
                model,
                train_idx,
                test_idx,
                train_out,
                test_out,
                objective_metric,
                direction,
            )

            fold_train_outs.append(train_out)
            fold_test_outs.append(test_out)

            self.tracker.log_metrics({"fold_id": fold_id, **metrics})
            self.tracker.finish()

            completed_folds.add(fold_id)
            with open(completed_folds_file, "w") as f:
                json.dump(sorted(list(completed_folds)), f)

        return fold_train_outs, fold_test_outs
