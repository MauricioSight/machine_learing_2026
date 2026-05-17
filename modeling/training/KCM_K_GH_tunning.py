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


class KCMKGHTunningTrain:

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
        y,
        model,
        verbose=True,
    ):
        if verbose:
            self.logger.info(f"Starting Fold {fold_id + 1}")

        # ------------------------------------------------------
        # reset
        # ------------------------------------------------------

        model.reset_weights()

        # ------------------------------------------------------
        # fit
        # ------------------------------------------------------

        self.fit(model, X, y, verbose=verbose)

        # ------------------------------------------------------
        # validation
        # ------------------------------------------------------

        train_out = self.test(model, X, y)

        return train_out

    def save_best_model(
        self,
        phase,
        metrics_handler,
        best_model_metric,
        model,
        train_out,
        objective_metric,
        direction,
    ):
        target_metric = (
            self.config.get("modeling", {}).get("training", {}).get("target_metric")
        )

        metrics = metrics_handler.get_run_metrics(
            train_out[0], train_out[1], verbose=False
        )

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
                },
                self.run_dir / f"{phase}_predictions.pt",
                pickle_protocol=pickle.HIGHEST_PROTOCOL,
            )

        return metrics

    def train(self, model, X: np.ndarray, y: pd.DataFrame, metrics_handler):
        n_repeats = (
            self.config.get("modeling", {}).get("training", {}).get("n_repeats", 100)
        )
        phase = self.config.get("phase")

        best_model_metric = None

        tune_config_name = (
            self.config.get("modeling", {}).get("training", {}).get("tunning_config")
        )
        tune_config = load_config(default_file_name=tune_config_name)
        direction = tune_config.get("direction")
        objective_metric = tune_config.get("objective_metric")

        n_clusters_set = tune_config.get("search_space").get(
            "modeling-structure-n_clusters"
        )[1:]

        y_labels = y["label"].values

        best_c = None
        best_c_metric = None
        for n_clusters in n_clusters_set:
            updated_config = apply_trial_to_config(
                self.config, {"modeling-structure-n_clusters": n_clusters}
            )

            self.logger.debug("Initializing tracker...")
            flat_config = flatten_dict(updated_config)
            self.tracker = WandBTracker(config=flat_config)

            best_model = None
            best_model_metric = None
            for i in range(n_repeats):
                inner_model = ModelingStructureFactory().get(
                    updated_config, self.logger, self.device, self.tracker
                )
                inner_model.compile()

                self.tracker.start_run(inner_model, fold_id=n_clusters)

                train_out = self.train_epoch(
                    i,
                    X,
                    y_labels,
                    inner_model,
                    verbose=False,
                )

                metrics = metrics_handler.get_run_metrics(
                    train_out[0], (X, train_out[1]), verbose=False
                )

                if (
                    best_model_metric is None
                    or best_model_metric > metrics[objective_metric]
                ):
                    best_model = inner_model
                    best_model_metric = metrics[objective_metric]

                self.tracker.log_metrics(
                    {"n_clusters": n_clusters, "repeat": i, **metrics}
                )
                self.tracker.finish()

            train_out = self.test(best_model, X, y_labels)

            metrics = metrics_handler.get_run_metrics(
                train_out[0], (X, train_out[1]), verbose=False
            )

            if best_c is None or best_c_metric > metrics[objective_metric]:
                best_c = n_clusters
                best_c_metric = metrics[objective_metric]

        self.logger.info(f"Best c: {best_c}")
        updated_config = apply_trial_to_config(
            self.config, {"modeling-structure-n_clusters": best_c}
        )
        self.config = updated_config

        model = ModelingStructureFactory().get(self.config, self.logger, self.device)
        model.compile()

        train_out = self.train_epoch(best_c, X, y_labels, model)
        out = (train_out[0], (X, train_out[1]))

        metrics = self.save_best_model(
            phase,
            metrics_handler,
            None,
            model,
            out,
            objective_metric,
            direction,
        )

        return [out], [out]
