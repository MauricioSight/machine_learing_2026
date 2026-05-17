import numpy as np
import torch

from torch import device as TorchDevice

from tracker.base_tracker import BaseTracker
from modeling.structure.classificador_bayesiano import BayesClassifier
from modeling.structure.classificador_bayesiano_kvizinhos import KNNBayesian
from modeling.structure.classificador_bayesiano_parzen import ParzenWindowBayesian
from modeling.structure.classificador_logistic_regression import (
    LogisticRegressionClassifier,
)


# --- 1. Definição da Classe do Voto Majoritário ---
class MajorityVotingClassifier:
    def __init__(
        self,
        config,
        logger,
        device: TorchDevice,
        tracker: BaseTracker,
    ):
        self.config = config
        self.logger = logger
        self.tracker = tracker
        self.device = device

        bayes_classifier = BayesClassifier(config, device)
        knn_bayesian = KNNBayesian(config, device)
        parzen_window_bayesian = ParzenWindowBayesian(config, device)
        logistic_regression = LogisticRegressionClassifier(
            config, logger, device, tracker
        )

        self.classifiers = [
            bayes_classifier,
            knn_bayesian,
            parzen_window_bayesian,
            logistic_regression,
        ]

    def save(self):
        for clf in self.classifiers:
            clf.save()

    def load(self):
        for clf in self.classifiers:
            clf.load()

    def reset_weights(self):
        for clf in self.classifiers:
            clf.reset_weights()

    def compile(self):
        for clf in self.classifiers:
            clf.compile()

    def fit(
        self,
        X_train: np.ndarray,
        y_train,
        X_val: np.ndarray = None,
        y_val=None,
        verbose=True,
    ):
        # Treina individualmente cada um dos classificadores contidos na lista
        for clf in self.classifiers:
            clf.fit(X_train, y_train, X_val, y_val, verbose)

    def predict(self, X):
        predictions = []

        for clf in self.classifiers:
            # Executa o predict e move o tensor resultante para o device central
            pred = clf.predict(X).to(self.device)
            predictions.append(pred)

        # Empilha as predições. Formato resultante: [num_modelos, num_amostras]
        prediction_pile = torch.stack(predictions, dim=0)

        # Calcula a Moda (voto majoritário) ao longo do eixo dos modelos (dim=0)
        final_vote, _ = torch.mode(prediction_pile, dim=0)

        return final_vote

    def predict_proba(self, X):
        predictions = []

        for clf in self.classifiers:
            # Executa o predict e move o tensor resultante para o device central
            pred = clf.predict(X).to(self.device)
            predictions.append(pred)

        # Empilha as predições. Formato resultante: [num_modelos, num_amostras]
        prediction_pile = torch.stack(predictions, dim=0)

        # Calcula a Moda (voto majoritário) ao longo do eixo dos modelos (dim=0)
        final_vote = (prediction_pile.float().mean(dim=0) >= 0.5).long()

        return final_vote
