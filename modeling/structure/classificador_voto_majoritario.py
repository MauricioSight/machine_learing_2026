import torch
import pandas as pd
import numpy as np
from classificador_bayesiano import BayesClassifier
from classificador_bayesiano_kvizinhos import KNNBayesian
from classificador_bayesiano_parzen import ParzenWindowBayesianCUDA
from classificador_logistic_regression import LogisticRegressionClassifier

# --- 1. Definição da Classe do Voto Majoritário ---
class MajorityVotingClassifier:
    def __init__(self, classifiers, device='cuda'):
        """
        classifiers: Lista de instâncias dos seus classificadores.
                     Ex: [modelo_bayes, modelo_knn, modelo_parzen, modelo_logistica]
        """
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.classifiers = classifiers

    def fit(self, X, y):
        # Treina individualmente cada um dos classificadores contidos na lista
        for clf in self.classifiers:
            clf.fit(X, y)

    def predict(self, X):
        X = X.to(self.device)
        lista_predicoes = []

        for clf in self.classifiers:
            clf.device = self.device
            
            # Executa o predict e move o tensor resultante para o device central
            preds = clf.predict(X).to(self.device)
            lista_predicoes.append(preds)

        # Empilha as predições. Formato resultante: [num_modelos, num_amostras]
        predicoes_empilhadas = torch.stack(lista_predicoes, dim=0)

        # Calcula a Moda (voto majoritário) ao longo do eixo dos modelos (dim=0)
        voto_final, _ = torch.mode(predicoes_empilhadas, dim=0)

        return voto_final