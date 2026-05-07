import torch
import torch.nn as nn
from torch import device as TorchDevice
import numpy as np
import pandas as pd

from logging import Logger

from modeling.structure.pytorch_base import PytorchModelStructure


class LogisticRegressionModel(PytorchModelStructure):
    def __init__(
        self,
        config: dict,
        logger: Logger,
        device: TorchDevice
    ):
        super().__init__(config, logger, device)

        input_dim  = config.get('modeling', {}).get('structure', {}).get('input_dim')
        num_classes  = config.get('modeling', {}).get('structure', {}).get('num_classes')

        # Regressão logística multinomial:
        # y = Wx + b
        self.linear = nn.Linear(input_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Tensor de entrada com shape:
            (batch_size, input_dim)

        Returns
        -------
        torch.Tensor
            Logits com shape:
            (batch_size, num_classes)
        """

        x = x.float()

        logits = self.linear(x)

        return logits

    def featuring(
        self,
        values: np.ndarray,
        labels: pd.DataFrame
    ) -> tuple[np.ndarray, np.ndarray]:
        return values, labels

    def reset_weights(self):
        for layer in self.children():
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()