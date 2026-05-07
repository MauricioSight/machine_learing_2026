from torch import device as TorchDevice

from modeling.structure.pytorch_base import PytorchModelStructure
from logger.base import Logger

class ModelingStructureFactory:
    """
    Base class for model structure factory.
    """
    def get(self, config: dict, logger: Logger, device: TorchDevice) -> PytorchModelStructure:
        name = config.get('modeling', {}).get('structure', {}).get('name')

        if name == 'mlp':
            from modeling.structure.mlp import MLP

            return MLP(config, logger, device)
        
        if name == 'logistic_regression':
            from modeling.structure.logistic_regression import LogisticRegressionModel

            return LogisticRegressionModel(config, logger, device)

        else:
            raise ValueError(
                f"Unsupported ModelingStructureFactory name: {name}")