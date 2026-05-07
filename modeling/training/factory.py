from torch import device as TorchDevice

from tracker.base_tracker import BaseTracker
from modeling.training.pytorch_base import PytorchTrainingAlgorithm
from logger.base import Logger

class ModelingTrainingFactory:
    """
    Base class for model training factory.
    """
    def get(self, config: dict, logger: Logger, device: TorchDevice, tracker: BaseTracker) -> PytorchTrainingAlgorithm:
        name = config.get('modeling', {}).get('training', {}).get('name')

        if name == 'dnn':
            from modeling.training.dnn_train import DNNTrain

            return DNNTrain(config, logger, device, tracker)
        
        if name == 'stratified_k_fold':
            from modeling.training.stratified_k_fold_train import StratifiedKFoldTrain

            return StratifiedKFoldTrain(config, logger, device, tracker)

        else:
            raise ValueError(
                f"Unsupported ModelingTrainingFactory name: {name}")