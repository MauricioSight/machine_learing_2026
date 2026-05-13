from torch import device as TorchDevice

from tracker.base_tracker import BaseTracker
from logger.base import Logger


class ModelingTrainingFactory:
    """
    Base class for model training factory.
    """

    def get(
        self, config: dict, logger: Logger, device: TorchDevice, tracker: BaseTracker
    ):
        name = config.get("modeling", {}).get("training", {}).get("name")

        if name == "stratified_k_fold":
            from modeling.training.stratified_k_fold_train import StratifiedKFoldTrain

            return StratifiedKFoldTrain(config, logger)

        else:
            raise ValueError(f"Unsupported ModelingTrainingFactory name: {name}")
