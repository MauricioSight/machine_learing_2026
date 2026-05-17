from torch import device as TorchDevice

from logger.base import Logger
from tracker.base_tracker import BaseTracker


class ModelingStructureFactory:
    """
    Base class for model structure factory.
    """

    def get(
        self,
        config: dict,
        logger: Logger,
        device: TorchDevice,
        tracker: BaseTracker = None,
    ):
        name = config.get("modeling", {}).get("structure", {}).get("name")

        if name == "logistic_regression":
            from modeling.structure.classificador_logistic_regression import (
                LogisticRegressionClassifier,
            )

            return LogisticRegressionClassifier(config, logger, device, tracker)

        if name == "bayes_class":
            from modeling.structure.classificador_bayesiano import (
                BayesClassifier,
            )

            return BayesClassifier(config=config, device=device)

        if name == "bayes_class_knn":
            from modeling.structure.classificador_bayesiano_kvizinhos import (
                KNNBayesian,
            )

            return KNNBayesian(config=config, device=device)

        if name == "bayes_class_parzen":
            from modeling.structure.classificador_bayesiano_parzen import (
                ParzenWindowBayesian,
            )

            return ParzenWindowBayesian(config=config, device=device)
        if name == "KCM_K_GH":
            from modeling.structure.KCM_K_GH import (
                KCM_K_GH,
            )

            return KCM_K_GH(config, logger, device, tracker)

        if name == "majority_voting_classifier":
            from modeling.structure.classificador_voto_majoritario import (
                MajorityVotingClassifier,
            )

            return MajorityVotingClassifier(config, logger, device, tracker)

        else:
            raise ValueError(f"Unsupported ModelingStructureFactory name: {name}")
