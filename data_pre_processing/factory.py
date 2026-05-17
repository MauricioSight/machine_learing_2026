from data_pre_processing.base import DataPrePrecessing
from logger.base import Logger


class DataPreProcessingFactory:
    """
    Base class for pre_processing factory.
    """

    def get(self, config: dict, logger: Logger) -> DataPrePrecessing:
        name = config.get("pre_processing", {}).get("name")

        if name == "norm":
            from data_pre_processing.norm_pre_precessing import NormPrePrecessing

            return NormPrePrecessing(config, logger)

        if name == "none":
            from data_pre_processing.none_pre_processing import NonePreProcessing

            return NonePreProcessing(config, logger)

        if name == "rmv_const_pre_processing":
            from data_pre_processing.rmv_const_pre_processing import (
                RmvConstPreProcessing,
            )

            return RmvConstPreProcessing(config, logger)

        else:
            raise ValueError(f"Unsupported DataPreProcessingFactory name: {name}")
