from logger.base import Logger


class MetricsFactory:
    """
    Base class for metrics factory.
    """

    def get(self, config: dict, logger: Logger, device):
        name = config.get("metrics", {}).get("name")

        if name == "logistic_classifier_metrics":
            from metrics.logistic_classifier_metrics import LogisticClassifierMetrics

            return LogisticClassifierMetrics(config, logger, device)

        if name == "bayes_classifier_metrics":
            from metrics.bayes_classifier_metrics import BayesClassifierMetrics

            return BayesClassifierMetrics(config, logger, device)

        else:
            raise ValueError(f"Unsupported MetricsFactory name: {name}")
