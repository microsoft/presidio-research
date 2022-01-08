import hashlib
import json
from typing import Dict


class ExperimentTracker:
    def __init__(self):
        self.parameters = None
        self.metrics = None
        self.dataset_info = None
        self.dataset_hash = None

    def log_parameter(self, key: str, value: object):
        self.parameters[key] = value

    def log_parameters(self, parameters: Dict):
        for k, v in parameters.values():
            self.log_parameter(k, v)

    def log_metric(self, key: str, value: object):
        self.metrics[key] = value

    def log_metrics(self, metrics: Dict):
        for k, v in metrics.values():
            self.log_metric(k, v)

    def log_dataset_hash(self, dataset_hash):
        self.dataset_hash = dataset_hash

    def log_dataset_info(self, dataset_info):
        self.dataset_info = dataset_info

    def __str__(self):
        return json.dumps(self.__dict__)

    def start(self):
        pass

    def end(self):
        pass