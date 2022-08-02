import json
import time
from typing import Dict, List


class ExperimentTracker:
    def __init__(self):
        self.parameters = dict()
        self.metrics = dict()
        self.dataset_info = None
        self.confusion_matrix = None
        self.labels = None

    def log_parameter(self, key: str, value: object):
        self.parameters[key] = value

    def log_parameters(self, parameters: Dict):
        for k, v in parameters.items():
            self.log_parameter(k, v)

    def log_metric(self, key: str, value: object):
        self.metrics[key] = value

    def log_metrics(self, metrics: Dict):
        for k, v in metrics.items():
            self.log_metric(k, v)

    def log_dataset_hash(self, data: str):
        pass

    def log_dataset_info(self, name: str):
        self.dataset_info = name

    def __str__(self):
        return json.dumps(self.__dict__)

    def log_confusion_matrix(
        self,
        matrix: List[List[int]],
        labels=List[str],
    ):
        self.confusion_matrix = matrix
        self.labels = labels

    def start(self):
        pass

    def end(self):
        datetime_val = time.strftime("%Y%m%d-%H%M%S")
        filename = f"experiment_{datetime_val}.json"
        print(f"saving experiment data to {filename}")
        with open(filename, 'w') as json_file:
            json.dump(self.__dict__, json_file)
