import json
import time
import os
from pathlib import Path
from typing import Dict, List, Union


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

    def end(self, output_dir: Union[str, Path] = None):
        datetime_val = time.strftime("%Y%m%d-%H%M%S")
        filename = f"experiment_{datetime_val}.json"
        
        # Convert to Path object if string
        if isinstance(output_dir, str):
            output_dir = Path(output_dir)
        elif output_dir is None:
            output_dir = Path.cwd()
            
        # Ensure output directory exists
        output_dir.mkdir(parents=True, exist_ok=True)
            
        # Create full file path using proper path joining
        output_path = output_dir / filename
        print(f"saving experiment data to {output_path}")
        
        with open(output_path, 'w') as json_file:
            json.dump(self.__dict__, json_file)
