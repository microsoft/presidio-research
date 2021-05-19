import collections
import json
import os
from pathlib import Path
from typing import List

import xmltodict
from tqdm import tqdm

from presidio_evaluator import InputSample
from presidio_evaluator.data_objects import Span
from presidio_evaluator.dataset_formatters import DatasetFormatter


class I2B22014Formatter(DatasetFormatter):
    def __init__(
        self,
        files_path=Path("../../data/i2b2").resolve(),
    ):
        self.files_path = files_path

    @staticmethod
    def _create_span(item):
        span = Span(
            entity_type=item["@TYPE"],
            entity_value=item["@text"],
            start_position=int(item["@start"]),
            end_position=int(item["@end"]),
        )
        return span

    def to_input_samples(self) -> List[InputSample]:
        input_samples = []
        for root, dirs, files in tqdm(os.walk(self.files_path),desc="Reading files..."):
            for file in files:
                spans = []
                filename = os.path.join(root, file)
                xml_content = open(filename, "r").read()

                ordered_dict = xmltodict.parse(xml_content)
                data = dict(ordered_dict["deIdi2b2"])
                text = data["TEXT"]
                tags = data["TAGS"]
                for item in tags.items():
                    if type(item[1]) is collections.OrderedDict:
                        spans.append(self._create_span(item[1]))
                    else:
                        for sub in item[1]:
                            spans.append(self._create_span(sub))
                input_samples.append(
                    InputSample(full_text=text, spans=spans, create_tags_from_span=True)
                )
        return input_samples

    @staticmethod
    def dataset_to_json(input_path, output_path):

        formatter = I2B22014Formatter(files_path=input_path)
        train_samples = formatter.to_input_samples()
        json_dataset = [example.to_dict() for example in train_samples]

        with open("{}".format(output_path), "w+", encoding="utf-8") as f:
            json.dump(json_dataset, f, ensure_ascii=False, indent=4)


if __name__ == "__main__":

    # Data is assumed to be on the data folder (repo root) under i2b2/2014
    # train 1
    input_path1 = Path("../../data/i2b2/2014/training-PHI-Gold-Set1")
    output_path1 = Path("../../data/i2b2/2014/training-PHI-Gold-Set1.json")
    I2B22014Formatter.dataset_to_json(input_path1, output_path1)

    # train 2
    input_path2 = Path("../../data/i2b2/2014/training-PHI-Gold-Set2")
    output_path2 = Path("../../data/i2b2/2014/training-PHI-Gold-Set2.json")
    I2B22014Formatter.dataset_to_json(input_path2, output_path2)

    # test
    input_path3 = Path("../../data/i2b2/2014/testing-PHI-Gold-fixed")
    output_path3 = Path("../../data/i2b2/2014/testing-PHI-Gold-fixed.json")
    I2B22014Formatter.dataset_to_json(input_path3, output_path3)
