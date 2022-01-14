from abc import ABC, abstractmethod
from typing import List, Dict, Optional

from presidio_evaluator import InputSample, io_to_scheme


class BaseModel(ABC):
    def __init__(
        self,
        labeling_scheme: str = "BIO",
        entities_to_keep: List[str] = None,
        entity_mapping: Optional[Dict[str, str]] = None,
        verbose: bool = False,
    ):

        """
        Abstract class for evaluating NER models and others
        :param entities_to_keep: Which entities should be evaluated? All other
        entities are ignored. If None, none are filtered
        :param labeling_scheme: Used to translate (if needed)
        the prediction to a specific scheme (IO, BIO/IOB, BILUO)
        :param entity_mapping: Dictionary for mapping this model's input and output with the expected.
        Keys should be the input entity types (from the input dataset),
        values should be the model's supported entity types.
        :param verbose: Whether to print more debug info


        """
        self.entities = entities_to_keep
        self.labeling_scheme = labeling_scheme
        self.entity_mapping = entity_mapping
        self.verbose = verbose

    @abstractmethod
    def predict(self, sample: InputSample) -> List[str]:
        """
        Abstract. Returns the predicted tokens/spans from the evaluated model
        :param sample: Sample to be evaluated
        :return: List of tags in self.labeling_scheme format
        """
        pass

    def align_entity_types(self, sample: InputSample) -> None:
        """
        Translates the sample's tags to the ones requested by the model
        :param sample: Input sample
        :return: None
        """
        if self.entity_mapping:
            sample.translate_input_sample_tags(dictionary=self.entity_mapping)

    def align_prediction_types(self, tags: List[str]) -> List[str]:
        """
        Turns the model's output from the model tags to the input tags.
        :param tags: List of tags (entity names in IO or "O")
        :return: New tags
        """
        if not self.entity_mapping:
            return tags

        inverse_mapping = {v: k for k, v in self.entity_mapping.items()}
        new_tags = [
            InputSample.translate_tag(
                tag, dictionary=inverse_mapping, ignore_unknown=True
            )
            for tag in tags
        ]
        return new_tags

    def filter_tags_in_supported_entities(self, tags: List[str]) -> List[str]:
        """
        Replaces tags of unwanted entities with O.
        :param tags: Lits of tags
        :return: List of tags where tags not in self.entities are considered "O"
        """
        if not self.entities:
            return tags
        return [tag if self._tag_in_entities(tag) else "O" for tag in tags]

    def to_scheme(self, tags: List[str]):
        """
        Translates IO tags to BIO/BILUO based on the input labeling_scheme
        :param tags: Current tags in IO
        :return: Tags in labeling scheme
        """

        io_tags = [self._to_io(tag) for tag in tags]

        return io_to_scheme(io_tags=io_tags, scheme=self.labeling_scheme)

    @staticmethod
    def _to_io(tag):
        if "-" in tag:
            return tag[2:]
        return tag

    def to_log(self) -> Dict:
        """
        Returns a dictionary of parameters for logging purposes.
        :return:
        """
        return {
            "labeling_scheme": self.labeling_scheme,
            "entities_to_keep": self.entities,
        }

    def _tag_in_entities(self, tag: str):
        if not self.entities:
            return True

        if tag == "O":
            return True

        if tag[1] == "-":  # BIO/BILUO
            return tag[2:] in self.entities
        else:  # IO
            return tag in self.entities
