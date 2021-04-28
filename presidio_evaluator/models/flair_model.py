from typing import List, Optional, Dict

import spacy

try:
    from flair.data import Sentence
    from flair.models import SequenceTagger
    from flair.tokenization import SpacyTokenizer
except ImportError:
    print("Flair is not installed by default")

from presidio_evaluator.data_objects import PRESIDIO_SPACY_ENTITIES
from presidio_evaluator import InputSample
from presidio_evaluator.models import BaseModel


class FlairModel(BaseModel):
    def __init__(
        self,
        model=None,
        model_path: str = None,
        entities_to_keep: List[str] = None,
        verbose: bool = False,
    ):
        """
        Evaluator for Flair models
        :param model: model of type SequenceTagger
        :param model_path:
        :param entities_to_keep:
        :param verbose:
        and model expected entity types
        """
        super().__init__(
            entities_to_keep=entities_to_keep,
            verbose=verbose,
        )
        if model is None:
            if model_path is None:
                raise ValueError("Either model_path or model object must be supplied")
            self.model = SequenceTagger.load(model_path)
        else:
            self.model = model

        self.spacy_tokenizer = SpacyTokenizer(model=spacy.load("en_core_web_lg"))

    def predict(self, sample: InputSample) -> List[str]:

        sentence = Sentence(text=sample.full_text, use_tokenizer=self.spacy_tokenizer)
        self.model.predict(sentence)

        tags = self.get_tags_from_sentence(sentence)
        if len(tags) != len(sample.tokens):
            print("mismatch between previous tokens and new tokens")

        if self.entities:
            tags = [tag for tag in tags if tag in self.entities]

        return tags

    @staticmethod
    def get_tags_from_sentence(sentence):
        tags = []
        for token in sentence:
            tags.append(token.get_tag("ner").value)

        new_tags = []
        for tag in tags:
            new_tags.append("PERSON" if tag == "PER" else tag)

        return new_tags
