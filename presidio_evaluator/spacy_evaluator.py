from typing import List

from presidio_evaluator import ModelEvaluator, InputSample
import spacy

from spacy.language import Language

from presidio_evaluator.data_objects import PRESIDIO_SPACY_ENTITIES


class SpacyEvaluator(ModelEvaluator):

    def __init__(self,
                 model: spacy.language.Language = None,
                 model_name: str = None,
                 entities_to_keep: List[str] = None,
                 verbose: bool = False,
                 labeling_scheme: str = "BIO",
                 compare_by_io: bool = True,
                 translate_to_spacy_ents = True):
        super().__init__(entities_to_keep=entities_to_keep,
                         verbose=verbose,
                         labeling_scheme=labeling_scheme,
                         compare_by_io=compare_by_io)

        if model is None:
            if model_name is None:
                raise ValueError("Either model_name or model object must be supplied")
            self.model = spacy.load(model_name)
        else:
            self.model = model

        self.translate_to_spacy_ents = translate_to_spacy_ents
        if self.translate_to_spacy_ents:
            print("Translating entites using this dictionary: {}".format(PRESIDIO_SPACY_ENTITIES))

    def predict(self, sample: InputSample) -> List[str]:
        if self.translate_to_spacy_ents:
            sample.translate_input_sample_tags()

        doc = self.model(sample.full_text)
        tags = self.get_tags_from_doc(doc)
        if len(doc) != len(sample.tokens):
            print("mismatch between input tokens and new tokens")

        return tags

    @staticmethod
    def get_tags_from_doc(doc):
        tags = [token.ent_type_ if token.ent_type_ != "" else "O" for token in doc]
        return tags

