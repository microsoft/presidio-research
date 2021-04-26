from typing import List

from presidio_analyzer import AnalyzerEngine

from presidio_evaluator import InputSample, span_to_tag
from presidio_evaluator.models import BaseModel


class PresidioAnalyzerWrapper(BaseModel):
    def __init__(
        self,
        analyzer_engine=AnalyzerEngine(),
        entities_to_keep: List[str] = None,
        verbose: bool = False,
        labeling_scheme="BIO",
        score_threshold=0.4,
    ):
        """
        Evaluation wrapper for the Presidio Analyzer
        :param analyzer_engine: object of type AnalyzerEngine (from presidio-analyzer)
        """
        super().__init__(
            entities_to_keep=entities_to_keep,
            verbose=verbose,
            labeling_scheme=labeling_scheme,
        )
        self.analyzer_engine = analyzer_engine

        self.score_threshold = score_threshold

    def predict(self, sample: InputSample) -> List[str]:

        results = self.analyzer_engine.analyze(
            text=sample.full_text,
            entities=self.entities,
            language="en",
            score_threshold=self.score_threshold,
        )
        starts = []
        ends = []
        scores = []
        tags = []
        #
        for res in results:
            starts.append(res.start)
            ends.append(res.end)
            tags.append(res.entity_type)
            scores.append(res.score)

        response_tags = span_to_tag(
            scheme=self.labeling_scheme,
            text=sample.full_text,
            start=starts,
            end=ends,
            tokens=sample.tokens,
            scores=scores,
            tag=tags,
        )
        return response_tags

    # Mapping between dataset entities and Presidio entities. Key: Dataset entity, Value: Presidio entity
    presidio_entities_map = {
        "PERSON": "PERSON",
        "EMAIL_ADDRESS": "EMAIL_ADDRESS",
        "CREDIT_CARD": "CREDIT_CARD",
        "FIRST_NAME": "PERSON",
        "PHONE_NUMBER": "PHONE_NUMBER",
        "BIRTHDAY": "DATE_TIME",
        "DATE_TIME": "DATE_TIME",
        "DOMAIN": "DOMAIN",
        "CITY": "LOCATION",
        "ADDRESS": "LOCATION",
        "NATIONALITY": "LOCATION",
        "IBAN": "IBAN_CODE",
        "URL": "DOMAIN_NAME",
        "US_SSN": "US_SSN",
        "IP_ADDRESS": "IP_ADDRESS",
        "ORGANIZATION": "ORG",
        "O": "O",
    }
