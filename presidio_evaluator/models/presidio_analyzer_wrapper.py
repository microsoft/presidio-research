from typing import List, Optional, Dict

from presidio_analyzer import AnalyzerEngine

from presidio_evaluator import InputSample, span_to_tag
from presidio_evaluator.models import BaseModel


class PresidioAnalyzerWrapper(BaseModel):
    def __init__(
        self,
        analyzer_engine: Optional[AnalyzerEngine] = None,
        entities_to_keep: List[str] = None,
        verbose: bool = False,
        labeling_scheme: str = "BIO",
        score_threshold: float = 0.4,
        language: str = "en",
        entity_mapping: Optional[Dict[str, str]] = None,
    ):
        """
        Evaluation wrapper for the Presidio Analyzer
        :param analyzer_engine: object of type AnalyzerEngine (from presidio-analyzer)
        """
        super().__init__(
            entities_to_keep=entities_to_keep,
            verbose=verbose,
            labeling_scheme=labeling_scheme,
            entity_mapping=entity_mapping,
        )
        self.score_threshold = score_threshold
        self.language = language

        if not analyzer_engine:
            analyzer_engine = AnalyzerEngine()
            self._update_recognizers_based_on_entities_to_keep(analyzer_engine)
        self.analyzer_engine = analyzer_engine

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
            scheme="IO",
            text=sample.full_text,
            starts=starts,
            ends=ends,
            tokens=sample.tokens,
            scores=scores,
            tags=tags,
        )
        return response_tags

    # Mapping between dataset entities and Presidio entities. Key: Dataset entity, Value: Presidio entity
    presidio_entities_map = {
        "PERSON": "PERSON",
        "GPE": "LOCATION",
        "EMAIL_ADDRESS": "EMAIL_ADDRESS",
        "CREDIT_CARD": "CREDIT_CARD",
        "FIRST_NAME": "PERSON",
        "LAST_NAME": "PERSON",
        "PHONE_NUMBER": "PHONE_NUMBER",
        "BIRTHDAY": "DATE_TIME",
        "DATE_TIME": "DATE_TIME",
        "DOMAIN_NAME": "DOMAIN_NAME",
        "CITY": "LOCATION",
        "ADDRESS": "LOCATION",
        "NATIONALITY": "LOCATION",
        "LOCATION": "LOCATION",
        "IBAN_CODE": "IBAN_CODE",
        "URL": "DOMAIN_NAME",
        "US_SSN": "US_SSN",
        "IP_ADDRESS": "IP_ADDRESS",
        "ORGANIZATION": "ORG",
        "US_DRIVER_LICENSE": "US_DRIVER_LICENSE",
        "NRP": "NRP",
        "TITLE": "O",  # not supported
        "PREFIX": "O",  # not supported
        "STREET_ADDRESS": "O",  # not supported
        "ZIP_CODE": "O",  # not supported
        "AGE": "O",  # not supported
        "O": "O",
    }

    def _update_recognizers_based_on_entities_to_keep(
        self, analyzer_engine: AnalyzerEngine
    ):
        """Check if there are any entities not supported by this presidio instance.
        Add ORGANIZATION as it is removed by default

        """
        supported_entities = analyzer_engine.get_supported_entities(
            language=self.language
        )
        print("Entities supported by this Presidio Analyzer instance:")
        print(", ".join(supported_entities))

        if not self.entities:
            self.entities = supported_entities

        for entity in self.entities:
            if entity not in supported_entities:
                print(
                    f"Entity {entity} is not supported by this instance of Presidio Analyzer Engine"
                )

        if "ORGANIZATION" in self.entities and "ORGANIZATION" not in supported_entities:
            recognizers = analyzer_engine.get_recognizers()
            spacy_recognizer = [
                rec
                for rec in recognizers
                if rec.name == "SpacyRecognizer" or rec.name == "StanzaRecognizer"
            ]
            if len(spacy_recognizer):
                spacy_recognizer = spacy_recognizer[0]
                spacy_recognizer.supported_entities.append("ORGANIZATION")
                self.entities.append("ORGANIZATION")
                print("Added ORGANIZATION as a supported entity from spaCy/Stanza")
