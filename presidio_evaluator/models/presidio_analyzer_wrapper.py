from typing import List, Optional, Dict

from presidio_analyzer import AnalyzerEngine, EntityRecognizer, BatchAnalyzerEngine
from presidio_anonymizer import RecognizerResult

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
        ad_hoc_recognizers: Optional[List[EntityRecognizer]] = None,
        context: Optional[List[str]] = None,
        allow_list: Optional[List[str]] = None,
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
        self.ad_hoc_recognizers = ad_hoc_recognizers
        self.context = context
        self.allow_list = allow_list

        if not analyzer_engine:
            analyzer_engine = AnalyzerEngine()

        self.analyzer_engine = analyzer_engine

        self.print_discrepancies()

    def predict(self, sample: InputSample, **kwargs) -> List[str]:
        self.__update_kwargs(kwargs)

        results = self.analyzer_engine.analyze(
            text=sample.full_text,
            **kwargs,
        )
        response_tags = self.__recognizer_results_to_tags(results, sample)
        return response_tags

    def batch_predict(self, dataset: List[InputSample], **kwargs) -> List[List[str]]:
        self.__update_kwargs(kwargs)
        texts = [sample.full_text for sample in dataset]
        batch_analyzer = BatchAnalyzerEngine(analyzer_engine=self.analyzer_engine)
        analyzer_results = batch_analyzer.analyze_iterator(texts=texts, **kwargs)

        predictions = []
        for prediction, sample in zip(analyzer_results, dataset):
            predictions.append(self.__recognizer_results_to_tags(prediction, sample))

        return predictions

    @staticmethod
    def __recognizer_results_to_tags(
        results: List[RecognizerResult], sample: InputSample
    ) -> List[str]:
        starts = []
        ends = []
        scores = []
        tags = []
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

    def __update_kwargs(self, kwargs):
        kwargs["language"] = kwargs.get("language", self.language)
        kwargs["score_threshold"] = kwargs.get("score_threshold", self.score_threshold)
        kwargs["ad_hoc_recognizers"] = kwargs.get(
            "ad_hoc_recognizers", self.ad_hoc_recognizers
        )
        kwargs["context"] = kwargs.get("context", self.context)
        kwargs["allow_list"] = kwargs.get("allow_list", self.allow_list)
        kwargs["entities"] = kwargs.get("entities", self.entities)

    # Mapping between dataset entities and Presidio entities. Key: Dataset entity, Value: Presidio entity
    presidio_entities_map = dict(
        # Names
        PER="PERSON",
        PERSON="PERSON",
        FIRST_NAME="PERSON",
        LAST_NAME="PERSON",
        PATIENT="PERSON",
        STAFF="PERSON",
        HCW="PERSON",
        # Locations, GPE
        LOC="LOCATION",
        LOCATION="LOCATION",
        GPE="LOCATION",
        FACILITY="LOCATION",
        CITY="LOCATION",
        ADDRESS="LOCATION",
        STREET_ADDRESS="LOCATION",
        NATIONALITY="LOCATION",
        ZIP="ZIP_CODE",
        ZIP_CODE="ZIP_CODE",
        # Organizations, norps
        ORG="ORGANIZATION",
        ORGANIZATION="ORGANIZATION",
        VENDOR="ORGANIZATION",
        NORP="NRP",
        NRP="NRP",
        HOSP="ORGANIZATION",
        PATORG="ORGANIZATION",
        HOSPITAL="ORGANIZATION",
        # Generic
        AGE="AGE",
        ID="ID",
        TITLE="TITLE",
        PREFIX="TITLE",
        # Financial
        CREDIT_CARD="CREDIT_CARD",
        IBAN_CODE="IBAN_CODE",
        IBAN="IBAN_CODE",
        # Dates, times, birthdays
        DATE="DATE_TIME",
        TIME="DATE_TIME",
        DATE_TIME="DATE_TIME",
        BIRTHDAY="DATE_TIME",
        DATE_OF_BIRTH="DATE_TIME",
        DOB="DATE_TIME",
        PHONE="PHONE_NUMBER",
        PHONE_NUMBER="PHONE_NUMBER",
        # Internet
        DOMAIN_NAME="URL",
        URL="URL",
        DOMAIN="URL",
        EMAIL="EMAIL_ADDRESS",
        EMAIL_ADDRESS="EMAIL_ADDRESS",
        IP_ADDRESS="IP_ADDRESS",
        # US
        SSN="US_SSN",
        US_SSN="US_SSN",
        US_DRIVER_LICENSE="US_DRIVER_LICENSE",
        O="O",
    )

    def print_discrepancies(self):
        supported_entities = self.analyzer_engine.get_supported_entities(
            language=self.language
        )

        if not self.entities:
            self.entities = supported_entities

        for entity in self.entities:
            if entity not in supported_entities:
                print(
                    f"Warning: Entity {entity} is not supported by this instance of Presidio Analyzer Engine"
                )
        print("--------")
        print("Entities supported by this Presidio Analyzer instance:")
        print(", ".join(supported_entities))

    def _update_recognizers_based_on_entities_to_keep(self):
        """Add ORGANIZATION as it is removed by default."""

        supported_entities = self.analyzer_engine.get_supported_entities(
            language=self.language
        )

        if "ORGANIZATION" in self.entities and "ORGANIZATION" not in supported_entities:
            recognizers = self.analyzer_engine.get_recognizers()
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
