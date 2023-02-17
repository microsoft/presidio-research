from typing import List, Optional, Dict

from presidio_analyzer import AnalyzerEngine

from presidio_evaluator import InputSample, Span
from presidio_evaluator.models_2 import BaseModel
from presidio_evaluator.evaluator_2 import ModelPrediction


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

        if analyzer_engine:
            assert language in analyzer_engine.supported_languages, \
                f'Provided analyzer supports {analyzer_engine.supported_languages} ' \
                f'which does not include {language}'
        else:
            analyzer_engine = AnalyzerEngine(supported_languages=[language])
        self.language = language
        self._update_recognizers_based_on_entities_to_keep(analyzer_engine)
        self.analyzer_engine = analyzer_engine

    def predict(self, sample: InputSample) -> ModelPrediction:
        """
        Returns the predicted tokens/spans from the evaluated model
        :param sample: Sample to be evaluated
        :return: ModelPrediction object
        """
        results = self.analyzer_engine.analyze(
            text=sample.full_text,
            entities=self.entities,
            language=self.language,
            score_threshold=self.score_threshold,
        )
        # Convert Presidio results to the List of Span format
        response_spans = []

        for res in results:
            # Create output for span-level evaluation
            predict_span = Span(entity_type=res.entity_type,
                                start_position=res.start,
                                end_position=res.end,
                                entity_value=sample.full_text[res.start:res.end],
                                score=res.score)
            response_spans.append(predict_span)

        return ModelPrediction(
            input_sample=sample,
            predicted_spans=response_spans
        )

    # Mapping between dataset entities and Presidio entities.
    # Key: Dataset entity, Value: Presidio entity
    presidio_entities_map = dict(PERSON="PERSON",
                                 GPE="LOCATION",
                                 EMAIL_ADDRESS="EMAIL_ADDRESS",
                                 CREDIT_CARD="CREDIT_CARD",
                                 FIRST_NAME="PERSON",
                                 LAST_NAME="PERSON",
                                 PHONE_NUMBER="PHONE_NUMBER",
                                 BIRTHDAY="DATE_TIME",
                                 DATE_TIME="DATE_TIME",
                                 DOMAIN_NAME="URL",
                                 CITY="LOCATION",
                                 ADDRESS="LOCATION",
                                 NATIONALITY="NRP",
                                 LOCATION="LOCATION",
                                 IBAN_CODE="IBAN_CODE",
                                 US_SSN="US_SSN",
                                 IP_ADDRESS="IP_ADDRESS",
                                 ORGANIZATION="O",
                                 US_DRIVER_LICENSE="US_DRIVER_LICENSE",
                                 NRP="NRP",
                                 TITLE="O",  # not supported
                                 PREFIX="O",  # not supported
                                 STREET_ADDRESS="O",  # not supported
                                 ZIP_CODE="O",  # not supported
                                 AGE="O",  # not supported
                                 O="O")

    def _update_recognizers_based_on_entities_to_keep(
            self, analyzer_engine: AnalyzerEngine
    ):
        """Check if there are any entities not supported by this presidio instance.
        Add ORGANIZATION as it is removed by default
        """
        supported_entities = analyzer_engine.get_supported_entities(
            language=self.language)
        print("Entities supported by this Presidio Analyzer instance:")
        print(", ".join(supported_entities))

        if not self.entities:
            self.entities = supported_entities
        for entity in self.entities:
            if entity not in supported_entities:
                print(
                    f"Entity {entity} is not supported by "
                    f"this instance of Presidio Analyzer Engine")

        if "ORGANIZATION" in self.entities and "ORGANIZATION" not in supported_entities:
            recognizers = analyzer_engine.get_recognizers()
            spacy_recognizer = [rec for rec in recognizers
                                if
                                rec.name == "SpacyRecognizer"
                                or rec.name == "StanzaRecognizer"]
            if len(spacy_recognizer):
                spacy_recognizer = spacy_recognizer[0]
                spacy_recognizer.supported_entities.append("ORGANIZATION")
                self.entities.append("ORGANIZATION")
                print("Added ORGANIZATION as a supported entity from spaCy/Stanza")
