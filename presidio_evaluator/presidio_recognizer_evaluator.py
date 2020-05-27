"""
Presidio Analyzer not yet on PyPI, therefore it cannot be referenced explicitly
"""

import math
from typing import List, Tuple, Dict

from presidio_analyzer.nlp_engine import SpacyNlpEngine

from presidio_evaluator import ModelEvaluator, InputSample, EvaluationResult
from presidio_evaluator.span_to_tag import span_to_tag


class PresidioRecognizerEvaluator(ModelEvaluator):
    def __init__(
        self,
        recognizer,
        nlp_engine,
        entities_to_keep=None,
        with_nlp_artifacts=False,
        verbose=False,
        compare_by_io=True,
    ):
        """
        Evaluator for one recognizer
        :param recognizer: An object of type EntityRecognizer (in presidion-analyzer)
        :param nlp_engine: An object of type NlpEngine, e.g. SpacyNlpEngine (in presidio-analyzer)
        """
        super().__init__(
            entities_to_keep=entities_to_keep,
            verbose=verbose,
            compare_by_io=compare_by_io,
        )
        self.withNlpArtifacts = with_nlp_artifacts
        self.recognizer = recognizer
        self.nlp_engine = nlp_engine

    #
    def __make_nlp_artifacts(self, text: str):
        return self.nlp_engine.process_text(text, "en")

    #
    def predict(self, sample: InputSample) -> List[str]:
        nlpArtifacts = None
        if self.withNlpArtifacts:
            nlpArtifacts = self.__make_nlp_artifacts(sample.full_text)
        results = self.recognizer.analyze(sample.full_text, self.entities, nlpArtifacts)
        starts = []
        ends = []
        tags = []
        scores = []
        for res in results:
            if not res.start:
                res.start = 0
            starts.append(res.start)
            ends.append(res.end)
            tags.append(res.entity_type)
            scores.append(res.score)
        response_tags = span_to_tag(
            scheme=self.labeling_scheme,
            text=sample.full_text,
            start=starts,
            end=ends,
            tag=tags,
            tokens=sample.tokens,
            scores=scores,
            io_tags_only=self.compare_by_io,
        )
        if len(sample.tags) == 0:
            sample.tags = ["0" for word in response_tags]
        return response_tags


def score_presidio_recognizer(
    recognizer, entities_to_keep, input_samples, withNlpArtifacts=False
) -> EvaluationResult:
    model = PresidioRecognizerEvaluator(
        recognizer=recognizer,
        entities_to_keep=entities_to_keep,
        nlp_engine=SpacyNlpEngine(),
        with_nlp_artifacts=withNlpArtifacts,
    )
    evaluated_samples = model.evaluate_all(input_samples[:])
    evaluation_result = model.calculate_score(evaluated_samples, beta=2.5)
    evaluation_result.print()
    if math.isnan(evaluation_result.pii_precision):
        evaluation_result.pii_precision = 0
    return evaluation_result
