from abc import ABC, abstractmethod

import pandas as pd

from presidio_evaluator import InputSample, Span, span_to_tag, tags_to_span_ids
from presidio_evaluator.entity_mapping.data_objects import (
    ANNOTATION_SPAN_ID,
    PREDICTION_SPAN_ID,
)
from presidio_evaluator.span_to_tag import split_scheme_prefix

#: Columns of the DataFrame returned by :meth:`BaseModel.predict_dataset`.
RESULTS_COLUMNS = [
    "sentence_id",
    "token",
    "annotation",
    "prediction",
    "start_indices",
    ANNOTATION_SPAN_ID,
    PREDICTION_SPAN_ID,
]


class BaseModel(ABC):
    def __init__(
        self,
        labeling_scheme: str = "IO",
        entities_to_keep: list[str] | None = None,
        entity_mapping: dict[str, str] | None = None,
        verbose: bool = False,
    ) -> None:
        """
        Abstract class for evaluating NER models and others
        :param entities_to_keep: Which entities should be evaluated? All other
        entities are ignored. If None, none are filtered
        :param labeling_scheme: Scheme of the tags returned by ``predict`` /
        ``batch_predict`` (IO, BIO/IOB, BILUO)
        :param entity_mapping: DEPRECATED. This parameter is no longer supported.
        Entity mapping should be passed to the evaluator instead.
        :param verbose: Whether to print more debug info
        """

        if entity_mapping is not None:
            raise ValueError(
                "The 'entity_mapping' parameter is deprecated and has been removed.\n"
                "Entity mapping is now handled by CanonicalMapper before evaluation.\n"
                "See notebooks/4_Evaluate_Presidio_Analyzer.ipynb for examples.",
            )

        self.entities = entities_to_keep
        self.labeling_scheme = labeling_scheme
        self.verbose = verbose
        self.name = self.__class__.__name__

    @abstractmethod
    def predict(self, sample: InputSample, **kwargs) -> list[str]:
        """
        Abstract. Returns the predicted tokens/spans from the evaluated model
        :param sample: Sample to be evaluated
        :return: List of tags in self.labeling_scheme format
        """
        pass

    @abstractmethod
    def batch_predict(self, dataset: list[InputSample], **kwargs) -> list[list[str]]:
        """Perform batch prediction if the model supports it."""

    def batch_predict_spans(
        self, dataset: list[InputSample], **kwargs
    ) -> list[list[Span]]:
        """Predict entity spans per sample, in character offsets of ``sample.full_text``.

        The default derives spans from :meth:`batch_predict` tags, so tag-only models
        get label-run boundaries; override to return the model's own spans.
        """
        return [
            self._tags_to_spans(sample, tags)
            for sample, tags in zip(
                dataset, self.batch_predict(dataset, **kwargs), strict=True
            )
        ]

    def _spans_to_tags(self, sample: InputSample, spans: list[Span]) -> list[str]:
        """Flatten spans to one tag per token in ``self.labeling_scheme``."""
        return span_to_tag(
            scheme=self.labeling_scheme,
            text=sample.full_text,
            starts=[s.start_position for s in spans],
            ends=[s.end_position for s in spans],
            tags=[s.entity_type for s in spans],
            scores=[s.score if s.score is not None else 0.5 for s in spans],
            tokens=[str(t) for t in sample.tokens],
            start_indices=_token_offsets(sample),
        )

    @staticmethod
    def _tags_to_spans(sample: InputSample, tags: list[str]) -> list[Span]:
        """Group a sample's tokens into spans by entity instance id."""
        tags = _pad(tags, len(sample.tokens), "O")
        spans: list[Span] = []
        for token, tag, start, span_id in zip(
            sample.tokens,
            tags,
            _token_offsets(sample),
            tags_to_span_ids(tags),
            strict=True,
        ):
            if span_id is None:
                continue
            text = str(token)
            end = start + len(text)
            if spans and span_id == len(spans) - 1:
                current = spans[-1]
                current.entity_value = sample.full_text[current.start_position : end]
                current.end_position = end
            else:
                _, entity_type = split_scheme_prefix(str(tag))
                spans.append(
                    Span(
                        entity_type=entity_type,
                        entity_value=text,
                        start_position=start,
                        end_position=end,
                    )
                )
        return spans

    def predict_dataset(self, dataset: list[InputSample]) -> pd.DataFrame:
        """Predict a dataset into a per-token results DataFrame with ``RESULTS_COLUMNS``.

        Labels are left unmapped; the span-id columns carry entity boundaries from the
        source spans so span evaluation does not depend on label runs.
        """
        predictions = self.batch_predict_spans(dataset)

        columns: dict[str, list] = {name: [] for name in RESULTS_COLUMNS}
        for i, (sample, spans) in enumerate(zip(dataset, predictions, strict=True)):
            n = len(sample.tokens)
            start_indices = _token_offsets(sample)
            pred_tags, pred_ids = span_to_tag(
                scheme="IO",
                text=sample.full_text,
                starts=[s.start_position for s in spans],
                ends=[s.end_position for s in spans],
                tags=[s.entity_type for s in spans],
                scores=[s.score if s.score is not None else 0.5 for s in spans],
                tokens=[str(t) for t in sample.tokens],
                start_indices=start_indices,
                return_span_ids=True,
            )
            sentence_id = sample.sample_id if sample.sample_id is not None else i
            columns["sentence_id"] += [sentence_id] * n
            columns["token"] += [str(t) for t in sample.tokens]
            columns["annotation"] += _pad(sample.tags, n, "O")
            columns["prediction"] += _pad(pred_tags, n, "O")
            columns["start_indices"] += start_indices
            columns[ANNOTATION_SPAN_ID] += _pad(sample.span_ids, n, None)
            columns[PREDICTION_SPAN_ID] += _pad(pred_ids, n, None)

        df = pd.DataFrame({name: columns[name] for name in RESULTS_COLUMNS})
        # int + None would be inferred as float64 (0.0 / NaN); keep ids as
        # int / None so equality-based grouping works.
        for name in (ANNOTATION_SPAN_ID, PREDICTION_SPAN_ID):
            df[name] = pd.Series(columns[name], dtype="object")
        return df

    def to_log(self) -> dict:
        """
        Returns a dictionary of parameters for logging purposes.
        :return:
        """
        return {
            "labeling_scheme": self.labeling_scheme,
            "entities_to_keep": self.entities,
        }


def _pad(values, n: int, fill) -> list:
    values = list(values or [])[:n]
    return values + [fill] * (n - len(values))


def _token_offsets(sample: InputSample) -> list[int]:
    """Character offset of each token.

    Uses ``sample.start_indices`` when complete; otherwise locates each token in
    ``full_text`` left to right (samples built by hand, without tokenization).
    """
    if len(sample.start_indices or []) == len(sample.tokens):
        return list(sample.start_indices)
    offsets, cursor = [], 0
    for token in sample.tokens:
        text = str(token)
        found = sample.full_text.find(text, cursor)
        if found < 0:
            found = cursor
        offsets.append(found)
        cursor = found + len(text)
    return offsets
