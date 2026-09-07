from abc import ABC, abstractmethod

import pandas as pd

from presidio_evaluator import InputSample, io_to_scheme


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
        :param labeling_scheme: Used to translate (if needed)
        the prediction to a specific scheme (IO, BIO/IOB, BILUO)
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

    def predict_dataset(self, dataset: list[InputSample]) -> pd.DataFrame:
        """Predict entities for a dataset and return results as a DataFrame.

        Calls batch_predict() internally and assembles the result into a
        flat DataFrame.  No entity mapping is applied — that is the mapper's job.

        When at least one sample carries gold spans, an extra
        ``annotation_span_id`` column is attached: the index of the gold span
        covering each token (None for ``O`` tokens, and for tokens of samples
        without spans). Downstream span reconstruction uses it to recover exact
        gold span boundaries instead of inferring them from label runs.

        :param dataset: List of InputSample objects (must have tokens and tags set).
        :return: DataFrame with columns:
            sentence_id, token, annotation, prediction, start_indices
            [, annotation_span_id]
        """
        from presidio_evaluator.entity_mapping.data_objects import (  # noqa: PLC0415
            ANNOTATION_SPAN_ID,
        )

        predictions = self.batch_predict(dataset)
        has_spans = any(sample.spans for sample in dataset)

        rows: list[dict] = []
        span_ids: list[int | None] = []
        for i, (sample, pred_tags) in enumerate(
            zip(dataset, predictions, strict=False)
        ):
            sentence_id = sample.sample_id if sample.sample_id is not None else i
            tokens = sample.tokens
            annotations = sample.tags
            start_indices = sample.start_indices
            if has_spans:
                span_ids.extend(self._annotation_span_ids(sample))
            for j in range(len(tokens)):
                rows.append(
                    {
                        "sentence_id": sentence_id,
                        "token": str(tokens[j]),
                        "annotation": annotations[j] if j < len(annotations) else "O",
                        "prediction": pred_tags[j] if j < len(pred_tags) else "O",
                        "start_indices": start_indices[j]
                        if j < len(start_indices)
                        else 0,
                    },
                )

        df = pd.DataFrame(
            rows,
            columns=[
                "sentence_id",
                "token",
                "annotation",
                "prediction",
                "start_indices",
            ],
        )
        if has_spans:
            # object dtype keeps ids as int/None; a numeric column would turn
            # None into NaN, and NaN != NaN breaks equality-based grouping.
            df[ANNOTATION_SPAN_ID] = pd.Series(span_ids, dtype="object")
        return df

    @staticmethod
    def _annotation_span_ids(sample: InputSample) -> list[int | None]:
        """Index of the gold span covering each token, None for O tokens.

        Ids follow the labels in ``sample.tags``, which span_to_tag already
        aligned to the tokens: a token labelled O gets None even if a span
        overlaps it, and when several spans overlap a token the one whose
        entity type matches the token's label wins, so a token's label and its
        span id never disagree.
        """
        spans = sample.spans or []
        tags = sample.tags
        start_indices = sample.start_indices
        ids: list[int | None] = []
        for j, token in enumerate(sample.tokens):
            label = tags[j] if j < len(tags) else "O"
            start = start_indices[j] if j < len(start_indices) else None
            if label == "O" or not spans or start is None:
                ids.append(None)
                continue
            end = start + len(str(token))
            base_label = (
                label.split("-", 1)[1]
                if label[:2] in {"B-", "I-", "L-", "U-"}
                else label
            )
            overlapping = [
                i
                for i, span in enumerate(spans)
                if span.start_position < end and start < span.end_position
            ]
            matching = [
                i for i in overlapping if spans[i].entity_type == base_label
            ]
            candidates = matching or overlapping
            ids.append(candidates[0] if candidates else None)
        return ids

    def filter_tags_in_supported_entities(self, tags: list[str]) -> list[str]:
        """
        Replaces tags of unwanted entities with O.
        :param tags: Lits of tags
        :return: List of tags where tags not in self.entities are considered "O"
        """
        if not self.entities:
            return tags
        return [tag if self._tag_in_entities(tag) else "O" for tag in tags]

    def to_scheme(self, tags: list[str]):
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

    def to_log(self) -> dict:
        """
        Returns a dictionary of parameters for logging purposes.
        :return:
        """
        return {
            "labeling_scheme": self.labeling_scheme,
            "entities_to_keep": self.entities,
        }

    def _tag_in_entities(self, tag: str) -> bool:
        """True if the tag is in the entities to keep."""

        if not self.entities:
            return True

        if tag == "O":
            return True

        if tag[1] == "-":  # BIO/BILUO
            return tag[2:] in self.entities
        else:  # IO
            return tag in self.entities
