import re
from dataclasses import dataclass
from typing import List, Dict

from faker import Generator
from presidio_anonymizer.entities import OperatorConfig

_re_token = re.compile(r"\{\{\s*(\w+)(:\s*\w+?)?\s*\}\}")


@dataclass(repr=True, order=True)
class Span:
    value: str
    start: int
    end: int


@dataclass(repr=True)
class SpansResult:
    fake: str
    spans: List[Span]

    def __str__(self):
        return self.fake


class SpanGenerator(Generator):
    """Generator which also returns the indices of fake values."""

    def parse(self, text) -> SpansResult:
        fake = super().parse(text)
        original_spans = self._match_to_span(text)

        # Reverse for easier index handling while replacing
        original_spans = sorted(original_spans, reverse=True, key=lambda x: x.start)

        new_spans = []
        for span in original_spans:
            prev_len = len(span.value) + 4  # adding two curly brackets

            formatted = str(self.format(span.value.strip()))
            new_len = len(formatted)
            start = span.start
            delta = new_len - prev_len
            end = span.end + delta

            # Update previously inserted spans
            for new_span in new_spans:
                new_span.start += delta
                new_span.end += delta

            new_spans.append(Span(value=formatted, start=start, end=end))

        return SpansResult(fake=fake, spans=new_spans)

    @staticmethod
    def _to_replace_operators(before_after_dict: Dict[str, str]):
        operators = {}
        for k, v in before_after_dict.items():
            operators[k] = OperatorConfig("replace", {"new_value": v})
        return operators

    @staticmethod
    def _match_to_span(text) -> List[Span]:
        matches = _re_token.finditer(text)

        results: List[Span] = []
        for match in matches:
            results.append(
                Span(
                    value=match.group()[2:-2],
                    start=match.start(),
                    end=match.end(),
                )
            )

        return results
