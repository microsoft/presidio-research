import dataclasses
import json
import re
from dataclasses import dataclass
from typing import List, Dict

from faker import Generator
from presidio_anonymizer.entities import OperatorConfig

_re_token = re.compile(r"\{\{\s*(\w+)(:\s*\w+?)?\s*\}\}")


@dataclass(eq=True)
class Span:
    value: str
    start: int
    end: int

    def __repr__(self):
        return json.dumps(dataclasses.asdict(self))



@dataclass()
class SpansResult:
    fake: str
    spans: List[Span]

    def __str__(self):
        return self.fake

    def __repr__(self):
        spans_dict = json.dumps([dataclasses.asdict(span) for span in self.spans])
        return json.dumps({"fake":self.fake, "spans": spans_dict})


class SpanGenerator(Generator):
    """Generator which also returns the indices of fake values.

    :example:
    >>>from faker import Faker
    >>>from presidio_evaluator.data_generator.faker_extensions import SpanGenerator

    >>>generator = SpanGenerator()
    >>>faker = Faker(generator=generator)
    >>>res = faker.address()

    >>>res.spans
    [{"value": "84272", "start": 36, "end": 41},
    {"value": "ME", "start": 33, "end": 35},
    {"value": "East Destiny", "start": 19, "end": 31},
    {"value": "52883 Murray Views", "start": 0, "end": 18}]

    >>>res.fake
    '0233 Nielsen Falls\nKellyborough, DC 81152'

    """

    def parse(self, text) -> SpansResult:
        fake = super().parse(text)
        original_spans = self._match_to_span(text)

        # Reverse for easier index handling while replacing
        original_spans = sorted(original_spans, reverse=True, key=lambda x: x.start)

        new_spans = []
        for span in original_spans:
            old_len = len(span.value) + 4  # adding two curly brackets

            formatted = str(self.format(span.value.strip()))
            new_len = len(formatted)
            start = span.start
            delta = new_len - old_len
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
