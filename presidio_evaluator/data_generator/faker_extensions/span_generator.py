import dataclasses
import json
from dataclasses import dataclass
from typing import List, Union

from faker import Generator
from faker.generator import _re_token


@dataclass(eq=True)
class Span:
    """Span holds the start, end, value and type of every element replaced."""

    value: str
    start: int
    end: int
    type: str

    def __repr__(self):
        return json.dumps(dataclasses.asdict(self))


@dataclass()
class SpansResult:
    """SpanResult holds the full fake sentence
    and a list of spans for each element replaced."""

    fake: str
    spans: List[Span]

    def __str__(self):
        return self.fake

    def __repr__(self):
        spans_dict = json.dumps([dataclasses.asdict(span) for span in self.spans])
        return json.dumps({"fake": self.fake, "spans": spans_dict})


class SpanGenerator(Generator):
    """Generator which also returns the indices of fake values.

    :example:
    >>>from faker import Faker
    >>>from presidio_evaluator.data_generator.faker_extensions import SpanGenerator

    >>>generator = SpanGenerator()
    >>>faker = Faker(generator=generator)
    >>>res = faker.parse("My child's name is {{name}}", add_spans=True)

    >>>res.spans
        [{"value": "Daniel Gallagher", "start": 19, "end": 35, "type": "name"}]
    >>>res.fake
        "My child's name is Daniel Gallagher"
    >>>str(res)
        "My child's name is Daniel Gallagher"
    """

    def parse(self, text: str, add_spans: bool = False) -> Union[str, SpansResult]:
        if not add_spans:
            return super().parse(text)
        else:
            return self.parse_with_spans(text)

    def parse_with_spans(self, text: str) -> SpansResult:
        """Parses a Faker template and returns a `SpanResult` object.
        :param text: Text holding the faker template, e.g. "My name is {{name}}".
        """

        spans = self._match_to_span(text)

        # Reverse for easier index handling while replacing
        spans = sorted(spans, reverse=True, key=lambda x: x.start)

        for i, span in enumerate(spans):
            formatter = span.type
            old_len = len(formatter) + 4  # adding two curly brackets

            new_len = len(span.value)
            span.start = span.start
            delta = new_len - old_len
            span.end = span.end + delta
            span.type = formatter.strip()

            # Update previously inserted spans since indices shifted
            for j in range(0, i):
                spans[j].start += delta
                spans[j].end += delta

        before_after = dict([(span.type, span.value) for span in spans])
        # Create fake text using already sampled values
        fake_text = _re_token.sub(lambda mo: before_after[list(mo.groups())[0]], text)

        return SpansResult(fake=fake_text, spans=spans)

    def _match_to_span(self, text: str) -> List[Span]:
        matches = _re_token.finditer(text)

        results: List[Span] = []
        for match in matches:
            formatter = match.group()[2:-2]
            results.append(
                Span(
                    type=formatter,
                    start=match.start(),
                    end=match.end(),
                    value=super().format(formatter.strip()),
                )
            )

        return results
