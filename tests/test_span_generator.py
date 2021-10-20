import pytest
from faker import Faker
from faker.providers import BaseProvider

from presidio_evaluator.data_generator.faker_extensions import (
    SpanGenerator,
    Span,
    SpansResult,
)


@pytest.fixture(scope="session")
def test_provider():
    class TestProvider(BaseProvider):
        def foo(self) -> str:
            return "bar"

        def foofoofoo(self) -> str:
            return "bar"

        def foo2(self) -> str:
            return "barbar"

        def foo3(self) -> str:
            return "barbarbar"

    return TestProvider


@pytest.fixture(scope="session")
def faker(test_provider):
    generator = SpanGenerator()
    Faker.seed(42)
    faker = Faker(generator=generator)
    faker.add_provider(test_provider)

    return faker


@pytest.mark.parametrize(
    "pattern, expected",
    [
        ("My name is {{foo}}", "My name is bar"),
        ("My name is {{  foo   }}", "My name is bar"),
        ("my name is {{foofoofoo}}", "my name is bar"),
    ],
)
def test_one_replacement(faker, pattern, expected):

    res = faker.parse(pattern, add_spans=True)

    assert str(res) == expected
    assert res.fake == expected
    assert res.spans[0].start == expected.index("bar")
    assert res.spans[0].end == len(expected)
    assert res.spans[0].value == "bar"


def test_multiple_replacements(faker):
    pattern = "{{foo}} and then {{foo2}}, {{  foofoofoo  }} and finally {{foo3}}."
    expected = "bar and then barbar, bar and finally barbarbar."
    expected_spans = [
        Span(value="bar", start=0, end=3, type="foo"),
        Span(value="barbar", start=13, end=19, type="foo2"),
        Span(value="bar", start=21, end=24, type="foofoofoo"),
        Span(value="barbarbar", start=37, end=46, type="foo3"),
    ]

    res = faker.parse(pattern, add_spans=True)

    actual_spans = sorted(res.spans, key=lambda x: x.start)

    assert str(res) == expected
    assert res.fake == expected
    for expected, actual in zip(expected_spans, actual_spans):
        assert expected == actual


def test_spans_result_repr():
    sr = SpansResult(fake="momo", spans=[Span("momo", 0, 4, type="name")])
    expected = (
        '{"fake": "momo", "spans": "[{\\"value\\": \\"momo\\", '
        '\\"start\\": 0, '
        '\\"end\\": 4, '
        '\\"type\\": \\"name\\"}]"}'
    )

    assert sr.__repr__() == expected


def test_no_replacements(faker):
    pattern = "this is a sentence with no fields"

    res = faker.parse(pattern, add_spans=True)

    assert str(res) == pattern
    assert len(res.spans) == 0


def test_without_spans(faker):
    pattern = "this is a sentence with {{foo}}"
    expected = "this is a sentence with bar"
    res = faker.parse(pattern)

    assert type(res) == str
    assert res == expected


def test_generated_text_contains_spans_text(faker):
    pattern = "My name is {{name}} and i live in {{address}}."

    res = faker.parse(pattern, add_spans=True)

    for span in res.spans:
        assert span.value in res.fake
