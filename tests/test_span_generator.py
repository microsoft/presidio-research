import pytest
from faker import Faker
from faker.providers import BaseProvider

from presidio_evaluator.data_generator.faker_extensions import (
    SpanGenerator,
    Span,
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
    faker = Faker(generator=generator)
    faker.add_provider(test_provider)

    return faker


@pytest.mark.parametrize(
    "pattern, expected",
    [
        ("My name is {{foo}}", "My name is bar"),
        ("My name is {{  foo   }}", "My name is bar"),
        ( "my name is {{foofoofoo}}","my name is bar")
    ],
)
def test_one_replacement(faker, pattern, expected):

    res = faker.parse(pattern)

    assert str(res) == expected
    assert res.fake == expected
    assert res.spans[0].start == expected.index("bar")
    assert res.spans[0].end == len(expected)
    assert res.spans[0].value == "bar"



def test_multiple_replacements(faker):
    pattern = "{{foo}} and then {{foo2}}, {{ foofoofoo }} and finally {{foo3}}."
    expected = "bar and then barbar and finally barbarbar."
    expected_spans = [
        Span(value="bar", start=0, end=3),
        Span(value="barbar", start=13, end=19),
        Span(value="bar", start=13, end=19),
        Span(value="barbarbar", start=32, end=41),
    ]

    res = faker.parse(pattern)

    actual_spans = sorted(res.spans, key=lambda x: x.start)

    assert str(res) == expected
    assert res.fake == expected
    for expected, actual in zip(expected_spans, actual_spans):
        assert expected == actual


def test_no_replacements(faker):
    pattern = "this is a sentence with no fields"

    res = faker.parse(pattern)

    assert str(res) == pattern
    assert len(res.spans) == 0
