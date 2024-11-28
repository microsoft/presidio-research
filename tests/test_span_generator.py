import pytest
from faker import Faker
from faker.providers import BaseProvider

from presidio_evaluator.data_generator.faker_extensions import (
    SpanGenerator,
    FakerSpan,
    FakerSpansResult,
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
def span_faker(test_provider):
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
        ("my name is {{FOO}}", "my name is bar")
    ],
)
def test_one_replacement(span_faker, pattern, expected):

    res = span_faker.parse(pattern, add_spans=True)

    assert str(res) == expected
    assert res.fake == expected
    assert res.spans[0].start == expected.index("bar")
    assert res.spans[0].end == len(expected)
    assert res.spans[0].value == "bar"


def test_multiple_replacements(span_faker):
    pattern = "{{foo}} and then {{foo2}}, {{  foofoofoo  }} and finally {{foo3}}."
    expected = "bar and then barbar, bar and finally barbarbar."
    expected_spans = [
        FakerSpan(value="bar", start=0, end=3, type="foo"),
        FakerSpan(value="barbar", start=13, end=19, type="foo2"),
        FakerSpan(value="bar", start=21, end=24, type="foofoofoo"),
        FakerSpan(value="barbarbar", start=37, end=46, type="foo3"),
    ]

    res = span_faker.parse(pattern, add_spans=True)

    actual_spans = sorted(res.spans, key=lambda x: x.start)

    assert str(res) == expected
    assert res.fake == expected
    for expected, actual in zip(expected_spans, actual_spans):
        assert expected == actual


def test_spans_result_repr():
    sr = FakerSpansResult(fake="momo", spans=[FakerSpan("momo", 0, 4, type="name")])
    expected = (
        '{"fake": "momo", '
        '"spans": [{"value": "momo", "start": 0, "end": 4, "type": "name"}],'
        ' "template": null, '
        '"template_id": null, '
        '"sample_id": null}'
    )

    assert sr.__repr__() == expected


def test_no_replacements(span_faker):
    pattern = "this is a sentence with no fields"

    res = span_faker.parse(pattern, add_spans=True)

    assert str(res) == pattern
    assert len(res.spans) == 0


def test_without_spans(span_faker):
    pattern = "this is a sentence with {{foo}}"
    expected = "this is a sentence with bar"
    res = span_faker.parse(pattern)

    assert type(res) == str
    assert res == expected


def test_generated_text_contains_spans_text(span_faker):
    pattern = "My name is {{name}} and i live in {{address}}."

    res = span_faker.parse(pattern, add_spans=True)

    for span in res.spans:
        assert span.value in res.fake


@pytest.mark.parametrize(
    "pattern, non_element_text",
    [
        ("{{name}} My name is {{name}}", " My name is "),
        ("a b {{name}}{{name}}{{name}}", "a b "),
        ("...{{name}}{{name}} {{name}}...", "... ..."),
    ],
)
def test_generated_text_duplicate_types_returns_different_results(
    span_faker, pattern, non_element_text
):

    res = span_faker.parse(pattern, add_spans=True)

    # assert that span values exist in the text
    for span in res.spans:
        assert span.value in res.fake

    # assert that the non-element text is identical
    substring_indices = list(range(len(res.fake)))
    for span in res.spans:
        substring_indices = [
            ind for ind in substring_indices if ind not in range(span.start, span.end)
        ]

    actual_non_element_text = "".join(
        [res.fake[i] for i in range(len(res.fake)) if i in substring_indices]
    )
    assert actual_non_element_text == non_element_text

    # assert that names are different from each other
    for i in range(len(res.spans)):
        for j in range(i + 1, len(res.spans)):
            assert res.spans[i].value != res.spans[j].value
            assert res.spans[i].start != res.spans[j].start
            assert res.spans[i].end != res.spans[j].end
