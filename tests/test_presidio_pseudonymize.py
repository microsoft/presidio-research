import pytest
from faker import Faker
from faker.providers import DynamicProvider
from presidio_analyzer import RecognizerResult

from presidio_evaluator.data_generator import PresidioPseudonymization


@pytest.fixture(scope="session")
def fake_faker():

    faker = Faker()
    person_provider = DynamicProvider("PERSON", ["James"])
    location_provider = DynamicProvider("LOCATION", ["Africa"])
    faker.add_provider(person_provider)
    faker.add_provider(location_provider)

    return faker


@pytest.mark.parametrize(
    # fmt: off
    "text, entity1, entity2, start1, end1, start2, end2, value1, value2",
    [
        (
            "Hi I live in South Africa and my name is Toma",
            "LOCATION", "PERSON", 13, 25, 41, 45, "Africa", "James"
        ),
        ("Africa is my continent, James", "LOCATION", "PERSON", 0, 6, 24, 29, "Africa", "James"),
    ],
    # fmt: on
)
def test_presidio_pseudonymize_two_entities(
    text, entity1, entity2, start1, end1, start2, end2, value1, value2, fake_faker
):

    presidio_response = [
        RecognizerResult(entity_type=entity1, start=start1, end=end1, score=0.85),
        RecognizerResult(entity_type=entity2, start=start2, end=end2, score=0.85),
    ]
    presidio_pseudonymizer = PresidioPseudonymization(
        custom_faker=fake_faker, lower_case_ratio=0.0, map_to_presidio_entities=False
    )
    pseudonyms = presidio_pseudonymizer.pseudonymize(
        original_text=text, presidio_response=presidio_response, count=5
    )

    assert len(pseudonyms) == 5
    for pseudonym in pseudonyms:
        assert value1 in pseudonym
        assert value2 in pseudonym
        assert text[:start1].lower() in pseudonym.lower()
        assert text[end1:start2].lower() in pseudonym.lower()


def test_simple_scenario():
    original_text = "Hi my name is Doug Funny and this is my website: https://www.dougf.io" # noqa
    presidio_response = [
        RecognizerResult(entity_type="PERSON", start=14, end=24, score=0.85),
        RecognizerResult(entity_type="URL", start=49, end=69, score=0.95),
    ]

    PresidioPseudonymization().pseudonymize(original_text=original_text,
                                            presidio_response=presidio_response,
                                            count=5)
