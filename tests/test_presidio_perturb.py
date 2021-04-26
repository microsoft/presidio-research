import pytest
from presidio_analyzer import RecognizerResult

from presidio_evaluator.data_generator.presidio_perturb import PresidioPerturb
from tests import get_mock_fake_df

import pandas as pd


@pytest.mark.parametrize(
    # fmt: off
    "text, entity1, entity2, start1, end1, start2, end2",
    [
        (
            "Hi I live in South Africa and my name is Toma",
            "LOCATION", "PERSON", 13, 25, 41, 45,
        ),
        ("Africa is my continent, James", "LOCATION", "PERSON", 0, 6, 24, 29,),
    ],
    # fmt: on
)
def test_presidio_perturb_two_entities(
    text, entity1, entity2, start1, end1, start2, end2
):

    presidio_response = [
        RecognizerResult(entity_type=entity1, start=start1, end=end1, score=0.85),
        RecognizerResult(entity_type=entity2, start=start2, end=end2, score=0.85),
    ]
    presidio_perturb = PresidioPerturb(fake_pii_df=get_mock_fake_df())
    fake_df = presidio_perturb.fake_pii
    perturbations = presidio_perturb.perturb(
        original_text=text, presidio_response=presidio_response, count=5
    )

    assert len(perturbations) == 5
    for perturbation in perturbations:
        assert fake_df[entity1].str.lower()[0] in perturbation.lower()
        assert fake_df[entity2].str.lower()[0] in perturbation.lower()
        assert text[:start1].lower() in perturbation.lower()
        assert text[end1:start2].lower() in perturbation.lower()


def test_entity_translation():
    text = "My email is email@email.com"

    presidio_response = [
        RecognizerResult(entity_type="EMAIL_ADDRESS", start=12, end=27, score=0.5)
    ]

    presidio_perturb = PresidioPerturb(fake_pii_df=get_mock_fake_df())
    fake_df = presidio_perturb.fake_pii
    perturbations = presidio_perturb.perturb(
        original_text=text, presidio_response=presidio_response, count=1
    )

    assert fake_df["EMAIL_ADDRESS"].str.lower()[0] in perturbations[0]


def test_subset_perturbation():
    text = "My name is Dan"
    presidio_response = [
        RecognizerResult(entity_type="PERSON", start=11, end=14, score=0.5)
    ]

    fake_df = pd.DataFrame(
        {
            "FIRST_NAME": ["Neta", "George"],
            "LAST_NAME": ["Levy", "Harrison"],
            "GENDER": ["Female", "Male"],
            "NameSet": ["Hebrew", "English"],
        }
    )
    ignore_types = {"DATE", "LOCATION", "ADDRESS", "GENDER"}

    presidio_perturb = PresidioPerturb(fake_pii_df=fake_df, ignore_types=ignore_types)

    perturbations = presidio_perturb.perturb(
        original_text=text,
        presidio_response=presidio_response,
        namesets=["Hebrew"],
        genders=["Female"],
        count=5,
    )
    for pert in perturbations:
        assert "neta" in pert.lower()
