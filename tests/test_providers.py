from faker import Faker

from presidio_evaluator.data_generator.faker_extensions import (
    NationalityProvider,
    OrganizationProvider,
)


def test_nationality_provider():
    faker = Faker()
    faker.add_provider(NationalityProvider)
    element = faker.nation_man()
    assert element


def test_organization_provider():
    faker = Faker()
    faker.add_provider(OrganizationProvider)
    element = faker.organization()
    assert element
