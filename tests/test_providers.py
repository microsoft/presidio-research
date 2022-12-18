from faker import Faker

from presidio_evaluator.data_generator.faker_extensions import (
    NationalityProvider,
    OrganizationProvider,
    HospitalProvider
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


def test_hospital_provider():
    faker = Faker()
    faker.add_provider(HospitalProvider)
    element = faker.hospital_name()
    assert element
