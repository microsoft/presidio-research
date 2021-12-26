from collections import OrderedDict
from pathlib import Path
import random
from typing import Union

import pandas as pd
from faker.providers import BaseProvider
from faker.providers.address.en import AddressProvider


class NationalityProvider(BaseProvider):
    def __init__(self, generator, nationality_file: Union[str, Path] = None):
        super().__init__(generator=generator)
        if not nationality_file:
            nationality_file = Path(
                Path(__file__).parent.parent, "raw_data", "nationalities.csv"
            ).resolve()

        self.nationality_file = nationality_file
        self.nationalities = self.load_nationalities()

    def load_nationalities(self):
        return pd.read_csv(self.nationality_file)

    def country(self):
        self.random_element(self.nationalities["country"].tolist())

    def nationality(self):
        return self.random_element(self.nationalities["nationality"].tolist())

    def nation_man(self):
        return self.random_element(self.nationalities["man"].tolist())

    def nation_woman(self):
        return self.random_element(self.nationalities["woman"].tolist())

    def nation_plural(self):
        return self.random_element(self.nationalities["plural"].tolist())


class OrganizationProvider(BaseProvider):
    def __init__(
        self,
        generator,
        organizations_file: Union[str, Path] = None,
    ):
        super().__init__(generator=generator)
        if not organizations_file:
            organizations_file = Path(
                Path(__file__).parent.parent, "raw_data", "organizations.csv"
            ).resolve()
        self.organizations_file = organizations_file
        self.organizations = self.load_organizations()

    def load_organizations(self):
        return pd.read_csv(self.organizations_file, delimiter="\t")

    def organization(self):
        return self.random_element(self.organizations["organization"].tolist())


class UsDriverLicenseProvider(BaseProvider):
    def __init__(
        self,
        generator,
        us_driver_license_file: Union[str, Path] = None,
    ):
        super().__init__(generator=generator)
        if not us_driver_license_file:
            us_driver_license_file = Path(
                Path(__file__).parent.parent, "raw_data", "us_driver_licenses.csv"
            ).resolve()
        self.us_driver_license_file = us_driver_license_file
        self.us_driver_licenses = self.load_us_driver_licenses()

    def us_driver_license(self):
        return self.random_element(
            self.us_driver_licenses["us_driver_license"].tolist()
        )

    def load_us_driver_licenses(self):
        return pd.read_csv(self.us_driver_license_file, delimiter="\t")


class IpAddressProvider(BaseProvider):
    """Generating both v4 and v6 IP addresses."""

    def ip_address(self):
        if random.random() < 0.8:
            return self.generator.ipv4()
        else:
            return self.generator.ipv6()


class AddressProviderNew(AddressProvider):
    """
    Extending the Faker AddressProvider with additional templates
    """

    address_formats = OrderedDict(
        (
            (
                "{{building_number}} {{street_name}} {{secondary_address}} {{city}} {{state}}",
                5.0,
            ),
            (
                "{{building_number}} {{street_name}} {{secondary_address}} {{city}} {{state_abbr}}",
                5.0,
            ),
            (
                "{{building_number}} {{street_name}} {{secondary_address}} {{city}} {{country}}",
                5.0,
            ),
            (
                "{{building_number}} {{street_name}}\n {{secondary_address}}\n {{city}}\n {{country}}",
                5.0,
            ),
            (
                "{{building_number}} {{street_name}}\n {{secondary_address}}\n {{city}}\n {{country}} {{postcode}}",
                5.0,
            ),
            (
                "{{street_name}} {{street_name}}\n {{secondary_address}}\n {{city}}\n {{country}} {{postcode}}",
                5.0,
            ),
            ("the corner of {{street_name}} and {{street_name}}", 3.0),
            ("{{first_name}} and {{street_name}}", 3.0),
            ("{{street_address}}, {{city}}, {{country}}", 5.0),
            (
                "{{street_address}} {{secondary_address}}, {{city}}, {{country}} {{postcode}}",
                5.0,
            ),
            ("{{street_address}}\n{{city}}, {{state_abbr}} {{postcode}}", 25.0),
            ("{{street_address}}\n{{city}}\n, {{state_abbr}}\n {{postcode}}", 25.0),
            (
                "{{street_address}}\n{{city}}\n, {{state_abbr}}\n {{country}} {{postcode}}",
                25.0,
            ),
            #  military address formatting.
            ("{{military_apo}}\nAPO {{military_state}} {{postcode}}", 1.0),
            (
                "{{military_ship}} {{last_name}}\nFPO {{military_state}} {{postcode}}",
                1.0,
            ),
            ("{{military_dpo}}\nDPO {{military_state}} {{postcode}}", 1.0),
        )
    )
