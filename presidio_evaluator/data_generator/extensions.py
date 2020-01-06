import random
import pandas as pd
from faker import Faker
from haikunator import Haikunator

from presidio_evaluator.data_generator.nationality_generator import NationalityGenerator
from presidio_evaluator.data_generator.org_name_generator import OrgNameGenerator

fake = Faker()
haikunator = Haikunator()
IP_V4_RATIO = 0.8

org_name_generator = OrgNameGenerator()
nationality_generator = NationalityGenerator()

def generate_url(domain: pd.Series):
    def generate_url_postfix():
        length = random.randint(4, 8)
        delim = "/" if random.random() > 0.5 else ""
        postfix = haikunator.haikunate(delimiter=delim,
                                       token_chars='abcdefghijklmnopqrstuvwxyz0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ',
                                       token_length=length)
        return postfix

    def generate_url_prefix():
        rand = random.random()

        if rand < 0.3:
            return "http://"
        elif rand < 0.6:
            return "http://www."
        else:
            return ""

    def concat_url(prefix, domain, postfix):
        return "{}{}/{}".format(prefix, domain, postfix)

    return domain.apply(lambda x: concat_url(generate_url_prefix(), x.lower(), generate_url_postfix()))
    #
    # urls = []
    # for index, value in domain.items():
    #     url = "{}{}/{}".format(generate_url_prefix(), value.lower(), generate_url_postfix())
    #     urls.append(url)
    #
    # return urls


def generate_SSNs(length):
    return [fake.ssn() for _ in range(length)]


def generate_iban(country: pd.Series):
    def generate_one_iban(cntry):
        try:
            from schwifty.iban import _get_iban_spec, code_length, IBAN
            import math

            spec = _get_iban_spec(cntry)
            bank_code_length = code_length(spec, 'bank_code')
            branch_code_length = code_length(spec, 'branch_code')
            bank_and_branch_code_length = bank_code_length + branch_code_length
            account_code_length = code_length(spec, 'account_code')

            bank_code = random.randint(1, math.pow(10, bank_and_branch_code_length) - 1)
            account_code = random.randint(1, math.pow(10, account_code_length) - 1)
            iban = IBAN.generate(cntry, str(bank_code), str(account_code))
            return iban.formatted
        except ValueError as err:
            ## Failed to generate IBAN
            return "IL270126100000000544211"

    return country.apply(generate_one_iban)


def generate_company_names(length):
    return [org_name_generator.get_organization() for _ in range(length)]


def generate_ip_addresses(length):
    def generate_one():
        v = 4 if random.random() > IP_V4_RATIO else 6
        return fake.ipv4() if v == 4 else fake.ipv6()

    return [generate_one() for _ in range(length)]


def generate_title(gender=None):
    MALE_TITLES = ['Mr.', 'Dr.', 'Professor.', 'Eng.', 'Prof.', 'Doctor.']
    FEMALE_TITLES = ['Mrs.', 'Ms.', 'Miss', 'Dr.', 'Professor.', 'Eng.', 'Prof.', 'Doctor']

    if gender.lower() == 'male':
        return random.choices(MALE_TITLES, weights=[0.7, 0.1, 0.05, 0.05, 0.05, 0.05])[0]
    else:
        return random.choices(FEMALE_TITLES, weights=[0.3, 0.25, 0.20, 0.05, 0.05, 0.05, 0.05, 0.05])[0]


def generate_titles(gender: pd.Series):
    return gender.apply(generate_title)


def generate_roles(length):
    roles = ['President', 'Vice-president', 'Chief of staff', 'Chief Architect', 'CEO', 'CFO', 'Engineer', 'Accountant',
             'Attorney', 'Scientist', 'Journalist', 'Operator', 'CIO', "Chief Information Officer", "General Manager",
             "Manager", "Chief Executive Officer", 'Actuary', 'Secretary', 'Prime minister', 'Minister', 'Director']
    return [random.choice(roles) for _ in range(length)]


def generate_nationality(length):
    return [nationality_generator.get_nationality() for _ in range(length)]


def generate_country(length):
    return [nationality_generator.get_country() for _ in range(length)]


def generate_nation_woman(length):
    return [nationality_generator.get_nation_woman() for _ in range(length)]


def generate_nation_man(length):
    return [nationality_generator.get_nation_man() for _ in range(length)]

def generate_nation_plural(length):
    return [nationality_generator.get_nation_plural() for _ in range(length)]
