from presidio_evaluator.data_generator import generate, read_synth_dataset, FakeDataGenerator


def get_fake_generator(template, fake_pii_df):
    class MockFakeGenerator(FakeDataGenerator):
        """
        Mock class that doesn't add to the fake PII DF so you could inject entities yourself.
        """

        def __init__(self, **kwargs):
            super().__init__(**kwargs)

        def prep_fake_pii(self, df):
            return df

    return MockFakeGenerator(templates=[template],
                             fake_pii_df=fake_pii_df,
                             include_metadata=False,
                             span_to_tag=False,
                             dictionary_path=None,
                             lower_case_ratio=0)


def test_generator_correct_output():
    OUTPUT = "generated_test.txt"
    EXAMPLES = 3

    import os
    dir_path = os.path.dirname(os.path.realpath(__file__))
    fake_pii_csv = "{}/data/FakeNameGenerator.com_100.csv".format(dir_path)
    utterances_file = "{}/data/templates.txt".format(dir_path)
    dictionary = "{}/data/Dictionary_test.csv".format(dir_path)

    generate(fake_pii_csv=fake_pii_csv,
             utterances_file=utterances_file,
             dictionary_path=dictionary,
             output_file=OUTPUT,
             lower_case_ratio=0.3,
             num_of_examples=EXAMPLES)

    input_samples = read_synth_dataset(OUTPUT)

    for sample in input_samples:
        assert len(sample.tags) == len(sample.tokens)


def test_a_turned_to_an():
    fake_pii_df = get_mock_fake_df(GENDER="Ale")
    template = "I am a [GENDER] living in [COUNTRY]"
    bracket_location = template.find("[")
    fake_generator = get_fake_generator(fake_pii_df=fake_pii_df,
                                        template=template)

    examples = [x for x in fake_generator.sample_examples(1)]
    assert " an " in examples[0].full_text
    # entity location updated
    assert examples[0].spans[0].start_position == bracket_location + 1


def test_a_not_turning_into_an():
    fake_pii_df = get_mock_fake_df(GENDER="Male")
    template = "I am a [GENDER] living in [COUNTRY]"
    previous_bracket = template.find("[")
    fake_generator = get_fake_generator(fake_pii_df=fake_pii_df,
                                        template=template)

    examples = [x for x in fake_generator.sample_examples(1)]
    assert " an " not in examples[0].full_text
    assert examples[0].spans[0].start_position == previous_bracket


def test_A_turning_into_An():
    fake_pii_df = get_mock_fake_df(GENDER="ale")
    template = "A [GENDER] living in [COUNTRY]"
    previous_bracket = template.find("[")
    fake_generator = get_fake_generator(fake_pii_df=fake_pii_df,
                                        template=template)

    examples = [x for x in fake_generator.sample_examples(1)]
    assert "An " in examples[0].full_text
    assert examples[0].spans[0].start_position == previous_bracket + 1


def get_mock_fake_df(**kwargs):
    dict = {
        "Number": 1,
        "Gender": "Male",
        "NameSet": "English",
        "Title": "Mr.",
        "GivenName": "Dondo",
        "MiddleInitial": "N",
        "Surname": "Mondo",
        "StreetAddress": "Where I live 15",
        "City": "Amsterdam",
        "State": "",
        "StateFull": "",
        "ZipCode": "12345",
        "Country": "Netherlands",
        "CountryFull": "Netherlands",
        "EmailAddress": "dondo@mondo.net",
        "Username": "Dondo12",
        "Password": "123456",
        "TelephoneNumber": "+1412391",
        "TelephoneCountryCode": "14",
        "MothersMaiden": "",
        "Birthday": "15 Aug 1966",
        "Age": "200",
        "CCType": "astercard",
        "CCNumber": "12371832821",
        "CVV2": "123",
        "CCExpires": "19-19",
        "NationalID": "14124",
        "Occupation": "Hunter",
        "Company": "Lolo and sons",
        "Domain": "lolo.com"}

    dict.update(kwargs)

    import pandas as pd
    fake_pii_df = pd.DataFrame(dict, index=[0])
    return fake_pii_df
