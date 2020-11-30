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
