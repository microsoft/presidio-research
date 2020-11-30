import random
import os
from pathlib import Path


class UsDriverLicenseGenerator:
    def __init__(self, company_name_file_path="raw_data/us_driver_licenses.csv"):
        self.licenses = []
        dir_path = os.path.dirname(os.path.realpath(__file__))
        file_path = Path(dir_path, company_name_file_path)

        with open(str(file_path)) as file:
            self.licenses = file.read().splitlines()

    def get_driver_license_number(self):
        return random.choice(self.licenses)
