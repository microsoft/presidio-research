# -*- coding: utf-8 -*-
from setuptools import setup
import os
from os import path

this_directory = path.abspath(path.dirname(__file__))
with open(path.join(this_directory, "README.md"), encoding="utf-8") as f:
    long_description = f.read()

with open(os.path.join(this_directory, "VERSION")) as version_file:
    version = version_file.read().strip()


packages = [
    "presidio_evaluator",
    "presidio_evaluator.data_generator",
    "presidio_evaluator.data_generator.faker_extensions",
    "presidio_evaluator.dataset_formatters",
    "presidio_evaluator.evaluation",
    "presidio_evaluator.experiment_tracking",
    "presidio_evaluator.models",
]

package_data = {"": ["*"], "presidio_evaluator.data_generator": ["raw_data/*"]}

install_requires = [
    "azure-ai-textanalytics>=5.3.0,<6.0.0",
    "faker>=21.0,<22.0",
    "numpy>=1.22,<2.0",
    "pandas>=2.1.4,<3.0.0",
    "plotly>=5.18.0,<6.0.0",
    "presidio-analyzer>=2.2.351,<3.0.0",
    "presidio-anonymizer>=2.2.351,<3.0.0",
    "python-dotenv>=1.0.0,<2.0.0",
    "requests>=2.25,<3.0",
    "scikit-learn>=1.3.2,<2.0.0",
    "spacy>=3.5.0,<4.0.0",
    "tqdm>=4.60.0,<5.0.0",
    "xmltodict>=0.12.0,<0.13.0",
]

setup(
    name="presidio-evaluator",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://www.github.com/microsoft/presidio-research",
    version=version,
    license="MIT",
    packages=packages,
    package_data=package_data,
    install_requires=install_requires,
    python_requires=">=3.8,<4.0",
)