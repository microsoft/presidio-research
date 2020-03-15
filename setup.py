from setuptools import setup
import os.path
# read the contents of the README file
from os import path

this_directory = path.abspath(path.dirname(__file__))
with open(path.join(this_directory, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()
    # print(long_description)
__version__ = ""

with open(os.path.join(this_directory, 'VERSION')) as version_file:
    __version__ = version_file.read().strip()

setup(
    name='presidio-evaluator',
    long_description=long_description,
    long_description_content_type='text/markdown',
    version=__version__,
    packages=['presidio_evaluator', 'presidio_evaluator.data_generator'
              ],
    url='https://www.github.com/microsoft/presidio',
    license='MIT',
    description='PII dataset generator, model evaluator for Presidio and PII data in general',
    data_files=[('presidio_evaluator/data_generator/raw_data', ['presidio_evaluator/data_generator/raw_data/FakeNameGenerator.com_3000.csv', 'presidio_evaluator/data_generator/raw_data/templates.txt', 'presidio_evaluator/data_generator/raw_data/organizations.csv', 'presidio_evaluator/data_generator/raw_data/nationalities.csv'])],
    include_package_data=True,
    install_requires=[
        'spacy>=2.2.0',
        'requests==2.22.0',
        'numpy',
        'pandas',
        'tqdm>=4.32.1',
        'jupyter>=1.0.0',
        'pytest>=4.6.2',
        'haikunator',
        'schwifty',
        'faker',
        'sklearn_crfsuite']

)
