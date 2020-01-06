from presidio_evaluator import InputSample
from presidio_evaluator.data_generator import read_synth_dataset


def test_to_conll():
    import os
    dir_path = os.path.dirname(os.path.realpath(__file__))
    input_samples = read_synth_dataset(os.path.join(dir_path, "data/generated_small.txt"))

    conll = InputSample.create_conll_dataset(input_samples)

    sentences = conll['sentence'].unique()
    assert len(sentences) == len(input_samples)


def test_to_spacy_all_entities():
    import os
    dir_path = os.path.dirname(os.path.realpath(__file__))
    input_samples = read_synth_dataset(os.path.join(dir_path, "data/generated_small.txt"))

    spacy_ver = InputSample.create_spacy_dataset(input_samples)

    assert len(spacy_ver) == len(input_samples)


def test_to_spacy_all_entities_specific_entities():
    import os
    dir_path = os.path.dirname(os.path.realpath(__file__))
    input_samples = read_synth_dataset(os.path.join(dir_path, "data/generated_small.txt"))

    spacy_ver = InputSample.create_spacy_dataset(input_samples, entities=['PERSON'])

    spacy_ver_with_labels = [sample for sample in spacy_ver if len(sample[1]['entities'])]

    assert len(spacy_ver_with_labels) < len(input_samples)
    assert len(spacy_ver_with_labels) > 0


def test_to_spach_json():
    import os
    dir_path = os.path.dirname(os.path.realpath(__file__))
    input_samples = read_synth_dataset(os.path.join(dir_path, "data/generated_small.txt"))

    spacy_ver = InputSample.create_spacy_json(input_samples)

    assert len(spacy_ver) == len(input_samples)
    assert 'id' in spacy_ver[0]
    assert 'paragraphs' in spacy_ver[0]
