import logging
import pickle
import random
import sys
from pathlib import Path

import spacy
from azureml.core import Workspace, Experiment
from spacy.util import minibatch, compounding

from presidio_evaluator import SpacyEvaluator, InputSample

logging.basicConfig(level=logging.INFO)

root = logging.getLogger()
root.setLevel(logging.INFO)

handler = logging.StreamHandler(sys.stdout)
handler.setLevel(logging.INFO)
root.addHandler(handler)


class SpacyRetrainer:

    def __init__(self, original_model_name=None, experiment_name=None, n_iter=100, dropout=0.5,
                 aml_config='config.json', output_dir='../../model-outputs', train_pickle='../data/train.pickle',
                 test_pickle='../data/test.pickle'):
        self.experiment_name = experiment_name
        if aml_config:
            self.ws = Workspace.from_config(aml_config)
            self.experiment = Experiment(workspace=self.ws, name=experiment_name)
            self.aml_run = self.experiment.start_logging()
            self.has_aml = True
        else:
            self.has_aml = False

        self.model = original_model_name
        self.n_iter = n_iter
        self.output_dir = output_dir
        self.train_file = train_pickle
        self.test_file = test_pickle
        self.dropout = dropout

    def run(self):
        if self.has_aml:
            self.aml_run.log("model", self.model)
            self.aml_run.log("n_iter", self.n_iter)
            self.aml_run.log("train_file", self.train_file)
            self.aml_run.log("test_file", self.test_file)
            self.aml_run.log("dropout rate", self.dropout)
        model_path = self._train(self.model, self.output_dir, self.n_iter, self.train_file, self.experiment_name)
        self._score_validate(model_path, self.test_file)
        if self.has_aml:
            self.aml_run.complete()

    def print_scores(self, split, evaluation_result):
        """
        Logs results into experiment run.
        :param split: Name of this split. For ex 'train' or 'valid'
        :param evaluation_result: EvaluationResult containing various metrics
        :return: None. Writes to experiment runner and logs locally.
        """
        logging.info('SPLIT: {0}. PII_precision: {1}, PII_recall: {2},'
                     'Person_precision: {3}, Person_recall: {4}'. \
                     format(split, evaluation_result.pii_precision, evaluation_result.pii_recall,
                            evaluation_result.entity_precision_dict['PERSON'],
                            evaluation_result.entity_recall_dict['PERSON']))
        if self.has_aml:
            self.aml_run.log('Precision', evaluation_result.pii_precision, split)
            self.aml_run.log('Recall', evaluation_result.pii_recall, split)

    @staticmethod
    def _score(model, data):
        """
        Score the model against the data
        :param model: Trained model
        :param data: Data split which is being scored.
        :return: An EvaluationResult containing various metrics
        """

        spacy_evaluator = SpacyEvaluator(model=model)

        results = []
        for text, ground_truth_annotations in data:
            ground_truth_entities = ground_truth_annotations['entities']
            input_sample = InputSample.from_spacy(text, ground_truth_entities)
            results.append(spacy_evaluator.evaluate_sample(input_sample))

        return spacy_evaluator.calculate_score(evaluation_results=results)

    def _score_validate(self, model_path, test_data_file):
        """
        Validation step for the model. Also prints the scores.
        :param model_path: Path to trained model.
        :param test_data_file: Data file which has the dataset for this split.
        :return: None. Prints the scores.
        """
        with open(test_data_file, 'rb') as f:
            valid_data = pickle.load(f)
        nlp = spacy.load(model_path)
        self.print_scores('Valid', self._score(nlp, valid_data))

    # @plac.annotations(
    #     model=("Model name. Defaults to blank 'en' model.", "option", "m", str),
    #     output_dir=("Optional output directory", "option", "o", Path),
    #     n_iter=("Number of training iterations", "option", "n", int),
    #     train_file=("File containing pickled training Spacy NER formatted data", "option", "d", Path),
    #     test_file=("File containing pickled test Spacy NER formatted data", "option", "d", Path),
    #     exp_name=("Name of this experiment", "option", "e")
    # )

    def _train(self, model, output_dir, n_iter, train_file, exp_name):
        """Load the model, set up the pipeline and train the entity recognizer."""
        nlp = self.load_or_create_empty_model(model)

        if "ner" not in nlp.pipe_names:
            ner = nlp.create_pipe("ner")
            nlp.add_pipe(ner, last=True)
        else:
            ner = nlp.get_pipe("ner")

        with open(train_file, 'rb') as f:
            train_data = pickle.load(f)

        # DEBUG
        train_data = train_data[:50]

        # add labels
        for _, annotations in train_data:
            for ent in annotations.get("entities"):
                ner.add_label(ent[2])

        # get names of other pipes to disable them during training
        other_pipes = [pipe for pipe in nlp.pipe_names if pipe != "ner"]
        with nlp.disable_pipes(*other_pipes):  # only train NER
            # reset and initialize the weights randomly – but only if we're
            # training a new model
            if model is None:
                nlp.begin_training()
            for itn in range(n_iter):
                random.shuffle(train_data)
                losses = {}
                # batch up the examples using spaCy's minibatch
                batches = minibatch(train_data, size=compounding(4.0, 32.0, 1.001))
                for batch in batches:
                    texts, annotations = zip(*batch)
                    nlp.update(texts, annotations, drop=self.dropout, losses=losses, )
                logging.debug("Losses", losses)
                if self.has_aml:
                    self.aml_run.log('Losses', losses['ner'])
                self.print_scores('Itn {}'.format(itn), self._score(nlp, train_data))

        self.print_scores('Train', self._score(nlp, train_data))

        saved_model_path = self.save_model(exp_name, nlp, output_dir)
        return saved_model_path

    @staticmethod
    def save_model(exp_name, model, output_dir):
        """
        Saves model to disk for later use.
        :param exp_name: Name of the running experiment. This is used as folder name for storing the model.
        :param model: Model being saved
        :param output_dir: Directory where to save the model.
        :return: Full path to saved model.
        """
        saved_model_path = Path(output_dir, exp_name)
        if not saved_model_path.exists():
            saved_model_path.mkdir(parents=True)
        model.to_disk(saved_model_path)
        logging.info("Saved model to {}".format(output_dir))
        return saved_model_path

    @staticmethod
    def load_model(exp_name, model_dir):
        """
        Loads a spacy model from disk

        :param exp_name: Name of experiment under which the model was saved
        :param model_dir: path to saved model
        :return: spacy model
        """
        saved_model_path = Path(model_dir, exp_name)
        return spacy.load(saved_model_path)

    @staticmethod
    def load_or_create_empty_model(model=None):
        """
        Loads a given model or creates a blank english model.
        :param model: Optional Model to load.
        :return: Loaded or blank model.
        """
        if model:
            nlp = spacy.load(model)
            logging.debug("Loaded model {}".format(model))
        else:
            nlp = spacy.blank("en")
            logging.debug("Created blank 'en' model")
        return nlp


if __name__ == "__main__":
    spacy_retrainer = SpacyRetrainer(original_model_name='en_core_web_lg',
                                     experiment_name='spacy_new_ontonotes28',
                                     n_iter=500, dropout=0.5, aml_config=None)
    spacy_retrainer.run()
