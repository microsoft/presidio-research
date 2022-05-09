import pickle
from typing import List, Dict, Optional

try:
    import sklearn_crfsuite
except ImportError:
    sklearn_crfsuite = None

from presidio_evaluator import InputSample
from presidio_evaluator.models import BaseModel


class CRFModel(BaseModel):
    """
    Wrapper for a CRF model.
    :param model_pickle_path: Path to pickled trained model
    """

    def __init__(
        self,
        model_pickle_path: Optional[str] = None,
        entities_to_keep: List[str] = None,
        verbose: bool = False,
        entity_mapping: Dict[str, str] = None,
    ):
        super().__init__(
            entities_to_keep=entities_to_keep,
            verbose=verbose,
            entity_mapping=entity_mapping,
        )

        if model_pickle_path:
            with open(model_pickle_path, "rb") as f:
                self.model = pickle.load(f)
        else:
            self.model = None

    def fit(
        self,
        train_samples: List[InputSample],
        algorithm="lbfgs",
        c1=0.1,
        c2=0.1,
        max_iterations=100,
        all_possible_transitions=True,
        **kwargs
    ):
        """
        Trains a simple CRF model
        :param train_samples: Training samples to train with
        :param algorithm: see `sklearn_crfsuite.CRF`
        :param c1: see `sklearn_crfsuite.CRF`
        :param c2: see `sklearn_crfsuite.CRF`
        :param max_iterations: see `sklearn_crfsuite.CRF`
        :param all_possible_transitions: see `sklearn_crfsuite.CRF`
        :return:
        """

        if not sklearn_crfsuite:
            raise ValueError("sklearn_crfsuite not installed")

        # Ignore entities not requested
        train_samples_filtered = self._ignore_unwanted_entities(train_samples)

        X_train, y_train = self._to_feature_set(train_samples_filtered)

        self.model = sklearn_crfsuite.CRF(
            algorithm=algorithm,
            c1=c1,
            c2=c2,
            max_iterations=max_iterations,
            all_possible_transitions=all_possible_transitions,
            **kwargs
        )
        self.model.fit(X_train, y_train)

    def _to_feature_set(self, dataset: List[InputSample]):

        samples_conll = InputSample.create_conll_dataset(dataset)
        sentences = samples_conll.groupby("sentence")[["text", "pos", "label"]].apply(
            lambda x: x.values.tolist()
        )

        X_train = [self.sent2features(s) for s in sentences]
        y_train = [self.sent2labels(s) for s in sentences]
        return X_train, y_train

    def predict(self, sample: InputSample) -> List[str]:
        tags = CRFModel.crf_predict(sample, self.model)

        if len(tags) != len(sample.tokens):
            print("mismatch between previous tokens and new tokens")
        return tags

    @staticmethod
    def crf_predict(sample, model):
        sample.translate_input_sample_tags()

        conll = sample.to_conll(translate_tags=True)
        sentence = [(di["text"], di["pos"], di["label"]) for di in conll]
        features = CRFModel.sent2features(sentence)
        return model.predict([features])[0]

    @staticmethod
    def word2features(sent, i):
        word = sent[i][0]
        postag = sent[i][1]

        features = {
            "bias": 1.0,
            "word.lower()": word.lower(),
            "word[-3:]": word[-3:],
            "word[-2:]": word[-2:],
            "word.isupper()": word.isupper(),
            "word.istitle()": word.istitle(),
            "word.isdigit()": word.isdigit(),
            "postag": postag,
            "postag[:2]": postag[:2],
        }
        if i > 0:
            word1 = sent[i - 1][0]
            postag1 = sent[i - 1][1]
            features.update(
                {
                    "-1:word.lower()": word1.lower(),
                    "-1:word.istitle()": word1.istitle(),
                    "-1:word.isupper()": word1.isupper(),
                    "-1:postag": postag1,
                    "-1:postag[:2]": postag1[:2],
                }
            )
        else:
            features["BOS"] = True

        if i < len(sent) - 1:
            word1 = sent[i + 1][0]
            postag1 = sent[i + 1][1]
            features.update(
                {
                    "+1:word.lower()": word1.lower(),
                    "+1:word.istitle()": word1.istitle(),
                    "+1:word.isupper()": word1.isupper(),
                    "+1:postag": postag1,
                    "+1:postag[:2]": postag1[:2],
                }
            )
        else:
            features["EOS"] = True

        return features

    @staticmethod
    def sent2features(sent):
        return [CRFModel.word2features(sent, i) for i in range(len(sent))]

    @staticmethod
    def sent2labels(sent):
        return [label for token, postag, label in sent]

    @staticmethod
    def sent2tokens(sent):
        return [token for token, postag, label in sent]
