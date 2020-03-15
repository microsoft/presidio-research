from typing import List

from flair.data import Corpus, Sentence
from flair.datasets import ColumnCorpus
from flair.embeddings import TokenEmbeddings, WordEmbeddings, StackedEmbeddings, FlairEmbeddings, BertEmbeddings
from flair.models import SequenceTagger
from flair.trainers import ModelTrainer

from presidio_evaluator import InputSample
from presidio_evaluator.data_generator import read_synth_dataset
from os import path


class FlairTrainer:

    @staticmethod
    def to_flair_row(text, pos, label):
        return "{} {} {}".format(text, pos, label)

    def to_flair(self, df, outfile="flair_train.txt"):
        sentence = 0
        flair = []
        for row in df.itertuples():
            if row.sentence != sentence:
                sentence += 1
                flair.append("")
            else:
                flair.append(self.to_flair_row(row.text, row.pos, row.label))

        if outfile:
            with open(outfile, "w", encoding="utf-8") as f:
                for item in flair:
                    f.write("{}\n".format(item))

    def create_flair_corpus(self, train_samples_path, test_samples_path, val_samples_path):
        if not path.exists("flair_train.txt"):
            train_samples = read_synth_dataset(train_samples_path)
            train_tagged = [sample for sample in train_samples if len(sample.spans) > 0]
            print("Kept {} train samples after removal of non-tagged samples".format(len(train_tagged)))
            train_data = InputSample.create_conll_dataset(train_tagged)
            self.to_flair(train_data, outfile="flair_train.txt")

        if not path.exists("flair_test.txt"):
            test_samples = read_synth_dataset(test_samples_path)
            test_data = InputSample.create_conll_dataset(test_samples)
            self.to_flair(test_data, outfile="flair_test.txt")

        if not path.exists("flair_val.txt"):
            val_samples = read_synth_dataset(val_samples_path)
            val_data = InputSample.create_conll_dataset(val_samples)
            self.to_flair(val_data, outfile="flair_val.txt")

    @staticmethod
    def read_corpus(data_folder) -> Corpus:
        columns = {0: 'text', 1: 'pos', 2: 'ner'}
        corpus: Corpus = ColumnCorpus(data_folder, columns,
                                      train_file='flair_train.txt',
                                      test_file='flair_val.txt',
                                      dev_file='flair_test.txt')
        return corpus

    @staticmethod
    def train(corpus):
        print(corpus)

        # 2. what tag do we want to predict?
        tag_type = 'ner'

        # 3. make the tag dictionary from the corpus
        tag_dictionary = corpus.make_tag_dictionary(tag_type=tag_type)
        print(tag_dictionary.idx2item)

        # 4. initialize embeddings
        embedding_types: List[TokenEmbeddings] = [
            WordEmbeddings('glove'),
            FlairEmbeddings('news-forward'),
            FlairEmbeddings('news-backward')
        ]

        embeddings: StackedEmbeddings = StackedEmbeddings(embeddings=embedding_types)

        # 5. initialize sequence tagger

        tagger: SequenceTagger = SequenceTagger(hidden_size=256,
                                                embeddings=embeddings,
                                                tag_dictionary=tag_dictionary,
                                                tag_type=tag_type,
                                                use_crf=True)

        # 6. initialize trainer

        trainer: ModelTrainer = ModelTrainer(tagger, corpus)

        checkpoint = 'resources/taggers/presidio-ner/checkpoint.pt'
        # trainer = ModelTrainer.load_checkpoint(checkpoint, corpus)
        trainer.train('resources/taggers/presidio-ner',
                      learning_rate=0.1,
                      mini_batch_size=32,
                      max_epochs=150,
                      checkpoint=True)

        sentence = Sentence('I am from Jerusalem')
        # run NER over sentence
        tagger.predict(sentence)

        print(sentence)
        print('The following NER tags are found:')

        # iterate over entities and print
        for entity in sentence.get_spans('ner'):
            print(entity)


if __name__ == "__main__":
    train_samples = "../data/generated_train_November 12 2019.json"
    test_samples = "../data/generated_test_November 12 2019.json"
    val_samples = "../data/generated_validation_November 12 2019.json"

    trainer = FlairTrainer()
    trainer.create_flair_corpus(train_samples, test_samples, val_samples)

    corpus = trainer.read_corpus("")
    trainer.train(corpus)
