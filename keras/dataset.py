from data_utils import *
from keras.preprocessing.sequence import pad_sequences


#TODO: write doc strings for this
class LanguageDataSet(object):
    def __init__(self, examples, vocab, index):
        self.examples = examples
        self.matrix = pad_sequences(examples)  #fixed length examples
        self.vocab = vocab
        self.index = index
        self.vocab_size = len(vocab)
        self.max_length = self.matrix.shape[1]
        self.max_features = max(vocab.values())

    def sentence_to_ids(self, sentence):
        return sentence_to_token_ids(sentence, self.vocab)

    def ids_to_sentence(self, ids):
        return token_ids_to_sentence(ids, self.index)

    def pad_sequences(self, ids):
        return pad_sequences(ids, self.max_length)

