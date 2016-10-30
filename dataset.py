from data_utils import *


#TODO: write doc strings for this
class LanguageDataSet(object):
    def __init__(self, examples, vocab, index):
        self.examples = examples
        self.vocab = vocab
        self.index = index
        self.vocab_size = len(vocab)
        self.max_length = examples.shape[1]
        self.max_features = max(vocab.values())

    def sentence_to_ids(self, sentence):
        return sentence_to_token_ids(sentence, self.vocab)

    def ids_to_sentence(self, ids):
        return token_ids_to_sentence(ids, self.index)

