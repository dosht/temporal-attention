from tensorflow.models.rnn.translate import data_utils
import re


_PAD = "_PAD"
_GO = "_GO"
_EOS = "_EOS"
_UNK = "_UNK"
_START_VOCAB = [_PAD, _GO, _EOS, _UNK]

PAD_ID = 0
GO_ID = 1
EOS_ID = 2
UNK_ID = 3

# Regular expressions used to tokenize.
_WORD_SPLIT = re.compile("([.,!?\"':;)(])")
_DIGIT_RE = re.compile(r"\d{3,}")

def read_vocab(vocab_path):
    vocab_list = []
    vocab_list.extend(_START_VOCAB)

    with open(vocab_path, 'br') as f:
        vocab_list.extend([s.decode("utf-8").strip() for s in f.readlines()])

    words_to_ids = {w:i for (i, w) in enumerate(vocab_list)}
    ids_to_words = {i:w for (w, i) in words_to_ids.items()}
    return ids_to_words, words_to_ids


def basic_tokenizer(sentence):
    """Very basic tokenizer: split the sentence into a list of tokens."""
    words = []
    for space_separated_fragment in sentence.strip().split():
        words.extend(re.split(_WORD_SPLIT, space_separated_fragment))
    return [w.lower() for w in words if w]


def sentence_to_token_ids(sentence, vocabulary,
                          tokenizer=None, normalize_digits=True):
    """Convert a string to list of integers representing token-ids.

    For example, a sentence "I have a dog" may become tokenized into
    ["I", "have", "a", "dog"] and with vocabulary {"I": 1, "have": 2,
    "a": 4, "dog": 7"} this function will return [1, 2, 4, 7].

    Args:
    sentence: a string, the sentence to convert to token-ids.
    vocabulary: a dictionary mapping tokens to integers.
    tokenizer: a function to use to tokenize each sentence;
      if None, basic_tokenizer will be used.
    normalize_digits: Boolean; if true, all digits are replaced by 0s.

    Returns:
    a list of integers, the token-ids for the sentence.
    """
    if tokenizer:
        words = tokenizer(sentence)
    else:
        words = basic_tokenizer(sentence)
    if not normalize_digits:
        return [vocabulary.get(w, UNK_ID) for w in words]

    # Normalize digits by 0 before looking words up in the vocabulary.
    return [vocabulary.get(re.sub(_DIGIT_RE, "0", w), UNK_ID) for w in words]

def token_ids_to_sentence(ids, vocab_index):
    maybe_words = [vocab_index.get(_id) for _id in ids]
    return " ".join([w for w in maybe_words if w])


def read_data(path):
    "reads a file of ids and return a list of lists of ids"
    with open(path, 'r') as f:
        return [[int(x) for x in line.split(" ")] for line in f.readlines()]


#TODO: complete the docstring
def prepare_date(data_dir, vocab_size, sample=True):
    """
    Downloads english/fench pair and returns the data set as an instace of LanguagePairDataSet
    """
    from dataset import LanguageDataSet
    pathes = data_utils.prepare_wmt_data(data_dir, vocab_size, vocab_size)
    en2_path, fr2_path, en2013_path, fr2013_path, en_vocab_path, fr_vocab_path = pathes
    en_index, en_vocab = read_vocab(en_vocab_path)
    fr_index, fr_vocab = read_vocab(fr_vocab_path)
    #FIXME: some non-ascii charachters
    en_vocab_size = len(en_vocab) + 1
    fr_vocab_size = len(fr_vocab) + 1

    if sample:
        en_ids = read_data(en2013_path)
        fr_ids = read_data(fr2013_path)

    else:
        print("reading the full dataset")
        en_ids = read_data(en2_path)
        fr_ids = read_data(fr2_path)

    #Make it the same length (= the max length of the sentences) with zeros for shorter sentences
    return LanguageDataSet(en_ids, en_vocab, en_index), LanguageDataSet(fr_ids, fr_vocab, fr_index)


if __name__ == "__main__":
    #TODO: read from args
    data_dir = "/data/translate" #TODO: You may need to change that or create a sympolic link
    vocab_size = 20000
    en, fr = prepare_date(data_dir, vocab_size)
    ids = en.sentence_to_ids("Good morning")
    sendtence = en.ids_to_sentence(ids)
    print('ids of sentence "Good morning" = %s' % ids)
    print('sendtence of ids %s = %s' % (ids, sendtence))

