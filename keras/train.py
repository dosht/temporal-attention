from keras.layers.embeddings import Embedding
from keras.layers import Dense, Merge, Dropout, RepeatVector, Permute, Activation, recurrent, LSTM, GRU
from keras.models import Sequential
from keras.layers.wrappers import TimeDistributed, Bidirectional
from data_utils import *
from dataset import *


def main():
    print("\n\nLoading data...")
    data_dir = "/data/translate"
    vocab_size = 20000
    en, fr = prepare_date(data_dir, vocab_size)

    print("\n\nbuilding the model...")
    embedding_size = 64
    hidden_size = 32
    model = Sequential()
    model.add(Embedding(en.max_features, embedding_size, input_length=en.max_length, mask_zero=True))
    model.add(Bidirectional(GRU(hidden_size), merge_mode='sum'))
    model.add(RepeatVector(fr.max_length))
    model.add(GRU(embedding_size))
    model.add(Dense(fr.max_length, activation="softmax"))
    model.compile('rmsprop', 'mse')
    print(model.get_config())

    print("\n\nFitting the model...")
    model.fit(en.examples, fr.examples)

    print("\n\nEvaluation...")
    #TODO


if __name__ == '__main__':
    main()

