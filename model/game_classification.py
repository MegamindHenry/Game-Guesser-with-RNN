import argparse
import fasttext
import json
import nltk
import numpy as np
import pickle
from keras import Sequential, Input
from keras.callbacks import EarlyStopping
from keras.layers import Bidirectional, LSTM, Dense, Dropout, Embedding
from keras.models import load_model

import os

from keras_preprocessing.sequence import pad_sequences
from keras_preprocessing.text import Tokenizer

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = ""

EMBEDDING_DIM = 128
TOKENS_MAX_LENGTH = 64
BATCH_SIZE = 128
MAX_NUM_WORDS = 20000
THRESHOLD = 0.3
K = 5
IS_SHORT = True
APPEND = True


def train_embedding(input_file, output_file):
    model = fasttext.train_unsupervised(input_file, model='skipgram', dim=EMBEDDING_DIM)
    model.save_model(output_file)


def read_file(file, is_short=True, emode='fasttext', append=True):
    sources = []
    targets = []

    with open(file, 'r', encoding='utf8') as fp:
        data = json.load(fp)

    if emode == 'fasttext':
        for d in data:
            if is_short:
                description = d['short_description']
            else:
                description = d['detailed_description']

            tokens = nltk.tokenize.word_tokenize(description)

            if append:
                while len(tokens) > TOKENS_MAX_LENGTH:
                    sub_d = tokens[:TOKENS_MAX_LENGTH]
                    sources.append(sub_d)
                    targets.append(d['genres'])
                    tokens = tokens[TOKENS_MAX_LENGTH:]
            else:
                tokens = tokens[:TOKENS_MAX_LENGTH]

            sources.append(tokens)
            targets.append(d['genres'])
    elif emode == 'keras':
        for d in data:
            if is_short:
                description = d['short_description']
            else:
                description = d['detailed_description']

            sources.append(description)
            targets.append(d['genres'])
    else:
        raise Exception('You should pass a emode')

    return sources, targets


def read_genres(file):
    with open(file, 'rb') as fp:
        return pickle.load(fp, encoding='utf8')


class DescriptionEncoder:
    """ Utility to transform character sequences to numbers and back.
    """
    __slots__ = ['model']

    def __init__(self, bin_file):
        self.model = fasttext.load_model(bin_file)

    def word2int(self, word):
        return self.model[word]

    def transform(self, sources, max_len=TOKENS_MAX_LENGTH, dim=EMBEDDING_DIM):
        encoded_sources = []
        pad = np.zeros((dim,))

        for s in sources:
            s_vec = []
            for token in s:
                s_vec.append(self.model[token])

            while len(s_vec) < max_len:
                s_vec.append(pad)

            encoded_sources.append(s_vec)
        return np.array(encoded_sources)


class GenresEncoder:
    __slots__ = ['genres2int', 'int2genres', 'num_genres']

    def __init__(self, genres_file):
        genres = read_genres(genres_file)
        self.num_genres = len(genres)
        self.genres2int = {v: i for i, v in enumerate(genres)}
        self.int2genres = {i: v for i, v in enumerate(genres)}

    def transform(self, targets):

        y = np.zeros((len(targets), self.num_genres))

        for i, t in enumerate(targets):
            for x in t:
                y[i][self.genres2int[x]] = 1

        return y


def train(X, y, input_shape, output_dim, batch_size=BATCH_SIZE,
          encoder_dim=128, decoder_dim=256, dropout=0.2, max_epoch=100, use_es=True,
          emode='fasttext'):
    if emode == 'fasttext':
        model = Sequential()
        model.add(Bidirectional(LSTM(encoder_dim, activation='relu',
                                     return_sequences=False), input_shape=input_shape))
        model.add(Dropout(dropout))
        model.add(Dense(decoder_dim, activation='relu'))
        model.add(Dropout(dropout))
        model.add(Dense(output_dim, activation='sigmoid'))
        model.add(Dropout(dropout))
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        model.summary()

        es = EarlyStopping(patience=5, monitor='val_loss', restore_best_weights=True)

        if use_es:
            model.fit(X, y, batch_size=batch_size, epochs=max_epoch, validation_split=0.2, callbacks=[es])
        else:
            model.fit(X, y, batch_size=batch_size, epochs=max_epoch, validation_split=0.2)
        return model
    elif emode == 'keras':
        model = Sequential()
        model.add(Embedding(MAX_NUM_WORDS, EMBEDDING_DIM, input_length=TOKENS_MAX_LENGTH, mask_zero=True))
        model.add(Bidirectional(LSTM(encoder_dim, activation='relu',
                                     return_sequences=False), input_shape=input_shape))
        model.add(Dropout(dropout))
        model.add(Dense(decoder_dim, activation='relu'))
        model.add(Dropout(dropout))
        model.add(Dense(output_dim, activation='sigmoid'))
        model.add(Dropout(dropout))
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        model.summary()

        es = EarlyStopping(patience=5, monitor='val_loss', restore_best_weights=True)

        if use_es:
            model.fit(X, y, batch_size=batch_size, epochs=max_epoch, validation_split=0.2, callbacks=[es])
        else:
            model.fit(X, y, batch_size=batch_size, epochs=max_epoch, validation_split=0.2)
        return model
    else:
        raise Exception('You should pass a emode')


def argmax_top_k(x, k):
    """
    given a numpy array return the top k argumet index
    :param x: numpy array
    :param k: top k
    :return: list of index
    """
    ind = x.argsort(axis=1)
    ind = ind[..., ::-1]

    return ind[:, :k]


def evaluate(model, ge, X, gold, threshold=THRESHOLD, tmode='threshold', k=K):
    y_predict = model.predict(X)

    if tmode=='threshold':
        predicts = []

        for x in y_predict:
            label_predict = []
            for i, label in enumerate(x):
                if label > threshold:
                    label_predict.append(ge.int2genres[i])

            predicts.append(label_predict)
    elif tmode == 'k':
        predicts = []
        i_m = argmax_top_k(y_predict, k)

        for i_v in i_m:
            label_predict = []
            for i in i_v:
                label_predict.append(ge.int2genres[i])

            predicts.append(label_predict)

    tp = 0
    fp = 0
    fn = 0
    total = len(gold) * ge.num_genres

    for i, x in enumerate(gold):
        for label in x:
            if label in predicts[i]:
                tp += 1
            else:
                fn += 1

    for i, x in enumerate(predicts):
        for label in x:
            if label not in gold[i]:
                fp += 1

    tn = total - tp - fp - fn

    print('True Positive: {}'.format(tp))
    print('True Negative: {}'.format(tn))
    print('False Positive: {}'.format(fp))
    print('False Negative: {}'.format(fn))

    accuracy = (tp + tn)/(tp + tn + fp + fn)
    precision = tp/(tp+fp)
    recall = tp/(tp+fn)
    f1 = 2*(recall*precision)/(recall+precision)

    print('Accuracy: {:0.4f}'.format(accuracy))
    print('Precision: {:0.4f}'.format(precision))
    print('Recall: {:0.4f}'.format(recall))
    print('F1 Score: {:0.4f}'.format(f1))


if __name__ == "__main__":
    argp = argparse.ArgumentParser(add_help=True)
    argp.add_argument('command', choices=('embedding', 'training', 'evaluating'))
    argp.add_argument('--emode', help='embedding model: onehot or fasttext', choices=('keras', 'fasttext'))
    argp.add_argument('--edim', help='embedding dimension', type=int)
    argp.add_argument('--bsize', help='batch size', type=int)
    argp.add_argument('--k', help='threshold k', type=int)
    argp.add_argument('--threshold', help='threshold value', type=float)
    argp.add_argument('--tmode', help='threshold mode', choices=('threshold', 'k'))
    argp.add_argument('--detail', action="store_true", default=False)
    argp.add_argument('--append', action="store_true", default=False)
    args = argp.parse_args()

    if args.command == 'embedding':
        if args.edim:
            EMBEDDING_DIM = args.edim
        train_embedding('../processed_data/corpus.txt', '../processed_data/embedding_{}.bin'.format(EMBEDDING_DIM))
    elif args.command == 'training':
        if args.edim:
            EMBEDDING_DIM = args.edim

        if args.bsize:
            BATCH_SIZE = args.bsize

        if args.detail:
            IS_SHORT = False
        else:
            IS_SHORT = True

        if args.append:
            APPEND = True
        else:
            APPEND = False

        if args.emode == 'fasttext':
            print('Embedding...')
            sources_train, targets_train = read_file('../processed_data/games_train.json', is_short=IS_SHORT,
                                                     append=APPEND)
            sources_test, targets_test = read_file('../processed_data/games_test.json', is_short=IS_SHORT,
                                                   append=APPEND)
            de = DescriptionEncoder('../processed_data/embedding_{}.bin'.format(EMBEDDING_DIM))

            X_train = de.transform(sources_train, dim=EMBEDDING_DIM)

            ge = GenresEncoder('../processed_data/genres')

            y_train = ge.transform(targets_train)

            print(X_train.shape)
            print(y_train.shape)

            print(len(sources_train[100]))
            # quit()

            model = train(X_train, y_train, (TOKENS_MAX_LENGTH, EMBEDDING_DIM),
                          ge.num_genres, batch_size=BATCH_SIZE, max_epoch=100, use_es=True,
                          emode='fasttext')

            model.save('fasttext_{}.h5'.format(EMBEDDING_DIM))

            X_test = de.transform(sources_test, dim=EMBEDDING_DIM)

            if args.tmode == 'threshold':
                if args.threshold:
                    THRESHOLD = args.threshold
                evaluate(model, ge, X_test, targets_test, tmode='threshold', threshold=THRESHOLD)
            elif args.tmode == 'k':
                if args.k:
                    K = argp.k
                evaluate(model, ge, X_test, targets_test, tmode='k', k=K)
            else:
                raise Exception('You should pass a threshold mode')
        elif args.emode == 'keras':
            sources_train, targets_train = read_file('../processed_data/games_train.json', emode='keras')
            sources_test, targets_test = read_file('../processed_data/games_test.json', emode='keras')

            tokenizer = Tokenizer(num_words=MAX_NUM_WORDS,
                                  filters='!"#$%&()*+,-./:;<=>?@[\]^_`{|}~', lower=True)
            tokenizer.fit_on_texts(sources_train)
            word_index = tokenizer.word_index
            x_train = tokenizer.texts_to_sequences(sources_train)
            x_train = pad_sequences(x_train, maxlen=TOKENS_MAX_LENGTH)

            ge = GenresEncoder('../processed_data/genres')
            y_train = ge.transform(targets_train)

            print(x_train.shape)
            print(y_train.shape)

            model = train(x_train, y_train, (TOKENS_MAX_LENGTH,),
                          ge.num_genres, batch_size=BATCH_SIZE, max_epoch=100, use_es=True,
                          emode='keras')

            x_test = tokenizer.texts_to_sequences(sources_test)
            x_test = pad_sequences(x_test, maxlen=TOKENS_MAX_LENGTH)

            evaluate(model, ge, x_test, targets_test)
        else:
            raise Exception('You should pass a embedding model')
    elif args.command == 'evaluating':
        model = load_model('fasttext_{}.h5'.format(EMBEDDING_DIM))

        sources_test, targets_test = read_file('../processed_data/games_test.json')
        de = DescriptionEncoder('../processed_data/embedding_{}.bin'.format(EMBEDDING_DIM))
        X_test = de.transform(sources_test, dim=EMBEDDING_DIM)
        ge = GenresEncoder('../processed_data/genres')

        if args.tmode == 'threshold':
            if args.threshold:
                THRESHOLD = args.threshold
            evaluate(model, ge, X_test, targets_test, tmode='threshold', threshold=THRESHOLD)
        elif args.tmode == 'k':
            if args.k:
                K = args.k
            evaluate(model, ge, X_test, targets_test, tmode='k', k=K)
        else:
            raise Exception('You should pass a threshold mode')
    else:
        raise Exception('You should pass a command')