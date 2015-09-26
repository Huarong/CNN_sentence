#!/usr/bin/env python
# -*- coding: utf-8 -*-


import codecs
import cPickle
from collections import defaultdict
import sys
import re

import numpy as np
import pandas as pd
import theano

# for gpu purpose
theano.config.floatX = 'float32'


def build_data_cv(train_path, cv=10, clean_string=False):
    """
    Loads data and split into 10 folds.
    split range is from 0 to cv-1
    : return: revs, vocab
    revs is a list of data dicts. One line corresponds one dict.
    vocab is a dict. The key is word and value is word count.
    """
    revs = []
    vocab = defaultdict(float)
    with codecs.open(train_path, "rb", encoding='gb18030') as f:
        for i, line in enumerate(f):
            tokens = line.strip('\n\r').split('\t')
            if len(tokens) != 2:
                sys.stderr.write('data file in line %s should have exactly two columns.' % (i + 1))
                continue
            label, sent = tokens
            if clean_string:
                orig_rev = clean_str(sent)
            else:
                orig_rev = sent.lower()
            words = set(orig_rev.split())
            for word in words:
                vocab[word] += 1
            datum = {"y": int(label),
                     "text": orig_rev,
                     "num_words": len(orig_rev.split()),
                     "split": np.random.randint(0, cv)}
            revs.append(datum)
    return revs, vocab


def get_W(word_vecs, k=300):
    """
    Get word matrix. W[i] is the vector for word indexed by i
    : return W, word_idx_map
    W is a numpy matrix.
    word_idx_map is a dict. The key is the word and the value is the word index.
    The index starts with 1.
    """
    vocab_size = len(word_vecs)
    word_idx_map = dict()
    W = np.zeros(shape=(vocab_size + 1, k), dtype=theano.config.floatX)
    W[0] = np.zeros(k, dtype=theano.config.floatX)
    i = 1
    for word in word_vecs:
        W[i] = word_vecs[word]
        word_idx_map[word] = i
        i += 1
    return W, word_idx_map


def load_bin_vec(fname, vocab):
    """
    Loads 300x1 word vecs from Google (Mikolov) word2vec.
    Ignore words not in vocab.
    : return: a dict. The key is the word, and the value is the word vector of numpy array type.
    """
    word_vecs = {}
    with open(fname, "rb") as f:
        header = f.readline()
        vocab_size, layer1_size = map(int, header.split())
        binary_len = np.dtype('float32').itemsize * layer1_size
        for line in xrange(vocab_size):
            word = []
            while True:
                ch = f.read(1)
                if ch == ' ':
                    word = ''.join(word)
                    break
                if ch != '\n':
                    word.append(ch)
            if word in vocab:
                word_vecs[word] = np.fromstring(f.read(binary_len), dtype='float32')
            else:
                f.read(binary_len)
    return word_vecs


def add_unknown_words(word_vecs, vocab, min_df=1, k=300):
    """
    For words that occur in at least min_df documents, create a separate word vector.
    0.25 is chosen so the unknown vectors have (approximately) same variance as pre-trained ones
    """
    for word in vocab:
        if word not in word_vecs and vocab[word] >= min_df:
            word_vecs[word] = np.random.uniform(-0.25, 0.25, k)


def clean_str(string, TREC=False):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Every dataset is lower cased except for TREC

    !!! do not use it
    This function will convert all the Chinese sentents to null string.

    """
    print '##### Warning: Using clean_str'
    # string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    # string = re.sub(r"\'s", " \'s", string)
    # string = re.sub(r"\'ve", " \'ve", string)
    # string = re.sub(r"n\'t", " n\'t", string)
    # string = re.sub(r"\'re", " \'re", string)
    # string = re.sub(r"\'d", " \'d", string)
    # string = re.sub(r"\'ll", " \'ll", string)
    # string = re.sub(r",", " , ", string)
    # string = re.sub(r"!", " ! ", string)
    # string = re.sub(r"\(", " \( ", string)
    # string = re.sub(r"\)", " \) ", string)
    # string = re.sub(r"\?", " \? ", string)
    # string = re.sub(r"\s{2,}", " ", string)
    return string.strip() if TREC else string.strip().lower()


# def clean_str_sst(string):
#     """
#     Tokenization/string cleaning for the SST dataset
#     """
#     string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
#     string = re.sub(r"\s{2,}", " ", string)
#     return string.strip().lower()


def main():
    # word2vec dict
    # w2v_file = sys.argv[1]
    # traning file
    # two columns. The first one is label and the second one is sentences.
    # train_path = sys.argv[2]
    w2v_file = 'data/muying_answer.seg.utf8.bin'
    train_path = 'data/train500'
    print "loading data...",
    revs, vocab = build_data_cv(train_path, cv=10, clean_string=False)
    # The max lenghth of all sentences.
    max_l = np.max(pd.DataFrame(revs)["num_words"])
    print "data loaded!"
    print "number of sentences: " + str(len(revs))
    print "vocab size: " + str(len(vocab))
    print "max sentence length: " + str(max_l)
    print "loading word2vec vectors...",
    w2v = load_bin_vec(w2v_file, vocab)
    print "word2vec loaded!"
    print "num words already in word2vec: " + str(len(w2v))
    add_unknown_words(w2v, vocab)
    W, word_idx_map = get_W(w2v)
    rand_vecs = {}
    add_unknown_words(rand_vecs, vocab)
    W2, _ = get_W(rand_vecs)
    cPickle.dump([revs, W, W2, word_idx_map, vocab], open("mr.p", "wb"))
    print "dataset created!"
    return None


if __name__ == '__main__':
    main()
