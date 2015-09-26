import os
import sys
import json
import codecs
import logging
import numpy as np
import theano
from theano import tensor as T


class ConfigError(Exception):
    def __init__(self, message):
        self.message = message

    def __str__(self):
        return repr(self.message)


class ConfigBase(object):
    def __init__(self, conf_path):
        self.path = os.path.abspath(conf_path)
        self.name = os.path.basename(conf_path)
        self.ext = self.name.split('.')[-1]

    def cast(self, key, value):
        init_value = self.__dict__[key]
        if isinstance(init_value, int):
            value = int(value)
        elif isinstance(init_value, float):
            value = float(value)
        elif isinstance(init_value, (list)):
            tokens = value.split(',')
            if init_value:
                element_type = type(init_value[0])
            value = [element_type(t) for t in tokens]
            if type(init_value) == tuple:
                value = tuple(value)
            elif type(init_value) == set:
                value = set(value)
        else:
            pass
        return value

    def load_conf(self):
        if self.ext == 'json':
            self.__load_json_conf()
        else:
            self.__load_shell_conf()
        return None

    def __load_shell_conf(self):
        with codecs.open(self.path, encoding='utf-8') as fc:
            for line in fc:
                if not line.strip():
                    continue
                if line.lstrip().startswith('#'):
                    continue
                tokens = line.rstrip().split('=')
                if len(tokens) < 2:
                    logging.warning('invalid config line: %s' % line)
                key = tokens[0]
                value = ''.join(tokens[1:])
                value = self.cast(key, value)
                self.__setattr__(key, value)
        return None

    def __load_json_conf(self):
        with codecs.open(self.path, encoding='utf-8') as fc:
            json_str = ''
            for line in fc:
                if not line.lstrip().startswith('//'):
                    json_str += line.rstrip('\n')
            jsn = json.loads(json_str)
            for key, value in jsn:
                value = self.cast(key, value)
                self.__setattr__(key, value)
        return None

    def dump(self, path):
        with codecs.open(path, 'wb', encoding='utf-8') as fp:
            for key, value in self.__dict__.items():
                fp.write('%s=%s\n' % (key, value))
        return None

    def log(self, logger):
        logger.info('log config:')
        for key, value in self.__dict__.items():
            logger.info('%s=%s' % (key, value))

    def __str__(self):
        return str(self.__dict__)

    def __unicode__(self):
        return self.__str__()


def init_log(logname, filename, level=logging.DEBUG, console=True):
    # make log file directory when not exist
    directory = os.path.dirname(filename)
    if directory:
        if not os.path.exists(directory):
            os.makedirs(directory)

    # set up logging to file - see previous section for more details
    logging.basicConfig(level=level,
                        format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s',
                        datefmt='%Y-%m-%d %H:%M',
                        filename=filename,
                        filemode='a')
    # Now, define a couple of other loggers which might represent areas in your
    # application:
    logger = logging.getLogger(logname)
    if console:
        # define a Handler which writes INFO messages or higher to the sys.stderr
        console = logging.StreamHandler()
        console.setLevel(logging.INFO)
        # set a format which is simpler for console use
        formatter = logging.Formatter('%(asctime)s %(name)-12s: %(levelname)-8s %(message)s')
        # tell the handler to use this format
        console.setFormatter(formatter)
        # add the handler to the the current logger
        logger.addHandler(console)
    return logger


def sigmoid(x):
    y = T.nnet.sigmoid(x)
    return(y)


def get_idx_from_sent(sent, word_idx_map, max_l=20, filter_h=5):
    """
    Transforms sentence into a list of indices. Pad with zeroes.

    The max length of the index list is max_l.
    """
    x = [0.0 for _ in range(max_l)]

    words = sent.split()
    idx = []
    for word in words:
        try:
            idx.append(word_idx_map[word])
        except KeyError:
            pass
    for i in range(min(len(idx), max_l)):
        x[i] = idx[i]
    return x


def make_idx_data_cv(revs, word_idx_map, cv, max_l=20, k=300):
    """
    Transforms sentences into a 2-d matrix.

    The matrix width is max_l + 2 * (filter_h - 1) + 1.
    The last columns of the matrix is the label
    """
    train_x, train_y, valid_x, valid_y = [], [], [], []
    for rev in revs:
        sent = get_idx_from_sent(rev["text"], word_idx_map, max_l)
        # Append the label
        label = rev['y']
        if rev["split"] == cv:
            valid_x.append(sent)
            valid_y.append(label)
        else:
            train_x.append(sent)
            train_y.append(label)

    def shared_dataset(data_xy, borrow=True):
        data_x, data_y = data_xy
        shared_x = theano.shared(np.asarray(data_x,
                                               dtype=theano.config.floatX),
                                 borrow=borrow)
        shared_y = theano.shared(np.asarray(data_y,
                                               dtype=theano.config.floatX),
                                 borrow=borrow)
        return shared_x, T.cast(shared_y, 'int32')

    train_x, train_y = shared_dataset((train_x, train_y))
    valid_x, valid_y = shared_dataset((valid_x, valid_y))
    return train_x, train_y, valid_x, valid_y


def make_predict_data(sents, word_idx_map, max_l=20, k=300):
    data_x = []
    for s in sents:
        sent = get_idx_from_sent(s, word_idx_map, max_l)
        data_x.append(sent)
    shared_x = theano.shared(
        value=np.asarray(data_x, dtype=theano.config.floatX),
        borrow='True'
        )
    return shared_x




def evaluate(ans, pred, labels=None):
    """
    ======================================
                 | True | False |
       positive  |  TP  |  FP   |  P
       negative  |  TN  |  FN   |  N
                 |  T   |  F    |
    ======================================
    Parameters
    ----------
    ans : array_like
        the correct answers
    pred: array_like
        the to evaluate results
    labels: array_like
        the label set. It is generated from `ans` if it is None.

    Returns
    -------
    ret : tuple
        (all_precison, label_precision, label_recall)
        The type of all_precison is float.
        The type of label_precision is a dict. The key is the label,
        and the value is the score.
        The label_recall is the same as label_precision
    """
    if len(ans) != len(pred):
        raise ValueError('The length of ans and pred must be the same, but got ans(%s) and pred(%s)' % (len(ans), len(pred)))
    if not labels:
        labels = set(ans)
    all_precison = np.mean(np.equal(ans, pred))
    label_precision = {}
    label_recall = {}
    size = len(ans)
    for label in labels:
        label_vec = np.ones(size) * label
        P = np.equal(pred, label_vec)
        T = np.equal(ans, label_vec)
        TP = np.logical_and(P, T)
        TP_sum = float(np.sum(TP))
        if TP_sum == 0:
            precision = 0.0
            recall = 0.0
        else:
            precision = TP_sum / np.sum(P)
            recall = TP_sum / np.sum(T)
        label_precision[label] = precision
        label_recall[label] = recall
    return all_precison, label_precision, label_recall


def mkdir(dir_name):
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
    return None
