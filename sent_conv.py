

import os
import cPickle as pickle
import numpy as np
from collections import defaultdict, OrderedDict
import theano
import theano.tensor as T
import re
import datetime
import codecs
import logging
import sys
import timeit
import ipdb

import util
from util import sigmoid, make_idx_data_cv, evaluate, get_idx_from_sent, ConfigBase, init_log
from layers import HiddenLayer, LogisticRegression

logger = None

# theano.config.optimizer = 'None'
theano.config.on_unused_input = 'ignore'


class Config(ConfigBase):
    def __init__(self, conf_path):
        super(Config, self).__init__(conf_path)
        self.train_path = ''
        self.cv_index = 0
        self.test_path = ''
        self.test_out_path = ''
        self.batch_size = 50
        self.n_epochs = 100
        self.filter_num = 100
        self.filter_hs = [3, 4, 5]
        self.L1_reg = 0.0
        self.L2_reg = 0.0
        self.n_hidden = 100
        self.out_root = './outdir'
        self.load_conf()
        self.auto_conf()

    def auto_conf(self):
        util.mkdir(self.out_root)
        self.out_dir = os.path.join(self.out_root, self.name)
        util.mkdir(self.out_dir)
        self.model_path = os.path.join(self.out_dir, 'model')
        self.log_path = os.path.join(self.out_dir, 'log')
        self.test_out_path = os.path.join(self.out_dir, 'test_out')




class ConvLayer(object):
    def __init__(self, rng, data, W=None, b=None, filter_h=2, filter_num=50, k=300):
        """
        :param data: a 3D tensor (sentence number, sentence length, word vector size).
        :param W: a matrix (filter_num, word vector size)
        :param filter_h: converlution operation window size.
        :param filter_num: the feature map number of each converlution window size.
        So the total feature maps are `filter_num`, which is
        also the size of the new vector representation of the sentence.
        """
        if W is None:
            W = np.asarray(rng.uniform(size=(filter_num, k * filter_h)),
                           dtype=theano.config.floatX
                           )
        self.W = theano.shared(value=W, name='W', borrow=True)
        # initialize the biases b as a vector of n_out 0s

        if b is None:
            b = np.asarray(rng.uniform(
                           size=(filter_num,)),
                           dtype=theano.config.floatX
                           )
        self.b = theano.shared(value=b, name='b', borrow=True)

        X_h, X_w = data.shape[1], data.shape[2]
        idx_range = T.arange(X_h - filter_h + 1)

        self.window_results, updates = theano.scan(fn=lambda i, X, filter_h: T.flatten(data[:, i: i + filter_h], outdim=2),
                                       sequences=idx_range,
                                       outputs_info=None,
                                       non_sequences=[data, filter_h]
                                       )
        self.window_results = T.transpose(self.window_results, axes=(1, 0, 2))

        c = sigmoid(T.dot(self.window_results, self.W.T) + self.b)
        # max pooling
        c_max = T.max(c, axis=1)
        self.c = c
        # c_max (sentence number, filter_num)
        self.c_max = c_max
        self.params = [self.W, self.b]


class SentConv(object):
    def __init__(self,
                 learning_rate=0.1,
                 L1_reg=0.00,
                 L2_reg=0.0001,
                 filter_hs=[3, 4, 5],
                 filter_num=100,
                 n_hidden=100,
                 n_out=2,
                 word_idx_map=None,
                 wordvec=None,
                 k=300,
                 adjust_input=False):
        """
        :type learning_rate: float
        :param learning_rate: learning rate used (factor for the stochastic
        gradient

        :type L1_reg: float
        :param L1_reg: L1-norm's weight when added to the cost (see
        regularization)

        :type L2_reg: float
        :param L2_reg: L2-norm's weight when added to the cost (see
        regularization)
        """
        self.learning_rate = learning_rate
        self.L1_reg = L1_reg
        self.L2_reg = L2_reg
        self.word_idx_map = word_idx_map
        rng = np.random.RandomState(3435)
        self.rng = rng
        self.k = k
        self.filter_num = filter_num
        self.filter_hs = filter_hs
        # Can be assigned at the fit step.
        self.batch_size = None

        self.Words = theano.shared(value=wordvec, name="Words")
        X = T.matrix('X')
        Y = T.ivector('Y')
        self.X = X
        self.Y = Y

        layer0_input = self.Words[T.cast(X.flatten(), dtype='int32')].reshape((X.shape[0], X.shape[1], self.Words.shape[1]))
        self.layer0_input = layer0_input
        c_max_list = []
        self.conv_layer_s = []
        test_case = []

        for filter_h in filter_hs:
            conv_layer = ConvLayer(rng, layer0_input, filter_h=filter_h, filter_num=filter_num, k=k)
            self.conv_layer_s.append(conv_layer)
            c_max_list.append(conv_layer.c_max)
        max_pooling_out = T.concatenate(c_max_list, axis=1)
        max_pooling_out_size = filter_num * len(filter_hs)

        self.hidden_layer = HiddenLayer(rng, max_pooling_out, max_pooling_out_size, n_hidden)

        self.lr_layer = LogisticRegression(
            input=self.hidden_layer.output,
            n_in=n_hidden,
            n_out=n_out,
        )
        # L1 norm ; one regularization option is to enforce L1 norm to
        # be small
        self.L1 = (
            sum([abs(conv_layer.W).sum() for conv_layer in self.conv_layer_s])
            + abs(self.hidden_layer.W).sum()
            + abs(self.lr_layer.W).sum()
        )

        # square of L2 norm ; one regularization option is to enforce
        # square of L2 norm to be small
        self.L2_sqr = (
            sum([(conv_layer.W ** 2).sum() for conv_layer in self.conv_layer_s])
            + (self.hidden_layer.W ** 2).sum()
            + (self.lr_layer.W ** 2).sum()
        )



        # the cost we minimize during training is the negative log likelihood of
        # the model plus the regularization terms (L1 and L2); cost is expressed
        # here symbolically
        self.cost = (
            self.negative_log_likelihood(Y)
            + self.L1_reg * self.L1
            + self.L2_reg * self.L2_sqr
        )

        # the parameters of the model are the parameters of the two layer it is
        # made out of
        self.params = []
        # also adjust the input word vectors
        if adjust_input:
            self.params.append(self.Words)
        for conv_layer in self.conv_layer_s:
            self.params += conv_layer.params
        self.params += self.hidden_layer.params
        self.params += self.lr_layer.params

    # negative log likelihood of the MLP is given by the negative
    # log likelihood of the output of the model, computed in the
    # logistic regression layer
    def negative_log_likelihood(self, Y):
        return self.lr_layer.negative_log_likelihood(Y)

    # same holds for the function computing the number of errors
    def errors(self, Y):
        return self.lr_layer.errors(Y)

    def fit(self, datasets, batch_size=50, n_epochs=400):
        train_x, train_y, valid_x, valid_y = datasets
        self.batch_size = batch_size

        # compute number of minibatches for training, validation and testing
        train_len = train_x.get_value(borrow=True).shape[0]
        valid_len = valid_x.get_value(borrow=True).shape[0]
        n_train_batches = train_len / batch_size
        if train_len % batch_size != 0:
            n_train_batches += 1
        n_valid_batches = valid_len / batch_size
        if valid_len % batch_size != 0:
            n_valid_batches += 1

        print 'number of train mini batch: %s' % n_train_batches

        ######################
        # BUILD ACTUAL MODEL #
        ######################
        print '... building the model'

        # allocate symbolic variables for the data
        index = T.lscalar()  # index to a [mini]batch
        X = self.X
        Y = self.Y
        lean_rate = T.scalar('Learning Rate')

        # compute the gradient of cost with respect to theta (sotred in params)
        # the resulting gradients will be stored in a list gparams
        gparams = [T.grad(self.cost, param) for param in self.params]

        # specify how to update the parameters of the model as a list of
        # (variable, update expression) pairs

        # given two lists of the same length, A = [a1, a2, a3, a4] and
        # B = [b1, b2, b3, b4], zip generates a list C of same size, where each
        # element is a pair formed from the two lists :
        #    C = [(a1, b1), (a2, b2), (a3, b3), (a4, b4)]
        updates = [
            (param, param - lean_rate * gparam)
            for param, gparam in zip(self.params, gparams)
        ]

        # compiling a Theano function `train_model` that returns the cost, but
        # in the same time updates the parameter of the model based on the rules
        # defined in `updates`
        train_model = theano.function(
            inputs=[index, lean_rate],
            outputs=self.cost,
            updates=updates,
            givens={
                X: train_x[index * batch_size: (index + 1) * batch_size],
                Y: train_y[index * batch_size: (index + 1) * batch_size]
            }
        )

        test_train_model = theano.function(
            inputs=[index],
            outputs=self.errors(Y),
            givens={
                X: train_x[index * batch_size: (index + 1) * batch_size],
                Y: train_y[index * batch_size: (index + 1) * batch_size]
            }
        )

        validate_model = theano.function(
            inputs=[index],
            outputs=self.errors(Y),
            givens={
                X: valid_x[index * batch_size:(index + 1) * batch_size],
                Y: valid_y[index * batch_size:(index + 1) * batch_size]
            }
        )

        ###############
        # TRAIN MODEL #
        ###############
        print '... training'

        # early-stopping parameters
        patience = 1000000  # look as this many examples regardless
        patience_increase = 2  # wait this much longer when a new best is
                               # found
        improvement_threshold = 0.9999  # a relative improvement of this much is
                                       # considered significant
        validation_frequency = min(n_train_batches, patience / 2)
                                      # go through this many
                                      # minibatche before checking the network
                                      # on the validation set; in this case we
                                      # check every epoch

        best_validation_loss = np.inf
        best_iter = 0
        test_score = 0.
        start_time = timeit.default_timer()

        epoch = 0
        done_looping = False
        last_cost = np.inf
        lean_rate = self.learning_rate
        sys.stdout.flush()

        while (epoch < n_epochs) and (not done_looping):
            epoch = epoch + 1
            avg_cost_list = []
            for minibatch_index in xrange(n_train_batches):

                minibatch_avg_cost = train_model(minibatch_index, lean_rate)
                avg_cost_list.append(minibatch_avg_cost)
                # print self.lr_layer.W.get_value()
                # iteration number
                iter = (epoch - 1) * n_train_batches + minibatch_index

                if (iter + 1) % validation_frequency == 0:
                    # print self.lr_layer.W.get_value()
                    # print self.lr_layer.b.get_value()

                    # train_losses = [test_train_model(i) for i in xrange(n_train_batches)]
                    # this_train_loss = np.mean(train_losses)


                    # # compute zero-one loss on validation set
                    # validation_losses = [validate_model(i) for i
                    #                      in xrange(n_valid_batches)]
                    # this_validation_loss = np.mean(validation_losses)
                    train_all_precison, train_label_precision, train_label_recall = \
                        self.test(train_x, train_y.owner.inputs[0].get_value(borrow=True))
                    this_train_loss = 1 - train_all_precison

                    valid_all_precison, valid_label_precision, valid_label_recall = \
                        self.test(valid_x, valid_y.owner.inputs[0].get_value(borrow=True))
                    this_validation_loss = 1 - valid_all_precison

                    avg_cost = np.mean(avg_cost_list)
                    if avg_cost >= last_cost:
                        lean_rate *= 0.95
                    last_cost = avg_cost


                    logger.info(
                        'epoch %i, learning rate: %f, avg_cost: %f, train P: %f %%, valid P: %f %%, train_1_P: %s, train_1_R: %s, valid_1_P: %s, valid_1_R: %s' %
                        (
                            epoch,
                            lean_rate,
                            avg_cost,
                            (1 - this_train_loss) * 100,
                            (1 - this_validation_loss) * 100.,
                            train_label_precision[1],
                            train_label_recall[1],
                            valid_label_precision[1],
                            valid_label_recall[1]
                        )
                    )
                    sys.stdout.flush()

                    # if we got the best validation score until now
                    if this_validation_loss < best_validation_loss:
                        #improve patience if loss improvement is good enough
                        if (
                            this_validation_loss < best_validation_loss *
                            improvement_threshold
                        ):
                        # Increase patience_increase times based on the current iteration.
                            patience = max(patience, iter * patience_increase)

                        best_validation_loss = this_validation_loss
                        best_iter = iter

                if patience <= iter:
                    done_looping = True
                    break

        end_time = timeit.default_timer()
        logger.info(('Optimization complete. Best validation score of %f %% '
               'obtained at iteration %i') %
              (( 1 - best_validation_loss) * 100., best_iter + 1))
        logger.info('The code for file ' +
                              os.path.split(__file__)[1] +
                              ' ran for %.2fm' % ((end_time - start_time) / 60.))

    def save(self, path):
        with open(path, 'wb') as f:
            pickle.dump(self, f, -1)
        logger.info('save model to path %s' % path)
        return None

    @classmethod
    def load(self, path):
        with open(path, 'rb') as f:
            return pickle.load(f)

    def predict(self, shared_x, batch_size=None):
        if not batch_size:
            batch_size = self.batch_size
        shared_x_len = shared_x.get_value(borrow=True).shape[0]
        n_batches = shared_x_len / batch_size
        if shared_x_len % batch_size != 0:
            n_batches += 1

        index = T.lscalar()  # index to a [mini]batch
        X = self.X

        predict_model = theano.function(
            inputs=[index],
            outputs=self.lr_layer.y_pred,
            givens={
                X: shared_x[index * batch_size:(index + 1) * batch_size]
            }
        )
        pred_y = np.concatenate([predict_model(i) for i in range(n_batches)])
        return pred_y

    def test(self, shared_x, data_y, out_path=None):
        pred_y = self.predict(shared_x)
        if out_path:
            with codecs.open(out_path, 'wb') as f:
                f.writelines(['%s\t%s\n' % (x, y) for x, y in zip(data_y, pred_y)])
        return evaluate(data_y, pred_y)

    def test_from_file(self, path, out_path=None, encoding='utf-8'):
        data_x = []
        data_y = []
        with codecs.open(path, 'rb', encoding=encoding) as f:
            for i, line in enumerate(f):
                tokens = line.strip('\n').split('\t')
                if len(tokens) != 2:
                    raise ValueError('invalid line %s' % (i+1))
                label = int(tokens[0])
                sent = tokens[1]
                s = get_idx_from_sent(sent, self.word_idx_map)
                data_x.append(s)
                data_y.append(label)
        shared_x = theano.shared(
            value=np.asarray(data_x, dtype=theano.config.floatX),
            borrow='True'
        )
        return self.test(shared_x, data_y, out_path=out_path)



def main():
    # mode = sys.argv[1]
    # word_vectors = sys.argv[2]
    conf_path = sys.argv[1]
    conf = Config(conf_path)
    global logger
    logger = init_log(__file__, conf.log_path)
    conf.log(logger)
    mode = '-static'
    word_vectors = '-word2vec'
    logger.info("loading data...")
    if mode not in ('-nonstatic', '-static'):
        raise ValueError('invalid parameter mode %s' % mode)
    if word_vectors not in ('-rand', '-word2vec'):
        raise ValueError('invalid parameter word_vectors %s' % word_vectors)

    with open(conf.train_path, 'rb') as f:
        x = pickle.load(f)
    revs, W, W2, word_idx_map, vocab = x[0], x[1], x[2], x[3], x[4]
    logger.info("data loaded!")
    if mode == "-nonstatic":
        print "model architecture: CNN-non-static"
        non_static = True
    elif mode == "-static":
        print "model architecture: CNN-static"
        non_static = False
    if word_vectors == "-rand":
        print "using: random vectors"
        U = W2
    elif word_vectors == "-word2vec":
        print "using: word2vec vectors"
        U = W
    sys.stdout.flush()
    results = []
    # cross validation
    datasets = make_idx_data_cv(revs, word_idx_map, conf.cv_index, max_l=15, k=300)
    sc = SentConv(filter_hs=conf.filter_hs, filter_num=conf.filter_num, n_hidden=conf.filter_num, n_out=2, word_idx_map=word_idx_map, wordvec=U, adjust_input=False)
    try:
        sc.fit(datasets, batch_size=conf.batch_size, n_epochs=conf.n_epochs)
    except KeyboardInterrupt:
        logger.warning('Got control C. Quit.')
        return
    finally:
        sc.save(conf.model_path)
    # sc = SentConv.load(conf.model_path)
    test_result = sc.test_from_file(conf.test_path, encoding='gb18030', out_path=conf.test_out_path)
    logger.info('test result of %s' % conf.test_path)
    logger.info(test_result)
    logger.info('test out path is %s' % conf.test_out_path)


if __name__ == '__main__':
    main()
