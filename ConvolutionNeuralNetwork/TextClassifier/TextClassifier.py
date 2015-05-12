# -*- coding: utf-8 -*-

import numpy
import re
from nltk.corpus import stopwords
from sklearn.base import BaseEstimator
import cPickle as pickle
from math import sqrt
import theano
import theano.tensor as T
from theano.tensor.nnet import conv
import pandas as pd
from gensim.models import Word2Vec

import sys
sys.path.insert(0, '../ImageClassifier_v2')
import Layers

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

theano.config.exception_verbosity='high'

class ConvLayerForSentences(object):
    """Свёрточный слой для классификации предложений"""

    def __init__(self, rng, input_data, filter_shape=(10, 1, 5, 100),
                 sentences_shape=None, activation=T.tanh):
        """
        Инициализирует ConvLayerForSentences с общими переменными внутренних параметров.

        :type rng: numpy.random.RandomState
        :param rng: генератор случайных чисел для инициализации весов

        :type input_data: theano.tensor.dtensor4
        :param input_data: символичный тензор предложений формата sentences_shape

        :type filter_shape: tuple или list длины 4
        :param filter_shape: (количество фильтров, количество входных каналов (для первой свёртки = 1),
                              высота фильтра = окно слов, ширина фильтра = размерность вектора слова)

        # NOTE: вероятно не нужно это указывать, если только каналов > 1, то тут нужно заморачиваться
        :type sentences_shape: tuple или list длины 4
        :param sentences_shape: (количество предложений = 1(всегда), количество каналов - обычно 1,
                                 высота изображения = длина предложения,
                                 ширина изображения = размерность ветора слова)

        :param activation: активачионная функция
        """
        if sentences_shape is not None:
            # проверяю совпадение размерности вектора слова
            assert sentences_shape[4] == filter_shape[4]
        self.input = input_data

        W_bound = int(1)
        # каждая карта входных признаков соединена с каждым фильтром,
        # поэтому и такая размерность у матрицы весов
        self.W = theano.shared(
            numpy.asarray(rng.uniform(low=-W_bound, high=W_bound,
                                      size=filter_shape),
                          dtype=theano.config.floatX
            ),
            borrow=True
        )

        # символическое выражение, выполняющее операцию свёртки с помощью фильтров
        conv_out = conv.conv2d(
            input=input_data,
            filters=self.W,
            filter_shape=filter_shape,
            image_shape=sentences_shape
        )

        # смещения - 1D тензор (вектор) - одно смещение для одного фильтра
        # filter_shape[0] - количество фильтров
        b_values = numpy.zeros((filter_shape[0],), dtype=theano.config.floatX)
        self.b = theano.shared(value=b_values, borrow=True)

        # дабавляем смещения. Изначально это 1D вектор,
        # который мы преобразовываем в тензор (1, n_filters, 1, 1)
        self.output = activation(conv_out + self.b.dimshuffle('x', 0, 'x', 'x'))

        # сохраним параметры этого слоя
        self.params = [self.W, self.b]


class MaxOverTimePoolingLayer(object):
    def __init__(self, pooling_input, n_filters):
        """
        Записывет максимальные значения по оси, отвечающей результату одного фильтра
        Один максимум для одного фильтра - для простейшего варианта - ширина фильтра =
        размерности слова
        :type pooling_input: символичный тензор размера 2:
                            (количество входных карт признаков(число фильтров) *
                            высота - итоговое количество окон предложения для одного фильтра)
        :param pooling_input: вход для пулинга
        """
        max_args = T.argmax(pooling_input, axis=1)
        self.output = pooling_input[xrange(n_filters), max_args]


class KMaxPoolingLayer(object):
    def __init__(self, pooling_input, k):
        """
        Записывает в self.output k максимальных значений входа
        :param pooling_input:  символичный тензор для пулинга

        :param k: int, количество максимальных элементов
        """
        # axis=1 так как нам нужна сортировка строк, а не столбцов
        pooling_args_sorted = T.argsort(pooling_input, axis=1)
        args_of_k_max = pooling_args_sorted[:, -k:]
        # axis=1 так как мы сортируем стоку
        args_of_k_max_sorted = T.sort(args_of_k_max, axis=1)
        # TODO: check it!
        ii = T.repeat(T.arange(pooling_input.shape[0], 'int64'), k)
        jj = args_of_k_max_sorted.flatten()

        self.output = pooling_input[ii, jj]


class CNNForSentences(object):
    """
    Свёрточная сеть с одним свёрточным слоем
    """
    def __init__(self, input, n_out, n_hidden, output_type,
                 n_filters, window, word_dimension, activation=T.tanh, seed=0):
        """
        :type input: theano.tensor.dtensor4
        :param input: символичный тензор предложений формата:
                      (количество предложений = 1(всегда), количество каналов - обычно 1,
                       высота = длина предложения,
                       ширина = размерность ветора слова)

        :param n_out: количество целевых классов классификации
        :param n_hidden:  число нейронов скрытого полносвязного слоя
        :param n_filters: число фильтров свёртки
        :type output_type: string
        :param output_type: type of decision 'softmax', 'binary', 'real'
        :param window: размер окна для фильтров
        :param activation: активационная функция
        :param seed: начальное значение для генератора случайных чисел
        """
        self.output_type = output_type
        self.softmax = T.nnet.softmax

        rng = numpy.random.RandomState(seed)
        # assert word_dimension == input.shape[3]

        self.layer0 = ConvLayerForSentences(rng, input_data=input, filter_shape=(n_filters, 1,
                                                                                 window,
                                                                                 word_dimension))
        layer1_input = self.layer0.output.dimshuffle(1, 2, 0, 3)
        layer1_input = layer1_input.flatten(2)
        # TODO: check it!
        self.layer1 = MaxOverTimePoolingLayer(layer1_input, n_filters)
        layer2_input = self.layer1.output #.flatten(1)
        # После этого слоя осталось ровно n_filters элементов
        self.layer2 = Layers.FullyConnectedLayer(rng, layer2_input,
                                                 n_in=n_filters,
                                                 n_out=n_hidden, activation=activation)

        # classify the values of the fully-connected sigmoidal layer
        self.layer3 = Layers.SoftmaxLayer(input=self.layer2.output,
                                   n_in=n_hidden, n_out=n_out)

        # CNN regularization
        self.L1 = self.layer3.L1
        self.L2_sqr = self.layer3.L2_sqr

        # create a list of all model parameters to be fit by gradient descent
        self.params = self.layer3.params + self.layer2.params + self.layer0.params

        self.y_pred = self.layer3.y_pred
        self.p_y_given_x = self.layer3.p_y_given_x

        if self.output_type == 'real':
            self.loss = lambda y: self.mse(y)
        elif self.output_type == 'binary':
            self.loss = lambda y: self.nll_binary(y)
        elif self.output_type == 'softmax':
            # push through softmax, computing vector of class-membership
            # probabilities in symbolic form
            self.loss = lambda y: self.nll_multiclass(y)
        else:
            raise NotImplementedError

    def mse(self, y):
        # error between output and target
        return T.mean((self.y_pred - y) ** 2)

    def nll_binary(self, y):
        # negative log likelihood based on binary cross entropy error
        return T.mean(T.nnet.binary_crossentropy(self.p_y_given_x, y))

    #same as negative-log-likelikhood
    def nll_multiclass(self, y):
        # negative log likelihood based on multiclass cross entropy error
        # y.shape[0] is (symbolically) the number of rows in y, i.e.,
        # number of time steps (call it T) in the sequence
        # T.arange(y.shape[0]) is a symbolic vector which will contain
        # [0,1,2,... n-1] T.log(self.p_y_given_x) is a matrix of
        # Log-Probabilities (call it LP) with one row per example and
        # one column per class LP[T.arange(y.shape[0]),y] is a vector
        # v containing [LP[0,y[0]], LP[1,y[1]], LP[2,y[2]], ...,
        # LP[n-1,y[n-1]]] and T.mean(LP[T.arange(y.shape[0]),y]) is
        # the mean (across minibatch examples) of the elements in v,
        # i.e., the mean log-likelihood across the minibatch.
        return -T.mean(T.log(self.p_y_given_x)[0, y[0]])

    def errors(self, y):
        """Return a float representing the number of errors in the sequence
        over the total number of examples in the sequence ; zero one
        loss over the size of the sequence

        :type y: theano.tensor.TensorType
        :param y: corresponds to a vector that gives for each example the
                  correct label
        """
        # check if y has same dimension of y_pred
        if y.ndim != self.y_out.ndim:
            raise TypeError('y should have the same shape as self.y_out',
                ('y', y.type, 'y_pred', self.y_pred.type))

        if self.output_type in ('binary', 'softmax'):
            # check if y is of the correct datatype
            if y.dtype.startswith('int'):
                # the T.neq operator returns a vector of 0s and 1s, where 1
                # represents a mistake in prediction
                return T.mean(T.neq(self.y_pred, y))
            else:
                raise NotImplementedError()


class TextClassifier(BaseEstimator):

    def __init__(self, learning_rate=0.1, n_epochs=3, activation='tanh', window=5,
                 n_hidden=10, n_filters=25, pooling_type='max_overtime',
                 output_type='binary', L1_reg=0.00, L2_reg=0.00, n_out=2, word_dimension=100):
        self.learning_rate = learning_rate
        self.n_hidden = n_hidden
        self.n_epochs = int(n_epochs)
        self.L1_reg = float(L1_reg)
        self.L2_reg = float(L2_reg)
        self.activation = activation
        self.output_type = output_type
        self.n_out = n_out
        self.n_filters = n_filters
        self.pooling_type = pooling_type
        self.window = window
        self.word_dimension = word_dimension

    def ready(self):
        """
        this function is called from "fit"
        """
        #input
        self.x = T.tensor4('x')
        #output (a label)
        self.y = T.ivector('y')

        if self.activation == 'tanh':
            activation = T.tanh
        elif self.activation == 'sigmoid':
            activation = T.nnet.sigmoid
        elif self.activation == 'relu':
            activation = lambda x: x * (x > 0)
        elif self.activation == 'cappedrelu':
            activation = lambda x: T.minimum(x * (x > 0), 6)
        else:
            raise NotImplementedError

        self.cnn = CNNForSentences(input=self.x,
                       n_out=self.n_out, activation=activation,
                       n_hidden=self.n_hidden, n_filters=self.n_filters,
                       output_type=self.output_type, window=self.window,
                       word_dimension=self.word_dimension)

        #self.cnn.predict expects one input.
        #we wrap those functions and pad as necessary in 'def predict' and 'def predict_proba'
        self.predict_wrap = theano.function(inputs=[self.x], outputs=self.cnn.y_pred)
        self.predict_proba_wrap = theano.function(inputs=[self.x], outputs=self.cnn.p_y_given_x)

    def score(self, X, y):
        """Returns the mean accuracy on the given test data and labels.

        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]
            Training set.

        y : array-like, shape = [n_samples]
            Labels for X.

        Returns
        -------
        z : float

        """
        return numpy.mean(self.predict(X) == y)

    def fit(self, x_train, y_train, x_test=None, y_test=None,
            validation_frequency=2, n_epochs=None):
        """ Fit model

        Pass in X_test, Y_test to compute test error and report during
        training.

        x_train : ndarray (T x n_in)
        y_train : ndarray (T x n_out)

        validation_frequency : int
            in terms of number of sequences (or number of weight updates)
        n_epochs : None (used to override self.n_epochs from init.
        """
        #prepare the CNN
        self.n_out = len(numpy.unique(y_train))
        self.ready()
        #input
        #x = T.matrix('x')
        #output (a label)
        #y = T.ivector('y')
        self.compute_error = theano.function(inputs=[self.x, self.y],
                                             outputs=self.cnn.loss(self.y))

        cost = self.cnn.loss(self.y)
            #+ self.L1_reg * self.cnn.L1\
            #+ self.L2_reg * self.cnn.L2_sqr

        # create a list of all model parameters to be fit by gradient descent
        self.params = self.cnn.params


        # Создаём список градиентов для всех параметров модели
        grads = T.grad(cost, self.params)

        # train_model это функция, которая обновляет параметры модели с помощью SGD
        # Так как модель имеет много парамметров, было бы утомтельным вручную создавать правила обновления
        # для каждой модели, поэтому мы создали updates list для автоматического прохождения по парам
        # (params[i], grads[i])
        updates = [
            (param_i, param_i - self.learning_rate * grad_i)
            for param_i, grad_i in zip(self.params, grads)
        ]

        self.train_model = theano.function([self.x, self.y], cost, updates=updates)

        if x_test is not None:
            assert(x_test is not None)
            interactive = True
        else:
            interactive = False

        # early-stopping parameters
        patience = 300  # look as this many examples regardless
        patience_increase = 2  # wait this much longer when a new best is
                               # found
        improvement_threshold = 0.995  # a relative improvement of this much is
                                       # considered significant
        validation_frequency = min(x_train.shape[0], patience / 2)
                                  # go through this many
                                  # minibatche before checking the network
                                  # on the validation set; in this case we
                                  # check every epoch
        best_test_loss = numpy.inf
        best_iter = 0
        epoch = 0
        done_looping = False

        if n_epochs is None:
            n_epochs = self.n_epochs

        n_train_samples = x_train.shape[0]
        if interactive:
            n_test_samples = x_test.shape[0]

        y_train = y_train.reshape(y_train.shape[0], 1)
        print y_train
        while (epoch < n_epochs) and (not done_looping):
            epoch += 1
            for idx in xrange(n_train_samples):

                iter = epoch * n_train_samples + idx

                x_current_input = x_train[idx].reshape(1, 1, x_train[idx].shape[0],
                                                       x_train[idx].shape[1])
                cost_ij = self.train_model(x_current_input, y_train[idx])

                if iter % validation_frequency == 0:
                    # compute loss on training set
                    train_losses = [self.compute_error(x_train[i].reshape(1, 1, x_train[idx].shape[0],
                                                       x_train[idx].shape[1]), y_train[i])
                                    for i in xrange(n_train_samples)]
                    this_train_loss = numpy.mean(train_losses)
                    if interactive:
                        test_losses = [self.compute_error(x_test[i], y_test[i])
                                       for i in xrange(n_test_samples)]
                        this_test_loss = numpy.mean(test_losses)
                        note = 'epoch %i, seq %i/%i, tr loss %f '\
                        'te loss %f lr: %f' % \
                        (epoch, idx + 1, n_train_samples,
                         this_train_loss, this_test_loss, self.learning_rate)
                        print note

                        if this_test_loss < best_test_loss:
                            #improve patience if loss improvement is good enough
                            if this_test_loss < best_test_loss *  \
                                    improvement_threshold:
                                patience = max(patience, iter * patience_increase)

                            # save best validation score and iteration number
                            best_test_loss = this_test_loss
                            best_iter = iter
                    else:
                        print "epoch %d, review %d: this train losses: %f"\
                              % (epoch, idx, this_train_loss)

                if patience <= iter:
                    done_looping = True
                    break

    def predict(self, data):
        if isinstance(data, list):
            data = numpy.array(data)
        if data.ndim == 1:
            data = numpy.array([data])
        return [self.predict_wrap(data[i]) for i in xrange(data.shape[0])]

    def predict_proba(self, data):
        if isinstance(data, list):
            data = numpy.array(data)
        if data.ndim == 1:
            data = numpy.array([data])
        return [self.predict_proba_wrap(data[i]) for i in xrange(data.shape[0])]

    def shared_dataset(self, data_xy):
        """ Load the dataset into shared variables """

        data_x, data_y = data_xy
        #shared_x = theano.shared(numpy.asarray(data_x, dtype=theano.config.floatX))
        shared_x = theano.shared(data_x)
        #shared_y = theano.shared(numpy.asarray(data_y, dtype=theano.config.floatX))
        shared_y = theano.shared(data_y)
        # TODO:
        if self.output_type in ('binary', 'softmax'):
            return shared_x, T.cast(shared_y, 'int32')
        else:
            return shared_x, shared_y

    def __getstate__(self):
        """ Return state sequence."""

        #check if we're using ubc_AI.classifier wrapper,
        #adding it's params to the state
        if hasattr(self, 'orig_class'):
            superparams = self.get_params()
            #now switch to orig. class (MetaCNN)
            oc = self.orig_class
            cc = self.__class__
            self.__class__ = oc
            params = self.get_params()
            for k, v in superparams.iteritems():
                params[k] = v
            self.__class__ = cc
        else:
            params = self.get_params()  #sklearn.BaseEstimator
        if hasattr(self, 'cnn'):
            weights = [p.get_value() for p in self.cnn.params]
        else:
            weights = []
        state = (params, weights)
        return state

    def _set_weights(self, weights):
        """ Set fittable parameters from weights sequence.

        Parameters must be in the order defined by self.params:
            W, W_in, W_out, h0, bh, by
        """
        i = iter(weights)
        if hasattr(self, 'cnn'):
            for param in self.cnn.params:
                param.set_value(i.next())

    def __setstate__(self, state):
        """ Set parameters from state sequence.

        Parameters must be in the order defined by self.params:
            W, W_in, W_out, h0, bh, by
        """
        params, weights = state
        # we may have several classes or superclasses
        for k in ['n_comp', 'use_pca', 'feature']:
            if k in params:
                self.set_params(**{k: params[k]})
                params.pop(k)

        # now switch to MetaCNN if necessary
        if hasattr(self, 'orig_class'):
            cc = self.__class__
            oc = self.orig_class
            self.__class__ = oc
            self.set_params(**params)
            self.ready()
            if len(weights) > 0:
                self._set_weights(weights)
            self.__class__ = cc
        else:
            self.set_params(**params)
            self.ready()
            self._set_weights(weights)

    def load(self, path):
        """ Load model parameters from path. """
        with open(path, 'rb') as f:
            state = pickle.load(f)
        self.__setstate__(state)

    def save_state(self, path):
        with open(path, 'w') as f:
            pickle.dump(self.__getstate__(), f)
