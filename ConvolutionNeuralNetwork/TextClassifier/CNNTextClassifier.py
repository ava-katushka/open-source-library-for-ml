# -*- coding: utf-8 -*-
import numpy
import re
from nltk.corpus import stopwords
from sklearn.base import BaseEstimator
import cPickle as pickle
import theano
import theano.tensor as T
from theano.tensor.nnet import conv
from gensim.models import Word2Vec

import sys
sys.path.insert(0, '../../ImageClassifier_v2')
sys.path.insert(0, '../ImageClassifier_v2')
sys.path.insert(0, '../ConvolutionNeuralNetwork/ImageClassifier_v2')
import Layers


theano.config.exception_verbosity = 'high'

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)


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
                                 ширина изображения = размерность вектора слова)

        :param activation: активационная функция
        """
        if sentences_shape is not None:
            # проверяю совпадение размерности вектора слова
            assert sentences_shape[4] == filter_shape[4]
        self.input = input_data

        W_bound = 0.5
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
        # self.p_y_given_x - матрица, состоящая из 1ой строки с вероятностями каждого класса
        # T.log(self.p_y_given_x)[0, y[0]] - это вероятность настоящего класса
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
        return -T.mean(T.log(self.p_y_given_x)[0][y[0]])

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


def text_to_word_list(text, remove_stopwords=False):
    text = re.sub("[^a-zA-Z]", " ", text)
    words = text.lower().split()
    # Optionally remove stop words (false by default)
    if remove_stopwords:
        stops = set(stopwords.words("english"))
        words = [w for w in words if not w in stops]
    # 5. Return a list of words
    return words


def make_feature_matrix(words, model):
    # feature_matrix = np.zeros((len(words), num_features), dtype="float32")
    feature_matrix = []
    # counter = 0.
    for word in words:
        if word in model.vocab:
            feature_matrix.append(list(model[word]))
            # feature_matrix[counter] = model[word]
            # counter += 1
        else:
            feature_matrix.append([0 for i in xrange(model.layer1_size)])
    feature_matrix = numpy.array(feature_matrix)
    return feature_matrix


def text_to_matrix(text, model):
    words = text_to_word_list(text, remove_stopwords=False)
    matrix = make_feature_matrix(words, model)
    return matrix


class CNNTextClassifier(BaseEstimator):

    def __init__(self, learning_rate=0.1, n_epochs=3, activation='tanh', window=5,
                 n_hidden=10, n_filters=25, pooling_type='max_overtime',
                 output_type='softmax', L1_reg=0.00, L2_reg=0.00, n_out=2, word_dimension=100,
                 seed=0, model_path=None):
        """
        :param learning_rate: темп обучения
        :param n_epochs: количество эпох обучения
        :type activation: string, варианты: 'tanh', 'sigmoid', 'relu', 'cappedrelu'
        :param activation: вид активационной функции
        :param window: размер "окна" для обработки близких друг к другу слов
        :param n_hidden: число нейронов в скрытом слое
        :param n_filters: число фильтров
        :param pooling_type: тип пулинга, пока что доступен только max_overtime пулинг
        :param output_type:
        :param L1_reg:
        :param L2_reg:
        :param n_out: количество классов для классификации
        :param word_dimension: размерность слов
        :param seed: начальное значение для генератора случайных чисел
        :type model_path: string / None
        :param model_path: путь к сохранённой модели word2vec, если путь не указан, используется
                        стандартная предобученная модель
        """
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
        self.seed = seed
        self.is_ready = False
        self.model_path = model_path
        # TODO:
        if model_path is None:
            if word_dimension == 100:
                self.model_path = "100features_40minwords_10context"
            elif word_dimension == 400:
                self.model_path = "word2vec.model"
            else:
                print "There is no prepared model with dimension = %d" % word_dimension
                return
        print "Model word2vec is loading from %s." % self.model_path
        self.model = Word2Vec.load(self.model_path)
        print "Model word2vec was loaded."
        assert self.model.layer1_size == self.word_dimension

    def ready(self):
        """
        this function is called from "fit"
        """
        #input
        self.x = T.tensor4('x')
        #output (a label)
        self.y = T.lvector('y')

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
                       word_dimension=self.word_dimension, seed=self.seed)

        #self.cnn.predict expects one input.
        #we wrap those functions and pad as necessary in 'def predict' and 'def predict_proba'
        self.predict_wrap = theano.function(inputs=[self.x], outputs=self.cnn.y_pred)
        self.predict_proba_wrap = theano.function(inputs=[self.x], outputs=self.cnn.p_y_given_x)
        self.is_ready = True

    def score(self, x_data, y):
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
        return numpy.mean(self.predict(x_data) == y)

    def fit(self, x_train, y_train, x_test=None, y_test=None, n_epochs=None):
        """ Fit model

        Pass in X_test, Y_test to compute test error and report during
        training.
        :type x_train: list(string) или numpy.array(string)
        :param x_train: входные данные - список из текстов
        :type y_train: list(int)
        :param y_train: целевые значения для каждого текста

        :type n_epochs: int/None
        :param n_epochs: used to override self.n_epochs from init.
        """
        assert max(y_train) < self.n_out
        assert min(y_train) >= 0
        assert len(x_train) == len(y_train)
        print "Feature selection..."
        x_train_matrix = self.__feature_selection(x_train)
        if x_test is not None:
            assert(y_test is not None)
            assert len(x_test) == len(y_test)
            interactive = True
            x_test_matrix = self.__feature_selection(x_test)
        else:
            interactive = False
        print "Feature selection finished"


        # подготовим CNN
        if not self.is_ready:
            self.ready()
        #input
        #x = T.matrix('x')
        #output (a label)
        #y = T.lvector('y')
        # TODO: на самом деле self.y - это не вектор, а всего лишь одно число - номер класса
        self.compute_error = theano.function(inputs=[self.x, self.y],
                                             outputs=self.cnn.loss(self.y))

        cost = self.cnn.loss(self.y)
            #+ self.L1_reg * self.cnn.L1\
            #+ self.L2_reg * self.cnn.L2_sqr


        # Создаём список градиентов для всех параметров модели
        grads = T.grad(cost, self.cnn.params)

        # train_model это функция, которая обновляет параметры модели с помощью SGD
        # Так как модель имеет много парамметров, было бы утомтельным вручную создавать правила обновления
        # для каждой модели, поэтому мы создали updates list для автоматического прохождения по парам
        # (params[i], grads[i])
        updates = [
            (param_i, param_i - self.learning_rate * grad_i)
            for param_i, grad_i in zip(self.cnn.params, grads)
        ]

        self.train_model = theano.function([self.x, self.y], cost, updates=updates)

        if n_epochs is None:
            n_epochs = self.n_epochs

        n_train_samples = x_train_matrix.shape[0]
        if interactive:
            n_test_samples = x_test_matrix.shape[0]

        y_train_as_matrix = y_train.reshape(y_train.shape[0], 1)
        if interactive:
            y_test_as_matrix = y_test.reshape(y_test.shape[0], 1)

        visualization_frequency = min(2000, n_train_samples - 1)
        epoch = 0
        while epoch < n_epochs:
            epoch += 1
            # compute loss on training set
            print "start epoch %d: this TRAIN SCORE: %f"\
                  % (epoch, float(self.score(x_train, y_train)))

            for idx in xrange(n_train_samples):

                x_current_input = x_train_matrix[idx].reshape(1, 1, x_train_matrix[idx].shape[0],
                                                              x_train_matrix[idx].shape[1])
                cost_ij = self.train_model(x_current_input, y_train_as_matrix[idx])

                if idx % visualization_frequency == 0 and idx > 0:
                    # print "train cost_ij = ", cost_ij
                    if interactive:
                        test_losses = [self.compute_error(x_test_matrix[i].reshape(1, 1,
                                                                                   x_test_matrix[i]
                                                                                   .shape[0],
                                                          x_test_matrix[i].shape[1]), y_test_as_matrix[i])
                                       for i in xrange(n_test_samples)]
                        this_test_loss = numpy.mean(test_losses)
                        print "epoch %d, review %d: this test losses(score): %f, this TEST MEAN " \
                              "SCORE: %f" % (epoch, idx, float(this_test_loss),
                                             float(self.score(x_test, y_test)))

                    else:
                        # compute loss on training set
                        train_losses = [self.compute_error(x_train_matrix[i].reshape(1, 1,
                                                                                     x_train_matrix[i].shape[0],
                                                           x_train_matrix[i].shape[1]), y_train_as_matrix[i])
                                        for i in xrange(n_train_samples)]
                        this_train_loss = numpy.mean(train_losses)
                        print self.score(x_train, y_train)
                        # print "cost_ij = ", cost_ij
                        print "epoch %d, review %d: this test losses: %f"\
                              % (epoch, idx, float(this_train_loss))

        print "Fitting was finished. Test score:"
        print self.score(x_test, y_test)

    def predict(self, data):
        if isinstance(data[0], str) or isinstance(data[0], unicode):
            matrix_data = self.__feature_selection(data)
        else:
            print type(data[0])
            matrix_data = data
        if isinstance(matrix_data, list):
            matrix_data = numpy.array(matrix_data)
        return [self.predict_wrap(matrix_data[i].reshape(1, 1, matrix_data[i].shape[0],
                                  matrix_data[i].shape[1])) for i in xrange(matrix_data.shape[0])]

    def predict_proba(self, data):
        if isinstance(data[0], str):
            matrix_data = self.__feature_selection(data)
        else:
            matrix_data = data
        if isinstance(data, list):
            matrix_data = numpy.array(matrix_data)
        return [self.predict_proba_wrap(matrix_data[i].reshape(1, 1, matrix_data[i].shape[0],
                                        matrix_data[i].shape[1])) for i in xrange(matrix_data.shape[0])]

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
        else:
            print "Error in function _set_weights: there is no cnn"

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
            else:
               print "Error in function __setstate__: there is no weights"
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

    def __feature_selection(self, text_data):
        text_data_as_matrix = []
        for text in text_data:
            if not isinstance(text, str) and not isinstance(text, numpy.unicode):
                print type(text)
                raise AttributeError("feature selection error: not string format")
            text_data_as_matrix.append(text_to_matrix(text, self.model))
        text_data_as_matrix = numpy.array(text_data_as_matrix)
        return text_data_as_matrix

    def get_cnn_params(self):
        if hasattr(self, 'cnn'):
            return self.cnn.params
        else:
            print "Error in function _set_weights: there is no cnn"
