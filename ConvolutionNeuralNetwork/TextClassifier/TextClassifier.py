# -*- coding: utf-8 -*-

import numpy

import theano
import theano.tensor as T
from theano.tensor.nnet import conv

from gensim.models import Word2Vec
import sys
sys.path.insert(0, '../ImageClassifier_v2')
import Layers

class ConvLayerForSentences(object):
    """Свёрточный слой для классификации предложений"""

    def __init__(self, rng, input, filter_shape=(10, 1, 5, 400),
                 sentences_shape=None, activation=T.tanh):
        """
        Инициализирует ConvLayerForSentences с общими переменными внутренних параметров.

        :type rng: numpy.random.RandomState
        :param rng: генератор случайных чисел для инициализации весов

        :type input: theano.tensor.dtensor4
        :param input: символичный тензор предложений формата sentences_shape

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
        self.input = input

        # "num input feature maps * filter height * filter width"
        fan_in = numpy.prod(filter_shape[1:])

        # "num output feature maps * filter height * filter width"
        fan_out = (filter_shape[0] * numpy.prod(filter_shape[2:]))
        # инициализация весов рандомными числами
        W_bound = numpy.sqrt(6. / (fan_in + fan_out))
        # каждая карта входных признаков соединена с каждым фильтром,
        # поэтому и такая размерность у матрицы весов
        self.W = theano.shared(
            numpy.asarray(
                rng.uniform(low=-W_bound, high=W_bound, size=filter_shape),
                dtype=theano.config.floatX
            ),
            borrow=True
        )

        # символическое выражение, выполняющее операцию свёртки с помощью фильтров
        conv_out = conv.conv2d(
            input=input,
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
    def __init__(self, pooling_input):
        """
        Записывет максимальные значения по оси, отвечающей результату одного фильтра
        Один максимум для одного фильтра - для простейшего варианта - ширина фильтра =
        размерности слова
        :type pooling_input: tuple размерности 3: количество входных карт признаков,
                            высота - итоговое количество окон для одного фильтра. ширина = 1
        :param pooling_input: выход фильтрации
        """
        assert pooling_input.shape[2] == 1
        pooling_input.reshape(pooling_input.shape[0], pooling_input.shape[1])
        nums_filters = pooling_input.shape[0]
        max_args = T.argmax(pooling_input, axis=1)
        self.output = pooling_input[range(nums_filters), max_args]


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
                 n_filters, window, activation=T.tanh, seed=0):
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
        word_dimension = input.shape[3]

        self.layer0 = ConvLayerForSentences(rng, input=input, filter_shape=(n_filters, 1, window,
                                                                            word_dimension))

        self.layer1 = MaxOverTimePoolingLayer(self.layer0.output)
        # После этого слоя осталось ровно n_filters элементов

        self.layer2 = Layers.FullyConnectedLayer(rng, input=self.layer1.output,
                                                 n_in=self.layer1.output.shape[0],
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
        return -T.mean(T.log(self.p_y_given_x)[T.arange(y.shape[0]), y])

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