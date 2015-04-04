# -*- coding: utf-8 -*-
import os
import sys
import time

import numpy

import theano
import theano.tensor as T
from theano.tensor.signal import downsample
from theano.tensor.nnet import conv

from LR import LogisticRegression, load_data
from MLP import HiddenLayer


class LeNetConvPoolLayer(object):
    """Свёрточный слой, объединённых с пулингом """

    def __init__(self, rng, input, filter_shape, image_shape, poolsize=(2, 2)):
        """
        Инициализирует LeNetConvPoolLayer с общими переменными внутренних параметров.

        :type rng: numpy.random.RandomState
        :param rng: генератор случайных чисел для инициализации весов

        :type input: theano.tensor.dtensor4
        :param input: символичный тензор изображения формата image_shape

        :type filter_shape: tuple или list длины 4
        :param filter_shape: (количество фильтров, количество каналов входного изображения,
                              высота фильтра, ширина фильтра)

        :type image_shape: tuple или list длины 4
        :param image_shape: (размер пакета, количество входных каналов(карт признаков),
                             высота изображения, ширина изображения)

        :type poolsize: tuple или list длины 2
        :param poolsize: коэффициенты уменьшения размерности(pooling) (#rows, #cols)
        """

        assert image_shape[1] == filter_shape[1]
        self.input = input

        # there are "num input feature maps * filter height * filter width"
        # inputs to each hidden unit
        # размерность входа одного объеква из пакета, numpy.prod
        fan_in = numpy.prod(filter_shape[1:])
        # each unit in the lower layer receives a gradient from:
        # "num output feature maps * filter height * filter width" /
        #   pooling size
        # размерность выхода одного объеква из пакета
        fan_out = (filter_shape[0] * numpy.prod(filter_shape[2:]) /
                   numpy.prod(poolsize))
        # инициализация весов рандомными числами
        # достаточно странная граница
        W_bound = numpy.sqrt(6. / (fan_in + fan_out))
        # каждый канал соединён с каждым фильтром, поэтому и такая размерность у матрицы весов
        self.W = theano.shared(
            numpy.asarray(
                rng.uniform(low=-W_bound, high=W_bound, size=filter_shape),
                dtype=theano.config.floatX
            ),
            borrow=True
        )

        # смещения - 1D тензор (вектор) -- одно смещение для одной карты признаков
        # filter_shape[0] - количество фильтров
        b_values = numpy.zeros((filter_shape[0],), dtype=theano.config.floatX)
        self.b = theano.shared(value=b_values, borrow=True)

        # символическое выражение, выполняющее операцию свёртки с помощью фильтров
        conv_out = conv.conv2d(
            input=input,
            filters=self.W,
            filter_shape=filter_shape,
            image_shape=image_shape
        )

        # символическое выражение, уменьшающее размерность каждой карты признаков отдельно (каждого канала)
        # используя maxpooling
        pooled_out = downsample.max_pool_2d(
            input=conv_out,
            ds=poolsize,
            ignore_border=True
        )

        # дабавляем смещения. Изначально это 1D вектор, который мы преобразовываем в тензор (1, n_filters, 1, 1)
        # Each bias will thus be broadcasted across mini-batches and feature map
        # width & height
        self.output = T.tanh(pooled_out + self.b.dimshuffle('x', 0, 'x', 'x'))
        # Не обязательно тут использовать T.tanh, можно любую другую, например:
        # self.output = T.nnet.sigmoid(pooled_out + self.b.dimshuffle('x', 0, 'x', 'x'))

        # сохраним параметры этого слоя
        self.params = [self.W, self.b]


class LeNet5(object):
    """ Свёрточная нейронная сеть (LeNet - 5), изначально архитектура была описана в статье 1998 года:
        http://yann.lecun.com/exdb/publis/pdf/lecun-98.pdf
        Затем была немного упрощена и описана а этом источнике:
        http://deeplearning.net/tutorial/lenet.html
        Нами была добавлен более гибкий выбор гимерпараметров сети и возможность изменения её архитектуры
        в процессе инициализации
    """

    def __init__(self, n_conv_layers=2, filter_shapes=((20, 1, 5, 5), (50, 20, 5, 5)),
                 image_shape=(500, 1, 28, 28), poolsize=(2, 2),  n_hidden_neurons = 500,
                 learning_rate=0.1, dataset='mnist.pkl.gz'):
        """
        Инициализирует сеть, в соответствии с переданными параметрами.

        :type n_conv_layers: int > 0
        :param n_conv_layers: количество свёрточных слоёв

        :type filter_shapes: tuple, длины n_conv_layers, состоящий из описаний фильтров(tuple длины 4)
        :param filter_shapes: каждый filter_shape имеет следующий формат:
                              (количество фильтров, количество входных каналов, высота фильтра, ширина фильтра)
                              количество фильтров соотвествует количеству выходных каналов

        :type image_shape: tuple или list длины 4
        :param image_shape: (размер одного пакета, количество входных каналов(карт признаков),
                             высота изображения, ширина изображения)

        :param poolsize: tuple длины 2

        :type n_hidden_neurons: int > 0
        :param n_hidden_neurons: количество нейронов в полносвязном скрытом слое

        :type learning_rate: double
        :param learning_rate: параметр, отвественный за скорость обучения методом градиентного спуска
        """

        # Проверка корректности входных параметров
        assert len(image_shape) == 4
        assert len(filter_shapes) == n_conv_layers
        assert len(poolsize) == 2

        self.batch_size = image_shape[0]

        self.rng = numpy.random.RandomState(23455)

        # резервируем символические переменные для данных

        ######################
        # BUILD ACTUAL MODEL #
        ######################
        print '... building the model'

        x = T.matrix('x')   # the data is presented as rasterized images
        y = T.ivector('y')  # the labels are presented as 1D vector of [int] labels

        layer_input = x.reshape(image_shape)

        # лист из параметров скрытых слоёв
        params = []

        # Инициализирую сверточные слоя, в соответствии с архитектурой, указанной в параметрах
        for i in xrange(n_conv_layers):
            batch_size, n_filter_maps, image_height, image_weight = image_shape
            n_filters, n_input_filter_maps, filer_heigth, filter_weight = filter_shapes[i]
            pool_height, pool_weight = poolsize

            # Построение свёртного слоя + пулинг
            conv_layer = LeNetConvPoolLayer(
                self.rng,
                input=layer_input,
                image_shape=image_shape,
                filter_shape=filter_shapes[i],
                poolsize=poolsize
            )

            layer_input = conv_layer.output
            # Сохраняю параметры сети
            params.append(conv_layer.params)

            # Фильтр сокращает размер изображение, новый размер: (28-5+1 , 28-5+1) = (24, 24)
            image_height = image_height - filer_heigth + 1
            image_weight = image_weight - filter_weight + 1

            # maxpooling также сокращает размер, новый размер: (24/2, 24/2) = (12, 12)
            image_height /= pool_height
            image_weight /= pool_weight

            # Таким образом новый размер тензора изображения: (batch_size, 20, 12, 12)
            image_shape = (batch_size, n_filters, image_height, image_weight)

        # Результат прохождения изображений через свёрточные слои записан в layer_input
        # Размер потока данный теперь соответсвует image_shape
        # Если в сети два слоя с дефолтными параметрами, то выходное изображением имеет формат:
        # filtering reduces the image size to (12-5+1, 12-5+1) = (8, 8)
        # maxpooling reduces this further to (8/2, 8/2) = (4, 4)
        # 4D output tensor is thus of shape (batch_size, 50, 4, 4)
        batch_size, n_filter_maps, image_height, image_weight = image_shape

        # Полносвязный скытый слой принимает на вход 2D матрицу, но у нас есть 4D тензор
        # Поэтому превращаем тензор в матрицу размера (batch_size, n_filters * image_height * image_weight)
        fully_connected_layer_input = layer_input.flatten(2)

        n_filter_maps = image_shape[1]
        # n_in: размерность входа
        # n_out: количество нейронов в скрытом слое
        # Построение полносвязного слоя
        # TODO: тут по-хорошему можно задавать активационную функцию в качестве параметра
        fully_connected_layer = HiddenLayer(
            self.rng,
            input=fully_connected_layer_input,
            n_in=n_filter_maps * image_height * image_weight,
            n_out=n_hidden_neurons,
            activation=T.tanh
        )

        # classify the values of the fully-connected sigmoidal layer
        logistic_regression_layer = LogisticRegression(input=fully_connected_layer.output,
                                                       n_in=n_hidden_neurons, n_out=10)

        # the cost we minimize during training is the NLL of the model
        cost = logistic_regression_layer.negative_log_likelihood(y)

        self.load(dataset)
        index = T.lscalar()  # индекс пакета
        # Создаём функцию, подсчитывающую ошибку модели
        self.test_model = theano.function(
            [index],
            logistic_regression_layer.errors(y),
            givens={
                x: self.test_set_x[index * self.batch_size: (index + 1) * self.batch_size],
                y: self.test_set_y[index * self.batch_size: (index + 1) * self.batch_size]
            }
        )

        self.validate_model = theano.function(
            [index],
            logistic_regression_layer.errors(y),
            givens={
                x: self.valid_set_x[index * self.batch_size: (index + 1) * self.batch_size],
                y: self.valid_set_y[index * self.batch_size: (index + 1) * self.batch_size]
            }
        )

        self.inverted_params = logistic_regression_layer.params + fully_connected_layer.params
        for i in xrange(n_conv_layers - 1, -1, -1):
            self.inverted_params += params[i]

        # Создаём список градиентов для всех параметров модели
        grads = T.grad(cost, self.inverted_params)

        # train_model это функция, которая обновляет параметры модели с помощью SGD
        # Так как модель имеет много парамметров, было бы утомтельным вручную создавать правила обновления
        # для каждой модели, поэтому мы создали updates list для автоматического прохождения по парам
        # (params[i], grads[i])
        updates = [
            (param_i, param_i - learning_rate * grad_i)
            for param_i, grad_i in zip(self.inverted_params, grads)
        ]

        self.train_model = theano.function(
            [index],
            cost,
            updates=updates,
            givens={
                x: self.train_set_x[index * self.batch_size: (index + 1) * self.batch_size],
                y: self.train_set_y[index * self.batch_size: (index + 1) * self.batch_size]
            }
        )

    def load(self, dataset='mnist.pkl.gz'):
        self.datasets = load_data(dataset)
        self.train_set_x, self.train_set_y = self.datasets[0]
        self.valid_set_x, self.valid_set_y = self.datasets[1]
        self.test_set_x, self.test_set_y = self.datasets[2]

        # Определяем количество мини-пакетов для обучения, валидации и тестирования
        self.n_train_batches = self.train_set_x.get_value(borrow=True).shape[0]
        self.n_valid_batches = self.valid_set_x.get_value(borrow=True).shape[0]
        self.n_test_batches = self.test_set_x.get_value(borrow=True).shape[0]
        self.n_train_batches /= self.batch_size
        self.n_valid_batches /= self.batch_size
        self.n_test_batches /= self.batch_size

    def train(self, n_epochs=200):
        """
        Обучение модели
        :type n_epochs: int > 0
        :param n_epochs: максимальное количество эпох для обучения

        """
        ###############
        # TRAIN MODEL #
        ###############
        print '... training'
        # early-stopping parameters
        patience = 10000  # look as this many examples regardless
        patience_increase = 2  # wait this much longer when a new best is
                               # found
        improvement_threshold = 0.995  # a relative improvement of this much is considered significant
        validation_frequency = min(self.n_train_batches, patience / 2)
        # go through this many minibatche before checking the network
        # on the validation set; in this case we check every epoch

        best_validation_loss = numpy.inf
        best_iter = 0
        test_score = 0.
        start_time = time.clock()

        epoch = 0
        done_looping = False

        while (epoch < n_epochs) and (not done_looping):
            epoch = epoch + 1
            for minibatch_index in xrange(self.n_train_batches):

                iter = (epoch - 1) * self.n_train_batches + minibatch_index

                if iter % 10 == 0:
                    print 'training @ iter = ', iter

                cost_ij = self.train_model(minibatch_index)

                if (iter + 1) % validation_frequency == 0:

                    # compute zero-one loss on validation set
                    validation_losses = [self.validate_model(i) for i
                                         in xrange(self.n_valid_batches)]
                    this_validation_loss = numpy.mean(validation_losses)
                    print('epoch %i, minibatch %i/%i, validation error %f %%' %
                          (epoch, minibatch_index + 1, self.n_train_batches,
                           this_validation_loss * 100.))

                    # if we got the best validation score until now
                    if this_validation_loss < best_validation_loss:

                        #improve patience if loss improvement is good enough
                        if this_validation_loss < best_validation_loss *  \
                           improvement_threshold:
                            patience = max(patience, iter * patience_increase)

                        # save best validation score and iteration number
                        best_validation_loss = this_validation_loss
                        best_iter = iter

                        # test it on the test set
                        test_losses = [
                            self.test_model(i)
                            for i in xrange(self.n_test_batches)
                        ]
                        test_score = numpy.mean(test_losses)
                        print(('     epoch %i, minibatch %i/%i, test error of '
                               'best model %f %%') %
                              (epoch, minibatch_index + 1, self.n_train_batches,
                               test_score * 100.))

                if patience <= iter:
                    done_looping = True
                    break

        end_time = time.clock()
        print('Optimization complete.')
        print('Best validation score of %f %% obtained at iteration %i, '
              'with test performance %f %%' %
              (best_validation_loss * 100., best_iter + 1, test_score * 100.))
        print >> sys.stderr, ('The code for file ' +
                              os.path.split(__file__)[1] +
                              ' ran for %.2fm' % ((end_time - start_time) / 60.))

    # TODO:
    def save_result(self):
        pass

if __name__ == '__main__':
    network = LeNet5()
    network.train()

'''
# Изначально пыталась отделить инициализацию сети от загрузки модели
# но возникли проблемы с theano - выражениями
    def complete_model(self):
        """
        Вызывается после метода load, конкретизирует модель
        """
        print '... complete model'
        self.test_model = theano.function(
            [self.index],
            self.model_error(self.index, self.test_set_x, self.test_set_y)
        )

        self.validate_model = theano.function(
            [self.index],
            self.model_error(self.index, self.valid_set_x, self.valid_set_y)
        )

        self.train_model = theano.function(
            [self.index],
            self.train_model(self.index, self.train_set_x, self.train_set_y)
        )
'''
