# -*- coding: utf-8 -*-

from Layers import SoftmaxLayer, FullyConnectedLayer, ConvLayer, ReLULayer, DropoutHiddenLayer, MaxPoolingLayer
import numpy

import theano
import os
import theano.tensor as T
from pandas import read_csv

def load_data(test_path = './../Data/test_data.csv', train_path='../Data/train_data.csv'):
    numpy.random.seed(42)

    #подгружаем ксв-файлы
    test_set = read_csv(test_path).as_matrix()
    train_set = read_csv(train_path).as_matrix()
    numpy.random.shuffle(test_set)
    numpy.random.shuffle(train_set)
    def shared_dataset(data_x, data_y, borrow=True):
        shared_x = theano.shared(numpy.asarray(data_x,dtype=theano.config.floatX),borrow=borrow)
        shared_y = theano.shared(numpy.asarray(data_y,dtype=theano.config.floatX),borrow=borrow)
        return shared_x, T.cast(shared_y, 'int32')

    test_set_x, test_set_y = shared_dataset(test_set[:,:-1], test_set[:,-1])
    train_set_x, train_set_y = shared_dataset(train_set[:,:-1], train_set[:,-1])
    rval = [(train_set_x, train_set_y),(test_set_x, test_set_y)]
    return rval

#насчет темпа обучения и числа фильтров - вопрос
def evaluate_lenet5(learning_rate=1.0, n_epochs=20, nkerns=[48, 96, 128, 128], batch_size=40):


    rng = numpy.random.RandomState(23455)
    datasets = load_data()

    train_set_x, train_set_y = datasets[0]
    test_set_x, test_set_y = datasets[1]

    # compute number of minibatches for training, validation and testing
    n_train_batches = train_set_x.get_value(borrow=True).shape[0]
    n_test_batches = test_set_x.get_value(borrow=True).shape[0]
    n_train_batches /= batch_size
    n_test_batches /= batch_size
    index = T.lscalar()  # index to a [mini]batch

    x = T.matrix('x')
    y = T.ivector('y')

    print '... building the model'

    #попробовал специфическую структуру отсюда
    #https://www.kaggle.com/c/datasciencebowl/forums/t/11279/public-start-guide-of-deep-network-with-a-score-of-1-382285
    layer0_input = x.reshape((batch_size, 1, 48, 48))

    layer0 = ConvLayer(
        rng,
        input=layer0_input,
        image_shape=(batch_size, 1, 48, 48),
        filter_shape=(nkerns[0], 1, 4, 4),

    )

    layer1 = ReLULayer(
        input=layer0.output
    )

    layer2 = MaxPoolingLayer(
        input=layer1.output,
        poolsize=(3,3)
    )

    layer3 = ConvLayer(
        rng,
        input=layer2.output,
        image_shape=(batch_size, nkerns[0], 22, 22),
        filter_shape=(nkerns[1], nkerns[0], 2, 2)
    )

    layer4 = ReLULayer(
        input=layer3.output
    )

    layer5 = ConvLayer(
        rng,
        input=layer4.output,
        image_shape=(batch_size, nkerns[1], 21, 21),
        filter_shape=(nkerns[2], nkerns[1], 2, 2)
    )

    layer6 = ReLULayer(
        input=layer5.output
    )

    layer7 = ConvLayer(
        rng,
        input=layer6.output,
        image_shape=(batch_size, nkerns[2], 20, 20),
        filter_shape=(nkerns[3], nkerns[2], 2, 2)
    )

    layer8 = MaxPoolingLayer(
        input=layer7.output,
        poolsize=(3,3)
    )

    layer9_input = layer8.output.flatten(2)

    layer9 = FullyConnectedLayer(
        rng,
        input=layer9_input,
        n_in=10368,
        n_out=batch_size,
        #activation=T.tanh
    )

    layer10 = SoftmaxLayer(
        input=layer9.output,
        n_in=batch_size,
        n_out=3
    )

    cost = layer10.negative_log_likelihood(y)
    params = layer10.params + layer9.params + layer7.params + layer5.params + layer3.params + layer0.params
    grads = T.grad(cost, params)

    updates = [(param_i, param_i - learning_rate * grad_i) for param_i, grad_i in zip(params, grads)]


    train_model = theano.function(
        [index],
        cost,
        updates=updates,
        givens={
            x: train_set_x[index * batch_size: (index + 1) * batch_size],
            y: train_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )

    test_model = theano.function(
        [index],
        layer10.errors(y),
        givens={
            x: test_set_x[index * batch_size: (index + 1) * batch_size],
            y: test_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )
    epoch = 0
    while (epoch < n_epochs):
        epoch = epoch + 1
        for minibatch_index in xrange(n_train_batches):
            iter = (epoch - 1) * n_train_batches + minibatch_index
            cost_ij = train_model(minibatch_index)
            if iter % 10 == 0:
                print 'training @ iter = ', iter
                print 'train err ', cost_ij
                test_losses = [test_model(i) for i in xrange(n_test_batches)]
                test_score = numpy.mean(test_losses)
                print(('     epoch %i, minibatch %i/%i, test error of '
                       'best model %f %%') %
                      (epoch, minibatch_index + 1, n_train_batches,test_score * 100.))

            if iter % 100 == 0:
                learning_rate = 0.1*learning_rate
                updates = [(param_i, param_i - learning_rate * grad_i) for param_i, grad_i in zip(params, grads)]
    #итог - тестовая ошибка не меняется
evaluate_lenet5()