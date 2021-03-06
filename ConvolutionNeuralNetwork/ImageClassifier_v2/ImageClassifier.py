import cPickle as pickle
from sklearn.base import BaseEstimator
import theano
import theano.tensor as T
import numpy as np


from Layers import SoftmaxLayer, LeNetConvPoolLayer, FullyConnectedLayer

class CNN(object):
    """
    Conformal Neural Network, 
    backend by Theano, but compliant with sklearn interface.

    This class holds the actual layers, while MetaCNN does
    the fit/predict routines. You should init with MetaCNN.
    
    There are three layers:
    layer0 : a convolutional filter making filters[0] shifted copies,
             then downsampled by max pooling in grids of poolsize[0]
             (N, 1, nx, ny)
             --> (N, nkerns[0], nx1, ny1)  (nx1 = nx - filters[0][0] + 1)
                                  (ny1 = ny - filters[0][1] + 1)
             --> (N, nkerns[0], nx1/poolsize[0][1], ny1/poolsize[0][1])
    layer1 : a convolutional filter making filters[1] shifted copies,
             then downsampled by max pooling in grids of poolsize[1]
             (N, nkerns[0], nx1/2, ny1/2)
             --> (N, nkerns[1], nx2, ny2) (nx2 = nx1 - filters[1][0] + 1)
             --> (N, nkerns[1], nx3, ny3) (nx3 = nx2/poolsize[1][0], ny3=ny2/poolsize[1][1])
    layer2 : hidden layer of nkerns[1]*nx3*ny3 input features and n_hidden hidden neurons
    layer3 : final LR layer with n_hidden neural inputs and n_out outputs/classes
             

    """
    def __init__(self, input, n_in, n_out, activation,
                 nkerns,
                 filters,
                 poolsize,
                 n_hidden,
                 output_type, batch_size,
                 num_layers):

        """

        :rtype : int
        n_in : width (or length) of input image (assumed square)
        n_out : number of class labels
        
        :type nkerns: list of ints
        :param nkerns: number of kernels on each layer
        
        :type filters: list of ints
        :param filters: width of convolution

        :type poolsize: list of 2-tuples
        :param poolsize: maxpooling in convolution layer (index-0),
                         and direction x or y (index-1)

        :type n_hidden: int
        :param n_hidden: number of hidden neurons
        
        :type output_type: string
        :param output_type: type of decision 'softmax', 'binary', 'real'
        
        :type batch_size: int
        :param batch_size: number of samples in each training batch. Default 200.
        """    
        self.activation = activation
        self.output_type = output_type

        #shape of input images
        nx, ny = n_in, n_in

        self.softmax = T.nnet.softmax

        # Reshape matrix of rasterized images of shape (batch_size, nx*ny)
        # to a 4D tensor, compatible with our LeNetConvPoolLayer
        _input = input.reshape((batch_size, 1, nx, ny))

        # Construct the first convolutional pooling layer:
        # filtering reduces the image size to (nx-5+1,ny-5+1)=(24,24)
        # maxpooling reduces this further to (nx/2,ny/2) = (12,12)
        # 4D output tensor is thus of shape (batch_size,nkerns[0],12,12)
        nim = filters[0]
        rng = np.random.RandomState(42)

        self.params = []
        self.layers = []

        input_size = 1
        out_size = filters[0]
        for i in range(num_layers):
            self.layers.append(LeNetConvPoolLayer(rng, input=_input,
                                      image_shape=(batch_size, input_size, nx, ny),
                                      filter_shape=(out_size, input_size, filters[i], filters[i]),
                                             poolsize=poolsize[i]))
            self.params.extend(self.layers[-1].params)
            nx = (nx - filters[i] + 1)/poolsize[i][0]
            ny = (ny - filters[i] + 1)/poolsize[i][1]
            input_size = out_size
            if i+1 < num_layers:
                out_size = nkerns[i+1]
            _input = self.layers[-1].output

        fullyconnected_input = self.layers[-1].output.flatten(2)
        self.fullyconnected = FullyConnectedLayer(rng, input=fullyconnected_input,
                                  n_in=out_size*ny*nx,
                                  n_out=n_hidden, activation=T.tanh)

        # classify the values of the fully-connected sigmoidal layer
        self.softmax = SoftmaxLayer(input=self.fullyconnected.output,
                                         n_in=n_hidden, n_out=n_out)

        self.params.extend(self.fullyconnected.params)
        self.params.extend(self.softmax.params)
        self.params.reverse()
        # CNN regularization
        self.L1 = self.softmax.L1
        self.L2_sqr = self.softmax.L2_sqr
        
        # create a list of all model parameters to be fit by gradient descent
        #self.params = self.softmax.params + self.layer2.params\
        #    + self.layer1.params + self.layer0.params

        self.y_pred = self.softmax.y_pred
        self.p_y_given_x = self.softmax.p_y_given_x

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


class ImageClassifier(BaseEstimator):
    """
    the actual CNN is not init-ed until .fit is called.
    We determine the image input size (assumed square images) and
    the number of outputs in .fit from the training data

    """
    def __init__(self, learning_rate=0.1,
                 n_epochs=3, batch_size=500, activation='tanh',
                 nkerns=[20,50],
                 n_hidden=500,
                 filters=[5,5],
                 poolsize=[(2,2),(2,2)],
                 output_type='softmax',
                 L1_reg=0.00, L2_reg=0.00,
                 n_in=50, n_out=2,
                 num_layers=2):
        self.learning_rate = learning_rate
        self.nkerns = nkerns
        self.n_hidden = n_hidden
        self.filters = filters
        self.poolsize = poolsize
        self.n_epochs = int(n_epochs)
        self.batch_size = int(batch_size)
        self.L1_reg = float(L1_reg)
        self.L2_reg = float(L2_reg)
        self.activation = activation
        self.output_type = output_type
        self.n_in = n_in
        self.n_out = n_out
        self.num_layers = num_layers
    def ready(self):
        """
        this routine is called from "fit" since we determine the
        image size (assumed square) and output labels from the training data.

        """
        #input
        self.x = T.matrix('x')
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
        
        self.cnn = CNN(input=self.x, 
                       n_in=self.n_in, 
                       n_out=self.n_out, activation=activation, 
                       nkerns=self.nkerns,
                       filters=self.filters,
                       n_hidden=self.n_hidden,
                       poolsize=self.poolsize,
                       output_type=self.output_type,
                       batch_size=self.batch_size,
                       num_layers=self.num_layers
                       )
        
        #self.cnn.predict expects batch_size number of inputs. 
        #we wrap those functions and pad as necessary in 'def predict' and 'def predict_proba'
        self.predict_wrap = theano.function(inputs=[self.x],
                                            outputs=self.cnn.y_pred
                                            )
        self.predict_proba_wrap = theano.function(inputs=[self.x],
                                                  outputs=self.cnn.p_y_given_x
                                                  )


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
        return np.mean(self.predict(X) == y)


    def fit(self, X_train, Y_train, X_test=None, Y_test=None,
            validation_frequency=2, n_epochs=None):
        """ Fit model

        Pass in X_test, Y_test to compute test error and report during
        training.

        X_train : ndarray (T x n_in)
        Y_train : ndarray (T x n_out)

        validation_frequency : int
            in terms of number of sequences (or number of weight updates)
        n_epochs : None (used to override self.n_epochs from init.
        """
        #prepare the CNN 
        self.n_in = int(np.sqrt(X_train.shape[1]))
        self.n_out = len(np.unique(Y_train))
        self.ready()

        if X_test is not None:
            assert(Y_test is not None)
            interactive = True
            test_set_x, test_set_y = self.shared_dataset((X_test, Y_test))
        else:
            interactive = False

        train_set_x, train_set_y = self.shared_dataset((X_train, Y_train))

        n_train_batches = train_set_x.get_value(borrow=True).shape[0]
        n_train_batches /= self.batch_size

        if interactive:
            n_test_batches = test_set_x.get_value(borrow=True).shape[0]
            n_test_batches /= self.batch_size

        ######################
        # BUILD ACTUAL MODEL #
        ######################

        index = T.lscalar('index')    # index to a [mini]batch

        cost = self.cnn.loss(self.y)
            #+ self.L1_reg * self.cnn.L1\
            #+ self.L2_reg * self.cnn.L2_sqr

        compute_train_error = theano.function(inputs=[index, ],
                                              outputs=self.cnn.loss(self.y),
                                              givens={
                self.x: train_set_x[index * self.batch_size: (index + 1) * self.batch_size],
                self.y: train_set_y[index * self.batch_size: (index + 1) * self.batch_size]})

        if interactive:
            compute_test_error = theano.function(inputs=[index, ],
                                                 outputs=self.cnn.loss(self.y),
                                                 givens={
                self.x: test_set_x[index * self.batch_size: (index + 1) * self.batch_size],
                self.y: test_set_y[index * self.batch_size: (index + 1) * self.batch_size]})

        # create a list of all model parameters to be fit by gradient descent
        self.params = self.cnn.params

        # create a list of gradients for all model parameters
        self.grads = T.grad(cost, self.params)
        
        # train_model is a function that updates the model parameters by
        # SGD Since this model has many parameters, it would be tedious to
        # manually create an update rule for each model parameter. We thus
        # create the updates dictionary by automatically looping over all
        # (params[i],grads[i]) pairs.
        self.updates = {}
        for param_i, grad_i in zip(self.params, self.grads):
            self.updates[param_i] = param_i - self.learning_rate * grad_i

        train_model = theano.function([index], cost, updates=self.updates,
                                      givens={
                self.x: train_set_x[index * self.batch_size: (index + 1) * self.batch_size],
                self.y: train_set_y[index * self.batch_size: (index + 1) * self.batch_size]}
                                      )

        # early-stopping parameters
        patience = 1000  # look as this many examples regardless
        patience_increase = 2  # wait this much longer when a new best is
                               # found
        improvement_threshold = 0.995  # a relative improvement of this much is
                                       # considered significant
        validation_frequency = min(n_train_batches, patience / 2)
                                  # go through this many
                                  # minibatche before checking the network
                                  # on the validation set; in this case we
                                  # check every epoch
        best_test_loss = np.inf
        best_iter = 0
        epoch = 0
        done_looping = False 

        if n_epochs is None:
            n_epochs = self.n_epochs

        while (epoch < n_epochs) and (not done_looping):
            epoch = epoch + 1
            for idx in xrange(n_train_batches):

                iter = epoch * n_train_batches + idx

                cost_ij = train_model(idx)
                
                if iter % validation_frequency == 0:
                    # compute loss on training set
                    train_losses = [compute_train_error(i)
                                    for i in xrange(n_train_batches)]
                    this_train_loss = np.mean(train_losses)
                    if interactive:
                        test_losses = [compute_test_error(i)
                                    for i in xrange(n_test_batches)]
                        this_test_loss = np.mean(test_losses)
                        note = 'epoch %i, seq %i/%i, tr loss %f '\
                        'te loss %f lr: %f' % \
                        (epoch, idx + 1, n_train_batches,
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


                if patience <= iter:
                    done_looping = True
                    break


    def predict(self, data):
        """
        the CNN expects inputs with Nsamples = self.batch_size.
        In order to run 'predict' on an arbitrary number of samples we
        pad as necessary.

        """
        if isinstance(data, list):
            data = np.array(data)
        if data.ndim == 1:
            data = np.array([data])

        nsamples = data.shape[0]
        n_batches = nsamples//self.batch_size
        n_rem = nsamples%self.batch_size
        if n_batches > 0:
            preds = [list(self.predict_wrap(data[i*self.batch_size:(i+1)*self.batch_size]))\
                                           for i in range(n_batches)]
        else:
            preds = []
        if n_rem > 0:
            z = np.zeros((self.batch_size, self.n_in * self.n_in))
            z[0:n_rem] = data[n_batches*self.batch_size:n_batches*self.batch_size+n_rem]
            preds.append(self.predict_wrap(z)[0:n_rem])
        
        return np.hstack(preds).flatten()
    
    def predict_proba(self, data):
        """
        the CNN expects inputs with Nsamples = self.batch_size.
        In order to run 'predict_proba' on an arbitrary number of samples we
        pad as necessary.

        """
        if isinstance(data, list):
            data = np.array(data)
        if data.ndim == 1:
            data = np.array([data])

        nsamples = data.shape[0]
        n_batches = nsamples//self.batch_size
        n_rem = nsamples%self.batch_size
        if n_batches > 0:
            preds = [list(self.predict_proba_wrap(data[i*self.batch_size:(i+1)*self.batch_size]))\
                                           for i in range(n_batches)]
        else:
            preds = []
        if n_rem > 0:
            z = np.zeros((self.batch_size, self.n_in * self.n_in))
            z[0:n_rem] = data[n_batches*self.batch_size:n_batches*self.batch_size+n_rem]
            preds.append(self.predict_proba_wrap(z)[0:n_rem])
        
        return np.vstack(preds)
        

    def shared_dataset(self, data_xy):
        """ Load the dataset into shared variables """

        data_x, data_y = data_xy
        shared_x = theano.shared(np.asarray(data_x,
                                            dtype=theano.config.floatX))

        shared_y = theano.shared(np.asarray(data_y,
                                            dtype=theano.config.floatX))

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
        #we may have several classes or superclasses
        for k in ['n_comp', 'use_pca', 'feature']:
            if k in params:
                self.set_params(**{k:params[k]})
                params.pop(k)

        #now switch to MetaCNN if necessary
        if hasattr(self,'orig_class'):
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
        file = open(path, 'rb')
        state = pickle.load(file)
        self.__setstate__(state)
        file.close()

