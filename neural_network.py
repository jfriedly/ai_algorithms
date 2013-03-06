"""
=====================
``neural_network.py``
=====================

Contents
--------

* ``Neuron`` A single Neuron in a neural network.  Accepts a function passed
             to it to be it's activation function.  Input and output layers
             are also composed of Neuron objects, as well as bias terms.

* ``NeuralNetwork`` A feed-forward network of Neuron objects, where each
                    neuron feeds every Neuron object in the next layer. Takes
                    a layer_config argument and accepts either a single
                    activation_func argument or a list of activation functions
                    with the same dimensionality as the layer config (minus
                    the input layer).  Supports inertia for MLPs and accepts a
                    learning rate.

* ``NeuralNetworkManager`` Instantiates a NeuralNetwork, given an iterable of
                           training samples, a function that returns the
                           desired output for a given sample, and a
                           layer_config.  Stops when the desired error is
                           reached or after the desired number of epochs.
                           Accepts the same activation function, learning
                           rate, and inertia arguments as NeuralNetwork and
                           passes them along.

"""
#TODO(jfriedly): rewrite using enumerate instead of xrange and indexes
#TODO(jfriedly): rewrite the matrix algebra stuff using numpy or something
#TODO(jfriedly): use functools instead of lambdas (not any faster, but more
#                readable)
#TODO(jfriedly): rewrite old code to support new module layout
#TODO(jfriedly): document this module better
#TODO(jfriedly): Make bias neurons simply have an activation function that
#                always returns 1 rather than special-casing them.

from utils import sigmoid

import random
import operator
import datetime
import types
import sys
try:
    import pylab
    GRAPHS_DRAWABLE = True
except ImportError:
    print "Cannot import matplotlib so graphs will not be drawable.\n"
    GRAPHS_DRAWABLE = False


class Neuron():
    """Class to represent a single neuron.
    """

    def __init__(self, activation_func=None, bias=False, input=False):
        """Initialize the neuron.

        bias and input should not both be true at the same time, even for
        the bias neuron to the first hidden layer.

        :param activation_func: Activation function for this neuron.
        :param bias: Boolean indicating that the neuron is a bias neuron.
        :param input: If False, the neuron will not be an input neuron;
                      otherwise the value of input will be this input
                      neuron's value.
        """
        if activation_func:
            self.activate = activation_func
        self.bias_neuron = bias
        if bias:
            self.value = 1
        self.input_neuron = False
        if input is not False:
            self.input_neuron = True
            self.value = input

    def __repr__(self):
        if self.bias_neuron:
            return "<Bias Neuron>"
        if self.input_neuron:
            return "<Input Neuron %s>" % self.value
        return "<Regular Neuron>"


class NeuralNetwork():
    """Class to represent a network of neurons.
    """

    def __init__(self, layer_config, activation_func=sigmoid, eta=0.05,
                 use_inertia=False, alpha=1.0, activation_funcs=None):
        """Initialize the network.

        layer_config will be a list containing the number of neurons in each
        consecutive layer.  The input layer is counted as a layer and all
        non-output layers will be given one extra neuron, a bias neuron.

        Example layer_config argument (for CSE 5526 lab1): [4, 4, 1]

        :param layer_config: The layer configuration, as described above.
        :param activation_func: The single activation function to give all
                                hidden layer neurons.
        :param eta: The learning rate (sometimes called alpha).
        :param alpha: The weight applied to inertia on an MLP (if inertia is
                      enabled)
        :param use_inertia: Boolean of whether or not to apply inertia to an
                            MLP.
        :param activation_funcs: The list of activation functions to give the
                                 hidden layer neurons.  Must have the same
                                 dimensions as layer_config excluding the
                                 first element in layer_config, which will be
                                 the input layer.

        self.num_layers is the number of layers (including the input layer)
        self.m is the number of input vectors
        See reset_weights() for documentation on self.weights
        """
        self.eta = eta
        self.alpha = alpha
        self.use_inertia = use_inertia
        self.layer_config = layer_config
        self.num_layers = len(layer_config)
        self.m = self.layer_config[0]
        self._create_layers(activation=activation_funcs or activation_func)
        self.reset_weights()

        self.weight_updates = self._create_weight_matrix(0)

        if self.use_inertia:
            self.old_updates = self._create_weight_matrix(0)

    def _create_layers(self, activation=None):
        """Creates the layers by filling them with Neuron objects.

        The zeroth layer will be the input layer.
        Each non-output layer will have a bias neuron as the last neuron in
        the layer.

        :param activation: List of activation functions to assign to the
                           hidden layer neurons or a single activation
                           function to assign to every hidden layer neuron.
        """
        # create hidden layers and output layer, using a unique activation
        # function for each neuron if provided, otherwise using the same
        # for each neuron.
        try:
            iter(activation)
            self.layers = [[Neuron(activation_func=activation[l][n]) for n in
                            xrange(self.layer_config[l + 1])] for l in
                           xrange(len(self.layer_config[1:]))]
        except TypeError:
            self.layers = [[Neuron(activation_func=activation) for n in
                            xrange(self.layer_config[l + 1])] for l in
                           xrange(len(self.layer_config[1:]))]
        # insert the input layer
        self.layers.insert(0,
                           [Neuron(input=True) for n in
                            xrange(self.layer_config[0])])
        # for each layer that isn't the output layer, insert a bias neuron at
        # the end
        for layer in self.layers[:-1]:
            layer.insert(len(layer), Neuron(bias=True))

    def _create_weight_matrix(self, value):
        """Returns a weight matrix with every weight set to the given value.
        See the docstring for reset_weights() below for more info.

        :param value: A static value to assign all the weights to or a
                      function taking the neuron's location and returning
                      the desired weight.
        """
        weights = []
        # create a 2D matrix of weights for all but the input layer
        for i in xrange(1, self.num_layers):
            layer_weights = []
            # create a list of neuron weights to go to every neuron in the
            # layer
            for j in xrange(self.layer_config[i]):
                # create a weight for each neuron in the previous layer
                if hasattr(value, '__call__'):
                    neuron_weights = [value(i, j) for x in
                                      xrange(len(self._prev_layer(i)))]
                else:
                    neuron_weights = [value for x in
                                      xrange(len(self._prev_layer(i)))]
                layer_weights.append(neuron_weights)
            weights.append(layer_weights)
        return weights

    def reset_weights(self):
        """Resets the weights to random values.

        self.weights is a 3D matrix and assumes that every neuron connects to
        every other neuron:
            1) A weight for each input to...
            2) a node for each node in...
            3) each layer.

        So, self.weights[0] is the weight matrix from the input layer to the
        first hidden layer.
            self.weights[0][0] is the list of weights from the input layer to
            the first neuron in the first hidden layer
                self.weights[0][0][0] is the weight from the first input neuron
                to the first neuron in the first hidden layer.
        """
        self.weights = self._create_weight_matrix(self._rand)

    def _next_layer(self, layer):
        """Returns the next layer of neurons.

        :param layer: Index of the current layer.
        """
        return self.layers[layer + 1]

    def _prev_layer(self, layer):
        """Returns the previous layer of neurons.

        :param layer: Index of the current layer.
        """
        return self.layers[layer - 1]

    def _weights_to(self, layer):
        """Returns the weight matrix to a given layer.

        :param layer: Index of the current layer.
        """
        return self.weights[layer - 1]

    def _weights_from(self, layer):
        """Returns the weight matrix from a given layer.

        :param layer: Index of the current layer.
        """
        return self.weights[layer]

    def set_inputs(self, inputs):
        """Sets the values of the input neurons.

        :param inputs: List of inputs to go on the input neurons (not
                       including the bias neuron in the input layer).
        """
        # sanity check
        assert(len(inputs) == self.m)
        for i in xrange(self.m):
            self.layers[0][i].value = inputs[i]

    def forward_propagate(self):
        """Runs forward propogation from input layer to output layer.
        Requires the input neurons to have already been set.
        Returns a list of the final output.
        """
        # Outer for loop goes self.num_layers - 1 times because we don't
        # forward propagate from the output layer.
        for i in xrange(self.num_layers - 1):
            # Inner for loop goes self.layer_config[i+1] times because
            # we want to iterate over every node in the next layer, but not
            # bias nodes (if they exist)
            next_layer = self._next_layer(i)
            for j in xrange(self.layer_config[i + 1]):
                weighted_inputs = map(operator.mul,
                                      (neuron.value for neuron in
                                       self.layers[i]),
                                      self.weights[i][j])
                weighted_inputs = sum(weighted_inputs)
                next_layer[j].value = next_layer[j].activate(weighted_inputs)
        return [neuron.value for neuron in self.layers[-1]]

    def natural_error(self, output, desired):
        """Implements natural error-based learning.  Only works for SLPs.

        :param output: Iterable of the output layer's values.
        :param desired: Iterable of the desired output values.
        """
        error = map(lambda y, x: y - x, desired, output)
        for i in xrange(len(self.weights[0])):
            for j in xrange(len(self.weights[0][i])):
                self.weights[0][i][j] += (self.eta * error[i] *
                                          self.layers[0][j].value)

    # FIXME this won't work outside of lab2
    def lms(self, output, desired):
        """Implements LMS learning, specifically for CSE 5526 lab2 (not
        reusable).

        :param output: Iterable of the output layer's values.
        :param desired: Iterable of the desired output values.
        """
        error = desired[0] - output[0]
        for i in xrange(len(self.weights[1][0])):
            value = self.layers[1][i].value
            self.weights[1][0][i] += self.eta * value * error

    def backpropagate(self, output, desired):
        """Runs backpropagation starting with the output layer back until
        it reaches the input layer, batch updating weights along the way.

        :param output: Iterable of the output layer's values.
        :param desired: Iterable of the desired output values.
        """
        # [::-1] iterates backwards through the list, and we iterate over each
        # 2D weight matrix
        for i in range(len(self.weights))[::-1]:
            # We now iterate over every list of weights to a neuron in the
            # next layer
            for j in xrange(len(self.weights[i])):
                local_gradient = self._calc_sigmoid_local_gradient_(i, j,
                                                                    output,
                                                                    desired)
                # We now iterate over every weight to a neuron in the next
                # layer
                for k in xrange(len(self.weights[i][j])):
                    self.weight_updates[i][j][k] = (self.eta * local_gradient *
                                                    self.layers[i][k].value)
        self._batch_update_weights()

    def use(self, sample):
        """Takes a real-life sample (not one from the training_set) and
        returns the trained neural networks' output for that sample.

        :param sample: Iterable of values to place on the input layer (not
                       including the bias neuron to the first hidden layer).
        """
        self.set_inputs(sample)
        return self.forward_propagate()

    def _batch_update_weights(self):
        """Update all the weights at once as a batch so that backpropagation
        doesn't influence itself.
        """
        for i in xrange(len(self.weights)):
            for j in xrange(len(self.weights[i])):
                for k in xrange(len(self.weights[i][j])):
                    if self.use_inertia:
                        inertia = self.alpha * self.old_updates[i][j][k]
                        self.weight_updates[i][j][k] += inertia
                        # because pep8 flags on the next line otherwise
                        pep8_temp = self.weight_updates[i][j][k]
                        self.old_updates[i][j][k] = pep8_temp
                    self.weights[i][j][k] += self.weight_updates[i][j][k]

    def _calc_sigmoid_local_gradient(self, i, j, output, desired, a=1):
        """Calculates the error gradient at a neuron, assuming its activation
        function was a sigmoid function.

        :param i: The index of the layer in self.weights
        :param j: The index of the neuron in the layer
        :param output: Iterable of the output layer's values.
        :param desired: Iterable of the desired output values.
        :param a: Weight to apply to the gradient.
        """
        # If we're the first layer behind the output layer, calculate the
        # gradient directly
        if i == len(self.layer_config) - 2:
            return a * (desired[j] - output[j]) * output[j] * (1 - output[j])
        # Otherwise, use recursion to get the gradient.
        gradient = (a * self.layers[i + 1][j].value *
                    (1 - self.layers[i + 1][j].value))
        summation = sum(map(operator.mul,
                            (neuron_weights[j] for neuron_weights in
                             self.weights[i + 1]),
                            (self._calc_sigmoid_local_gradient(i + 1, new_j,
                                                            output, desired)
                             for new_j in xrange(len(self.weights[i + 1])))))
        return gradient * summation

    def _rand(self, *args):
        """Returns a random value between -1 and 1.  Ignores any arguments.
        """
        return random.random() - random.random()

    def __repr__(self):
        base = "Neural Network with %d inputs\n%d Layers:\n%s\n%d Weights:\n%s"
        return (base % (self.m, len(self.layers),
                "".join(["\tLayer Neurons: %s\n" % layer for layer in
                         self.layers]),
                len(self.weights),
                "".join(["\tLayer Weights: %s\n" % layer for layer in
                         self.weights])))


class NeuralNetworkManager():
    """Class for managing a network of neurons.  Used for setting inputs,
    running through epochs, timing, etc.
    """

    def __init__(self, training_set, desired, layer_config,
                 stop_on_desired_error=False, desired_error=0.0,
                 activation_func=sigmoid, eta=0.05, use_inertia=False,
                 alpha=0.9, activation_funcs=None, max_epochs=sys.maxint):
        """Initializes the NeuralNetworkManager.

        layer_config will be a list containing the number of neurons in each
        consecutive layer.  The input layer is counted as a layer and all
        non-output layers will be given one extra neuron, a bias neuron.

        Example layer_config argument (for CSE 5526 lab1): [4, 4, 1]

        :param training_set: An iterable of all the training samples (each an
                             iterable itself).
        :param desired: A function that, given an input vector, returns the
                        desired output vector as a list.
        :param layer_config: The layer configuration, as described above.
        :param stop_on_desired_error: Boolean indicating whether or not the
                                      algorithm should terminate when a
                                      desired error is reached.
        :param desired_error: The absolute error allowed for each sample in
                              the training set.
        :param activation_func: The single activation function to give all
                                hidden layer neurons.
        :param eta: The learning rate (sometimes called alpha).
        :param alpha: The weight applied to inertia on an MLP (if inertia is
                      enabled)
        :param use_inertia: Boolean of whether or not to apply inertia to an
                            MLP.
        :param activation_funcs: The list of activation functions to give the
                                 hidden layer neurons.  Must have the same
                                 dimensions as layer_config excluding the
                                 first element in layer_config, which will be
                                 the input layer.
        :param max_epochs: Integer of the maximum number of epochs the
                           algorithm should run for.
        """
        self.training_set = training_set
        self._elapsed_epochs = 0
        self.max_epochs = max_epochs
        self.init_time = datetime.datetime.now()
        self.desired = desired
        self.stop_on_desired_error = stop_on_desired_error
        self.desired_error = desired_error
        self.network = NeuralNetwork(layer_config, activation_func, eta=eta,
                                     use_inertia=use_inertia, alpha=alpha,
                                     activation_funcs=activation_funcs)

    @property
    def elapsed_epochs(self):
        """Turns the self._elapsed_epochs variable into a public one nicely.
        """
        return self._elapsed_epochs

    @elapsed_epochs.setter
    def elapsed_epochs(self, num_epochs):
        """Gives public write access to the self._elapsed_epochs variable.

        :param num_epochs: Number of epochs to set self._elapsed_epochs to.
        """
        self._elapsed_epochs = num_epochs

    def get_inputs(self):
        """Gets the next set of inputs for an epoch.
        """
        return self.training_set[self.random_order.pop()]

    def initialize_epoch(self):
        """Sets the random order that inputs will be drawn in.
        """
        self.random_order = random.sample(range(len(self.training_set)),
                                          len(self.training_set))
        self.errors = []

    def finalize_epoch(self):
        """Increments elapsed_epochs, check for exit condition
        """
        if self.stop_on_desired_error:
            self.errors = [abs(x[0]) for x in self.errors]
            if all(map(lambda x: x <= self.desired_error, self.errors)):
                return True
        if self.elapsed_epochs == self.max_epochs:
            return True
        self.elapsed_epochs += 1

    def run_epoch(self, method):
        """Runs through an epoch using the desired method (see below).

        :param method: Method of the NeuralNetwork class to use as the
                       training algorithm.
        """
        self.initialize_epoch()
        for x in xrange(len(self.training_set)):
            inputs = self.get_inputs()
            self.network.set_inputs(inputs)
            y = self.network.forward_propagate()
            d = self.desired(inputs)
            # Don't bother to calculate error if we're ignoring it
            if self.stop_on_desired_error:
                self.errors.append(map(operator.sub, d, y))
            method(y, d)
        print "%d" % self.elapsed_epochs,
        sys.stdout.flush()
        if self.finalize_epoch():
            return True

    def learn(self, method):
        """Runs through epochs until an exit condition is reached using the
        desired method (currently backpropagation or LMS).  This can be
        referenced using the public ``network`` attribute of a
        NeuralNetworkManager instance::

            nnm = NeuralNetworkManager(*args, **kwargs)
            nnm.learn(nnm.network.backpropagation)

        :param method: Method of the NeuralNetwork class to use as the
                       training algorithm.
        """
        self.network.reset_weights()
        while not self.run_epoch(method):
            pass
        print "-" * 720
        print "Trained in %d epochs after %ds." % (self.elapsed_epochs,
            (datetime.datetime.now() - self.init_time).seconds)
        self.elapsed_epochs = 0

    def __repr__(self):
        return ("<NeuralNetworkManager of %d layer network after %d epochs"
                " and %d seconds" % (len(self.network.layers) - 1,
                self.elapsed_epochs,
                (datetime.datetime.now() - self.init_time).seconds))
