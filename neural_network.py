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
#TODO(jfriedly): Rewrite using enumerate instead of xrange and indexes.
#TODO(jfriedly): Rewrite the matrix algebra stuff using numpy or something.
#TODO(jfriedly): Use functools instead of lambdas (not any faster, but more
#                readable).
#TODO(jfriedly): Rewrite old code to support new module layout.
#TODO(jfriedly): Document this module better.
#TODO(jfriedly): Make bias neurons simply have an activation function that
#                always returns 1 rather than special-casing them.
#TODO(jfriedly): Implement more debugging level stuff.
#TODO(jfriedly): Fix the kernel perceptron to support multiclass

import random
import operator
import datetime
import sys
import numpy

from ai_algorithms import utils


class Neuron():
    """Class to represent a single neuron.
    """

    def __init__(self, activation_func=None, bias=False, input_neuron=False):
        """Initialize the neuron.

        bias and input should not both be true at the same time, even for
        the bias neuron to the first hidden layer.

        :param activation_func: Activation function for this neuron.
        :param bias: Boolean indicating that the neuron is a bias neuron.
        :param input_neuron: If False, the neuron will not be an input neuron;
                             otherwise the value of input will be this input
                             neuron's value.
        """
        if activation_func:
            self.activate = activation_func
        self.bias_neuron = bias
        if bias:
            self.value = 1
        self.input_neuron = False
        if input_neuron is not False:
            self.input_neuron = True
            self.value = input_neuron

    def __repr__(self):
        if self.bias_neuron:
            return "<Bias Neuron>"
        if self.input_neuron:
            return "<Input Neuron %s>" % self.value
        return "<Regular Neuron>"


class NeuralNetwork():
    """Class to represent a network of neurons.
    """

    def __init__(self, layer_config, activation_func=utils.sigmoid, eta=0.05,
                 use_inertia=False, alpha=1.0, activation_funcs=None,
                 kernel_func=None):
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
        :param kernel_func: If this NN should be sped up by using a kernel,
                            pass the kernel function to be used here.  Kernel
                            MLPs are not supported, only SLPs.

        self.num_layers is the number of layers (including the input layer)
        self.m is the number of input vectors
        See reset_weights() for documentation on self.weights
        """
        self.eta = eta
        self.alpha = alpha
        self.use_inertia = use_inertia
        self.layer_config = layer_config
        self.kernel_func = kernel_func
        self.num_layers = len(layer_config)
        self.m = self.layer_config[0]
        self._create_layers(activation=activation_funcs or activation_func)

        if self.kernel_func is not None:
            self._kernel = {}
            self.kernel_mutable = True

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
        if hasattr(activation, '__iter__'):
            self.layers = [[Neuron(activation_func=activation[l][n]) for n in
                            xrange(self.layer_config[l + 1])] for l in
                           xrange(len(self.layer_config[1:]))]
        else:
            self.layers = [[Neuron(activation_func=activation) for n in
                            xrange(self.layer_config[l + 1])] for l in
                           xrange(len(self.layer_config[1:]))]
        # insert the input layer
        self.layers.insert(0, [Neuron(input_neuron=True) for n in
                               xrange(self.layer_config[0])])
        # for each layer that isn't the output layer, insert a bias neuron at
        # the end
        for layer in self.layers[:-1]:
            layer.insert(len(layer), Neuron(bias=True))

    def _create_weight_matrix(self, value):
        """Returns a weight matrix with every weight set to the given value or
        a value derived from the given function.
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

    def reset_weights(self, value=None):
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

        :param value: A static value to assign all the weights to or a
                      function taking the neuron's location and returning
                      the desired weight.
        """
        if value is None:
            self.weights = self._create_weight_matrix(self._rand)
        else:
            self.weights = self._create_weight_matrix(value)

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
        if self._kernel is not None:
            # Skip the last neuron because it's a bias neuron.
            sample = [x.value for x in self.layers[0]][:self.m]
            kernel_values = [self.kernel(x, sample) for x in self.training_set]
            weighted_inputs = numpy.dot(self.alphas, kernel_values)
            #FIXME(jfriedly): This doesn't support multiclass!
            return [self.layers[-1][0].activate(weighted_inputs)]
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
    def lms_learn(self, output, desired, index=None):
        """Implements LMS learning, specifically for CSE 5526 lab2 (not
        reusable).

        :param output: Iterable of the output layer's values.
        :param desired: Iterable of the desired output values.
        :param index: Not used.  Only here to support the API.
        """
        error = desired[0] - output[0]
        for i in xrange(len(self.weights[1][0])):
            value = self.layers[1][i].value
            self.weights[1][0][i] += self.eta * value * error

    def backprop_learn(self, output, desired, index=None):
        """Runs backpropagation starting with the output layer back until
        it reaches the input layer, batch updating weights along the way.

        :param output: Iterable of the output layer's values.
        :param desired: Iterable of the desired output values.
        :param index: Not used.  Only here to support the API.
        """
        # [::-1] iterates backwards through the list, and we iterate over each
        # 2D weight matrix
        for i in range(len(self.weights))[::-1]:
            # We now iterate over every list of weights to a neuron in the
            # next layer
            for j in xrange(len(self.weights[i])):
                local_gradient = self._calc_sigmoid_local_gradient(i, j,
                                                                   output,
                                                                   desired)
                # We now iterate over every weight to a neuron in the next
                # layer
                for k in xrange(len(self.weights[i][j])):
                    self.weight_updates[i][j][k] = (self.eta * local_gradient *
                                                    self.layers[i][k].value)
        self._batch_update_weights()

    def kernel_learn(self, output, desired, index=None):
        """Updates the alphas used in the kernel perceptron.

        :param output: Iterable of the output layer's values.
        :param desired: Iterable of the desired output values.
        :param index: The index of the current training sample.
        """
        if numpy.dot(output, desired) <= 0:
            #FIXME(jfriedly): This only works for two class perceptrons!
            self.alphas[index] += desired[0]

    def use(self, sample):
        """Takes a real-life sample (not one from the training_set) and
        returns the trained neural networks' output for that sample.

        :param sample: Iterable of values to place on the input layer (not
                       including the bias neuron to the first hidden layer).
        """
        self.set_inputs(sample)
        return self.forward_propagate()

    def kernel(self, sample1, sample2, **kwargs):
        """Handles kernel access for the Neural Network.  The dot product of
        the kernel_func applied to each sample is computed, stored in the
        matrix, and returned.  If the pair is already in the matrix, it is not
        recomputed.  Once training is complete, the kernel becomes immutable.

        :param sample1: The first sample to examine.
        :param sample2: The second sample to examine.
        :param kwargs: Key word arguments passed on to the kernel function
        """
        if (tuple(sample1), tuple(sample2)) in self._kernel:
            return self._kernel[(tuple(sample1), tuple(sample2))]
        value = self.kernel_func(sample1, sample2, **kwargs)
        if self.kernel_mutable is True:
            self._kernel[(tuple(sample1), tuple(sample2))] = value
        return value

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
                 activation_func=utils.sigmoid, eta=0.05, use_inertia=False,
                 alpha=0.9, activation_funcs=None, desired_error=None,
                 max_epochs=utils.MAXINT, debug_level=1, kernel_func=None,
                 **kernel_kwargs):
        """Initializes the NeuralNetworkManager.

        layer_config will be a list containing the number of neurons in each
        consecutive layer.  The input layer is counted as a layer and all
        non-output layers will be given one extra neuron, a bias neuron.

        Example layer_config argument (for CSE 5526 lab1): [4, 4, 1]

        The activation_func argument (if provided) will be the 

        The desired_error and max_epochs arguments are used to control the
        Neural Networks's stopping condition:

        * If you want your NN to stop after a certain number of epochs, simply
          pass max_epochs.

        * If you want your NN to stop after some error limit is reached on
          each sample, just pass desired_error.

        * You may pass both desired_error and max_epochs; the NN will stop
          as soon as either condition is reached.

        The debug_level argument controls how much debugging you see.  Level 0
        means no debugging output, level 1 means print epochs only, and further
        levels are planned.


        :param training_set: An iterable of all the training samples (each an
                             iterable itself).  Must support len().
        :param desired: An iterable of the same length as training_set and
                        containing all the labels for the training set in the
                        same order, or a function on an input vector that
                        returns the desired output vector as a list.
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
        :param desired_error: The absolute error allowed for each sample in
                              the training set.
        :param max_epochs: Integer of the maximum number of epochs the
                           algorithm should run for.
        :param debug_level: Integer controlling how much debugging info is
                            output.  Output grows from 0, which is silent.
        :param kernel_func: If this NN should be sped up by using a kernel,
                            pass the kernel function to be used here.  Kernel
                            MLPs are not supported, only SLPs.
        """
        self.training_set = training_set
        self.elapsed_epochs = 0
        self.init_time = datetime.datetime.now()
        self.desired = desired
        self.desired_iter = hasattr(desired, '__iter__')
        self._stop_on_desired_error = desired_error is not None
        self.desired_error = desired_error
        self.max_epochs = max_epochs
        self.debug_level = debug_level
        self.kernel_func = kernel_func
        self.kernel_kwargs = kernel_kwargs

        # Used for debugging out elapsed epochs, which can be slow if you
        # print out too many characters.
        self._digit_length = 1
        self._next_order = 10

        self.network = NeuralNetwork(layer_config, activation_func, eta=eta,
                                     use_inertia=use_inertia, alpha=alpha,
                                     activation_funcs=activation_funcs,
                                     kernel_func=kernel_func)
        # Sanity check
        stops = (self._stop_on_desired_error or
                 (self.max_epochs != utils.MAXINT))
        assert stops, ("No stopping condition was set!  Pass desired_error or "
                       "max_epochs.")

    def compute_kernel(self):
        """Computes the kernel for this network.
        """
        print "Computing kernel (%d)  " % len(self.training_set),
        for sample1 in self.training_set:
            for sample2 in self.training_set:
                self.network.kernel(sample1, sample2, **self.kernel_kwargs)
            self.elapsed_epochs += 1
            if self.debug_level:
                self.print_epochs(0, 0, 0)
        print
        self._digit_length = 1
        self._next_order = 10
        self.network.kernel_mutable = False
        self.network.training_set = self.training_set

    def print_epochs(self, inputs, y, d):
        if self.debug_level == 1:
            sys.stdout.write('\b' * self._digit_length +
                             str(self.elapsed_epochs))
            if self._next_order == self.elapsed_epochs:
                self._next_order *= 10
                self._digit_length += 1
            sys.stdout.flush()
        elif self.debug_level == 2:
            print "Epoch %d" % self.elapsed_epochs
            print "Inputs are %s" % inputs
            print "We output %s, correct was %s" % (y, d)
            print "Weights are: %s" % self.network.weights
            print "Weight updates are: %s" % self.network.weight_updates

    def get_inputs(self):
        """Gets the next set of inputs for an epoch.
        """
        randval = self.random_order.pop()
        return randval, self.training_set[randval]

    def initialize_epoch(self):
        """Sets the random order that inputs will be drawn in.
        """
        self.random_order = random.sample(range(len(self.training_set)),
                                          len(self.training_set))
        self.errors = []

    def finalize_epoch(self):
        """Increments elapsed_epochs, check for exit condition
        """
        if self._stop_on_desired_error:
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
        for do_not_use in xrange(len(self.training_set)):
            randval, inputs = self.get_inputs()
            self.network.set_inputs(inputs)
            y = self.network.forward_propagate()
            if self.desired_iter:
                d = self.desired[randval]
            else:
                d = self.desired(inputs)
            # Don't bother to calculate error if we're ignoring it
            if self._stop_on_desired_error:
                self.errors.append(map(operator.sub, d, y))
            method(y, d, index=randval)
        if self.debug_level:
            self.print_epochs(inputs, y, d)
        if self.finalize_epoch():
            return True

    def learn(self, method):
        """Runs through epochs until an exit condition is reached using the
        desired method (currently backpropagation or LMS).  This can be
        referenced using the public ``network`` attribute of a
        NeuralNetworkManager instance

        .. code:: python

            nnm = NeuralNetworkManager(*args, **kwargs)
            nnm.learn(nnm.network.backpropagation)

        :param method: Method of the NeuralNetwork class to use as the
                       training algorithm.
        """
        self.network.reset_weights()
        if self.kernel_func is not None:
            self.compute_kernel()
            self.network.alphas = [0] * len(self.training_set)
        if self.debug_level == 1:
            self.elapsed_epochs = 0
            print "Epoch: (%d)  " % self.max_epochs,
        while not self.run_epoch(method):
            pass
        print
        print "NN trained in %d epochs after %ds." % (self.elapsed_epochs,
            (datetime.datetime.now() - self.init_time).seconds)
        self.elapsed_epochs = 0

    def __repr__(self):
        return ("<NeuralNetworkManager of %d layer network after %d epochs"
                " and %d seconds" % (len(self.network.layers) - 1,
                self.elapsed_epochs,
                (datetime.datetime.now() - self.init_time).seconds))
