import random
import math
import operator
import datetime
import types
import numpy
import sys
try:
    import pylab
    GRAPHS_DRAWABLE = True
except ImportError:
    print "Cannot import matplotlib so graphs will not be drawable.\n"
    GRAPHS_DRAWABLE = False
#TODO(jfriedly): rewrite the matrix algebra stuff using numpy or something
#TODO(jfriedly): use functools instead of lambdas (not any faster, but more
#                readable)


def sigmoid(v, a=1):
    """Logistic sigmoid function (default activation for a network).
    """
    return 1 / (1 + math.exp(-a * v))


def piecewise_linear(v):
    """Piecewise linear activation function.
    """
    if v < -1:
        return -1
    elif v > 1:
        return 1
    return v


def binary_threshold(v, threshold=0.5):
    """Threshold activation function returning binary values.
    """
    return 1 if v > threshold else 0


def bipolar_threshold(v, threshold=0.5):
    """Threshold activation function returning bipolar values.
    """
    return 1 if v > threshold else -1


def euclidean_distance(input1, input2):
    """Returns the Euclidean Distance between two points.
    """
    return numpy.linalg.norm(numpy.array(input1) - numpy.array(input2))


class Neuron():
    """Class to represent a single neuron.
    """

    def __init__(self, activation_func=None, bias=False, input=False):
        """Initialize the neuron.

        bias and input should not both be true at the same time, even for
        the bias neuron to the first hidden layer.
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


class Cluster():
    """Class to represent a single cluster in the K-Means algorithm.
    """

    def __init__(self, center):
        """Initialize the cluster.
        """
        self.center = center
        self.assigned_inputs = []
        self.variance = 0

    def recalculate_center(self):
        """Recalculates the center of the cluster.
        """
        # if we don't have any assigned inputs after this K-Means epoch, leave
        # the center where it was
        if self.assigned_inputs:
            new_center = []
            for dimension in xrange(len(self.assigned_inputs[0])):
                total = reduce(operator.add,
                               [x[dimension] for x in self.assigned_inputs])
                new_center.append(float(total) / len(self.assigned_inputs))
            self.center = new_center


class NeuralNetwork():
    """Class to represent a network of neurons.
    """

    def __init__(self, layer_config, activation_func=sigmoid, eta=0.05,
                 use_inertia=False, alpha=1.0, activation_funcs=None,
                 **kwargs):
        """Initialize the network.

        layer_config will be a list containing the number of neurons in each
        consecutive layer.  Extra keyword arguments are passed to the
        activation function.  Input layer is counted as a layer and all
        non-output layers are assumed to have one extra node, a bias node.

        Example layer_config argument (for first lab): [4, 4, 1]

        eta is the learning rate (sometimes called alpha)
        alpha is the weight applied to inertia (if inertia is enabled)

        self.num_layers is the number of layers (including the input layer)
        self.m is the number of input vectors
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

        value can be a curried function.
        """
        weights = []
        # create a 2D matrix of weights for all but the input layer
        for i in xrange(1, self.num_layers):
            layer_weights = []
            # create a list of neuron weights to go to every neuron in the
            # layer
            for j in xrange(self.layer_config[i]):
                # create a weight for each neuron in the previous layer
                if isinstance(value, types.MethodType):
                    neuron_weights = [value() for x in
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
        """
        return self.layers[layer + 1]

    def _prev_layer(self, layer):
        """Returns the previous layer of neurons.
        """
        return self.layers[layer - 1]

    def _weights_to(self, layer):
        """Returns the weight matrix to a given layer.
        """
        return self.weights[layer - 1]

    def _weights_from(self, layer):
        """Returns the weight matrix from a given layer.
        """
        return self.weights[layer]

    def set_inputs(self, inputs):
        """Sets the values of the input neurons.
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
        """Implements natural error-based learning.  Only works for single-layer
        perceptrons.
        """
        error = map(lambda y, x: y - x, desired, output)
        for i in xrange(len(self.weights[0])):
            for j in xrange(len(self.weights[0][i])):
                self.weights[0][i][j] += (self.eta * error[i] *
                                          self.layers[0][j].value)

    # FIXME this won't work outside of lab2
    def lms(self, output, desired):
        """Implements LMS learning, specifically for lab2 (not reusable).
        """
        error = desired[0] - output[0]
        for i in xrange(len(self.weights[1][0])):
            value = self.layers[1][i].value
            self.weights[1][0][i] += self.eta * value * error

    def backpropagate(self, output, desired):
        """Runs backpropogation starting with the output layer back until
        it reaches the input layer, updating weights along the way.
        """
        # [::-1] iterates backwards through the list, and we iterate over each
        # 2D weight matrix
        for i in range(len(self.weights))[::-1]:
            # We now iterate over every list of weights to a neuron in the
            # next layer
            for j in xrange(len(self.weights[i])):
                local_gradient = self._calculate_local_gradient(i, j, output,
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
        """
        self.set_inputs(sample)
        return self.forward_propagate()

    def _batch_update_weights(self):
        """We need to update all the weights at once as a batch or else
        backpropogation will influence itself.
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

    def _calculate_local_gradient(self, i, j, output, desired, a=1):
        """Calculates the error gradient at a neuron, assuming its activation
        function was a sigmoid function.

        i is the index of the layer (in self.weights)
        j is the index of the neuron in the layer
        """
        if i == len(self.layer_config) - 2:
            return a * (desired[j] - output[j]) * output[j] * (1 - output[j])
        gradient = (a * self.layers[i + 1][j].value *
                    (1 - self.layers[i + 1][j].value))
        summation = sum(map(operator.mul,
                            (neuron_weights[j] for neuron_weights in
                             self.weights[i + 1]),
                            (self._calculate_local_gradient(i + 1, new_j,
                                                            output, desired)
                             for new_j in xrange(len(self.weights[i + 1])))))
        return gradient * summation

    def _rand(self):
        """Returns a random value between -1 and 1.
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
                 alpha=0.9, activation_funcs=None, max_epochs=sys.maxint,
                 **kwargs):
        """Initializes the NeuralNetworkManager.

        training_set is an iterable of all the training samples.
        desired is a function that returns the desired output as a list
        desired_error is the absolute error allowed for each sample in the
        training set
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
                                     activation_funcs=activation_funcs,
                                     **kwargs)

    @property
    def elapsed_epochs(self):
        """Turns the elapsed_epochs variable into a public one nicely.
        """
        return self._elapsed_epochs

    @elapsed_epochs.setter
    def elapsed_epochs(self, value):
        """Gives public write access to the elapsed_epochs variable.
        """
        self._elapsed_epochs = value

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
        desired method (currently backpropogation or LMS).
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


class KMeansClassifier():
    """K-Means classifier.
    """

    def __init__(self, inputs, K):
        """Initialize the class.

        inputs is expected to be a list of vectors, where each vector is one
        sample.
        """
        self.inputs = inputs
        self.K = K
        self.clusters = [Cluster(random.sample(inputs, 1)[0]) for i in
                         xrange(K)]
        self.last_clusters_centers = []

    def get_variances(self, one_variance=False):
        """Returns the squared variance for each cluster in a list of length K.

        If one_variance is set to True, it will return the same variance for
        each cluster as calculated by sigma^2 = d_max^2 / 2K
        Otherwise, the variance will be determined individually using the
        following formula:

        sigma^2 = sum ||x - mu||^2          if the cluster contains more than
                                            one input
        sigma^2 = avg(all other sigma^2's)  if the cluster contains only one
                                            input
        """
        if one_variance:
            d_max = 0
            for cluster in self.clusters:
                this_d_max = max([euclidean_distance(cluster.center, x.center)
                                  for x in self.clusters])
                if this_d_max > d_max:
                    d_max = this_d_max
            for cluster in self.clusters:
                cluster.variance = (d_max / (2 * self.K)) ** 2
        else:
            one_input_clusters = []
            for cluster in self.clusters:
                if len(cluster.assigned_inputs) == 1:
                    one_input_clusters.append(cluster)
                    continue
                cluster.variance = sum([euclidean_distance(x,
                                                           cluster.center) ** 2
                                        for x in cluster.assigned_inputs])
            avg_variance = sum([cluster.variance for cluster in self.clusters])
            avg_variance /= (len(self.clusters) - len(one_input_clusters))
            for cluster in one_input_clusters:
                cluster.variance = avg_variance
        return [cluster.variance for cluster in self.clusters]

    def run_epoch(self):
        """Runs one epoch of the K-Means algorithm, assigning each input
        to a cluster and recalculating cluster centers.
        """
        for c in self.clusters:
            c.assigned_inputs = []
        for i in self.inputs:
            closest_cluster = self.clusters[0]
            min_distance = euclidean_distance(i, self.clusters[0].center)
            for c in self.clusters:
                if euclidean_distance(i, c.center) < min_distance:
                    min_distance = euclidean_distance(i, c.center)
                    closest_cluster = c
            closest_cluster.assigned_inputs.append(i)
        for c in self.clusters:
            c.recalculate_center()

    def learn(self, debug_mode=False):
        """Runs epochs until the classifier converges.
        """
        while True:
            self.last_clusters_centers = [c.center for c in self.clusters]
            self.run_epoch()
            if debug_mode and GRAPHS_DRAWABLE:
                pylab.plot(self.inputs, [1 for i in self.inputs], 'r.')
                pylab.plot([c.center[0] for c in self.clusters],
                           [1 for c in xrange(self.K)], 'p')
                pylab.show()
            if debug_mode and not GRAPHS_DRAWABLE:
                print ("Unable to draw graphs because pylab cannot be "
                       "imported\n.")
            if all(map(lambda x, y: x.center == y, self.clusters,
                       self.last_clusters_centers)):
                break
