"""
==============
``k_means.py``
==============

Contents
--------

* ``Cluster`` A class representing one cluster in K-Means clustering,
              parameterized only by an initial center.

* ``KMeansClusterManager`` A class representing an entire run through,
                           parameterized by a list of data points and a
                           number of clusters.
"""
from ai_algorithms import utils

import random
import operator
try:
    import pylab
    GRAPHS_DRAWABLE = True
except ImportError:
    print "Cannot import matplotlib so graphs will not be drawable.\n"
    GRAPHS_DRAWABLE = False


class Cluster():
    """Class to represent a single cluster in the K-Means algorithm.
    """

    def __init__(self, center):
        """Initialize the cluster.

        :param center: Initial center for the cluster (typically a random
                       data point).
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

    def __repr__(self):
        return ("<Cluster centered at %s with assigned inputs: %s"
                % (self.center, self.assigned_inputs))


class KMeansClusterManager():
    """Manager class for K-Means clustering.
    """

    def __init__(self, inputs, K):
        """Initialize the clusters using Forgy initialization:  cluster
        centroids begin as random data points.

        :param inputs: A list of vectors, where each vector is one sample and
                       is represented as an iterable of numbers.
        :param K: The number of clusters to create.
        """
        self.inputs = inputs
        # Will throw an error if len(inputs) == 0
        self.dimensionality = len(inputs[0])
        self.K = K
        self.clusters = [Cluster(random.sample(inputs, 1)[0]) for i in
                         xrange(K)]
        self.last_clusters_centers = []
        self._elapsed_epochs = 0

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

        :param one_variance: Boolean of indicating how to calculate the
                             variances.
        """
        if one_variance:
            d_max = 0
            for cluster in self.clusters:
                this_d_max = max([utils.euclidean_distance(cluster.center,
                                                           x.center)
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
                cluster.variance = sum([utils.euclidean_distance(x,
                                                                 cluster.center)
                                        ** 2 for x in cluster.assigned_inputs])
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
            min_distance = utils.euclidean_distance(i, self.clusters[0].center)
            for c in self.clusters:
                if utils.euclidean_distance(i, c.center) < min_distance:
                    min_distance = utils.euclidean_distance(i, c.center)
                    closest_cluster = c
            closest_cluster.assigned_inputs.append(i)
        for c in self.clusters:
            c.recalculate_center()
        self._elapsed_epochs += 1

    def draw_graph(self):
        """Draws a graph of all the data points and all the cluster centroids
        in up to two dimensions.
        """
        if GRAPHS_DRAWABLE:
            # handle one dimensional inputs
            if self.dimensionality == 1:
                pylab.plot(self.inputs, [1 for i in self.inputs], 'r.')
                pylab.plot([c.center[0] for c in self.clusters],
                           [1 for c in xrange(self.K)], 'p')
            # FIXME if there's > 1 dimension, pretend there's only two
            else:
                inputs_zipped = zip(*self.inputs)
                pylab.plot(inputs_zipped[0], inputs_zipped[1], 'r.')
                pylab.plot([c.center[0] for c in self.clusters],
                           [c.center[1] for c in self.clusters], 'p')
            pylab.title('Graph after %d epochs' % self._elapsed_epochs)
            pylab.show()
        else:
            print "Unable to draw graphs because pylab cannot be imported.\n"

    def learn(self, debug_mode=False):
        """Runs epochs until the cluster centroids stop moving.

        :param debug_mode: Draws graphs of the clusters if True.  Note that
                           drawing the graphs pauses clustering after each
                           epoch.
        """
        if debug_mode:
            self.draw_graph()
        while True:
            self.last_clusters_centers = [c.center for c in self.clusters]
            self.run_epoch()
            if debug_mode:
                self.draw_graph()
            if all(map(lambda x, y: x.center == y, self.clusters,
                       self.last_clusters_centers)):
                break
