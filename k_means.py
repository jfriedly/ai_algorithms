"""
==============
``k_means.py``
==============

Contents
--------

* ``Cluster`` A class representing one cluster in a K-Means classifier,
              parameterized only by an initial center.

* ``KMeansClassifier`` A class representing an entire classifier, parameterized
                       by a list of data points and a number of clusters.
"""
from utils import euclidean_distance


class Cluster():
    """Class to represent a single cluster in the K-Means algorithm.
    """

    def __init__(self, center):
        """Initialize the cluster.

        :param center: Initial center for the cluster (typically a random
                       point).
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


class KMeansClassifier():
    """K-Means classifier.
    """

    def __init__(self, inputs, K):
        """Initialize the class.

        :param inputs: An list of vectors, where each vector is one sample and
                       is represented as an iterable of numbers.
        :param K: The number of clusters in this classifier.
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

        :param one_variance: Boolean of indicating how to calculate the
                             variances.
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

        :param debug_mode: Draws graphs of the clusters if True.  Note that
                           drawing the graphs pauses clustering after each
                           epoch.
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
