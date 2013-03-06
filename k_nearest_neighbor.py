"""
=========================
``k_nearest_neighbor.py``
=========================

Contents
--------

* ``KNNCLassifier`` A class representing an entire classifier, parameterized
                    by a list of data points and a number of nearest neighbors
                    to check.
"""
from utils import euclidean_distance

from collections import Counter


class KNNClassifier():
    """Classifier based on the K-Nearest Neighbor algorithm.
    """

    def __init__(self, inputs, classes, K=1):
        """Initializes the classifier.

        :param inputs: A list of vectors, where each vector is one sample and
                       is represented as an iterable of numbers.
        :param classes: A list of classes, where each class applies to the
                        corresponding input vector (the one with the same
                        index)
        :param K: The number of neighbors to inspect during classification.
        """
        # Sanity check
        assert(len(inputs) == len(classes))
        self.inputs = inputs
        self.classes = classes
        self.K = K

    def use(self, sample):
        """Classify a sample by finding the nearest K points to it and
        assigning it to the class that the plurality of the points have.
        If two classes tie for a plurality, the one with the lower index is
        returned.

        :param sample: An unclassified vector, represented by an iterable of
                       numbers
        """
        def distance_to_point(other_point):
            return euclidean_distance(sample, other_point)

        nearest_neighbors = sorted(self.inputs, key=distance_to_point)[:self.K]
        nn_classes = []
        for point in nearest_neighbors:
            nn_classes.append(self.classes[self.inputs.index(point)])
        counter = Counter(nn_classes)
        print counter.most_common()
        return counter.most_common(1)[0][0]

    def check(self, sample, correct_class):
        """Classify a sample and then return whether or not we got the right
        class.

        :param sample: An unclassified vector, represented by an iterable of
                       numbers
        :param correct_class: The class we should test ourselves against.
        """
        predicted_class = self.use(sample)
        return predicted_class == correct_class
