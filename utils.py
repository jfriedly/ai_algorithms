"""
============
``utils.py``
============

Contents
--------

* ``sigmoid()`` The logistic sigmoid activation function.

* ``piecewise_linear()`` A linear activation function from -1 to 1.

* ``binary_threshold()`` A binary threshold activation function (1 or 0).

* ``bipolar_threshold()`` A bipolar threshold activation function (1 or -1).

* ``euclidean_distance()`` A fast implementation of Euclidean distance.
"""
#TODO(jfriedly): Put a try-except around the import numpy and write a (slow)
# implementation that can be used if numpy isn't installed.
#TODO(jfriedly): Add the new functions to the module docstring.
#TODO(jfriedly): Make the gaussian kernel efficient.
import sys
import math
import numpy
import random


MAXINT = sys.maxint
MININT = -sys.maxint - 1


def sigmoid(v, a=1):
    """Logistic sigmoid function (default activation for a network).

    :param v: The value to perform the activation function on.
    :param a: If you'd like to weight the exponent, set a equal to some other
              value.  Typically used with currying.
    """
    try:
        return 1 / (1 + math.exp(-a * v))
    except OverflowError:
        # If we overflow, that means we got too large of an exponent.  Return
        # the function evaluated at infinity instead.
        if v < 0:
            return 0.0
        else:
            return 1.0


def piecewise_linear(v):
    """Piecewise linear activation function.  Returns -1 when v is less than
    one, returns 1 when v is greater than one, and returns v otherwise

    :param v: The value to perform the activation function on.
    """
    if v < -1:
        return -1
    elif v > 1:
        return 1
    return v


def binary_threshold(v, threshold=0.5):
    """Threshold activation function returning binary values (1 or 0).

    :param v: The value to perform the activation function on.
    :param threshold: If you'd like a threshold other than 0.5, set it here.
                      Typically used with currying.
    """
    return 1 if v > threshold else 0


def bipolar_threshold(v, threshold=0.5):
    """Threshold activation function returning bipolar values (1 or -1).

    :param v: The value to perform the activation function on.
    :param threshold: If you'd like a threshold other than 0.5, set it here.
                      Typically used with currying.
    """
    return 1 if v > threshold else -1


def bipolar_sigmoid(v, a=1, threshold=0.5):
    return bipolar_threshold(sigmoid(v, a=a), threshold=threshold)


def euclidean_distance(p1, p2):
    """Returns the Euclidean Distance between two points in *n* dimensions.

    :param p1: A point in *n* dimensions, represented as an iterable of
               numbers.
    :param p2: A point in *n* dimensions, represented as an iterable of
               numbers.
    """
    # For reasons I cannot fathom, numpy and scipy's recommended ways of
    # calculating Euclidean distance are slow as hell.  I reproduced the
    # results at http://goo.gl/EiQgt and tested scipy as well.  My results:
    #  1.1s for fastest_calc_dist
    #  1.3s for math_calc_dist (a modified form of this is below)
    # 27.3s for math.sqrt(sum(numpy.array(p1) - numpy.array(p2)))
    # 28.9s for numpy.linalg.norm(numpy.array(p1) - numpy.array(p2))
    # 30.0s for np.sqrt(sum(numpy.array(p1) - numpy.array(p2)))
    # 40.9s for scipy.spatial.distance.euclidean(p1, p2)
    #
    # I retested using numpy arrays for all the data points and got better
    # results, but still disappointing:
    #  2.8s for math_calc_dist
    #  5.3s for fastest_calc_dist
    # 14.5s for numpy.linalg.norm(p1 - p2)
    # 16.7s for numpy.linalg.norm(numpy.array(p1) - numpy.array(p2))
    # 30.0s for scipy.spatial.distance.euclidean(p1, p2)
    #
    # Speculation:  Maybe the fact that numpy/scipy have to handle n dimensions
    # made them slower?  Nope, see below.
    #
    # I retested by rewriting math_calc_dist and fastest_calc_dist to support
    # n dimensions and they're still blazing fast.  Below are the times for
    # data that came in as tuples and numpy arrays, respectively:
    # 2.6s and 4.2s for math_calc_dist
    # 2.2s and 7.9s for fastest_calc_dist
    n = len(p1)
    return math.sqrt(sum((math.pow((p1[i] - p2[i]), 2) for i in xrange(n))))


def squared_distance(p1, p2):
    """Returns the Euclidean Distance between two points in *n* dimensions.

    :param p1: A point in *n* dimensions, represented as an iterable of
               numbers.
    :param p2: A point in *n* dimensions, represented as an iterable of
               numbers.
    """
    # I performed similar testing for this function as I did for Euclidean
    # distance above.  My results are below for data as tuples and arrays,
    # respectively:
    #  2.4s and  4.0s for math_calc_dist
    #  2.2s and  7.8s for fastest_calc_dist
    # 36.0s and 21.0s for scipy.spatial.distance.sqeuclidean(p1, p2)
    n = len(p1)
    return sum((math.pow((p1[i] - p2[i]), 2) for i in xrange(n)))


def identity(value):
    """Trivial identity function:  returns v.  This is useful as an output
    layer activation function and was written to avoid lambdas (slow).

    :param value: The value to return.
    """
    return value


def argmax(pairs):
    """Returns the argmax of an iterable of pairs assuming they come in the
    form arg: value.

    :param pairs: Iterable of arg: value pairs to examine.
    """
    argmax = -1
    # Smallest integer value that can be represented on this system without
    # using long ints.
    maximum = MININT
    for arg, value in pairs:
        if value > maximum:
            argmax = arg
            maximum = value
    return argmax


def vectorize(start, dimensions, stop=None):
    """Turns two scalars start and stop into a sparse vector of d dimensions
    elements indexed from start to stop as ones.  If stop is not passed, only
    the element at the start index will be a one.  The vector is zero-indexed,
    so the ones start at the start-1'th element and stop at the stop-1'th.

    Example:

    .. code:: python

        >>> vectorize(4, 10, stop=7)
        [0, 0, 0, 0, 1, 1, 1, 1, 0, 0]

    :param start: 0-indexed integer value of the first 1 in the vector.
    :param dimensions: Integer dimensionality of the vector.
    :param stop: 0-indexed integer value of the last 1 in the vector.
    """
    if stop is None:
        stop = start
    assert stop >= start, "Stop cannot be less than start in vectorize"
    vector = [0] * dimensions
    for i in xrange(start, stop+1):
        vector[i] = 1
    return vector


def devectorize(vector):
    """Does the opposite of the above function.  Given a sparse vector
    that contains a string of ones, pick out the beginning and ending indices
    of the ones and return them.

    Example:

    .. code:: python

        >>> devectorize([0, 0, 0, 0, 1, 1, 1, 1, 0, 0])
        (4, 7)

    :param vector: Vector to examine for the largest value.
    :returns: tuple of (start, stop) where start is the index of the first one
              and stop is the index of the last one in the string.  Both are
              zero-indexed.
    """
    start = 0
    stop = 0
    ones_seen = False
    for i in xrange(len(vector)):
        if ones_seen == False and vector[i] == 1:
            ones_seen = True
            start = i
        if ones_seen == True and vector[i] != 1:
            stop = i - 1
            break
    return start, stop


def normalize(data):
    """Given a list of data vectors, scale them by dividing every value by
    the largest value observed for that column.

    :param data: List of data vectors.
    """
    data_dimensions = len(data[0])
    maximums = [MININT] * data_dimensions
    for vector in data:
        for i, value in enumerate(vector):
            if value > maximums[i]:
                maximums[i] = value

    for i, vector in enumerate(data):
        for j, value in enumerate(vector):
            data[i][j] /= maximums[j]

    return data


def gaussian(x, mean, stddev):
    """Given an x-value and the parameters to a gaussian, return the y-value.

    :param x: X-value to evaluate the Gaussian at.
    :param mean: Mean parameter to the Gaussian.
    :param stddev: Standard Deviation parameter to the Gaussian.  We use
                   std dev. to avoid calculating square roots unnecessarily.
    """
    # Hard-coded to avoid calculating this
    sqrt_2pi = 2.5066282746310002
    coeff = (1/(stddev * sqrt_2pi))
    exp = math.e ** (((-1)/(2 * stddev ** 2)) * (x - mean) ** 2)
    return coeff * exp


def linear_kernel(sample1, sample2, **kwargs):
    """Linear kernel function, aka the identity kernel.

    :param sample1: The first sample to the kernel.
    :param sample2: The second sample to the kernel.
    """
    return numpy.dot(sample1, sample2)


def gaussian_kernel(sample1, sample2, sigma=1, **kwargs):
    top = euclidean_distance(sample1, sample2) ** 2
    frac = top / (2.0 * sigma ** 2)
    return math.exp(-frac)


def approx_gaussian_kernel(sample1, sample2, p=5, randvectors=[], biases=[],
                           **kwargs):
    """Based on Bochner's theorem, we can form a low dimensional approximation
    of the Gaussian kernel using this function.
    """
    coef = math.sqrt(2.0 / p)
    phi1 = [None] * p
    phi2 = [None] * p
    for m in xrange(p):
        for phi, sample in ((phi1, sample1), (phi2, sample2)):
            phi[m] = coef * math.cos(numpy.dot(randvectors[m], sample) + biases[m])
    return numpy.dot(phi1, phi2)
