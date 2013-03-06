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
import math
import numpy


def sigmoid(v, a=1):
    """Logistic sigmoid function (default activation for a network).

    :param v: The value to perform the activation function on.
    :param a: If you'd like to weight the exponent, set a equal to some other
              value.  Typically used with currying.
    """
    return 1 / (1 + math.exp(-a * v))


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


def euclidean_distance(point1, point2):
    """Returns the Euclidean Distance between two points in *n* dimensions.
    Uses numpy's efficient implementation of Euclidean distance.

    :param point1: A point in *n* dimensions, represented as an iterable of
                   numbers.
    :param point2: A point in *n* dimensions, represented as an iterable of
                   numbers.
    """
    return numpy.linalg.norm(numpy.array(point1) - numpy.array(point2))
