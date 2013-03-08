from ai_algorithms import utils

import numpy


def two_class_LDA(cls1, cls2, point, w, w_not):
    return sigmoid(w.transpose().dot(numpy.array(point)) + w_not)


def mean(data):
    return numpy.array(data).mean(axis=0)


def sample_covariance(data):
    n = len(data)
    m = mean(data)
    data = numpy.array([numpy.array(point) for point in data])
    s = sum((point - m)[:,None] * (point - m) for point in data)
    return (1./(n - 1)) * s
