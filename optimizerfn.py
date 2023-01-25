import numpy as np


def hypothesis(x, theta):
    """
    :param x: features
    :param theta: predictors
    :return: $h_theta$
    """
    return x @ theta


def cost(x, y, theta):
    """
    :param x: features
    :param y: output
    :param theta: predictors
    :return: cost fn
    """
    y_err = hypothesis(x, theta) - y
    m = y_err.shape[0]
    return y_err.T @ y_err / (2 * m)


def gradient(x, y, theta):
    """
    :param x: feature
    :param y: results
    :param theta: predictor
    :return: gradient
    """
    gr = np.zeros((7, 1))
    y_est = hypothesis(x, theta)
    err = y_est - y
    m = err.shape[0]
    gr = x.T @ err / m
    return gr
