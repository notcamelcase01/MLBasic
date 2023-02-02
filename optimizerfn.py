import numpy as np


def normalise(x):
    cols = x.shape[1]
    xn = x.copy()
    xmean = np.mean(xn, axis=0)
    xrange = np.max(xn, axis=0) - np.min(xn, axis=0)
    for i in range(0, cols):
        xn[:, i] = (xn[:, i] - np.mean(x[:, i]))/(np.max(x[:, i]) - np.min(x[:, i]))
    return xn, xmean, xrange


def hypothesis(x, theta, is_logistic=False):
    """
    :param is_logistic: is logistic
    :param x: features
    :param theta: predictors
    :return: $h_theta$
    """
    if is_logistic:
        return 1/(1 + np.exp(-(x @ theta)))
    return x @ theta


def cost_d(x, y, theta):
    """
    :param x: feature
    :param y:  label
    :param theta: predictor
    :return: cost
    """
    yest = hypothesis(x, theta, True)
    m = yest.shape[0]
    lh1 = np.log(yest)
    lh2 = np.log(1 - yest)
    return -(lh1.T @ y + lh2.T @ (1 - y))/m


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
    y_est = hypothesis(x, theta)
    err = y_est - y
    m = err.shape[0]
    gr = x.T @ err / m
    return gr
