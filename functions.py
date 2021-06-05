import numpy as np


def squared_error_derivative(output, label):
    res = output - label
    return res


def binary_cross_entropy_derivative(output, label):
    output[output == 0] = 0.000000001
    output[output == 1] = 0.999999999
    res = -label/output + ((1 - label)/(1 - output))
    return res


def sigmoid_derivative(z):
    res = z * (1 - z)
    return res


def relu_derivative(z):
    res = z
    res[res < 0] = 0
    res[res > 0] = 1
    res[res == 0] = 0.5
    return res


def sigmoid(z):
    res = 1/(1 + np.exp(-z))
    return res


def relu(z):
    res = z
    res[res < 0] = 0
    return res


activation_function_dispatcher = {'sigmoid': sigmoid,
                                  'relu': relu}

activation_function_derivative_dispatcher = {'sigmoid': sigmoid_derivative,
                                             'relu': relu_derivative}

loss_function_derivative_dispatcher = {'se': squared_error_derivative,
                                       'bce': binary_cross_entropy_derivative}
