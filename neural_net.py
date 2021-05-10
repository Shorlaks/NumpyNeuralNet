import numpy as np


class NeuralNet:
    def __init__(self, architecture):
        self.num_layers = len(architecture)
        self.layers = [FullyConnectedLayer(architecture[i], architecture[i+1]) for i in range(self.num_layers-1)]

    def predict(self, np_vector):
        pass


class FullyConnectedLayer:
    def __init__(self, neurons_in_current_layer, neurons_in_next_layer):
        self.weights = np.random.uniform(low=-1.0, high=1.0, size=(neurons_in_current_layer, neurons_in_next_layer))
        self.dot_result = None
        self.activation_result = None

    def take_input_return_output(self, np_input):
        self.dot_result = np.dot(np_input, self.weights)
        # Sigmoid
        self.activation_result = 1/(1 + np.exp(-self.dot_result))
        return self.activation_result



a = FullyConnectedLayer(2, 4)
a.take_input_return_output(np.array([0.1, 0.2]))



