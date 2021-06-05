import numpy as np
np.seterr(all='raise')
import functions


class NeuralNet:
    def __init__(self, architecture, activations, loss):
        self.architecture = architecture
        self.num_layers = len(architecture)
        self.loss_function = loss
        self.learning_rate = 0.5
        self._layers = [FullyConnectedLayer(architecture[i],
                                            architecture[i+1],
                                            activations[i]) for i in range(self.num_layers-1)]

    def predict(self, np_vector):
        for layer in self._layers:
            np_vector = layer.take_input_return_output(np_vector)
        return np_vector

    def train(self, np_matrix, labels, epochs=1):
        for _ in range(epochs):
            for np_vector, label in zip(np_matrix, labels):
                output = self.predict(np_vector)
                self.backward_pass(output, label)

    def backward_pass(self, output, label):
        errors = self.calculate_loss(output, label)
        for layer in reversed(self._layers):
            activation_derivative = layer.calculate_derivative_of_activation_function()
            delta = errors * activation_derivative
            net = layer.layer_input
            layer.delta_weights = self.chain_rule(delta, net)
            errors = np.dot(layer.weights, delta)
        self.update_weights()

    # calculate partial derivatives of the total error with respect to neurons in the output layer
    def calculate_loss(self, output, label):
        return functions.loss_function_derivative_dispatcher[self.loss_function](output, label)

    def update_weights(self):
        for layer in self._layers:
            layer.weights -= self.learning_rate * layer.delta_weights

    @staticmethod
    def chain_rule(delta, net):
        delta_weights = delta.reshape((-1, 1)) * net
        return np.transpose(delta_weights)


class FullyConnectedLayer:
    def __init__(self, neurons_in_current_layer, neurons_in_next_layer, activation):
        self.weights = np.random.uniform(low=-1.0, high=1.0, size=(neurons_in_current_layer, neurons_in_next_layer))
        self.activation_function = activation
        self.delta_weights = None
        self.dot_result = None
        self.activation_result = None
        self.layer_input = None

    def take_input_return_output(self, np_input):
        self.layer_input = np_input
        self.dot_result = np.dot(np_input, self.weights)
        self.activation_result = functions.activation_function_dispatcher[self.activation_function](self.dot_result)
        return self.activation_result

    def calculate_derivative_of_activation_function(self):
        return functions.activation_function_derivative_dispatcher[self.activation_function](self.activation_result)
