import numpy as np
from random import random

# save activation and derivative
# implement backpropagation
# implement gradient descent
# implement train
# train our net with some dummy dataset
# make some prediction


class MLP(object):

    def __init__(self, num_inputs=3, hidden_layers=[3, 3], num_outputs=2):

        self.num_inputs = num_inputs
        self.hidden_layers = hidden_layers
        self.num_outputs = num_outputs

        layers = [num_inputs] + hidden_layers + [num_outputs]

        # initiate random weight
        weight = []
        for i in range(len(layers)-1):
            w = np.random.rand(layers[i], layers[i+1])
            weight.append(w)
        self.weight = weight

        activation = []
        for i in range(len(layers)):
            a = np.zeros(layers[i])
            activation.append(a)
        self.activation = activation

        derivative = []
        for i in range(len(layers)-1):
            d = np.zeros((layers[i], layers[i-1]))
            derivative.append(d)
        self.derivative = derivative

    def forward_propogate(self, inputs):

        activation = inputs
        self.activation[0] = inputs

        for i, w in enumerate(self.weight):
            # calculate net inputs
            net_inputs = np.dot(activation, w)

            # calculate the activation
            activation = self._sigmoid(net_inputs)
            self.activation[i+1] = activation

        return activation

    def back_propagate(self, error, verbose=False):

        for i in reversed(range(len(self.derivative))):
            activation = self.activation[i+1]
            delta = error * self._sigmoid_derivative(activation)
            delta_reshaped = delta.reshape(delta.shape[0], -1).T
            current_activation = self.activation[i]
            current_activation_reshaped = current_activation.reshape(current_activation.shape[0], -1)
            self.derivative[i] = np.dot(current_activation_reshaped, delta_reshaped)
            error = np.dot(delta, self.weight[i].T)

            if verbose:
                print("Derivative for W{}: {}".format(i, self.derivative[i]))

        return error

    def gradient_descent(self, learning_rate):
        for i in range(len(self.weight)):
            weight = self.weight[i]
            derivative = self.derivative[i]
            weight += weight + derivative * learning_rate

    def train(self, inputs, target, epochs, learning_rate):

        for i in range(epochs):
            sum_error = 0

            for inputs, target in zip(inputs, target):

                # forward prop
                output = self.forward_propogate(inputs)

                # calculate error
                error = target - output

                # back propagation
                self.back_propagate(error)

                # apply gradient descent
                self.gradient_descent(learning_rate)

                sum_error += self._mse(target, output)

            # report error
            print("Error: {} at epoch {}".format(sum_error / len(inputs), i))

    def _mse(self, target, output):
        return np.average((target - output)**2)

    def _sigmoid_derivative(self, x):
        return x * (1.0 - x)

    def _sigmoid(self, x):
        y = 1.0 / (1 + np.exp(-x))
        return y


if __name__ == "__main__":

    inputs = np.array([[random() / 2 for _ in range(2)] for _ in range(1000)])
    target = np.array([[i[0] + i[1]] for i in inputs])

    # create an MLP
    mlp = MLP(2, [5], 1)

    # train MLP
    mlp.train(inputs, target, 50, 0.1)