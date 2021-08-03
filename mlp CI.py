import numpy as np
import matplotlib.pyplot as plt
from fillspace import *
from cross_val import cross_validation_split
from re import X


class MLP(object):
    """A Multilayer Perceptron class."""


    output_l = []

    def __init__(self, num_inputs = 8, hidden_layers = [7], num_outputs = 1):

        self.num_inputs = num_inputs
        self.hidden_layers = hidden_layers
        self.num_outputs = num_outputs

        # create a generic representation of the layers
        layers = [num_inputs] + hidden_layers + [num_outputs]

        # create random connection weights for the layers
        weights = []
        for i in range(len(layers) - 1):
            w = np.random.rand(layers[i], layers[i + 1])
            weights.append(w)
        self.weights = weights

        # save derivatives per layer
        derivatives = []
        for i in range(len(layers) - 1):
            d = np.zeros((layers[i], layers[i + 1]))
            derivatives.append(d)
        self.derivatives = derivatives

        # save activations per layer
        activations = []
        for i in range(len(layers)):
            a = np.zeros(layers[i])
            activations.append(a)
        self.activations = activations


    def forward_propagate(self, inputs):
        # the input layer activation is just the input itself
        activations = inputs

        # save the activations for backpropogation
        self.activations[0] = activations

        # iterate through the network layers
        for i, w in enumerate(self.weights):
            # calculate matrix multiplication between previous activation and weight matrix
            net_inputs = np.dot(activations, w)

            # apply sigmoid activation function
            activations = self._sigmoid(net_inputs)

            # save the activations for backpropogation
            self.activations[i + 1] = activations

        # return output layer activation
        return activations


    def back_propagate(self, error):
        # iterate backwards through the network layers
        for i in reversed(range(len(self.derivatives))):

            # get activation for previous layer
            activations = self.activations[i+1]

            # apply sigmoid derivative function
            delta = error * self._sigmoid_derivative(activations)

            # reshape delta as to have it as a 2d array
            delta_re = delta.reshape(delta.shape[0], -1).T

            # get activations for current layer
            current_activations = self.activations[i]

            # reshape activations as to have them as a 2d column matrix
            current_activations = current_activations.reshape(current_activations.shape[0],-1)

            # save derivative after applying matrix multiplication
            self.derivatives[i] = np.dot(current_activations, delta_re)

            # backpropogate the next error
            error = np.dot(delta, self.weights[i].T)


    def train(self, inputs, targets, epochs, learning_rate):
        # now enter the training loop
        for i in range(epochs):
            sum_errors = 0

            # iterate through all the training data
            for j, input in enumerate(inputs):
                target = targets[j]

                # activate the network!
                output = self.forward_propagate(input)

                error = target - output

                self.back_propagate(error)

                # now perform gradient descent on the derivatives
                # (this will update the weights
                self.gradient_descent(learning_rate)

                # keep track of the MSE for reporting later
                sum_errors += self._mse(target, output)

            self.output_l.append(output)

            # Epoch complete, report the training error
            print("Error: {} at epoch {}".format(sum_errors / len(items), i+1))

        print("Training complete!")
        print("===========================================")


    def gradient_descent(self, learningRate=1):
        # update the weights by stepping down the gradient
        for i in range(len(self.weights)):
            weights = self.weights[i]
            derivatives = self.derivatives[i]
            weights += derivatives * learningRate


    def _sigmoid(self, x):
        y = 1.0 / (1 + np.exp(-x))
        return y

    def ReLU(self,Z):
        return np.maximum(Z, 0)

    def _sigmoid_derivative(self, x):
        return x * (1.0 - x)

    def ReLU_deriv(Z):
        return Z > 0


    def _mse(self, target, output):
        return np.average((target - output) ** 2)

# ------------------------------------------------------------------------------


error_log = []
desired = []
calculate = []

k=10
folds = cross_validation_split(m_x, k)

for i in range(k):
    
    _k = len(folds[i])
    # print(folds[i][0:_k-1])
    items = np.array([folds[i][j][0:len(folds[i][j])-1] for j in range(len(folds[i]))])
    # print(items)
    targets = np.array([folds[i][j][len(folds[i][j])-1] for j in range(len(folds[i]))])
    # print(targets)
    # create a Multilayer Perceptron with one hidden layer
    mlp = MLP(8, [7], 1)

    # train network
    mlp.train(items, targets, 1000, 0.5)

    _input = folds[i][_k-1][0:len(folds[i][_k-1])-1]
    input = np.array(_input)
    target = np.array((folds[i][_k-1][len(folds[i][_k-1])-1]))

    output = mlp.forward_propagate(input)
    desired.append(denomallize(target))
    calculate.append(denomallize(output))

    print()
    print("if station 1 have {} \nstation 2 have {} \nIn the next 7 hours the water level should be {}".format(denomallize(input[0:4]), denomallize(input[3:8]), denomallize(output)))
    print("but actually should be {}".format(denomallize(target)))
    print()

    error_log.append(abs(denomallize(target)-denomallize(output))*100/denomallize(target))


for i in range(10):
    print("error round {} : {:.2f}%".format(i,error_log[i][0]))
print()

plt.subplots()      
plt.ylabel('water level')
desired, = plt.plot(desired, "ro", label=f"desired output")
calculate, = plt.plot(calculate, "bo", label=f"calculate output")
plt.legend()
plt.show()