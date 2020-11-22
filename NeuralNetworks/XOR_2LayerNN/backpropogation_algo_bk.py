from matplotlib.colors import ListedColormap
import numpy as np
from random import seed
from math import exp
from random import random
import matplotlib.pyplot as plt
import matplotlib.animation as anime
plt.style.use('fivethirtyeight')
# plt.ion()
# # Activation function
# sigmoid_functn = lambda x: 1.0/(1.0 + np.exp(-x))
# transfer_derivative = lambda output: output * (1.0 - output)
sigmoid_functn = lambda x: (1.0 - np.exp(-2*x))/(1.0 + np.exp(-2*x))

transfer_derivative = lambda x: (1 + x)*(1 - x)
# plt.ion()


def find_weighted_sum_inps(weight, inps):
    """
    Each input is multiplied with the weight of a neuron focussing towards a neuron in the next layer.
    This weight is different if the focus of the current layer neuron is a neuron which is diagonally opposite
    to it.
    :param weight:
    :param inps:
    :return:
    """
    activation = weight[-1]
    for i in range(len(weight) - 1):
        activation += weight[i] * inps[i]
    return activation


def plot_data(training_data, expected):

    f, ax = plt.subplots(figsize=(7, 7))
    data_0 , data_1 = [], []
    for i in range(len(training_data)):
        if expected[i] == 0:
            data_0.append(training_data[i])
        else:
            data_1.append(training_data[i])

    ax.scatter(data_0[0][0], data_0[0][1], marker='o', color='green', s=40, alpha=0.5)
    ax.scatter(data_1[0][0], data_1[0][1], marker='^', color='red', s=60, alpha=0.7)
    ax.scatter(data_0[1][0], data_0[1][1], marker='o', color='green', s=40, alpha=0.5)
    ax.scatter(data_1[1][0], data_1[1][1], marker='^', color='red', s=60, alpha=0.7)
    # plt.show()
    return plt

class NeuralNetwork:
    """

    """
    def __init__(self, net_arch):
        self.activation_function = sigmoid_functn
        self.layers = len(net_arch)
        self.steps_per_epoch = 1
        self.arch = net_arch
        self.network = []
        self.weights = []

        self.initialize_nn()

    def train_network(self, train, l_rate=0.001, n_epoch=50, n_outputs=None):
        import math
        """
        Online learning is utilized here. that means weights are updated at each epochs.
        :param train: training dataset
        :param l_rate: learning rate used for modifying the weigths
        :param n_epoch:
        :param n_outputs:
        :return:
        """
        # # Add bias units to the input layer -
        # # add a "1" to the input data (the always-on bias neuron)
        # ones = np.ones((1, train.shape[0]))
        # train = np.concatenate((ones.T, train), axis=1)
        self.error_vs_epoch = np.zeros(shape=(n_epoch,1))
        expected = n_outputs
        # ani = anime.FuncAnimation(plt.gcf(), self.plot_decision_boundary, interval=1000)
        for epoch in range(n_epoch):
            sum_error = 0
            for row_itr in range(len(train)):
                outputs = self.forward_pass(train[row_itr])
                # Mean squared error
                sum_error += (expected[row_itr] - outputs[0])**2
                # self.plot_hidden_layer_output(train, n_outputs)
                # self.plot_decision_boundary()
                self.backward_propagate_error(expected)
                self.optimize_network(train[row_itr], l_rate)
                if sum_error < 0.05:
                    break
            self.error_vs_epoch[epoch] = sum_error
            if (epoch+1) % 1000 == 0:
                print('>epoch=%d, lrate=%.3f, error=%.6f' % (epoch, l_rate, sum_error))
        # self.plot_error()
        plt = plot_data(train, n_outputs)
        # # self.plot_hidden_layer_output(train, n_outputs)
        self.plot_decision_boundary(train, n_outputs)
        # plt.tight_layout()
        # plt.show()


    def initialize_nn(self):
        """
        CREATING LAYERS AND ITS CORRESPONDING WEIGHTS
        No. of weights depends upon the number of neurons in the previous layer and bias depends upon
        no. of neurons in the present layer.
        the weights created here are the parameters which effect the output of that layer when multiplied
        with the output from the previous layers. Lets consider the previous layer as input layer and its output
        is nothing but the input. Now, this o/p gets multiplied by the weights of the Hidden layer. The
        function of a neuron in a layer is to perform the weighted sum and apply a non-linear function on the
        sum to understand the liveliness of that neuron.
        :return:
        """

        hidden_layer = [{'weights': [random() for i in range(self.arch[0] + 1)]} for i in range(self.arch[1])]
        self.network.append(hidden_layer)
        # the output layer consist of a bias due to the fact that the activation function is a Sigmoid
        output_layer = [{'weights': [random() for i in range(self.arch[1] + 1)]} for i in range(self.arch[2])]
        self.network.append(output_layer)
        return

    def forward_pass(self, inps=None):
        """
        Take each input and forward it to each neuron. At each neuron weighted sum of all the inputs that comes from
        the nodes of the previous layer. Then the weighted sum is passed to the activation function to find the
        reaction of a neuron for a given set of inputs.
        :param network: inp neural network with layers and weights
        :param inps: list of inputs
        :return:
        """
        i = 1
        if inps is None:
            inps = [1, 1]
        for layers in self.network:
            new_inps = []
            for neurons in layers:
                weighted_sum = find_weighted_sum_inps(neurons['weights'], inps)
                neurons['output'] = self.activation_function(weighted_sum)
                new_inps.append(neurons['output'])
            inps = new_inps
            i += 1
        return inps

    # Backpropagate error and store in neurons
    def backward_propagate_error(self, expected):
        """
        :param expected:
        :return:
        """
        for i in reversed(range(len(self.network))):
            layer = self.network[i]
            errors = list()
            if i != len(self.network)-1:
                for j in range(len(layer)):
                    error = 0.0
                    for neuron in self.network[i + 1]:
                        error += (neuron['weights'][j] * neuron['delta'])
                    errors.append(error)
            else:
                # for the last layer neurons
                for j in range(len(layer)):
                    neuron = layer[j]
                    errors.append(expected[j] - neuron['output'])
            for j in range(len(layer)):
                neuron = layer[j]
                neuron['delta'] = errors[j] * transfer_derivative(neuron['output'])

    def optimize_network(self, inp, lr):
        """
        using stochastic gradient descent
        :param inp:
        :param lr: factor controlling the steps that weights need to take to be modified.
        :return:
        """
        for layer_itr in range(len(self.network)):
            inputs = inp[:-1]
            if layer_itr != 0:
                inputs = [neuron['output'] for neuron in self.network[layer_itr - 1]]
            for neuron in self.network[layer_itr]:
                for j in range(len(inputs)):
                    neuron['weights'][j] += lr * neuron['delta'] * inputs[j]
                neuron['weights'][-1] += lr * neuron['delta']

    def predict(self, X):
        Y = np.array([]).reshape(0, self.arch[-1])
        for x in X:
            y = np.array([[self.predict_single_data(x)]])
            Y = np.vstack((Y,y))
        return Y

    def predict_single_data(self, x):
        val = np.concatenate((np.ones(1).T, np.array(x)))
        for layers in self.network:
            l = []
            for neurons in layers:
                l.append(sigmoid_functn(np.dot(val, neurons['weights'])))
            val = np.array(l)
            val = np.concatenate((np.ones(1).T, np.array(val)))
        return val[1]

    # Plot data and results ############################
    def plot_error(self):
        fig, ax = plt.subplots()
        plt.plot(self.error_vs_epoch)

        ax.set(xlabel='Epochs', ylabel=' Error',
               title='Error Vs Epochs')
        ax.grid()
        plt.show()

    def plot_decision_boundary(self, X, y):
        resolution=0.02
        test_idx = None
        markers = ('s', 'x', 'o', '^', 'v')
        colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
        cmap = ListedColormap(colors[:len(np.unique(y))])

        # plot the decision surface
        x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                               np.arange(x2_min, x2_max, resolution))
        Z = self.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
        Z = Z.reshape(xx1.shape)
        plt.contourf(xx1, xx2, Z, alpha=0.4, cmap=cmap)
        plt.xlim(xx1.min(), xx1.max())
        plt.ylim(xx2.min(), xx2.max())

        # plot class samples
        for idx, cl in enumerate(np.unique(y)):
            plt.scatter(x=X[y == cl, 0], y=X[y == cl, 1],
                        alpha=0.8, c=cmap(idx),
                        marker=markers[idx], label=cl)

        # highlight test samples
        if test_idx:
            # plot all samples
            X_test, y_test = X[test_idx, :], y[test_idx]

            plt.scatter(X_test[:, 0],
                        X_test[:, 1],
                        c='',
                        alpha=1.0,
                        linewidths=1,
                        marker='o',
                        s=55, label='test set')

    def plot_hidden_layer_output(self, x, y):
        W1=np.array([
                    [self.network[0][0]['weights'][:-1][0], self.network[0][0]['weights'][:-1][1]],
                    [self.network[0][1]['weights'][:-1][0], self.network[0][1]['weights'][:-1][1]]
                   ])
        bias = np.array([self.network[0][0]['weights'][-1], self.network[0][0]['weights'][-1]])
        z1 = (np.dot(x, W1) + bias).T
        a1 = sigmoid_functn(z1) #Applying Sigmoid Non linearity
        plt.scatter(a1[0,:],a1[1,:],c=y,alpha=1)
        plt.title('Hidden Weights ')
        plt.xlabel('x')
        plt.ylabel('y')
        # plt.show()

if __name__ == "__main__":
    seed(1)
    train_data = np.array([
                            [1,1], [1,0],
                            [0,1], [0,0]
                          ])
    l_rate = 0.01  #0.00001
    n_epoch = 400000
    n_outputs = np.array([0, 1,
                          1, 0])

    n = NeuralNetwork([2, 2, 1])
    n.train_network(train_data, l_rate, n_epoch, n_outputs)
    plt.xlabel('x-axis')
    plt.ylabel('y-axis')
    plt.legend(loc='upper left')
    plt.tight_layout()
    plt.show()
    print("Predictions..")
    for s in train_data:
        print(s, n.predict_single_data(s))
