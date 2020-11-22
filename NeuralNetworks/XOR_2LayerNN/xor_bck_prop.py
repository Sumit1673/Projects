from matplotlib.colors import ListedColormap
import numpy as np
from random import seed
from math import exp
from random import random
import matplotlib.pyplot as plt

# Activation functions
sigmoid_functn = lambda x: 1/(1+ np.exp(-x))

sigmoid_derivative = lambda x: x * (1-x)

# Question(a): Analyzing the input to hidden weights by applying different types of distribution
# to generate the weights randomly: random weights, binomially distributed, uniformally distributed
# non-bounded normal distribution of the weights
def truncated_normal(mean=0, sd=1, low=0, upp=10):
    from scipy.stats import truncnorm
    return truncnorm(
        (low - mean) / sd, (upp - mean) / sd, loc=mean, scale=sd)


class XORNeuralNetwork:
    """
        Network Arch : 2 ,2, 1
        Network with 2 inputs, 2 neurons in a hidden layer, and one output
    """

    def __init__(self, arch):
        """CREATING LAYERS AND ITS CORRESPONDING WEIGHTS
        No. of weights depends upon the number of neurons in the previous layer and bias depends upon
        no. of neurons in the present layer.
        the weights created here are the parameters which effect the output of that layer when multiplied
        with the output from the previous layers. Lets consider the previous layer as input layer and its output
        is nothing but the input. Now, this o/p gets multiplied by the weights of the Hidden layer. The
        function of a neuron in a layer is to perform the weighted sum and apply a non-linear function on the
        sum to understand the liveliness of that neuron."""

        np.random.seed(0)

        # Initialized the weights, activation function and epochs
        self.activation_func = sigmoid_functn
        self.activitation_func_deriv = sigmoid_derivative
        self.layers = len(arch)
        self.steps_per_epoch = 100
        self.arch = arch
        self.weights = []

        # Intializing weights with random values b/w -1 ,1
        for each_layer in range(self.layers - 1): # for inp to hidden and hidden to out
            #-------------------- inp/hidden ----------hidden/out
            w = 2*np.random.rand(arch[each_layer] + 1, arch[each_layer+1]) - 1
            # w = 2*np.random.binomial(1, 0.1, size=(arch[each_layer] + 1, arch[each_layer+1])) -1

            # un-bounded Normally distributed weights, results change if standard deviation or mean values are changed
            # X = truncated_normal(mean=0, sd=1, low=-1, upp=1)
            # w = X.rvs([arch[each_layer] + 1, arch[each_layer+1]])
            self.weights.append(w)
        # plt.hist(self.weights[0])
        # plt.show()
        return

    def fit_model(self, training_set, expected_outputs, learning_rate=0.1, epochs=100):
        self.error_vs_epoch = np.zeros(shape=(epochs,1))
        # concatenating one for the bias with the existing weights of the neurons
        ones = np.ones((1, training_set.shape[0]))
        X = np.concatenate((ones.T, training_set), axis=1)

        self.plot_hidden_layer_output(training_set, expected_outputs, title="Before training")

        for each_epoch in range(epochs):
            # training randomly not in sequence
            random_train_sample = np.random.randint(training_set.shape[0])

            # Feeding values to the network with a forward pass
            _set = [X[random_train_sample]]

            y_hat = self.forward_pass(_set)

            # plot_inp_out_weigthts(self.weights[0])
            # Finding out the change needed to be made to the weights using back propogation
            y = expected_outputs[random_train_sample]
            error = self.back_propogation(y_hat, y, learning_rate)
            self.error_vs_epoch[each_epoch] = error[-1]
            if (each_epoch+1) % 10000 == 0:
                print('epochs: {}'.format(each_epoch + 1))
                print('Error: {}'.format(error[-1]))
        self.plot_decision_line(training_set, expected_outputs)
        self.plot_error()

    # Plot data and results ############################
    def plot_error(self):
        fig, ax = plt.subplots()
        plt.plot(self.error_vs_epoch)

        ax.set(xlabel='Epochs', ylabel=' Error',
               title='Error Vs Epochs')
        ax.grid()
        plt.show()

    def forward_pass(self, train_set):
        """
        Take each input and forward it to each neuron. At each neuron weighted sum of all the inputs that comes from
        the nodes of the previous layer. Then the weighted sum is passed to the activation function to find the
        reaction of a neuron for a given set of inputs.
        :param train_set:
        :return: will be a np. array with the output of all the neurons
        """

        for i in range(len(self.weights)-1):
            activation = np.dot(train_set[i], self.weights[i])
            activity = sigmoid_functn(activation)

            # add the bias for the next layer
            activity = np.concatenate((np.ones(1), np.array(activity)))
            train_set.append(activity)

        # last layer
        activation = np.dot(train_set[-1], self.weights[-1])
        activity = sigmoid_functn(activation)
        train_set.append(activity)
        return train_set

    def back_propogation(self, Y_hat, Y, lr):
        error = (Y - Y_hat[-1])

        # delta = dL/dz --> z is the output of the output neuron. This delta will be multiplied to all the neurons
        # plus the local gradient i.e. gradient of error w.r.t. to output of each neuron.
        delta_vec = [error * sigmoid_derivative(Y_hat[-1])]

        # Traversing backwards
        for i in range(self.layers-2, 0, -1):
            error = delta_vec[-1].dot(self.weights[i][1:].T)
            error = error * sigmoid_derivative(Y_hat[i][1:])
            delta_vec.append(error)

        delta_vec.reverse()

        # Stochastic gradient descent for weights optimization
        for i in range(len(self.weights)):
            layer = Y_hat[i].reshape(1, self.arch[i]+1)  # from (3,1) --> (1,3)
            delta = delta_vec[i].reshape(1, self.arch[i+1])
            # self.weights[i] += lr*layer.T.dot(delta)
            np.add(self.weights[i], lr*layer.T.dot(delta), out=self.weights[i], casting="unsafe")
        return error

    def predict(self, test_data):
        prediction = np.array([]).reshape(0, self.arch[-1])
        for data_points in test_data:
            y = np.array([self.predict_single_data(data_points)])
            prediction = np.vstack((prediction, y))
        return prediction

    def predict_single_data(self, x):
        # concatenating one to make the dimension of the i/p vector equivalent to the dimension of weight vector
        x = np.concatenate((np.ones(1).T, np.array(x)))
        # forwarding the data to all the weights of the neuron
        for each_weight in range(0, len(self.weights)):
            x = sigmoid_functn(np.dot(x, self.weights[each_weight]))
            x = np.concatenate((np.ones(1).T, np.array(x)))
        return x[1]

    def plot_hidden_layer_output(self, train_set, labels, title=None):
        bias = np.array([self.weights[0][0]]).T
        fired_op = neuron_operation(self.weights[0][1:], bias, train_set)
        # bias = np.array([self.weights[0][0]])
        # hidden_layer_out = np.dot(self.weights[0][1:], np.transpose(train_set)) + bias.T
        # print(hidden_layer_out)
        # fired_op = sigmoid_functn(hidden_layer_out)
        # print(fired_op)
        plt.scatter(fired_op[0,:], fired_op[1,:],c=np.reshape(labels,-1),alpha=1)
        plt.xlabel("X axis")
        plt.ylabel("Y axis")
        plt.title(title)
        plt.show()

    def plot_decision_line(self, train_set, labels):

        hidden_layer_op1 = neuron_operation(self.weights[0][1:], np.array([self.weights[0][0]]).T, train_set)
        hidden_layer_op2 = neuron_operation(self.weights[1][1:].T, np.array([self.weights[1][0]]), hidden_layer_op1.T)
        cx = self.weights[1][1]
        cy = self.weights[1][2]
        c = -self.weights[1][0]
        plt.scatter(hidden_layer_op1[0,:], hidden_layer_op1[1,:], c=np.reshape(labels,-1),alpha=1)
        plt.plot(([-1,2]),([(c/cy-cx*-1/cy).reshape(1,),(c/cy-cx*2/cy).reshape(1,)]),c='r',marker='x')
        plt.xlabel("X axis")
        plt.ylabel("Y axis")
        plt.show()


def neuron_operation(weights, bias, train_set):

    hidden_layer_out = np.dot(weights, np.transpose(train_set)) + bias
    print(hidden_layer_out)
    return sigmoid_functn(hidden_layer_out)


def decision_boundary_plot(X, y, nn, test_idx=None, resolution=0.02):
    # Two decision boundary are required to separate the non-linear XOR data
    # setup marker generator and color map
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('green', 'red', 'darkgreen', 'black', 'blue')
    cmap = ListedColormap(colors[:len(np.unique(y))])

    # plot the decision surface
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                           np.arange(x2_min, x2_max, resolution))

    Z = nn.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
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


# def plot_inp_out_weigthts(weights):
#     from matplotlib.animation import FuncAnimation
#     plt.style.use('seaborn-pastel')
#     fig = plt.figure()
#     ax = plt.axes(xlim=(0, 4), ylim=(-2, 2))
#     line, = ax.plot([], [], lw=3)

if __name__ == '__main__':
    seed(0)
    # train_data = np.array([[0, 0], [0, 1],
    #                        [1, 0], [1, 1]])
    train_data = np.array([[-1, 1], [-1, -1],
                           [1, 1], [1, -1]])
    l_rate = .001
    epochs = 8000000
    labels = np.array([1, -1, -1, 1])
    # labels = np.array([0, 1,
    #                    1, 0])

    xor = XORNeuralNetwork([2, 2, 1])
    xor.fit_model(train_data, labels, l_rate, epochs)
    decision_boundary_plot(train_data, labels, xor)

    plt.xlabel('x-axis')
    plt.ylabel('y-axis')
    plt.legend(loc='upper left')
    plt.tight_layout()
    plt.show()
    # print("Predictions..")
    # for s in train_data:
    #     print(s, xor.predict_single_data(s))

