"""
Function to find the gradient descent on a set of data points divided into 4 classes
Algorithm:

a(k + 1) = a(k) − η(k)∇J(a(k))
a is weight vector
η is learning rate.
J is the hypothesis function
∇ - gradient
k is the iteration or the sample number
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def grab_data(file_path):
    data_df = pd.read_csv(file_path)
    data_features = np.zeros([10,8])
    i=0
    for col in data_df.columns:
        data_features[:,i] = data_df[col]
        i+=1
    train_features, labels = create_dataset(data_features)

    return [train_features, labels]


def create_dataset(data):

    cls_1 = data[:, 0:2]
    cls_2 = -data[:, 2:4]
    features = np.concatenate((cls_1,cls_2))
    labels = np.array([1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0])
    return features, labels


def split_data():
    pass


def optimize_weight_vector(features):
    gradient_descent(epochs=35000, tr_label=1, train_set=features, lr=0.001)


def sigmoid_activation(x):
    # compute and return the sigmoid activation value for a
    # given input value
    return 1.0 / (1 + np.exp(-x))


def gradient_descent(*, epochs=50, train_set=None, tr_label=None, lr=0.1):
    """
    a(k + 1) = a(k) − η(k)∇J(a(k))
    ∇J(a) = y(k)

    :param epochs:
    :param train_set:
    :param tr_label:
    :param lr:
    :return:
    """
    loss_history = []
    training_data, labels = train_set[0], train_set[1]

    training_data_points = augument_train_vector(training_data)

    # initialize weights
    weights = np.random.uniform(size=(training_data_points.shape[1],))
    loss = 0
    grd_jbya = None  # augumented input vector y_i

    for each_epoch in np.arange(0, epochs):
        pred = find_hypothesis(training_data_points.dot(weights))
        # pred = training_data_points.dot(weights)
        error = pred - labels
        loss = np.sum(error ** 2)
        loss_history.append(loss)
        print("[INFO] epoch #{}, loss={:.7f}".format(each_epoch + 1, loss))

        gradient = training_data_points.T.dot(error) / training_data_points.shape[0]
        weights += -(lr * gradient)

    plot_results(weights, training_data, labels,loss_history, epochs)


def find_hypothesis(x):
    """
    using sigmoid function for the hypothesis
       J(a(k)) = dot(a'Y) > 0.5 -> clssified
                 dot(a'Y) < 0.5 -> Misclassified
    :return:
    """
    # compute and return the sigmoid activation value for a
    # given input value
    return 1.0 / (1 + np.exp(-x))


def augument_train_vector(data_points):
    """
    augument input/train vector x into y i.e. [x1, x2, x3 ...... xn, 1]

    :return:
    """
    return np.c_[np.ones((data_points.shape[0])), data_points]


def plot_results(W, X,labels, loss, epochs):
    Y = (-W[0] - (W[1] * X)) / W[2]

    # plot the original data along with our line of best fit
    plt.figure()
    plt.scatter(X[:, 0], X[:, 1], marker="o", c=labels)
    plt.plot(X, Y, "r-")

    # construct a figure that plots the loss over time
    fig = plt.figure()
    plt.plot(np.arange(0, epochs), loss)
    fig.suptitle("Training Loss")
    plt.xlabel("Epoch #")
    plt.ylabel("Loss")
    plt.show()


def classify():
    pass


def main():
    features = grab_data('sample_gd.csv')
    # split_data()
    optimize_weight_vector(features)
    # classify()


if __name__ == '__main__':
    main()