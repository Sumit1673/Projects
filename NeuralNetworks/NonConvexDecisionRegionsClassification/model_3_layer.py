import tensorflow.compat.v1 as tf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import datetime
import csv
import tensorflow.python.util.deprecation as deprecation
deprecation._PRINT_DEPRECATION_WARNINGS = False
tf.disable_v2_behavior()


def split_train_test(data, test_ratio):
    shuffled_indices = np.random.permutation(len(data))
    test_set_size = int(len(data) * test_ratio)
    test_indices = shuffled_indices[:test_set_size]
    train_indices = shuffled_indices[test_set_size:]
    return data.iloc[train_indices], data.iloc[test_indices]


def split_feature_labels(data):
    feature = np.zeros(shape=(len(data), 2))
    labels = np.zeros(shape=(len(data), 1))
    for i in range(0, len(data)):
        x = data[i]
        feature[i][0], feature[i][1] = x[0], x[1]
        labels[i] = x[2]
    return feature, labels

df_csv = pd.read_csv("data_non_convex.csv")
training_data = df_csv.values.tolist()
train_set, test_set = split_train_test(df_csv, 0.1)  # 10%
train_set_list = train_set.values.tolist()
train_X, labels = split_feature_labels(train_set_list)

# plt.scatter(train_feature[:, 0], train_feature[:, 1], c=labels.ravel(), cmap=plt.cm.coolwarm
#             , label="Training Data")
# plt.show()

# -----------------------Network Architecture Begins -------------------------------
# n_samples = train_feature.shape[0]
# define hyperparamters
n_epochs = 150000
learning_rate = 0.00007
n_inp_neurons = 2
hidden_layer1_neurons = 15
output_layer_neurons = 1
time = datetime.datetime.now()
log_dir = "./log/3_layer/4/"
if not os.path.exists(log_dir):
    os.mkdir(log_dir)
filename = time.strftime("%Y-%m-%d_%H:%M")
log_filename = log_dir + str(filename)+".csv"
log_file = open(log_filename, 'w+')
csv_file_writer = csv.writer(log_file)
csv_file_writer.writerow(["Epoch", "LR", "Loss", "Accuracy"])

def fc_layer(inp, channels_in, channels_out, name='name'):
    with tf.name_scope(name):
        W = tf.Variable(tf.truncated_normal([channels_in, channels_out], -1.0, 1.0), name="W")
        B = tf.Variable(tf.constant(0.1, shape=[channels_out]), name="B")
        act_firing = tf.sigmoid(tf.matmul(inp, W) + B)

        tf.summary.histogram('weights', W)
        tf.summary.histogram('biases', B)
        tf.summary.histogram('sigmoid_activation', act_firing)
        return act_firing


def compute_loss(Y, Y_hat):
    """
    Log Loss Loss is calculated to evaluate the prediction.
     logLoss=−1/N (∑Ni=1(y_i(log p_i)+(1−y_i)log(1−p_i)))
    Log Loss is a slight twist on something called the Likelihood Function. In fact, Log Loss is
    -1 * the log of the  likelihood function. So, we will start by understanding the likelihood function
    """
    cost = tf.reduce_mean(-Y * tf.log(Y_hat) - (1 - Y) * tf.log(1 - Y_hat))
    return cost


def compute_accuracy(Y, Y_hat):
    """ Find the accuracy of the model """
    # ans = tf.equal(tf.floor(Y_hat+0.5), Y)
    # acc = tf.reduce_mean(tf.cast(ans, tf.float32))
    acc = tf.reduce_mean(tf.cast(tf.equal(Y_hat, Y), dtype=tf.float32))
    return acc


# defining tensorflow variables
X = tf.placeholder(tf.float32, name='Train_X')
Y = tf.placeholder(tf.float32, name="label_Y")

# create the layers now:
fc1 = fc_layer(X, n_inp_neurons, hidden_layer1_neurons, name="fc1")

# prediction will be out in from layer 2
hypothesis = fc_layer(fc1, hidden_layer1_neurons, output_layer_neurons, name="fc2")

# compute loss
with tf.name_scope("log_loss"):
    cost = compute_loss(Y, hypothesis)
tf.summary.scalar('log_loss', cost)

# classification check if prediction is greater than 0.6
prediction = tf.cast(hypothesis > 0.9, dtype=tf.float32)

# computer accuracy:
with tf.name_scope("accuracy"):
    # accuracy = compute_accuracy(Y, hypothesis)
    accuracy = compute_accuracy(Y, prediction)
tf.summary.scalar('accuracy', accuracy)

# weight optimization the acutal training of the model.
with tf.name_scope("train"):
    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)

with tf.Session() as sess:
    merged_summary = tf.summary.merge_all()
    writer = tf.summary.FileWriter(log_dir)
    writer.add_graph(sess.graph)
    sess.run(tf.global_variables_initializer())

    for step in range(n_epochs):
        if step % 1000 == 0:
            acc = accuracy.eval({X: train_X, Y: labels})
            loss = sess.run(cost, feed_dict={X: train_X, Y: labels})
            csv_file_writer.writerow([step, learning_rate, cost, acc])
            print("Epoch: ", step, "lr: ", learning_rate,
                  " Loss: ",loss, " Acc: ", acc)
            if loss < 0.18:
                break
        if step % 5 == 0:
            s = sess.run(merged_summary, feed_dict={X: train_X, Y: labels})
            writer.add_summary(s, step)
        sess.run(optimizer, feed_dict={X: train_X, Y: labels})

    # print(sess.run(hypothesis, feed_dict={X: train_feature, Y: labels}))
    # plot_d/ecision_boundary(lambda t: sess.run(prediction, feed_dict={X: rain_feature}), train_feature, labels)
    # test_pre = int(np.array(sess.run(prediction, feed_dict={X: train_feature, Y: labels})))
    x_min, x_max = train_X[:, 0].min() - 1, train_X[:, 0].max() + 1
    y_min, y_max = train_X[:, 1].min() - 1, train_X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1), np.arange(y_min, y_max, 0.1))
    f, axarr = plt.subplots(1, 1, sharex='col', sharey='row', figsize=(10, 8))
    Z = sess.run(prediction, feed_dict={X: np.c_[xx.ravel(), yy.ravel()]})
    Z = Z.reshape(xx.shape)
    axarr.contourf(xx, yy, Z, alpha=0.4)
    axarr.scatter(train_X[:, 0], train_X[:, 1], c=labels.reshape(len(train_X),),
                  s=20, edgecolor='k')
    axarr.set_title("Non Convex data")

    plt.show()