import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data


def show():
    plt.scatter(x_data, y_data, c="r")
    plt.plot(x_data, sess.run(w) * x_data + sess.run(b))
    plt.show()


def linear():
    global x_data, y_data, w, b, sess
    num_points = 1000
    vectors_set = []
    for i in range(num_points):
        x1 = np.random.normal(0.0, 0.55)
        y1 = x1 * 0.1 + 0.3 + np.random.normal(0.0, 0.03)
        vectors_set.append([x1, y1])
    x_data = [v[0] for v in vectors_set]
    y_data = [v[1] for v in vectors_set]
    w = tf.Variable(tf.random_uniform([1], -1.0, 1.0), name="W")
    b = tf.Variable(tf.zeros([1], name="b"))
    y = w * x_data + b
    # 均方差
    loss = tf.reduce_mean(tf.square(y - y_data), name="loss")
    optimizer = tf.train.GradientDescentOptimizer(0.5)
    train = optimizer.minimize(loss, name="train")
    with tf.Session() as sess:
        init = tf.global_variables_initializer()
        sess.run(init)
        print("w=", sess.run(w), "b=", sess.run(b), "loss=", sess.run(loss))
        for iter in range(100):
            sess.run(train)
            print("w=", sess.run(w), "b=", sess.run(b), "loss=", sess.run(loss))

        show()


mnist = input_data.read_data_sets("data/", one_hot=True)


def minist():
    print("type of 'mnist'is{}".format(type(mnist)))
    print("number of train data is{}".format(mnist.train.num_examples))
    trainimg = mnist.train.images
    trainlable = mnist.train.labels
    testimg = mnist.test.images
    testlable = mnist.test.labels
    print("mnist load success")
    print(trainimg.shape)
    print(trainlable[0])

    x = tf.placeholder("float", [None, 784])
    y = tf.placeholder("float", [None, 10])

    W = tf.Variable(tf.zeros([784, 10]))
    b = tf.Variable(tf.zeros([10]))

    actv = tf.nn.softmax(tf.matmul(x, W) + b)
    cost = tf.reduce_mean(-tf.reduce_sum(y * tf.log(actv), reduction_indices=1))
    learing_rate = 0.05
    optm = tf.train.GradientDescentOptimizer(learing_rate).minimize(cost)

    pred = tf.equal(tf.argmax(actv, 1), tf.arg_max(y, 1))
    accur = tf.reduce_mean(tf.cast(pred, "float"))
    init = tf.global_variables_initializer()

    training_epochs = 50
    batch_size = 100
    display_step = 5

    sess = tf.Session()
    sess.run(init)
    for epoch in range(training_epochs):
        avg_cost = 0
        num_batch = int(mnist.train.num_examples / batch_size)
        for i in range(num_batch):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            sess.run(optm, feed_dict={x: batch_xs, y: batch_ys})
            feeds = {x: batch_xs, y: batch_ys}
            avg_cost += sess.run(cost, feed_dict=feeds) / num_batch
        if epoch % display_step == 0:
            feed_train = {x: batch_xs, y: batch_ys}
            feeds_test = {x: mnist.test.images, y: mnist.test.labels}
            train_acc = sess.run(accur, feed_dict=feed_train)
            test_acc = sess.run(accur, feed_dict=feeds_test)
            print("epoch:{}  {} cost:{} train_accur:{}test_accur:{}"
                  .format(epoch, training_epochs, avg_cost, train_acc, test_acc))


# minist()

# def network():
n_hidden_1 = 256
n_hidden_2 = 128
n_input = 784
n_classes = 10
with tf.name_scope('Input'):
    x = tf.placeholder("float", [None, n_input])
    y = tf.placeholder("float", [None, n_classes])

stddev = 0.1
with tf.name_scope('Inference'):
    weight = {
        "w1": tf.Variable(tf.random_normal([n_input, n_hidden_1], stddev=stddev)),
        "w2": tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2], stddev=stddev)),
        "out": tf.Variable(tf.random_normal([n_hidden_2, n_classes], stddev=stddev))
    }

    biases = {
        "b1": tf.Variable(tf.random_normal([n_hidden_1])),
        "b2": tf.Variable(tf.random_normal([n_hidden_2])),
        "out": tf.Variable(tf.random_normal([n_classes]))
    }

print("network ready")


def multilayer_perceptron(_X, _weight, _biases):
    layer_1 = tf.nn.relu(tf.add(tf.matmul(_X, _weight["w1"]), _biases["b1"]))
    layer_2 = tf.nn.relu(tf.add(tf.matmul(layer_1, _weight["w2"]), _biases["b2"]))
    return tf.nn.relu((tf.matmul(layer_2, _weight["out"]) + _biases["out"]))


pred = multilayer_perceptron(x, weight, biases)

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
tf.summary.scalar('loss',cost)
optm = tf.train.GradientDescentOptimizer(learning_rate=0.001).minimize(cost)
corr = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))

accr = tf.reduce_mean(tf.cast(corr, "float"))
tf.summary.scalar('accr',accr)

init = tf.global_variables_initializer()

training_epochs = 100
batch_size = 100
display_step = 2

saver = tf.train.Saver()
with tf.Session() as sess:
    sess.run(init)
    for epoch in range(training_epochs):
        avg_cost = 0
        num_batch = int(mnist.train.num_examples / batch_size)
        for i in range(num_batch):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            sess.run(optm, feed_dict={x: batch_xs, y: batch_ys})
            feeds = {x: batch_xs, y: batch_ys}
            avg_cost += sess.run(cost, feed_dict=feeds) / num_batch
        if epoch % display_step == 0:
            feed_train = {x: batch_xs, y: batch_ys}
            feeds_test = {x: mnist.test.images, y: mnist.test.labels}
            train_acc = sess.run(accr, feed_dict=feed_train)
            test_acc = sess.run(accr, feed_dict=feeds_test)
            print("-----------predict:{}"
                  .format(sess.run(pred, feed_dict={x: batch_xs, y: batch_ys})))
            print("epoch:{}  {} cost:{} train_accur:{}test_accur:{}"
                  .format(epoch, training_epochs, avg_cost, train_acc, test_acc))
        merged = tf.summary.merge_all()

        rs = sess.run(merged,feed_dict=feed_train)
        writer = tf.summary.FileWriter("data/", sess.graph)
        writer.add_summary(rs, epoch)
