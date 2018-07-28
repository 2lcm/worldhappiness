import tensorflow as tf
import numpy as np
import csv
import random

def main():
    num_class = 3
    training_epochs = 1000
    batch_size = 128
    X = tf.placeholder(dtype=tf.float32, shape=[None, 9], name='factors')
    Y = tf.placeholder(dtype=tf.int32, shape=[None, ], name='result')
    keep_prob = tf.placeholder(tf.float32)
    Y_one_hot = tf.one_hot(Y, num_class)
    Y_one_hot = tf.reshape(Y_one_hot, [-1, num_class])
    initializer = tf.contrib.layers.xavier_initializer()
    weights = {
        'h1' : tf.Variable(initializer([9, 4])),
        'h2' : tf.Variable(initializer([4, 64])),
        'h3' : tf.Variable(initializer([64, 32])),
        'out' : tf.Variable(initializer([32, num_class]))
    }
    biases = {
        'b1' : tf.Variable(initializer([4])),
        'b2' : tf.Variable(initializer([64])),
        'b3' : tf.Variable(initializer([32])),
        'out' : tf.Variable(initializer([num_class]))
    }
    layer_1 = tf.nn.relu(tf.matmul(X, weights['h1']) + biases['b1'])
    layer_1 = tf.nn.dropout(layer_1, keep_prob=keep_prob)
    layer_2 = tf.nn.relu(tf.matmul(layer_1, weights['h2']) + biases['b2'])
    layer_2 = tf.nn.dropout(layer_2, keep_prob=keep_prob)
    layer_3 = tf.nn.relu(tf.matmul(layer_2, weights['h3']) + biases['b3'])
    layer_3 = tf.nn.dropout(layer_3, keep_prob=keep_prob)
    logits = tf.matmul(layer_3, weights['out']) + biases['out']
    hypothesis = tf.nn.softmax(logits)

    reg2 = tf.nn.l2_loss(weights['h1']) + tf.nn.l2_loss(weights['h2'])\
           + tf.nn.l2_loss(weights['h3']) + tf.nn.l2_loss(weights['out'])
    loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=Y_one_hot))

    global_step = tf.Variable(0, trainable=False)
    learning_rate = tf.train.exponential_decay(1e-2, global_step, 1000, 0.9, staircase=True)

    train_op = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss_op, global_step=global_step)
    prediction = tf.argmax(hypothesis, 1)

    correc_prediction = tf.equal(prediction, tf.argmax(Y_one_hot, 1))
    accuracy = tf.reduce_mean(tf.cast(correc_prediction, tf.float32))
    init = tf.global_variables_initializer()

    x, y = fileread("dataset/all_data.csv")
    random_selection = random.sample(range(len(x)), int(len(x) * 0.9))
    train_x, train_y = x[random_selection], y[random_selection]
    # print(train_x.shape, train_y.shape)
    test_x, test_y = np.delete(x, random_selection, axis=0), np.delete(y, random_selection)
    # print(test_x.shape, test_y.shape)
    with tf.Session() as sess:
        sess.run(init)
        for epoch in range(training_epochs):
            avg_cost = 0
            total_batch = int(len(x)/ batch_size)
            for step in range(0, total_batch - 1):
                batch_x, batch_y = train_x[step * batch_size: (step + 1) * batch_size], train_y[step * batch_size: (step + 1) * batch_size]
                _, loss, accu = sess.run([train_op, loss_op, accuracy], feed_dict={X:batch_x, Y: batch_y, keep_prob:1})
                avg_cost += loss / total_batch
            if epoch%20 == 0 :
                print("Epoch : %d, loss : %f, accuracy : %f " % (epoch, avg_cost, accu))
        print("Test time!")

        _, accur = sess.run([train_op, accuracy], feed_dict={X:test_x, Y:test_y, keep_prob:1})
        print("Test Accuracy : %f" % (accur))


def fileread(path):
    xy = csv.reader(open(path))
    next(xy)
    X = []
    Y = []
    for line in xy:
        X.append(line[1:-1])
        Y.append(line[-1])
    X, Y = np.array(X), np.array(Y)
    X = X.astype(np.float).tolist()
    Y = Y.astype(np.int).tolist()
    X, Y = np.array(X), np.array(Y)
    shuffle = np.random.shuffle(np.arange(len(X)))
    X = X[shuffle][0]
    Y = Y[shuffle][0]
    X = normalize(X)
    return X, Y


def normalize(d):
    d -= np.min(d, axis=0)
    d = d / np.ptp(d, axis=0)
    return d


if __name__ == "__main__":
    main()