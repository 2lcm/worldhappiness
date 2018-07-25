import tensorflow as tf
import numpy as np
tf.set_random_seed(135)

# read csv file
xy = np.genfromtxt('dataset/all_data.csv', delimiter=',', skip_header=1)
for i in range(1, xy.shape[1] - 1):
    xy = xy[[not b for b in np.isnan(xy[:, i])], :]
x_data = xy[:, 1:-1]
y_data = xy[:, [-1]]

nb_x_col = x_data.shape[1]
nb_classes = 2

X = tf.placeholder(tf.float32, [None, nb_x_col])
Y = tf.placeholder(tf.int32, [None, 1])
Y_one_hot = tf.one_hot(Y, nb_classes)  # one hot
print("one_hot", Y_one_hot)
Y_one_hot = tf.reshape(Y_one_hot, [-1, nb_classes])
print("reshape", Y_one_hot)

W = tf.Variable(tf.random_normal([nb_x_col, nb_classes]), name='weight')
b = tf.Variable(tf.random_normal([nb_classes]), name='bias')

logits = tf.matmul(X, W) + b
hypothesis = tf.nn.softmax(logits)

cost_i = tf.nn.softmax_cross_entropy_with_logits(logits=logits,
                                                 labels=Y_one_hot)
cost = tf.reduce_mean(cost_i)
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(cost)

prediction = tf.argmax(hypothesis, 1)
correct_prediction = tf.equal(prediction, tf.argmax(Y_one_hot, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for step in range(10001):
        sess.run(optimizer, feed_dict={X: x_data, Y: y_data})
        if step % 100 == 0:
            loss, acc = sess.run([cost, accuracy], feed_dict={
                                 X: x_data, Y: y_data})
            print("Step: {:5}\tLoss: {:.3f}\tAcc: {:.2%}".format(
                step, loss, acc))
    #
    # # Let's see if we can predict
    # pred = sess.run(prediction, feed_dict={X: x_data})
    # # y_data: (N,1) = flatten => (N, ) matches pred.shape
    # for p, y in zip(pred, y_data.flatten()):
    #     print("[{}] Prediction: {} True Y: {}".format(p == int(y), p, int(y)))
