import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

tf.set_random_seed(135)
columns = ['Life Ladder','Log GDP per capita','Social support','Healthy life expectancy at birth',
           'Freedom to make life choices','Generosity','Perceptions of corruption',
           'Confidence in national government','Democratic Quality','Delivery Quality']

# read csv file
xy = np.genfromtxt('dataset/all_data.csv', delimiter=',', skip_header=1)

X = tf.placeholder(tf.float32, shape=[None])
Y = tf.placeholder(tf.float32, shape=[None])

W = tf.Variable(tf.random_normal([1]), name='weight')
b = tf.Variable(tf.random_normal([1]), name='bias')

hypothesis = X * W + b
cost = tf.reduce_mean(tf.square(hypothesis - Y))

optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
train = optimizer.minimize(cost)
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for i in range(1, xy.shape[1]):
        xy_tmp = xy[[not b for b in np.isnan(xy[:, i])], :]
        x_data = xy_tmp[:, i]
        min_x = min(x_data)
        max_x = max(x_data)
        x_data = [(x_data[j] - min_x) / (max_x - min_x) for j in range(len(x_data))]
        y_data = xy_tmp[:, 0]

        W_val = 0.
        b_val = 0.
        for step in range(2001):
            cost_val, W_val, b_val, _ = sess.run([cost, W, b, train], feed_dict={X: x_data, Y: y_data})
            if step % 200 == 0:
                print(step, cost_val, W_val, b_val)

        plt.plot(x_data, y_data, 'o')
        print(min(x_data))
        print(max(x_data))
        x = np.arange(min(x_data), max(x_data), 0.01)
        y = W_val * x + b_val
        plt.plot(x, y)
        plt.xlabel(columns[i])
        plt.ylabel(columns[0])
        plt.savefig('img/img'+str(i)+'.png')
        plt.show()