import tensorflow as tf
import numpy as np
import csv
import random

class model():
    def __init__(self,x, y, random_selection):
        self.x, self.y = x, y
        self.random_selection = random_selection
        # self.opin = opin
        self.num_class = 3
        self.training_epochs = 400
        self.batch_size = 128
        self.X = tf.placeholder(dtype=tf.float32, shape=[None, 9], name='factors')
        self.Y = tf.placeholder(dtype=tf.int32, shape=[None, ], name='result')
        self.keep_prob = tf.placeholder(tf.float32)
        self.Y_one_hot = tf.one_hot(self.Y, self.num_class)
        self.Y_one_hot = tf.reshape(self.Y_one_hot, [-1, self.num_class])
        initializer = tf.contrib.layers.xavier_initializer()
        self.weights = {
            'h1': tf.Variable(initializer([9, 128])),
            'h2': tf.Variable(initializer([128, 64])),
            'h3': tf.Variable(initializer([64, 32])),
            'out': tf.Variable(initializer([32, self.num_class]))
        }
        self.biases = {
            'b1': tf.Variable(initializer([128])),
            'b2': tf.Variable(initializer([64])),
            'b3': tf.Variable(initializer([32])),
            'out': tf.Variable(initializer([self.num_class]))
        }
    def train(self):

        layer_1 = tf.nn.relu(tf.matmul(self.X, self.weights['h1']) + self.biases['b1'])
        layer_1 = tf.nn.dropout(layer_1, keep_prob=self.keep_prob)
        layer_2 = tf.nn.relu(tf.matmul(layer_1, self.weights['h2']) + self.biases['b2'])
        layer_2 = tf.nn.dropout(layer_2, keep_prob=self.keep_prob)
        layer_3 = tf.nn.relu(tf.matmul(layer_2, self.weights['h3']) + self.biases['b3'])
        layer_3 = tf.nn.dropout(layer_3, keep_prob=self.keep_prob)
        logits = tf.matmul(layer_3, self.weights['out']) + self.biases['out']
        hypothesis = tf.nn.softmax(logits)

        reg2 = tf.nn.l2_loss(self.weights['h1']) + tf.nn.l2_loss(self.weights['h2'])\
               + tf.nn.l2_loss(self.weights['h3']) + tf.nn.l2_loss(self.weights['out'])
        loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=self.Y_one_hot)) + 0.001 * reg2

        global_step = tf.Variable(0, trainable=False)
        learning_rate = tf.train.exponential_decay(1e-3, global_step, 2000, 0.9, staircase=True)

        train_op = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss_op, global_step=global_step)
        prediction = tf.argmax(hypothesis, 1)

        correc_prediction = tf.equal(prediction, tf.argmax(self.Y_one_hot, 1))
        accuracy = tf.reduce_mean(tf.cast(correc_prediction, tf.float32))
        init = tf.global_variables_initializer()

        # x, y = fileread("dataset/all_data.csv")
        # random_selection = random.sample(range(len(x)), int(len(x) * 0.9))
        train_x, train_y = self.x[self.random_selection], self.y[self.random_selection]
        self.random_selection = random.sample(range(len(train_x)), int(len(train_x) * 0.9))
        train_x, train_y = train_x[self.random_selection], train_y[self.random_selection]
        # print(train_x.shape, train_y.shape)
        test_x, test_y = np.delete(self.x, self.random_selection, axis=0), np.delete(self.y, self.random_selection)
        # print(test_x.shape, test_y.shape)
        with tf.Session() as sess:
            sess.run(init)
            for epoch in range(self.training_epochs):
                avg_cost = 0
                total_batch = int(len(self.x)/ self.batch_size)
                for step in range(0, total_batch - 1):
                    batch_x, batch_y = train_x[step * self.batch_size: (step + 1) * self.batch_size],\
                                       train_y[step * self.batch_size: (step + 1) * self.batch_size]
                    _, loss, accu = sess.run([train_op, loss_op, accuracy], feed_dict={self.X:batch_x, self.Y: batch_y, self.keep_prob:0.9})
                    avg_cost += loss / total_batch
                # print("Epoch : %d, loss : %f, accuracy : %f " % (epoch, avg_cost, accu))
            print("Test time!")

            _, accur, pred, ans = sess.run([train_op, accuracy, prediction, self.Y_one_hot], feed_dict={self.X:test_x, self.Y:test_y, self.keep_prob:1})
            # hapos = sess.run(prediction, feed_dict={self.X : self.opin})
            print("Test Accuracy : %f" % (accur))
            # print("happy position : ", hapos)
            return pred, ans, pred[0:3]



def main():
    x, y = fileread("dataset/all_data.csv")
    opin = fileread("dataset/our_data.csv")
    random_selection = random.sample(range(3,len(x)), int(len(x) * 0.9))
    model1 = model(x, y, random_selection)
    model2 = model(x, y, random_selection)
    model3 = model(x, y, random_selection)
    model4 = model(x, y, random_selection)
    model5 = model(x, y, random_selection)
    # for i in range(5):

    model_1 = model1.train()
    model_2 = model2.train()
    model_3 = model3.train()
    model_4 = model4.train()
    model_5 = model5.train()
    ensemble_model =[model_1[0], model_2[0], model_3[0], model_4[0], model_5[0]]
    votes = []

    for i in range(len(ensemble_model[0])):
        num_model = len(ensemble_model)
        buffer = np.zeros(num_model)
        for j in range(num_model):
            buffer[ensemble_model[j][i]] += 1
        votes.append(np.argmax(buffer))
    answer = np.argmax(np.array(model_1[1]), axis=1)
    accuracy = sum(votes == answer)/len(answer)
    print("our opinion! : ", [model_1[2],model_2[2],model_3[2],model_4[2],model_5[2]])
    print("Test accuracy : ", accuracy)



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