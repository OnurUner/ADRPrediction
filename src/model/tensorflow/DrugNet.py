#!/usr/bin/env python2
# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np


class DrugNet:
    n_hidden_1 = 4096
    n_hidden_2 = 2048
    n_hidden_3 = 1024
    n_hidden_4 = 512

    dropout = 0.75

    def __init__(self, num_features, num_classes):
        self.learning_rate = 0.01
        self.display_step = 1
        self.num_features = num_features
        self.num_classes = num_classes
        self.batch_index = 0

    def calc_class_weights(self, Y):
        weights = []
        for i in range(Y.shape[1]):
            counts = np.bincount(Y[:, i].astype(int))
            num_neg = float(counts[0])
            num_pos = float(counts[1])
            weights.append(num_neg / num_pos)
        return weights

    def neural_net(self, x, reuse, is_training):

        with tf.variable_scope('DrugNet', reuse=reuse):
            layer_1 = tf.layers.dense(inputs=x, units=self.n_hidden_1, activation=tf.nn.relu, kernel_initializer=tf.contrib.layers.xavier_initializer(), bias_initializer=tf.contrib.layers.xavier_initializer())
            layer_2 = tf.layers.dense(inputs=layer_1, units=self.n_hidden_2, activation=tf.nn.relu, kernel_initializer=tf.contrib.layers.xavier_initializer(), bias_initializer=tf.contrib.layers.xavier_initializer())
            layer_3 = tf.layers.dense(inputs=layer_2, units=self.n_hidden_3, activation=tf.nn.relu, kernel_initializer=tf.contrib.layers.xavier_initializer(), bias_initializer=tf.contrib.layers.xavier_initializer())
            layer_4 = tf.layers.dense(inputs=layer_3, units=self.n_hidden_4, activation=tf.nn.relu, kernel_initializer=tf.contrib.layers.xavier_initializer(), bias_initializer=tf.contrib.layers.xavier_initializer())
            layer_5 = tf.layers.dropout(layer_4, rate=self.dropout, training=is_training)

            out = tf.layers.dense(inputs=layer_5, units=self.num_classes)

        return out

    def fit(self, x_train, y_train, epochs=30, batch_size=30, is_balanced=False):
        X = tf.placeholder(tf.float32, [batch_size, self.num_features])
        Y = tf.placeholder(tf.float32, [batch_size, self.num_classes])
        logits = self.neural_net(X, False, is_training=True)

        if is_balanced:
            weights = tf.placeholder(tf.float32, [self.num_classes])
            class_weights = self.calc_class_weights(y_train)
            entropy = tf.nn.weighted_cross_entropy_with_logits(logits=logits, targets=Y, pos_weight=weights)
        else:
            entropy = tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=Y)

        self.loss = tf.reduce_mean(entropy)  # computes the mean over examples in the batch

        optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.loss)

        init = tf.global_variables_initializer()
        with tf.Session() as sess:
            sess.run(init)
            n_batches = int(x_train.shape[0] / batch_size)
            for epoch in range(epochs):
                epoch_loss = []
                epoch_acc = []
                for i in range(n_batches):
                    start_index = i * batch_size
                    end_index = ((i + 1) * batch_size)
                    x_batch = x_train[start_index:end_index, :]
                    y_batch = y_train[start_index:end_index, :]
                    if is_balanced:
                        _, loss_batch, logits_batch = sess.run([optimizer, self.loss, logits],
                                                               feed_dict={X: x_batch, Y: y_batch,
                                                                          weights: class_weights})
                    else:
                        _, loss_batch, logits_batch = sess.run([optimizer, self.loss, logits],
                                                               feed_dict={X: x_batch, Y: y_batch})
                    y_pred = tf.nn.sigmoid(logits_batch)
                    correct_prediction = tf.equal(tf.round(y_pred), tf.cast(tf.round(y_batch), tf.float32))
                    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
                    epoch_loss.append(loss_batch)
                    epoch_acc.append(accuracy)

                if (epoch + 1) % self.display_step == 0:
                    tf_mean_loss = tf.reduce_mean(epoch_loss)
                    tf_mean_acc = tf.reduce_mean(epoch_acc)
                    mean_loss, mean_acc = sess.run([tf_mean_loss, tf_mean_acc])
                    print "Epoch:", '%04d' % (epoch + 1), "cost=", "{:.9f}".format(
                        mean_loss), "accuracy=", "{:.9f}".format(mean_acc)

    def predict_proba(self, x_test):
        X = tf.placeholder(tf.float32, [x_test.shape[0], self.num_features])
        logits = self.neural_net(X, True, False)
        init = tf.global_variables_initializer()
        with tf.Session() as sess:
            sess.run(init)
            logits_test = sess.run(logits, feed_dict={X: x_test})
            y_ = tf.nn.sigmoid(logits_test)
            y_pred = sess.run(y_)
        return y_pred

















