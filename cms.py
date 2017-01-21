import tensorflow as tf
import numpy as np

n_input = 0
n_hidden1 = 0
n_output = 0

weights = {
    'h1': tf.Variable(tf.random_normal([n_input, n_hidden1])),
    'out': tf.Variable(tf.random_normal([n_hidden1, n_output]))
}

biases = {
    'b1': tf.Variable(tf.random_normal([n_hidden1])),
    'out': tf.Variable(tf.random_normal([n_output]))
}


def RBF():
    pass


def preceptron(x, weights, biases):
    layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
    layer_1 = RBF(layer_1)
