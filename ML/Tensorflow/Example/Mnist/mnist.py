from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import tensorflow as tf

NUM_CLASSES = 10

IMAGE_SIZE = 28
IMAGE_PIXELS = IMAGE_SIZE * IMAGE_SIZE

def inference(images, hidden1_units, hidden2_units):
    with tf.name_scope('hidden1'):
        weights = tf.Variable(tf.truncated_normal([IMAGE_SIZE, hidden1_units], stddev=1.0/math.sqrt(float(IMAGE_PIXELS))), name='weights')
        bias = tf.Variable(tf.zeros([hidden1_units]), name='bias')
        hidden1 = tf.nn.relu(tf.matmul(images, weights) + bias)
    with tf.name_scope('hidden2'):
        weights = tf.Variable(tf.truncated_normal([hidden1_units, hidden2_units], stddev=1.0/math.sqrt(float(hidden1_units))), name='weights')
        bias = tf.Variable(tf.zeros([hidden2_units]), name='bias')
        hidden2 = tf.nn.relu(tf.matmul(hidden1, weights) + bias)
    with tf.name_scope('linear'):
        weights = tf.Variable(tf.truncated_normal([hidden2_units, NUM_CLASSES], stddev=1.0/math.sqrt(float(hidden2_units))), name='weights')
        bias = tf.Variable(tf.zeros([NUM_CLASSES]), name='bias')
        logits = tf.matmul(hidden2, weights) + bias
    return logits

def loss(logits, labels):
    labels = tf.to_int64(labels)
    cross_entropy = tf.nn.sparse_softmax_with_logits(logits, labels, name='xentropy')
    loss = tf.reduce_mean(cross_entropy, name='xentropy_mean')
    return loss

def training(loss, learning_rate):
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    global_step = tf.Variable(0, name='global_step', trainable=False)
    train_op = optimizer.minimize(loss, global_step=global_step)
    return train_op

def evaluation(logits, labels):
    # prediction = tf.argmax(logits, 1)
    # correct_prediction = tf.equal(prediction, labels)
    # accuracy = tf.reduce_sum(tf.cast(correct_prediction, tf.int32))
    correct = tf.nn.in_top_k(logits, labels, 1)
    return tf.reduce_sum(tf.cast(correct, tf.int32))
