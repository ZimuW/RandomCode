import tensorflow as tf
import numpy as np

from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
n_samples = mnist.train.num_examples

def xavier_init(fan_in, fan_out, constant=1):
    low = -constant * np.sqrt(6.0/(fan_in + fan_out))
    high = constant * np.sqrt(6.0/(fan_in + fan_out))
    return tf.random_uniform([fan_in, fan_out], minval=low, maxval=high, dtype=tf.float32)

class VariantionalAutoencoder(object):
    def __init__(self, network_architecture, transfer_func=tf.nn.softplus, learning_rate=0.00001, batch_size=100):
        self.network_architecture = network_architecture
        self.transfer_func = transfer_func
        self.learning_rate = learning_rate
        self.batch_size = batch_size

        self.x = tf.placeholder(tf.float32, [None, network_architecture["n_input"]])

        self.create_network()
        self.create_loss_optimizer()

        self.sess = tf.InteractiveSession()
        self.sess.run(tf.global_variables_initializer())

    def create_network(self):
        network_weights = self.initialize_weights(**self.network_architecture)
        self.z_mean, self.z_log_sigma_sq = self.recognition_network(network_weights["weights_recog"], network_weights["biases_recog"])
        n_z = self.network_architecture["n_z"]
        eps = tf.random_normal([self.batch_size, n_z], 0, 1, dtype=tf.float32)

        self.z = tf.add(self.z_mean, tf.mul(tf.sqrt(tf.exp(self.z_log_sigma_sq)), eps))

        self.reconstructed = self.generator_network(network_weights["weights_gener"], network_weights["biases_gener"])

    def initialize_weights(self, n_hidden_recog_1, n_hidden_recog_2, n_hidden_gener_1, n_hidden_gener_2, n_input, n_z):

        all_weights = dict()
        all_weights['weights_recog'] = {
            'h1': tf.Variable(xavier_init(n_input, n_hidden_recog_1)),
            'h2': tf.Variable(xavier_init(n_hidden_recog_1, n_hidden_recog_2)),
            'out_mean': tf.Variable(xavier_init(n_hidden_recog_2, n_z)),
            'out_log_sigma': tf.Variable(xavier_init(n_hidden_recog_2, n_z))}
        all_weights['biases_recog'] = {
            'b1': tf.Variable(tf.zeros([n_hidden_recog_1], dtype=tf.float32)),
            'b2': tf.Variable(tf.zeros([n_hidden_recog_2], dtype=tf.float32)),
            'out_mean': tf.Variable(tf.zeros([n_z], dtype=tf.float32)),
            'out_log_sigma': tf.Variable(tf.zeros([n_z], dtype=tf.float32))}
        all_weights['weights_gener'] = {
            'h1': tf.Variable(xavier_init(n_z, n_hidden_gener_1)),
            'h2': tf.Variable(xavier_init(n_hidden_gener_1, n_hidden_gener_2)),
            'out_mean': tf.Variable(xavier_init(n_hidden_gener_2, n_input))}
        all_weights['biases_gener'] = {
            'b1': tf.Variable(tf.zeros([n_hidden_gener_1], dtype=tf.float32)),
            'b2': tf.Variable(tf.zeros([n_hidden_gener_2], dtype=tf.float32)),
            'out_mean': tf.Variable(tf.zeros([n_input], dtype=tf.float32))}
        return all_weights

    def recognition_network(self, weights, biases):
        layer1 = self.transfer_func(tf.add(tf.matmul(self.x, weights["h1"]), biases["b1"]))
        layer2 = self.transfer_func(tf.add(tf.matmul(layer1, weights["h2"]), biases["b2"]))
        z_mean = tf.add(tf.matmul(layer2, weights["out_mean"]), biases["out_mean"])
        z_log_sigma_sq = tf.add(tf.matmul(layer2, weights["out_log_sigma"]), biases["out_log_sigma"])
        return z_mean, z_log_sigma_sq

    def generator_network(self, weights, biases):
        layer1 = self.transfer_func(tf.add(tf.matmul(self.z, weights["h1"]), biases["b1"]))
        layer2 = self.transfer_func(tf.add(tf.matmul(layer1, weights["h2"]), biases["b2"]))
        reconstructed = tf.nn.sigmoid(tf.add(tf.matmul(layer2, weights["out_mean"]), biases["out_mean"]))
        return reconstructed

    def create_loss_optimizer(self):
        reconstructed_loss = -tf.reduce_sum(self.x * tf.log(1e-10 + self.reconstructed) + (1 - self.x) * (tf.log(1e-10 + 1 - self.reconstructed)))

        latent_loss = -0.5 * tf.reduce_sum(1 + self.z_log_sigma_sq - tf.square(self.z_mean) - tf.exp(self.z_log_sigma_sq), 1)
        self.cost = tf.reduce_mean(reconstructed_loss + latent_loss)
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.cost)

    def minibatch(self, X):
        opt, cost = self.sess.run([self.optimizer, self.cost], feed_dict={self.x: X})
        return cost

    def transform(self, X):
        return self.session.run([self.z], feed_dict={self.x : X})

    def generate(self, z_mu=None):
        if z_mu is None:
            z_mu = np.random.normal(size=self.network_architecture["n_z"])
        return self.sess.run([self.reconstructed], feed_dict={self.z : z_mu})

    def reconstruct(self, X):
        return self.sess.run([self.reconstructed], feed_dict={self.x : X})
