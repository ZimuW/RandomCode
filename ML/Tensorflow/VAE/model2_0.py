import os
import time
import numpy as np
import tensorflow as tf
from ops import *
from tensorflow.contrib import layers
# import prettytensor as pt


class CNNVAE(object):

    def __init__(self,
                 data_name="Mnist",
                 batch_size=100,
                 input_height=32,
                 input_width=32,
                 output_height=32,
                 output_width=32,
                 z_dim=100,
                 c_dim=3,
                 gf_dim=64,
                 df_dim=64,
                 gfc_dim=1024,
                 dfc_dim=1024,
                 keep_prob=1.0,
                 learning_rate=0.0001,
                 learning_rate_vae=0.0001,
                 beta1=0.9,
                 model_name="cnnvae"
                 ):
        self.isMnist = False
        self.isCifar = False
        self.isImageNet = False
        if data_name == "Mnist":
            self.isMnist = True
            self.c_dim = 1
        elif data_name == "Cifar":
            self.isCifar = True
            self.c_dim = 3

        elif data_name == "ImageNet":
            self.isImageNet = True
            self.c_dim = 3
        self.batch_size = batch_size
        self.input_width = input_width
        self.input_height = input_height
        self.output_height = output_height
        self.output_width = output_width
        self.z_dim = z_dim

        self.gf_dim = gf_dim
        self.df_dim = df_dim
        self.gfc_dim = gfc_dim
        self.dfc_dim = dfc_dim
        self.keep_prob = keep_prob
        self.learning_rate = learning_rate
        self.learning_rate_vae = learning_rate_vae
        self.beta1 = beta1
        self.model_name = model_name

        self.g_bn0 = batch_norm(name="g_bn0")
        self.g_bn1 = batch_norm(name="g_bn1")
        self.g_bn2 = batch_norm(name="g_bn2")
        self.g_bn3 = batch_norm(name="g_bn3")

        self.build_model()

    def build_model(self):
        self.d_real = 0
        self.d_fake = 0
        image_dims = [self.input_height, self.input_width, self.c_dim]
        self.inputs = tf.placeholder(
            tf.float32, [self.batch_size] + image_dims, name="input_images")
        self.inputs_flatten = tf.reshape(self.inputs, [self.batch_size, -1])
        # self.z = tf.placeholder(tf.float32, [None, self.z_dim], name="z")
        self.n_points = self.input_width * self.input_height * self.c_dim

        self.z_mean, self.z_log_sigma_sq = self.encoder(self.inputs)
        eps = tf.random_normal(
            (self.batch_size, self.z_dim), 0, 1, dtype=tf.float32)

        self.z = tf.add(self.z_mean, tf.sqrt(tf.exp(self.z_log_sigma_sq)) * eps)

        self.G = self.generator()
        self.reconstructed_flatten = tf.reshape(self.G, [self.batch_size, -1])


        # create encoder loss
        self.reconstructed_loss=-tf.reduce_sum(self.inputs_flatten * tf.log(1e-10 + self.reconstructed_flatten) + (1 - self.inputs_flatten) * tf.log(1e-10 + 1 - self.reconstructed_flatten), 1)
        # self.intermediate = self.inputs_normalized * tf.log(1e-10 + self.reconstructed_flatten) + (1 - self.inputs_normalized) * tf.log(1e-10 + 1 - self.reconstructed_flatten)
        # self.reconstructed_loss = tf.reduce_sum(tf.square(self.reconstructed_flatten - self.inputs_flatten), 1) / self.input_width / self.input_height / self.c_dim
        self.latent_loss=-0.5 * \
            tf.reduce_sum(1 + tf.clip_by_value(self.z_log_sigma_sq, -10.0, 10.0) - \
                          tf.square(tf.clip_by_value(self.z_mean, -10.0, 10.0)) - tf.exp(tf.clip_by_value(self.z_log_sigma_sq, -10.0, 10.0)), 1)
        self.t_var=tf.trainable_variables()


        #self.e_loss = tf.clip_by_value(self.latent_loss + self.reconstructed_loss, -100, 100)
        self.e_loss = self.latent_loss + self.reconstructed_loss
        self.opt_E = tf.train.AdamOptimizer(0.001, epsilon=1.0).minimize(self.e_loss, var_list = self.t_var)
        self.sess=tf.InteractiveSession()
        self.sess.run(tf.global_variables_initializer())
        self.saver=tf.train.Saver(tf.global_variables())


    def generator(self, reuse=False):
        with tf.variable_scope(tf.get_variable_scope(), reuse=reuse):
            if self.isMnist:
                z2=dense(self.z, self.z_dim, 7 * 7 * self.gf_dim * 4, scope='g_h0_lin')
                h0=tf.nn.relu(self.g_bn0(tf.reshape(z2, [-1, 7, 7, self.gf_dim * 4])))  # 4x4x256
                h1=tf.nn.relu(self.g_bn1(conv_transpose(h0, [self.batch_size, 14, 14, self.gf_dim * 2], name="g_h1")))  # 8x8x128
                h2=conv_transpose(h1, [self.batch_size, 28, 28, 1], name="g_h2")
                return tf.nn.tanh(h2)
            if self.isCifar:
                z2=dense(self.z, self.z_dim, 4 * 4 * self.gf_dim * 4, scope='g_h0_lin')
                h0=tf.nn.relu(self.g_bn0(tf.reshape(z2, [-1, 4, 4, self.gf_dim * 4])))  # 4x4x256
                h1=tf.nn.relu(self.g_bn1(conv_transpose(h0, [self.batch_size, 8, 8, self.gf_dim * 2], name="g_h1")))  # 8x8x128
                h2=tf.nn.relu(self.g_bn2(conv_transpose(h1, [self.batch_size, 16, 16, self.gf_dim * 1], name="g_h2")))  # 16x16x64
                h3=conv_transpose(h2, [self.batch_size, 32, 32, 3], name="g_h3")
                return tf.nn.sigmoid(h3)
            if self.isImageNet:
                z2=dense(self.z, self.z_dim, self.gf_dim * 8 * 4 * 4, scope='g_h0_lin')
                h0=tf.nn.relu(self.g_bn0(tf.reshape(z2, [-1, 4, 4, self.gf_dim * 8])))
                h1=tf.nn.relu(self.g_bn1(conv_transpose(
                    h0, [self.batch_size, 8, 8, self.gf_dim * 4], name="g_h1")))
                h2=tf.nn.relu(self.g_bn2(conv_transpose(
                    h1, [self.batch_size, 16, 16, self.gf_dim * 2], name="g_h2")))
                h3=tf.nn.relu(self.g_bn3(conv_transpose(
                    h2, [self.batch_size, 32, 32, self.gf_dim * 1], name="g_h3")))
                h4=conv_transpose(h3, [self.batch_size, 64, 64, 3], name="g_h4")
                return tf.nn.sigmoid(h4)


    def encoder(self, input_tensor):
        e0=tf.reshape(
            self.inputs, [-1, self.input_height, self.input_width, self.c_dim])
        e1=lrelu(conv2d(e0, 64, name='e_1'))
        e2=lrelu(conv2d(e1, 128, name="e_2"))
        e3=lrelu(conv2d(e2, 256, name="e_3"))
        e4=tf.reshape(e3, [self.batch_size, -1])
        z_mean=fully_connected(e4, self.z_dim, scope="e_linear1")
        z_log_sigma_sq=fully_connected(e4,  self.z_dim, scope="e_linear2")
        return z_mean, z_log_sigma_sq

        # H1 = tf.nn.dropout(tf.nn.(linear(self.inputs_flatten, 512, 'e_lin1')), self.keep_prob)
        # H2 = tf.nn.dropout(tf.nn.sigmoid(linear(H1, 512, 'e_lin2')), self.keep_prob)
        # z_mean = linear(H2, self.z_dim, 'e_lin3_mean')
        # z_log_sigma_sq = linear(H2, self.z_dim, 'e_lin3_log_sigma_sq')
        # return (z_mean, z_log_sigma_sq)


    def partial_train(self, batch):
        _, vae_loss=self.sess.run(
            (self.opt_E, self.e_loss), feed_dict={self.inputs: batch})
        # print d_loss.shape
        # print g_loss.shape
        # print vae_loss.shape
        vae_loss = np.mean(vae_loss) / self.input_width / self.input_width / self.c_dim
        # print vae_loss
        return vae_loss

    def encode(self, X):
        return self.sess.run(self.z_mean, feed_dict={self.inputs: X})

    def generate(self, z=None):
        if z is None:
            z=np.random.normal(size=self.z_dim).astype(np.float32)

        z=np.reshape(z, (self.batch_size, self.z_dim))

        G=self.generator(reuse=True)

        image=self.sess.run(G, feed_dict={self.z: z})
        return image

    def save_model(self, checkpoint_path, epoch):
        self.saver.save(self.sess, checkpoint_path, global_step=epoch)

    def load_model(self, checkpoint_path):

        ckpt=tf.train.get_checkpoint_state(checkpoint_path)
        print "loading model: ", ckpt.model_checkpoint_path
        self.saver.restore(self.sess, ckpt.model_checkpoint_path)

    def close(self):
        self.sess.close()
