import os
import time
import numpy as np
import tensorflow as tf
from ops import *
from tensorflow.contrib import layers
# import prettytensor as pt


class DCGANVAE(object):

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
                 learning_rate_d=0.0001,
                 learning_rate_vae=0.0001,
                 beta1=0.9,
                 model_name="dcganvae"
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
        self.learning_rate_d = learning_rate_d
        self.learning_rate_vae = learning_rate_vae
        self.beta1 = beta1
        self.model_name = model_name

        self.d_bn1 = batch_norm(name="d_bn1")
        self.d_bn2 = batch_norm(name="d_bn2")
        self.d_bn3 = batch_norm(name="d_bn3")

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
        self.reconstructed_flatten_original = tf.reshape(self.G, [self.batch_size, -1])
        self.reconstructed_flatten = self.reconstructed_flatten_original
        self.D, self.ll_real = self.discriminator(self.inputs)
        self.D_fake , self.ll_fake= self.discriminator(self.G, reuse=True)

        # create discriminator loss
        # self.d_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(self.D_logits, tf.ones_like(self.D)))
        # self.d_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(self.D_logits_fake, tf.zeros_like(self.D_fake)))
        # self.d_loss = self.d_loss_fake + self.d_loss_real
        self.d_loss = tf.reduce_mean(-1.*(tf.log(tf.clip_by_value(self.D,1e-5,1.0)) +
                                    tf.log(tf.clip_by_value(1.0 - self.D_fake ,1e-5,1.0))))

        # create generator loss
        # self.g_loss=tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(self.D_logits_fake, tf.ones_like(self.D_fake)))
        self.g_loss = tf.reduce_mean(-1.*(tf.log(tf.clip_by_value(self.D_fake,1e-5,1.0))))
        self.ll_loss = tf.reduce_sum(tf.square(self.ll_real - self.ll_fake)) / self.input_width / self.input_height / self.c_dim

        # create encoder loss
        # self.reconstructed_loss=-tf.reduce_sum(self.inputs_flatten * tf.log(1e-10 + self.reconstructed_flatten) + (1 - self.inputs_flatten) * tf.log(1e-10 + 1 - self.reconstructed_flatten), 1)
        # self.intermediate = self.inputs_normalized * tf.log(1e-10 + self.reconstructed_flatten) + (1 - self.inputs_normalized) * tf.log(1e-10 + 1 - self.reconstructed_flatten)
        # self.reconstructed_loss = tf.reduce_sum(tf.square(self.reconstructed_flatten - self.inputs_flatten), 1) / self.input_width / self.input_height / self.c_dim
        self.latent_loss=-0.5 * \
            tf.reduce_sum(1 + tf.clip_by_value(self.z_log_sigma_sq, -10.0, 10.0) - \
                          tf.square(tf.clip_by_value(self.z_mean, -10.0, 10.0)) - tf.exp(tf.clip_by_value(self.z_log_sigma_sq, -10.0, 10.0)), 1) / self.input_width / self.input_height / self.c_dim
        self.t_var=tf.trainable_variables()
        self.encoder_var=[var for var in self.t_var if ("e_") in var.name]
        self.discriminator_var=[var for var in self.t_var if ("d_") in var.name]
        self.generator_var=[var for var in self.t_var if ("g_") in var.name]

        self.lr_D = tf.placeholder(tf.float32, shape=[])
        self.lr_G = tf.placeholder(tf.float32, shape=[])
        self.lr_E = tf.placeholder(tf.float32, shape=[])
        self.e_loss = tf.clip_by_value(self.latent_loss + self.ll_loss, -100, 100)
        self.g_loss = tf.clip_by_value(self.ll_loss + self.g_loss, -100, 100)
        self.d_loss = tf.clip_by_value(self.d_loss, -100, 100)
        self.opt_D = tf.train.AdamOptimizer(self.lr_D, epsilon=1.0).minimize(self.d_loss, var_list = self.t_var)
        self.opt_G = tf.train.AdamOptimizer(self.lr_G, epsilon=1.0).minimize(self.g_loss, var_list = self.t_var)
        self.opt_E = tf.train.AdamOptimizer(self.lr_E, epsilon=1.0).minimize(self.e_loss, var_list = self.t_var)
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
                z2=tf.nn.relu(dense(self.z, self.z_dim, 4 * 4 * self.gf_dim * 4, scope='g_h0_lin'))
                h0=tf.nn.relu(self.g_bn0(tf.reshape(z2, [-1, 4, 4, self.gf_dim * 4])))  # 4x4x256
                h1=tf.nn.relu(self.g_bn1(conv_transpose(h0, [self.batch_size, 8, 8, self.gf_dim * 2], name="g_h1")))  # 8x8x128
                h2=tf.nn.relu(self.g_bn2(conv_transpose(h1, [self.batch_size, 16, 16, self.gf_dim * 1], name="g_h2")))  # 16x16x64
                h3=tf.nn.relu(self.g_bn3(conv_transpose(h2, [self.batch_size, 32, 32, 3], name="g_h3")))
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
                return tf.nn.tanh(h4)


    def encoder(self, input_tensor):
        e0=tf.reshape(
            self.inputs, [-1, self.input_height, self.input_width, self.c_dim])
        e1=tf.nn.relu(conv2d(e0, 64, name='e_1'))
        e2=tf.nn.relu(conv2d(e1, 128, name="e_2"))
        e3=tf.nn.relu(conv2d(e2, 256, name="e_3"))
        e4=tf.reshape(e3, [self.batch_size, -1])
        self.half_encoded = e4
        z_mean=tf.nn.relu(fully_connected(e4, self.z_dim, scope="e_linear1"))
        z_log_sigma_sq=tf.nn.relu(fully_connected(e4,  self.z_dim, scope="e_linear2"))
        return z_mean, z_log_sigma_sq

        # H1 = tf.nn.dropout(tf.nn.(linear(self.inputs_flatten, 512, 'e_lin1')), self.keep_prob)
        # H2 = tf.nn.dropout(tf.nn.sigmoid(linear(H1, 512, 'e_lin2')), self.keep_prob)
        # z_mean = linear(H2, self.z_dim, 'e_lin3_mean')
        # z_log_sigma_sq = linear(H2, self.z_dim, 'e_lin3_log_sigma_sq')
        # return (z_mean, z_log_sigma_sq)

    def discriminator(self, image, reuse=False):
        with tf.variable_scope(tf.get_variable_scope(), reuse=reuse):

            if self.isMnist:
                h0=lrelu(conv2d(image, self.df_dim, name='d_h0_conv'))  # 16x16x64
                # 8x8x128
                h1=lrelu(self.d_bn1(conv2d(h0, self.df_dim * 2, name='d_h1_conv')))
                # 4x4x256
                h2=lrelu(self.d_bn2(conv2d(h1,  self.df_dim * 4, name='d_h2_conv')))
                h3=dense(tf.reshape(h2, [self.batch_size, -1]),
                         4 * 4 * self.df_dim * 4, 1, scope='d_h3_lin')
                return tf.nn.sigmoid(h3), h3
            if self.isCifar:
                h0=tf.nn.relu(conv2d(image, self.df_dim, name='d_h0_conv')) # 16x16x64
                # 8x8x128
                h1=tf.nn.relu(self.d_bn1(conv2d(h0, self.df_dim * 2, name='d_h1_conv')))
                # 4x4x256
                h2=tf.nn.relu(self.d_bn2(conv2d(h1,  self.df_dim * 4, name='d_h2_conv')))

                h3=tf.nn.relu(dense(tf.reshape(h2, [self.batch_size, -1]), 4 * 4 * self.df_dim * 4, 1024, scope='d_h3_lin'))
                h4 = dense(h3, 1024, 1, scope='d_h4_lin')
                return tf.nn.sigmoid(h4), h3
            if self.isImageNet:
                h0=lrelu(conv2d(image, self.df_dim, name='d_h0_conv'))
                h1=lrelu(self.d_bn1(conv2d(h0, self.df_dim * 2, name='d_h1_conv')))
                h2=lrelu(self.d_bn2(conv2d(h1, self.df_dim * 4, name='d_h2_conv')))
                h3=lrelu(self.d_bn3(conv2d(h2, self.df_dim * 8, name='d_h3_conv')))
                h4=dense(tf.reshape(h3, [self.batch_size, -1]),
                         4 * 4 * 512, 1, scope='d_h3_lin')
                return tf.nn.sigmoid(h4), h4

    def partial_train(self, batch):
        def sigmoid(x,shift,mult):
            """
            Using this sigmoid to discourage one network overpowering the other
            """
            return 1 / (1 + math.exp(-(x+shift)*mult))
        e_learning_rate = 1e-3
        g_learning_rate = 1e-3
        d_learning_rate = 1e-3
        e_current_lr = e_learning_rate*sigmoid(np.mean(self.d_real),-.5,15)
        g_current_lr = g_learning_rate*sigmoid(np.mean(self.d_real),-.5,15)
        d_current_lr = d_learning_rate*sigmoid(np.mean(self.d_fake),-.5,15)
        _, _, _, d_loss, g_loss, vae_loss, ll_loss, self.d_fake, self.d_real=self.sess.run(
            (self.opt_D, self.opt_E, self.opt_G, self.d_loss, self.g_loss, self.latent_loss, self.ll_loss, self.D_fake, self.D), feed_dict={self.inputs: batch,
                                                                                                     self.lr_E: e_current_lr,
                                                                                                     self.lr_D: d_current_lr,
                                                                                                     self.lr_G: g_current_lr})
        # print d_loss.shape
        # print g_loss.shape
        # print vae_loss.shape
        vae_loss = np.mean(vae_loss)
        # print vae_loss
        return d_loss, g_loss, vae_loss, ll_loss

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
