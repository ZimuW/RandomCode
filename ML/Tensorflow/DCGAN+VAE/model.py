import os
import time
import numpy as np
import tensorflow as tf
from ops import *

class DCGANVAE(object):
    def __init__(self,
                 batch_size=500,
                 input_height=28,
                 input_width=28,
                 output_height=28,
                 output_width=28,
                 z_dim=100,
                 c_dim=3,
                 gf_dim=64,
                 df_dim=64,
                 gfc_dim=1024,
                 dfc_dim=1024,
                 keep_prob=1.0,
                 learning_rate=0.01,
                 learning_rate_d=0.001,
                 learning_rate_vae=0.0001,
                 beta1=0.9,
                 model_name="dcganvae"
                 ):
        self.batch_size = batch_size
        self.input_width = input_width
        self.input_height = input_height
        self.output_height = output_height
        self.output_width = output_width
        self.z_dim = z_dim
        self.c_dim = c_dim
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
        image_dims = [self.input_height, self.input_width, self.c_dim]
        self.inputs = tf.placeholder(tf.float32, [self.batch_size] + image_dims, name="input_images")
        self.inputs_flatten = tf.reshape(self.inputs, [batch_size, -1])
        # self.z = tf.placeholder(tf.float32, [None, self.z_dim], name="z")
        self.n_points = self.input_width * self.input_height * self.c_dim
        self.z_mean, self.z_log_sigma_sq = self.encoder()
        eps = tf.random_normal((self.batch_size, self.z_dim), 0, 1, dtype=tf.float32)

        self.z = tf.add(self.z_mean, tf.mul(tf.sqrt(tf.exp(self.z_log_sigma_sq)), eps))

        self.G = self.generator()
        self.reconstructed_flatten = tf.reshape(self.G, [batch_size, -1])

        self.D, self.D_logits = self.discriminator(inputs)
        self.D_, self.D_logits_ = self.discriminator(self.G, reuse=True)

        # create discriminator loss
        self.d_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D_logits, labels=tf.ones_like(self.D)))
        self.d_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D_logits_, labels=tf.zeros_like(self.D_))
        self.total_d_loss = self.d_loss_fake + self.d_loss_real
        # create generator loss
        self.g_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D_logits_, labels=tf.ones_like(self.D_)))

        # create encoder loss
        self.reconstructed_loss = -tf.reduce_sum(self.inputs_flatten * tf.log(1e-10+self.reconstructed_flatten) + (1-self.inputs_flatten) * tf.log(1e-10 + 1-self.reconstructed_flatten), 1)
        self.latent_loss = -0.5 * tf.reduce_sum(1 + self.z_log_sigma_sq - tf.square(self.z_mean) -tf.exp(self.z_log_sigma_sq), 1)
        self.vae_loss = tf.reduce_mean(self.reconstructed_loss + self.latent_loss) / self.n_points

        self.combined_loss = self.g_loss_fake + self.vae_loss

        self.t_vars = tf.trainable_variables()

        self.encoder_var = [var for var in self.t_var if (self.model_name + "_e_") in var.name]
        self.discriminator_var = [var for var in self.t_var if (self.model_name + "_d_") in var.name]
        self.generator_var = [var for var in self.t_var if (self.model_name + "_g_") in var.name]
        self.d_opt = tf.train.AdamOptimizer(self.learning_rate_d, beta1=self.beta1).minimize(self.d_loss, var_list=self.d)
