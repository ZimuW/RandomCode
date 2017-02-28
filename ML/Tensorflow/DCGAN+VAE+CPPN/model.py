import os
import time
import numpy as np
import tensorflow as tf
from ops import *


class CPPNVAE(object):

    def __init__(self,
                 batch_size=1,
                 z_dim=32,
                 x_dim=26,
                 y_dim=26,
                 c_dim=1,
                 scale=8.0,
                 learning_rate=0.01,
                 learning_rate_d=0.001,
                 learning_rate_vae=0.0001,
                 beta1=0.9,
                 net_size_g=128,
                 net_depth_g=4,
                 net_size_q=512,
                 keep_prob=1.0,
                 df_dim=24,
                 model_name="cppnvae"):
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.learning_rate_d = learning_rate_d
        self.learning_rate_vae = learning_rate_vae
        self.beta1 = beta1
        self.net_size_g = net_size_g
        self.net_size_q = net_size_q
        self.x_dim = x_dim
        self.y_dim = y_dim
        self.scale = scale
        self.c_dim = c_dim
        self.z_dim = z_dim
        self.net_depth_g = net_depth_g
        self.model_name = model_name
        self.keep_prob = keep_prob
        self.df_dim = df_dim

        self.batch = tf.placeholder(
            tf.float32, [batch_size, x_dim, y_dim, c_dim])
        self.batch_flatten = tf.reshape(self.batch, [batch_size, -1])

        n_points = x_dim * y_dim
        self.n_points = n_points

        self.x_vec, self.y_vec, self.r_vec = self.coordinates(
            x_dim, y_dim, scale)

        self.x = tf.placeholder(tf.float32, [self.batch_size, None, 1])
        self.y = tf.placeholder(tf.float32, [self.batch_size, None, 1])
        self.r = tf.placeholder(tf.float32, [self.batch_size, None, 1])

        self.d_bn1 = batch_norm(batch_size, name=self.model_name + '_d_bn1')
        self.d_bn2 = batch_norm(batch_size, name=self.model_name + '_d_bn2')

        self.z_mean, self.z_log_sigma_sq = self.encoder()

        eps = tf.random_normal(
            (self.batch_size, self.z_dim), 0, 1, dtype=tf.float32)

        self.z = tf.add(self.z_mean, tf.mul(
            tf.sqrt(tf.exp(self.z_log_sigma_sq)), eps))

        self.G = self.generator()
        self.batch_reconstruct_flatten = tf.reshape(self.G, [batch_size, -1])

        self.D_right = self.discriminator(self.batch)
        self.D_wrong = self.discriminator(
            self.G, reuse=True)

        self.create_vae_loss_terms()
        self.create_gan_loss_terms()
        self.balanced_loss = 1.0 * self.g_loss + 1.0 * self.vae_loss

        self.t_vars = tf.trainable_variables()

        self.q_vars = [var for var in self.t_vars if (
            self.model_name + '_q_') in var.name]
        self.g_vars = [var for var in self.t_vars if (
            self.model_name + '_g_') in var.name]
        self.d_vars = [var for var in self.t_vars if (
            self.model_name + '_d_') in var.name]
        self.vae_vars = self.q_vars + self.g_vars

        # Use ADAM optimizer
        self.d_opt = tf.train.AdamOptimizer(self.learning_rate_d, beta1=self.beta1) \
                          .minimize(self.d_loss, var_list=self.d_vars)
        self.g_opt = tf.train.AdamOptimizer(self.learning_rate, beta1=self.beta1) \
                          .minimize(self.balanced_loss, var_list=self.vae_vars)
        self.vae_opt = tf.train.AdamOptimizer(self.learning_rate_vae, beta1=self.beta1) \
                          .minimize(self.vae_loss, var_list=self.vae_vars)

        # Initializing the tensor flow variables
        init = tf.initialize_all_variables()

        # Launch the session
        self.sess = tf.InteractiveSession()
        self.sess.run(init)
        self.saver = tf.train.Saver(tf.all_variables())

  def create_vae_loss_terms(self):
    reconstr_loss = \
        -tf.reduce_sum(self.batch_flatten * tf.log(1e-10 + self.batch_reconstruct_flatten)
                       + (1-self.batch_flatten) * tf.log(1e-10 + 1 - self.batch_reconstruct_flatten), 1)
    latent_loss = -0.5 * tf.reduce_sum(1 + self.z_log_sigma_sq
                                       - tf.square(self.z_mean)
                                       - tf.exp(self.z_log_sigma_sq), 1)
    self.vae_loss = tf.reduce_mean(reconstr_loss + latent_loss) / self.n_points

  def create_gan_loss_terms(self):
    self.d_loss_real = binary_cross_entropy_with_logits(tf.ones_like(self.D_right), self.D_right)
    self.d_loss_fake = binary_cross_entropy_with_logits(tf.zeros_like(self.D_wrong), self.D_wrong)
    self.d_loss = 1.0*(self.d_loss_real + self.d_loss_fake)/ 2.0
    self.g_loss = 1.0*binary_cross_entropy_with_logits(tf.ones_like(self.D_wrong), self.D_wrong)

  def coordinates(self, x_dim = 32, y_dim = 32, scale = 1.0):
    n_pixel = x_dim * y_dim
    x_range = scale*(np.arange(x_dim)-(x_dim-1)/2.0)/(x_dim-1)/0.5
    y_range = scale*(np.arange(y_dim)-(y_dim-1)/2.0)/(y_dim-1)/0.5
    x_mat = np.matmul(np.ones((y_dim, 1)), x_range.reshape((1, x_dim)))
    y_mat = np.matmul(y_range.reshape((y_dim, 1)), np.ones((1, x_dim)))
    r_mat = np.sqrt(x_mat*x_mat + y_mat*y_mat)
    x_mat = np.tile(x_mat.flatten(), self.batch_size).reshape(self.batch_size, n_pixel, 1)
    y_mat = np.tile(y_mat.flatten(), self.batch_size).reshape(self.batch_size, n_pixel, 1)
    r_mat = np.tile(r_mat.flatten(), self.batch_size).reshape(self.batch_size, n_pixel, 1)
    return x_mat, y_mat, r_mat

  def show_image(self, image):
    plt.subplot(1, 1, 1)
    y_dim = image.shape[0]
    x_dim = image.shape[1]
    if self.c_dim > 1:
      plt.imshow(image, interpolation='nearest')
    else:
      plt.imshow(image.reshape(y_dim, x_dim), cmap='Greys', interpolation='nearest')
    plt.axis('off')
    plt.show()

  def encoder(self):
    H1 = tf.nn.dropout(tf.nn.softplus(linear(self.batch_flatten, self.net_size_q, self.model_name+'_q_lin1')), self.keep_prob)
    H2 = tf.nn.dropout(tf.nn.softplus(linear(H1, self.net_size_q, self.model_name+'_q_lin2')), self.keep_prob)
    z_mean = linear(H2, self.z_dim, self.model_name+'_q_lin3_mean')
    z_log_sigma_sq = linear(H2, self.z_dim, self.model_name+'_q_lin3_log_sigma_sq')
    return (z_mean, z_log_sigma_sq)

  def discriminator(self, image, reuse=False):

    if reuse:
        tf.get_variable_scope().reuse_variables()

    h0 = lrelu(conv2d(image, self.df_dim, name=self.model_name+'_d_h0_conv'))
    h1 = lrelu(self.d_bn1(conv2d(h0, self.df_dim*2, name=self.model_name+'_d_h1_conv')))
    h2 = lrelu(self.d_bn2(conv2d(h1, self.df_dim*4, name=self.model_name+'_d_h2_conv')))
    h3 = linear(tf.reshape(h2, [self.batch_size, -1]), 1, self.model_name+'_d_h2_lin')

    return tf.nn.sigmoid(h3)

  def generator(self, gen_x_dim = 26, gen_y_dim = 26, reuse = False):

    if reuse:
        tf.get_variable_scope().reuse_variables()

    n_network = self.net_size_g
    gen_n_points = gen_x_dim * gen_y_dim

    z_scaled = tf.reshape(self.z, [self.batch_size, 1, self.z_dim]) * \
                    tf.ones([gen_n_points, 1], dtype=tf.float32) * self.scale
    z_unroll = tf.reshape(z_scaled, [self.batch_size*gen_n_points, self.z_dim])
    x_unroll = tf.reshape(self.x, [self.batch_size*gen_n_points, 1])
    y_unroll = tf.reshape(self.y, [self.batch_size*gen_n_points, 1])
    r_unroll = tf.reshape(self.r, [self.batch_size*gen_n_points, 1])

    U = fully_connected(z_unroll, n_network, self.model_name+'_g_0_z') + \
        fully_connected(x_unroll, n_network, self.model_name+'_g_0_x', with_bias = False) + \
        fully_connected(y_unroll, n_network, self.model_name+'_g_0_y', with_bias = False) + \
        fully_connected(r_unroll, n_network, self.model_name+'_g_0_r', with_bias = False)

    H = tf.nn.softplus(U)

    for i in range(1, self.net_depth_g):
      H = tf.nn.tanh(fully_connected(H, n_network, self.model_name+'_g_tanh_'+str(i)))

    output = tf.sigmoid(fully_connected(H, self.c_dim, self.model_name+'_g_'+str(self.net_depth_g)))

    result = tf.reshape(output, [self.batch_size, gen_y_dim, gen_x_dim, self.c_dim])

    return result


  def partial_train(self, batch):
    """Train model based on mini-batch of input data.
    Return cost of mini-batch.
    I should really seperate the below tricks into parameters, like number of times/pass
    and also the regulator threshold levels.
    """

    counter = 0
    '''
    for i in range(4):
      counter += 1
      _, vae_loss, g_loss = self.sess.run((self.g_opt, self.vae_loss, self.g_loss),
                              feed_dict={self.batch: batch, self.x: self.x_vec, self.y: self.y_vec, self.r: self.r_vec})
      if g_loss < 0.6:
        break
    '''

    for i in range(4):
      counter += 1
      _, vae_loss = self.sess.run((self.vae_opt, self.vae_loss),
                              feed_dict={self.batch: batch, self.x: self.x_vec, self.y: self.y_vec, self.r: self.r_vec})


    for i in range(4):
      counter += 1
      _, g_loss = self.sess.run((self.g_opt, self.g_loss),
                              feed_dict={self.batch: batch, self.x: self.x_vec, self.y: self.y_vec, self.r: self.r_vec})
      if g_loss < 0.6:
        break

    d_loss = self.sess.run(self.d_loss,
                              feed_dict={self.batch: batch, self.x: self.x_vec, self.y: self.y_vec, self.r: self.r_vec})

    if d_loss > 0.6 and g_loss < 0.75:
      for i in range(1):
        counter += 1
        _, d_loss = self.sess.run((self.d_opt, self.d_loss),
                                feed_dict={self.batch: batch, self.x: self.x_vec, self.y: self.y_vec, self.r: self.r_vec})
        if d_loss < 0.6:
          break

    return d_loss, g_loss, vae_loss, counter

  def encode(self, X):
      """Transform data by mapping it into the latent space."""
      # Note: This maps to mean of distribution, we could alternatively
      # sample from Gaussian distribution
      return self.sess.run(self.z_mean, feed_dict={self.batch: X})

  def generate(self, z=None, x_dim = 26, y_dim = 26, scale = 5.0):
    """ Generate data by sampling from latent space.
    If z is not None, data for this point in latent space is
    generated. Otherwise, z is drawn from prior in latent
    space.
    """
    if z is None:
        z = np.random.normal(size=self.z_dim).astype(np.float32)
    # Note: This maps to mean of distribution, we could alternatively
    # sample from Gaussian distribution

    z = np.reshape(z, (self.batch_size, self.z_dim))

    G = self.generator(gen_x_dim = x_dim, gen_y_dim = y_dim, reuse = True)
    gen_x_vec, gen_y_vec, gen_r_vec = self.coordinates(x_dim, y_dim, scale = scale)
    image = self.sess.run(G, feed_dict={self.z: z, self.x: gen_x_vec, self.y: gen_y_vec, self.r: gen_r_vec})
    return image

  def save_model(self, checkpoint_path, epoch):
    """ saves the model to a file """
    self.saver.save(self.sess, checkpoint_path, global_step = epoch)

  def load_model(self, checkpoint_path):

    ckpt = tf.train.get_checkpoint_state(checkpoint_path)
    print "loading model: ",ckpt.model_checkpoint_path

    self.saver.restore(self.sess, checkpoint_path+'/'+ckpt.model_checkpoint_path)
    # use the below line for tensorflow 0.7
    # self.saver.restore(self.sess, ckpt.model_checkpoint_path)

  def close(self):
    self.sess.close()
