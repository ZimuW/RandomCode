import numpy as np
import tensorflow as tf

import argparse
import time
import os
import cPickle

isMnist=False
isCifar=True
if isMnist:
    from tensorflow.examples.tutorials.mnist import input_data
    mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
    n_samples = mnist.train.num_examples
if isCifar:
    from cifar10 import DataLoader
    cifar = DataLoader()
from model import DCGANVAE

'''
dcgan vae:

compositional pattern-producing generative adversarial network

LOADS of help was taken from:

https://github.com/carpedm20/DCGAN-tensorflow
https://jmetzen.github.io/2015-11-27/vae.html

'''

def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('--training_epochs', type=int, default=100,
                     help='training epochs')
  parser.add_argument('--display_step', type=int, default=1,
                     help='display step')
  parser.add_argument('--checkpoint_step', type=int, default=1,
                     help='checkpoint step')
  parser.add_argument('--batch_size', type=int, default=500,
                     help='batch size')
  parser.add_argument('--learning_rate', type=float, default=0.005,
                     help='learning rate for G and VAE')
  parser.add_argument('--learning_rate_vae', type=float, default=0.001,
                     help='learning rate for VAE')
  parser.add_argument('--learning_rate_d', type=float, default=0.001,
                     help='learning rate for D')
  parser.add_argument('--keep_prob', type=float, default=1.00,
                     help='dropout keep probability')
  parser.add_argument('--beta1', type=float, default=0.65,
                     help='adam momentum param for descriminator')
  args = parser.parse_args()
  return train(args)

def train(args):

  learning_rate = args.learning_rate
  learning_rate_d = args.learning_rate_d
  learning_rate_vae = args.learning_rate_vae
  batch_size = args.batch_size
  training_epochs = args.training_epochs
  display_step = args.display_step
  checkpoint_step = args.checkpoint_step # save training results every check point step
  beta1 = args.beta1
  keep_prob = args.keep_prob
  dirname = 'save'
  if not os.path.exists(dirname):
    os.makedirs(dirname)

  with open(os.path.join(dirname, 'config.pkl'), 'w') as f:
    cPickle.dump(args, f)

  dcganvae = DCGANVAE(data_name="Cifar", batch_size=batch_size, learning_rate = learning_rate, learning_rate_d = learning_rate_d, learning_rate_vae = learning_rate_vae, beta1 = beta1, keep_prob = keep_prob)
  counter = 0


  for epoch in range(training_epochs):
    avg_d_loss = 0.
    avg_q_loss = 0.
    avg_vae_loss = 0.
    total_batch = 100
    for i in range(total_batch):
      if isMnist:
          batch_images = mnist.train.next_batch(batch_size)[0]
          print batch_images.shape
          batch_images = np.reshape(batch_images, [batch_size, 28, 28, 1])
      else:
          batch_images = cifar.next_batch(batch_size)
          print batch_images.shape
          batch_images = np.reshape(batch_images, [batch_size, 32, 32, 3])
      d_loss, g_loss, vae_loss, n_operations = dcganvae.partial_train(batch_images)
    #   print d_loss
    #   print g_loss
    #   print vae_loss
    #   print i
      assert( vae_loss < 1000000 ) # make sure it is not NaN or Inf
      assert( d_loss < 1000000 ) # make sure it is not NaN or Inf
      assert( g_loss < 1000000 ) # make sure it is not NaN or Inf
      if (counter+1) % display_step == 0:
        print "Sample:", '%d' % ((i+1)*batch_size), " Epoch:", '%d' % (epoch), \
              "d_loss=", "{:.4f}".format(d_loss), \
              "g_loss=", "{:.4f}".format(g_loss), \
              "vae_loss=", "{:.4f}".format(vae_loss), \
              "n_op=", '%d' % (n_operations)
      counter += 1
      avg_d_loss += d_loss / n_samples * batch_size
      avg_q_loss += g_loss / n_samples * batch_size
      avg_vae_loss += vae_loss / n_samples * batch_size

    if epoch >= 0:
      print "Epoch:", '%04d' % (epoch), \
            "avg_d_loss=", "{:.6f}".format(avg_d_loss), \
            "avg_q_loss=", "{:.6f}".format(avg_q_loss), \
            "avg_vae_loss=", "{:.6f}".format(avg_vae_loss)

    if epoch >= 0 and epoch % checkpoint_step == 0:
      checkpoint_path = os.path.join('save', 'model.ckpt')
      dcganvae.save_model(checkpoint_path, epoch)
      print "model saved to {}".format(checkpoint_path)

  dcganvae.save_model(checkpoint_path, 0)

if __name__ == '__main__':
  main()
