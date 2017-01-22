import tensorflow as tf
import numpy as np

from model import VariantionalAutoencoder
import matplotlib.pyplot as plt
from cifar10 import DataLoader


cifar = DataLoader()
isMnist = False
if isMnist:
    from tensorflow.examples.tutorials.mnist import input_data
    mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
    n_samples = mnist.train.num_examples
def train(network_architecture, learning_rate=0.001, batch_size=100, training_epochs=10, display_step=5):
    vae = VariantionalAutoencoder(network_architecture, learning_rate=learning_rate, batch_size=batch_size)
    for epoch in range(training_epochs):
        avg_cost = 0
        total_batch = 600
        for i in range(total_batch):
            if isMnist:
                batch_xs, _ = mnist.train.next_batch(batch_size)
            else:
                batch_xs = cifar.next_batch(batch_size)
            cost = vae.minibatch(batch_xs)
            avg_cost += cost
        print("Epoch:", '%04d' % (epoch), "cost=", "{:.9f}".format(avg_cost))

    return vae

network_architecture = \
    dict(n_hidden_recog_1=500, # 1st layer encoder neurons
         n_hidden_recog_2=500, # 2nd layer encoder neurons
         n_hidden_gener_1=500, # 1st layer decoder neurons
         n_hidden_gener_2=500, # 2nd layer decoder neurons
         n_input=28*28 if isMnist else 32*32*3, # MNIST data input (img shape: 28*28)
         n_z=20)  # dimensionality of latent space

vae = train(network_architecture, training_epochs=5)
if isMnist:
    x_sample = mnist.test.next_batch(100)[0]
else:
    x_sample = cifar.next_batch()

x_reconstruct = vae.reconstruct(x_sample)[0]
print(x_sample.shape)
print(x_reconstruct.shape)
plt.figure(figsize=(8, 12))
for i in range(5):

    plt.subplot(5, 2, 2*i + 1)
    plt.imshow(x_sample[i].reshape(28, 28), vmin=0, vmax=1, cmap="gray")
    plt.title("Test input")
    plt.colorbar()
    plt.subplot(5, 2, 2*i + 2)
    plt.imshow(x_reconstruct[i].reshape(28, 28), vmin=0, vmax=1, cmap="gray")
    plt.title("Reconstruction")
    plt.colorbar()
plt.tight_layout()
plt.show()
