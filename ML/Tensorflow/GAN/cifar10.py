from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf
from datetime import datetime
import os.path
import time
import gzip
import os
import re
import sys
import tarfile
import numpy as np
from six.moves import xrange
from six.moves import urllib
data_dir = './cifar10_train'
import numpy as np
import matplotlib.pyplot as plt
import cPickle
import random as random

class DataLoader(object):
    def __init__(self, batch_size=100, test_batch=False, all_images=True):
        self.data_dir = "./cifar-10-batches-py"
        self.batch_size = batch_size
        self.test_batch = test_batch
        self.all_images = all_images
        datafiles = ['data_batch_1', 'data_batch_2', 'data_batch_3', 'data_batch_4', 'data_batch_5']

        if test_batch == True:
          datafiles = ['test_batch']

        if all_images == True:
          datafiles = ['data_batch_1', 'data_batch_2', 'data_batch_3', 'data_batch_4', 'data_batch_5', 'test_batch']

        def unpickle(f):
          fo = open(f, 'rb')
          d = cPickle.load(fo)
          fo.close()
          return d

        self.data = []

        for f in datafiles:
          d = unpickle(self.data_dir+'/'+f)
          data = d["data"]
          labels = np.array(d["labels"])
          nsamples = len(data)
          for d in data:
            k = d.reshape(3, 32, 32).transpose(1, 2, 0)

            self.data.append(k)

        self.data = np.array(self.data, dtype=np.float32)
       

        self.num_examples = len(self.data)

        self.pointer = 0

        self.shuffle_data()

    def show_random_image(self):
        pos = 1
        for i in range(10):
          for j in range(10):
            plt.subplot(10, 10, pos)
            img = random.choice(self.data)
            # (channel, row, column) => (row, column, channel)
            plt.imshow(np.clip(img, 0.0, 1.0), interpolation='none')
            plt.axis('off')
            pos += 1
        plt.show()

    def show_image(self, image):
        '''
        image is in [height width depth]
        '''
        plt.subplot(1, 1, 1)
        plt.imshow(np.clip(image, 0.0, 1.0), interpolation='none')
        plt.axis('off')
        plt.show()

    def next_batch(self, batch_size):
        self.pointer += batch_size
        if self.pointer >= self.num_examples:
          self.pointer = 0
        result = []
        # def random_flip(x):
        #   if np.random.rand(1)[0] > 0.5:
        #     return np.fliplr(x)
        #   return x
        # for data in self.data[self.pointer:self.pointer+batch_size]:
        #   result.append(random_flip(data))
        # return self.distort_batch(np.array(result, dtype=np.float32))
        for data in self.data[self.pointer : self.pointer + batch_size]:
            result.append(data)
        
        return np.array(result, dtype=np.float32)

    def distort_batch(self, batch):
        batch_size = len(batch)
        row_distort = np.random.randint(0, 3, batch_size)
        col_distort = np.random.randint(0, 3, batch_size)
        result = np.zeros(shape=(batch_size, 30, 30, 3), dtype=np.float32)
        for i in range(batch_size):
          result[i, :, :, :] = batch[i, row_distort[i]:row_distort[i]+30, col_distort[i]:col_distort[i]+30, :]
        return result

    def shuffle_data(self):
        self.data = np.random.permutation(self.data)


def maybe_download_and_extract():
  """Download and extract the tarball from Alex's website."""
  DATA_URL = 'http://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz'
  dest_directory = data_dir
  if not os.path.exists(dest_directory):
    os.makedirs(dest_directory)
  filename = DATA_URL.split('/')[-1]
  filepath = os.path.join(dest_directory, filename)
  if not os.path.exists(filepath):
    def _progress(count, block_size, total_size):
      sys.stdout.write('\r>> Downloading %s %.1f%%' % (filename,
          float(count * block_size) / float(total_size) * 100.0))
      sys.stdout.flush()
    filepath, _ = urllib.request.urlretrieve(DATA_URL, filepath, _progress)
    print()
    statinfo = os.stat(filepath)
    print('Successfully downloaded', filename, statinfo.st_size, 'bytes.')

  tarfile.open(filepath, 'r:gz').extractall(dest_directory)
