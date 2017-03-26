from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import time
import gzip
import os
import re
import sys
import tarfile
import numpy as np
from six.moves import xrange
from six.moves import urllib
import numpy as np
import matplotlib.pyplot as plt
import cPickle
import random as random
from glob import glob
import numpy as np
import scipy.misc
import h5py


def imread(path):
    return scipy.misc.imread(path).astype(np.float)

def resize_width(image, width=64.):
    h, w = np.shape(image)[:2]
    return scipy.misc.imresize(image,[int((float(h)/w)*width),width])

def center_crop(x, height=64):
    h= np.shape(x)[0]
    j = int(round((h - height)/2.))
    return x[j:j+height,:,:]

def get_image(image_path, width=64, height=64):
    return center_crop(resize_width(imread(image_path), width = width),height=height)

class CelebLoader(object):
    def __init__(self, batch_size=100):
        self.data = glob(os.path.join("../../data/img_align_celeba", "*.jpg"))
        self.data = np.sort(self.data)
        self.num_examples = len(self.data)
        self.pointer = 0
        self.shuffle_data()
        self.data = self.data[:100000]
        self.img_data = []
        with h5py.File(''.join(['../datasets/faces_dataset_new.h5']), 'r') as hf:
            self.img_data = hf['images'].value
        self.num_examples = len(self.img_data)
    def dataset_size(self):
        return self.num_examples

    def show_random_image(self):
        pos = 1
        for i in range(10):
            for j in range(10):
                plt.subplot(10, 10, pos)
                img = random.choice(self.img_data)
                # img = get_image(img)
                # img = img / 255.0
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

    def shuffle_data(self):
        self.data = np.random.permutation(self.data)

    def next_batch(self, batch_size=100):
        self.pointer += batch_size
        if self.pointer >= self.num_examples:
            self.pointer = 0
        result = []
        for data in self.img_data[self.pointer:self.pointer + batch_size]:
            tmp = data / 255.0
            result.append(tmp)
        return np.array(result, dtype=np.float32)
