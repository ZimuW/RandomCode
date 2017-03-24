# [Variational Autoencoder(Link to wiki page)](https://github.com/Fishest/RandomCode/wiki/Variational-Autoencoder)
## Basic model
The basic model feeds the image batch into two or three fully-connected layers to encode and also use fully-connected layers to decode. 
To be able to run the cifar dataset, download the dataset and decompressed them into this repository. [Link to the dataset](https://www.cs.toronto.edu/~kriz/cifar.html) and [download link](https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz)
To run the basic model 
```
python old_train.py
```
If you want to test on the cifar dataset. Change **IsMnist** in ```old_train.py``` to be *False*.

From my own experience, the network converges pretty fast, so about 10 epochs should give a decent result regardless of the dimension you choose for the latent space.

## Convolutional model
This model uses convolutional layers in encoding and deconvolutional layers in decoding.
To run the model
```
python train.py
```
To train on different dataset you have to modify ```train.py``` and ```model2_0.py``` to adjust the name of the dataset and dimensionality of the input image.
