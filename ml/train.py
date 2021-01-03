"""
Unsupervised image classification module
"""
import os
import pandas as pd
import numpy as np
from PIL import Image
import requests
from io import BytesIO
import keras
from keras import layers
from keras.datasets import mnist
import matplotlib.pyplot as plt
import plaidml.keras
plaidml.keras.install_backend()
from time import time
import numpy as np
import keras.backend as K
from keras.engine.topology import Layer, InputSpec
from keras.layers import Dense, Input
from keras.models import Model
from keras.optimizers import SGD
from keras import callbacks
from keras.initializers import VarianceScaling
from sklearn.cluster import KMeans

os.environ["KERAS_BACKEND"] = "plaidml.keras.backend"
n = 0
def io(url):
    try:
        response = requests.get(url)
        img_array = (Image.open(BytesIO(response.content)).convert('L')).resize((400,400))
        img_array = img_to_array(img_array)
    except:
        empty_img = Image.new('L', (400, 400))
        img_array = empty_img.resize((400,400))
        img_array = img_to_array(img_array)

    return img_array


def add_img_array(data):
    data['img_array'] = data['url'].map(io)
    data.to_hdf('reddit_img.h5', key='data')

def auto_encoder(dims, act='relu', init='glorot_uniform'):
    """
    Fully connected auto-encoder model, symmetric.
    Arguments:
        dims: list of number of units in each layer of encoder. dims[0] is input dim, dims[-1] is units in hidden layer.
            The decoder is symmetric with encoder. So number of layers of the auto-encoder is 2*len(dims)-1
        act: activation, not applied to Input, Hidden and Output layers
    return:
        (ae_model, encoder_model), Model of autoencoder and model of encoder
    """
    n_stacks = len(dims) - 1
    # input
    input_img = Input(shape=(dims[0],), name='input')
    x = input_img
    # internal layers in encoder
    for i in range(n_stacks-1):
        x = Dense(dims[i + 1], activation=act, kernel_initializer=init, name='encoder_%d' % i)(x)

    # hidden layer
    encoded = Dense(dims[-1], kernel_initializer=init, name='encoder_%d' % (n_stacks - 1))(x)  # hidden layer, features are extracted from here

    x = encoded
    # internal layers in decoder
    for i in range(n_stacks-1, 0, -1):
        x = Dense(dims[i], activation=act, kernel_initializer=init, name='decoder_%d' % i)(x)

    # output
    x = Dense(dims[0], kernel_initializer=init, name='decoder_0')(x)
    decoded = x
    return Model(inputs=input_img, outputs=decoded, name='AE'), Model(inputs=input_img, outputs=encoded, name='encoder')

def auto_encoder_old():
    # This is the size of our encoded representations
    encoding_dim = 32  # 32 floats -> compression of factor 24.5, assuming the input is 784 floats

    # This is our input image
    input_img = keras.Input(shape=(400,))
    # "encoded" is the encoded representation of the input
    encoded = layers.Dense(encoding_dim, activation='relu')(input_img)
    # "decoded" is the lossy reconstruction of the input
    decoded = layers.Dense(400, activation='sigmoid')(encoded)

    # This model maps an input to its reconstruction
    autoencoder = keras.Model(input_img, decoded)

    # This model maps an input to its encoded representation
    encoder = keras.Model(input_img, encoded)
    # This is our encoded (32-dimensional) input
    encoded_input = keras.Input(shape=(encoding_dim,))
    # Retrieve the last layer of the autoencoder model
    decoder_layer = autoencoder.layers[-1]
    # Create the decoder model
    decoder = keras.Model(encoded_input, decoder_layer(encoded_input))

    autoencoder.compile(optimizer='adam', loss='binary_crossentropy')
    return autoencoder, encoder, decoder

def mnist_test():
    (x_train, _), (x_test, _) = mnist.load_data()
    print(x_train)
    x_train = x_train.astype('float32') / 255.
    x_test = x_test.astype('float32') / 255.
    x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
    x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))
    print(x_train.shape)
    print(x_test.shape)
    return x_test, x_train




def train(autoencoder, x_train, x_test):
    autoencoder.fit(x_train, x_train,
                epochs=50,
                batch_size=256,
                shuffle=True,
                validation_data=(x_test, x_test))

def visualize(encoder, decoder, x_train, x_test):
    # Encode and decode some digits
    # Note that we take them from the *test* set
    encoded_imgs = encoder.predict(x_test)
    decoded_imgs = decoder.predict(encoded_imgs)
    n = 10  # How many digits we will display
    plt.figure(figsize=(20, 4))
    for i in range(n):
        # Display original
        ax = plt.subplot(2, n, i + 1)
        plt.imshow(x_test[i].reshape(28, 28))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        # Display reconstruction
        ax = plt.subplot(2, n, i + 1 + n)
        plt.imshow(decoded_imgs[i].reshape(28,28))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    plt.show()

if __name__ == "__main__":
    try:
        print('try')
        topics_data = pd.read_hdf('./reddit_img.h5', key='data')

    except Exception as e:
        print( "Error: %s" % e )
        topics_data = pd.read_csv('../redditretriever/reddit.csv')
        add_img_array(topics_data)
    

    x_train = topics_data['img_array']
    print(x_train[3])
    x_train = x_train / 255.
    x_train = x_train.values
    print(x_train[0].shape)
    print(x_train[1].shape)
    print(x_train[2].shape)
    print(x_train[3].shape)

    x_train = np.concatenate(x_train)
    print(len(x_train))
    print(np.prod(x_train.shape[1:]))
    x_train = x_train.reshape((len(x_train), int(np.prod(x_train.shape[1:]))))

    print(x_train.shape)
   

    
    """
    autoencoder, encoder, decoder = auto_encoder_old()

    train(autoencoder, x_train, x_train)
    visualize(encoder, decoder, x_train, x_train)
    """
    n_clusters = 10
    kmeans = KMeans(n_clusters=n_clusters, n_init=20, n_jobs=4)
    y_pred_kmeans = kmeans.fit_predict(x_train)
    print(y_pred_kmeans)
    print(y_pred_kmeans.shape)




    """
    image = Image.fromarray(img_array)
    print(image.format)
    print(image.size)
    print(image.mode)
    image.show()
    x_train = np.array(topics_data['img_array'].tolist())
    x_train, x_test = mnist_test()
    """