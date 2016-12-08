from keras.layers import Input, Dense
from keras.models import Model
from keras.datasets import mnist
import numpy as np
import matplotlib.pyplot as plt


def main():
    # network structure
    input_layer = Input(shape=(784,))
    encoded = Dense(128, activation='relu')(input_layer)
    encoded = Dense(64, activation='relu')(encoded)
    encoded = Dense(32, activation='relu')(encoded)

    decoded = Dense(64, activation='relu')(encoded)
    decoded = Dense(128, activation='relu')(decoded)
    decoded = Dense(784, activation='sigmoid')(decoded)
    autoencoder = Model(input=input_layer, output=decoded)
    autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')

    # build an extra model for encoder
    # encoder = Model(input=input_layer, output=encoded)

    # build an extra model for decoder
    # encoded_input = Input(shape=(32,))
    # decoder_layer = autoencoder.layers[-3](encoded_input)
    # decoder_layer = autoencoder.layers[-2](decoder_layer)
    # decoder_layer = autoencoder.layers[-1](decoder_layer)
    # decoder = Model(input=encoded_input, output=decoder_layer)

    (x_train, _), (x_test, _) = mnist.load_data()
    x_train = x_train.astype('float32') / 255.
    x_test = x_test.astype('float32') / 255.
    x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
    x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))

    noise_factor = 0.5
    x_train_noisy = x_train + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=x_train.shape) 
    x_test_noisy = x_test + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=x_test.shape) 

    x_train_noisy = np.clip(x_train_noisy, 0., 1.)
    x_test_noisy = np.clip(x_test_noisy, 0., 1.)

    print x_train.shape
    print x_test.shape

    autoencoder.fit(x_train, x_train,
                    nb_epoch=50,
                    batch_size=256,
                    shuffle=True,
                    validation_data=(x_test_noisy, x_test))

    # encoded_imgs = encoder.predict(x_test)
    # decoded_imgs = decoder.predict(encoded_imgs)
    decoded_imgs = autoencoder.predict(x_test)

    n = 10 
    plt.figure(figsize=(20, 4))
    for i in range(n):
        # display original
        ax = plt.subplot(2, n, i + 1)
        plt.imshow(x_test_noisy[i].reshape(28, 28))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        # display reconstruction
        ax = plt.subplot(2, n, i + 1 + n)
        plt.imshow(decoded_imgs[i].reshape(28, 28))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    plt.show()
    return


if __name__ == '__main__':
    main()