#!/usr/bin/env python

"""
Script to use the autoencoder from state_ae.py for learning MNIST image 
encodings.
"""

from __future__ import print_function

import sys
import os
import time

import numpy as np
import theano

import lasagne
import state_ae
import pickle

import matplotlib.pyplot as plt

# ##################################################################
# Use the autoencoder to learn image encodings.

def learn(x_train, trainfunc, x_test):

    for it in range(state_ae.NUM_EPOCHS):

        p = 0
        count = 0
        avg_cost = 0
        while True:
            x, y = state_ae.gen_data(x_train, p)
            p += len(x)
            count += 1
            avg_cost += trainfunc(x, y)

            if (p == len(x_train)):
                break

        print("Epoch {} average loss = {}".format(it, avg_cost / count))


# ############################## Main program ################################
# The main program loads the dataset, builds the network using state_ae,
# trains the model and tests it.

def main():
    # Load the dataset
    print("Loading data...")

    f = open('x_train.mnist', 'rb')
    x_train = pickle.load(f)
    f.close()
    f = open('x_test.mnist', 'rb')
    x_test = pickle.load(f)
    f.close()

    trainfunc, predict, encode = state_ae.build_network()
    learn(x_train, trainfunc, x_test)
    check_model(x_test, predict, encode)
    
##############################################################
# Test the model and plot the images

def check_model(x_test, predict, encode):

    encoded_imgs = encode(x_test[:, :])
    decoded_imgs = predict(x_test[:, :])

    n = 10  # how many digits we will display
    plt.figure(figsize=(40, 8))
    for i in range(n):
        # display original
        ax = plt.subplot(3, n, i + 1)
        plt.imshow(x_test[i].reshape(28, 28))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        if (i == n / 2):
            ax.set_title("Original images")

        # display reconstruction
        ax = plt.subplot(3, n, i + n + 1)
        plt.imshow(decoded_imgs[i].reshape(28, 28))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        if (i == n / 2):
            ax.set_title("Reconstructed images")

        # display encodings
        ax = plt.subplot(3, n, i + 2*n + 1)
        plt.imshow(encoded_imgs[i].reshape(4, 8))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        if (i == n / 2):
            ax.set_title("Encoded images")

    plt.show()
    

if __name__ == '__main__':
    main()
