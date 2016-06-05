import pickle
from keras.datasets import mnist
import numpy as np
(x_train, _), (x_test, _) = mnist.load_data()

x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))

f = open('x_train.mnist', 'wb')
pickle.dump(x_train, f)
f.close()
f = open('x_test.mnist', 'wb')
pickle.dump(x_test, f)
f.close()
