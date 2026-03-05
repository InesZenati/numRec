#reading and processing data
import numpy as np 

#deep learning libraries
import tensorflow as tf 
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import Dense

#chargement de la bdd
(X_train, Y_train), (X_test, Y_test) = mnist.load_data()
print(X_train.shape, Y_train.shape)
print(X_test.shape, Y_test.shape)

print(X_train[0])

#normalisation 
X_train = X_train/255.0
X_test = X_test/255.0 