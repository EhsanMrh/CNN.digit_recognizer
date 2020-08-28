# Import libraries
import numpy as np
import pandas as pd
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D

# import dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# PreProcessing data
from preprocessing import preprocessing_data as ppd
ppd_res = ppd(x_train, y_train, x_test, y_test)
x_train = ppd_res['x_train']
y_train = ppd_res['y_train']
x_test = ppd_res['x_test']
y_test = ppd_res['y_test']
