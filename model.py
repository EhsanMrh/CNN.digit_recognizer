# Import libraries
import keras
from keras.datasets import mnist


# import dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# PreProcessing data
from preprocessing import preprocessing_data as ppd
ppd_res = ppd(x_train, y_train, x_test, y_test)
x_train = ppd_res['x_train']
y_train = ppd_res['y_train']
x_test = ppd_res['x_test']
y_test = ppd_res['y_test']

x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)

    
num_classes = 10
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)
    
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255

# CNN Model
from cnn_model import f_cnn_model
model = f_cnn_model()

batch_size = 128
epochs = 10
model.fit(x_train, 
          y_train, 
          batch_size = batch_size,
          epochs = epochs,
          verbose=1,
          validation_data = (x_test, y_test))
print("The model has successfully trained :)")

model.save('pickle_model.h5')
print('Saving the model as pickle_model.h5')