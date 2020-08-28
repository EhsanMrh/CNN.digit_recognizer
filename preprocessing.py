# Preprocess and Reshape the dataset to 4 dimension

import keras

def preprocessing_data(x_train, y_train, x_test, y_test):
    x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
    x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
    
    input_shape = (28, 28, 1)
    
    num_classes = 10
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)
    
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255
    
    return {"x_train": x_train,
            "y_train": y_train,
            "x_test": x_test,
            "y_test": y_test}