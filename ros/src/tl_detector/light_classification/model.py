import pickle
import tensorflow as tf
import numpy as np
from keras.models import Model, Sequential
from keras.callbacks import ModelCheckpoint, Callback
from keras.layers import Input, Dense, Activation, Flatten, Dropout, Lambda
from keras.layers.convolutional import Convolution2D, Cropping2D
from keras.layers.pooling import MaxPooling2D
from keras.optimizers import Adam
from keras.regularizers import l1,l2
from keras import backend as K
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from keras.utils import np_utils
import cv2, random
#import seaborn as sns

flags = tf.app.flags
FLAGS = flags.FLAGS

# command line flags
flags.DEFINE_string('testing', False, "Testing flag") # will show a random processed image if true
flags.DEFINE_integer('epochs', 10, "The number of epochs.")

# some image processing for the files used for training
# to help with generalization - images are darkened and blurred
def process_image(image, training=True):
    image_br = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image_br

def load_data(testing=False):

    import csv
    path = 'data/'
    car_images = []
    light = []

    file_row_array = []
    lights = np.array([])
    with open('data/log.csv', 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            light = float(row[1].strip())
            file_row_array.append([row[0].strip(), row[1].strip()])
            lights = np.append(lights, light)

    return file_row_array

# generator that loads data in batches to avoid memory problems
# this is also where the data is augmented
# if the training parameter is true the images are also blurred and darkened
# in process_image(..)
def generator(samples, batch_size=32):
    num_samples = len(samples)

    while 1: # Loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            car_images = []
            light_values = []
            for batch_sample in batch_samples:
                name = batch_sample[0]
                img_center = process_image(cv2.imread(name), True)
              	car_images.extend([img_center])
		if (int(batch_sample[1]) == 0): # red
		  light = [0, 1]
		else:
		  light = [1, 0]
  		light_values.extend([light])

            X_train = np.array(car_images)
            y_train = np.array(light_values)

            yield shuffle(X_train, y_train)


def main(_):
    testing = FLAGS.testing
    row_data = load_data(testing)
    # split data
    train_samples, validation_samples = train_test_split(row_data, train_size=0.8)

    # compile and train the model using the generator function
    train_generator = generator(train_samples, batch_size=32)
    validation_generator = generator(validation_samples, batch_size=32)

    model = Sequential()
    model.add(Cropping2D(cropping=((40,15), (0,0)), input_shape=(600,800,3)))

    # Visualize the cropping for testing purposes
    if testing:
        layer_output = K.function([model.layers[0].input],
                                      [model.layers[0].output])
        input_image = cv2.imread(train_samples[0][0])
        input_image = np.uint8(process_image(input_image))
	cropped_image = np.uint8(layer_output([input_image[None,...]])[0][0,...])
        plt.figure()
        plt.imshow(input_image)
	plt.figure()
	plt.imshow(cropped_image)
        plt.show()

    # construct the model - l2 regularization is used to help generalize
    model.add(Lambda(lambda x: (x / 255.0) - 0.5))
    model.add(Convolution2D(2, 2, 2, W_regularizer=l1(0.000001)))
    model.add(MaxPooling2D((2, 2)))
    model.add(Activation('elu'))
    model.add(Convolution2D(4, 2, 2, W_regularizer=l1(0.000001)))
    model.add(MaxPooling2D((2, 2)))
    model.add(Activation('elu'))
    model.add(Flatten())
    model.add(Dense(32, name="dense2"))
    model.add(Activation('elu'))
    model.add(Dense(2, activation='softmax', name="dense1"))

    if not testing:
        model.compile(optimizer=Adam(lr=0.0001), loss="categorical_crossentropy")
        checkpoint = ModelCheckpoint('model{epoch:02d}.h5')
        history_object = model.fit_generator(generator = train_generator, steps_per_epoch = len(train_samples), validation_data = validation_generator, validation_steps=len(validation_samples), nb_epoch=FLAGS.epochs, verbose=1, callbacks=[checkpoint])
        model.save('model.h5')

        # performance visualization
        print(history_object.history.keys())

        ### plot the training and validation loss for each epoch
        plt.figure()
        plt.plot(history_object.history['loss'])
        plt.plot(history_object.history['val_loss'])
        plt.title('model mean squared error loss')
        plt.ylabel('mean squared error loss')
        plt.xlabel('epoch')
        plt.legend(['training set', 'validation set'], loc='upper right')
        plt.show()


# parses flags and calls the `main` function above
if __name__ == '__main__':
    tf.app.run()
