import csv
import cv2
import numpy as np
import os
import sklearn
import tensorflow
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, BatchNormalization, LeakyReLU
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from preprocess import preprocess_image

samples = []
with open('./data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        samples.append(line)

train_samples, validation_samples = train_test_split(samples, test_size=0.2)

def generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:
                name = './data/IMG/{}'.format(batch_sample[0].split('/')[-1])
                center_image = preprocess_image(cv2.imread(name))
                center_angle = float(batch_sample[3])
                images.append(center_image)
                angles.append(center_angle)

            # trim image to only see section with road
            X_train = np.array(images)
            y_train = np.array(angles)
            yield sklearn.utils.shuffle(X_train, y_train)

# compile and train the model using the generator function
train_generator = generator(train_samples, batch_size=32)
validation_generator = generator(validation_samples, batch_size=32)

#ch, row, col = 3, 80, 320  # Trimmed image format
ch, row, col = 1, 160, 320  # Trimmed image format

model = Sequential()
# Preprocess incoming data, centered around zero with small standard deviation
model.add(Convolution2D(6, 5, 5, input_shape=(row, col, ch)))
model.add(BatchNormalization(axis=1))
model.add(LeakyReLU())
model.add(MaxPooling2D())
model.add(Convolution2D(6, 5, 5))
model.add(BatchNormalization(axis=1))
model.add(LeakyReLU())
model.add(MaxPooling2D())
model.add(Flatten())
model.add(Dense(120))
model.add(Dense(84))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
model.fit_generator(train_generator, samples_per_epoch= \
            len(train_samples), validation_data=validation_generator, \
            nb_val_samples=len(validation_samples), nb_epoch=1)
model.save('model.h5')
