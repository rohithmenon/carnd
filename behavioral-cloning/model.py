import collections
import csv
import cv2
import numpy as np
import os
import sklearn
import tensorflow
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Dropout
from keras.layers.convolutional import Convolution2D, Cropping2D
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

Correction = collections.namedtuple('Correction', 'center left right')

samples = []
n_augmented_samples = 0
train_test_split_ratio = 0.1
# Read driving log given the directory and create samples.
def add_to_samples(dir_path, center_correction=None, left_correction=None, right_correction=None):
    global n_augmented_samples
    with open('{}/driving_log.csv'.format(dir_path)) as csvfile:
        reader = csv.reader(csvfile)
        for line in reader:
            samples.append((line, dir_path, Correction(center_correction, left_correction, right_correction)))
            # Two samples will be added for center, right and left images
            n_augmented_samples += 2
            n_augmented_samples += 2 if left_correction else 0
            n_augmented_samples += 2 if right_correction else 0

# Add samples from center drive
add_to_samples('./track1_center', left_correction=0.2, right_correction=-0.2)
# Add samples from left side drive
add_to_samples('./track1_left', center_correction=0.3)
# Add samples from right side drive
add_to_samples('./track1_right', center_correction=-0.3)

# Split train/validation samples
train_samples, validation_samples = train_test_split(samples, test_size=train_test_split_ratio)

# Expands samples with augmented data.
# 1. Corrections
# 2. Flips
def augment(samples, batch_size=32):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]
            images = []
            angles = []
            for batch_sample_tuple in batch_samples:
                batch_sample, dir_path, correction = batch_sample_tuple
                center_image = cv2.imread('{}/IMG/{}'.format(dir_path, batch_sample[0].split('/')[-1]))
                center_angle = float(batch_sample[3]) + correction.center if correction.center else 0.0
                images.append(center_image)
                angles.append(center_angle)
                images.append(np.flip(center_image, 1))
                angles.append(-center_angle)

                if correction.left:
                    left_image = cv2.imread('{}/IMG/{}'.format(dir_path, batch_sample[1].split('/')[-1]))
                    left_angle = center_angle + correction.left
                    images.append(left_image)
                    angles.append(left_angle)
                    images.append(np.flip(left_image, 1))
                    angles.append(-left_angle)

                if correction.right:
                    right_image = cv2.imread('{}/IMG/{}'.format(dir_path, batch_sample[2].split('/')[-1]))
                    right_angle = center_angle + correction.right
                    images.append(right_image)
                    angles.append(right_angle)
                    images.append(np.flip(right_image, 1))
                    angles.append(-right_angle)

            yield shuffle(images, angles)

# Generate batches of images with angles from augmented generator
def generator(samples, batch_size=32):
    batch_images = []
    batch_angles = []
    for images, angles in augment(samples, batch_size):
        for image, angle in zip(images, angles):
            batch_images.append(image)
            batch_angles.append(angle)
            # If after adding images, we hit the batch_size, return
            if len(batch_images) == batch_size:
                X_train = np.array(batch_images)
                y_train = np.array(batch_angles)
                batch_images = []
                batch_angles = []
                yield (X_train, y_train)

# Compile and train the model using the generator function
train_generator = generator(shuffle(train_samples), batch_size=32)
validation_generator = generator(shuffle(validation_samples), batch_size=32)

ch, row, col = 3, 160, 320 # Image dimensions

# Nvidia end-to-end self-driving network
model = Sequential()
# Preprocess incoming data, centered around zero with small standard deviation
model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=(row, col, ch)))
# Crop top 60 pixels and bottom 20 pixels to remove non track parts of the image
model.add(Cropping2D(cropping=((60, 20), (0, 0))))
model.add(Convolution2D(24, 5, 5, subsample=2, activation='relu'))
model.add(Convolution2D(36, 5, 5, subsample=2, activation='relu'))
model.add(Convolution2D(48, 5, 5, subsample=2, activation='relu'))
model.add(Convolution2D(64, 3, 3, activation='relu'))
model.add(Convolution2D(64, 3, 3, activation='relu'))
model.add(Flatten())
model.add(Dense(100, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(50, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(10, activation='relu'))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
model.fit_generator(train_generator, samples_per_epoch= \
            n_augmented_samples * (1.0 - train_test_split_ratio), \
            validation_data=validation_generator, \
            nb_val_samples=len(validation_samples), nb_epoch=3)
model.save('model.h5')
