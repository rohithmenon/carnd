import csv
import cv2
import numpy as np
import os
import sklearn
import tensorflow
from keras.models import Sequential
from keras.layers import ELU, Flatten, Dense, Lambda, BatchNormalization, Dropout
from keras.layers.convolutional import Convolution2D, MaxPooling2D, Cropping2D
from keras import backend as K
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from preprocess import preprocess_image

samples = []
# Read driving log given the directory and create samples.
def add_to_samples(dir_path, correction):
    with open('{}/driving_log.csv'.format(dir_path)) as csvfile:
        reader = csv.reader(csvfile)
        for line in reader:
            samples.append((line, dir_path, correction))

# Add samples from center drive
add_to_samples('./c_data', None)
# Add samples from left side drive
add_to_samples('./ls_data', 0.5)
# Add samples from right side drive
add_to_samples('./rs_data', -0.5)
# Add samples from curve track
#add_to_samples('./curve_c_data', None)
# Add samples from curve left side drive
#add_to_samples('./curve_data_ls', 0.5)
# Add samples from curve right side drive
#add_to_samples('./curve_data_rs', -0.5)

# Split train/validation samples
train_samples, validation_samples = train_test_split(samples, test_size=0.1)

def generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]
            images = []
            angles = []
            for batch_sample_tuple in batch_samples:
                batch_sample, dir_path, correction = batch_sample_tuple
                center_image = preprocess_image(cv2.imread('{}/IMG/{}'.format(dir_path, batch_sample[0].split('/')[-1])))
                center_angle = correction if correction else float(batch_sample[3])
                images.append(center_image)
                angles.append(center_angle)
                images.append(np.flip(center_image, 1))
                angles.append(-center_angle)
        
                if not correction:
                    left_image = preprocess_image(cv2.imread('{}/IMG/{}'.format(dir_path, batch_sample[1].split('/')[-1])))
                    left_angle = center_angle + 0.2
                    images.append(left_image)
                    angles.append(left_angle)
                    images.append(np.flip(left_image, 1))
                    angles.append(-left_angle)

                    right_image = preprocess_image(cv2.imread('{}/IMG/{}'.format(dir_path, batch_sample[2].split('/')[-1])))
                    right_angle = center_angle - 0.2
                    images.append(right_image)
                    angles.append(right_angle)
                    images.append(np.flip(right_image, 1))
                    angles.append(-right_angle)

            X_train = np.array(images)
            y_train = np.array(angles)
            yield shuffle(X_train, y_train)

# compile and train the model using the generator function
train_generator = generator(shuffle(train_samples), batch_size=32)
validation_generator = generator(shuffle(validation_samples), batch_size=32)

#ch, row, col = 3, 80, 320  # Trimmed image format
ch, row, col = 3, 160, 320 # Trimmed image format

model = Sequential()
# Preprocess incoming data, centered around zero with small standard deviation
model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=(row, col, ch)))
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
            len(train_samples) / 32, validation_data=validation_generator, \
            nb_val_samples=len(validation_samples), nb_epoch=5)
model.save('model.h5')
