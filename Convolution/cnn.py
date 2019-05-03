# My first convolutional NN

# Importing dependencies
from keras.preprocessing import image
import numpy as np
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.callbacks import ModelCheckpoint, CSVLogger

file_name = 'model_checkpoint.h5'
logs_file = 'logs.txt'
# Create 2 layer convolutional model
classifier = Sequential()
classifier.add(Conv2D(32, (3, 3), input_shape = (64, 64, 3), activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))
classifier.add(Conv2D(32, (3, 3), activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))
classifier.add(Flatten())
classifier.add(Dense(units = 128, activation = 'relu'))
classifier.add(Dense(units = 1, activation = 'sigmoid'))
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Data augmentation
from keras.preprocessing.image import ImageDataGenerator
train_datagen = ImageDataGenerator(rescale = 1./255, shear_range = 0.2, zoom_range = 0.2, horizontal_flip = True)

test_datagen = ImageDataGenerator(rescale = 1./255)

# Resize images so I can process them in my model
training_set = train_datagen.flow_from_directory('dataset/training_set', target_size = (64, 64), batch_size = 32, class_mode = 'binary')
# Do the same for test_set
test_set = test_datagen.flow_from_directory('dataset/test_set', target_size = (64, 64), batch_size = 32, class_mode = 'binary')

# classifier.fit_generator(training_set, steps_per_epoch = 8000, epochs = 10  , validation_data = test_set, validation_steps = 2000)
history = classifier.fit_generator(training_set,
                                    steps_per_epoch=8000,
                                    validation_data=test_set,
                                    validation_steps=2000,
                                    epochs=15,
                                    callbacks=[ModelCheckpoint(file_name,
                                                            monitor='val_acc',
                                                            save_best_only=True,
                                                            mode='max'),
                                                CSVLogger(logs_file,
                                                        append=False,
                                                        separator=';')])

# serialize model to JSON
model_json = classifier.to_json()
with open("model_final.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
classifier.save_weights("model_final.h5")
print("Saved model to disk")



