
#libraries 

import keras,os
from keras import optimizers
from keras.models import Sequential
from keras.models import Model
from keras.layers import Dense, Conv2D, MaxPool2D , Flatten, Dropout
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.image import image
import cv2
import numpy as np

#Allowing use a part of the GPU memory
# import tensorflow as tf
# from keras.backend.tensorflow_backend import set_session
# config = tf.ConfigProto()vertical_flip=True, horizontal_flip=True, rotation_range=20
# config.gpu_options.per_process_gpu_memory_fraction = 0.33
# set_session(tf.Session(config=config))


#Labeling
#folders training and testing data

#trdata = ImageDataGenerator(featurewise_center=True, featurewise_std_normalization=True,  horizontal_flip=True)
trdata = ImageDataGenerator(vertical_flip=True, horizontal_flip=True, rotation_range=360, width_shift_range=[-20,20], height_shift_range=[-20,20])
#trdata = ImageDataGenerator()
traindata = trdata.flow_from_directory(directory="dataset3class/training_set/training_set",target_size=(224,224),batch_size=64) #batch size


tsdata = ImageDataGenerator(vertical_flip=True, horizontal_flip=True, rotation_range=360, width_shift_range=[-20,20], height_shift_range=[-20,20])
testdata = tsdata.flow_from_directory(directory="dataset3class/test_set/test_set", target_size=(224,224),batch_size=64)


#architecture
#model and layers


########## FROM SCRATCH  #####################
model = Sequential()

model.add(Conv2D(input_shape=(224,224,3),filters=32,kernel_size=(3,3),padding="same", activation="relu")) 
model.add(MaxPool2D(pool_size=(2,2),strides=(3,3)))

model.add(Conv2D(filters=64,kernel_size=(3,3),padding="same", activation="relu")) 
model.add(MaxPool2D(pool_size=(4,4),strides=(3,3)))

model.add(Conv2D(filters=128, kernel_size=(3,3), padding="same", activation="relu"))
model.add(MaxPool2D(pool_size=(4,4),strides=(3,3)))


#softmax for classification for 3 classes
model.add(Flatten())
model.add(Dense(units=512,activation="relu"))
model.add(Dense(units=3, activation="softmax"))

#compilation with adam optimizier
#and cross entropy loss

from keras.optimizers import Adam

#visualize the model
model.summary()

model_final=model

#COMPILE
kwargs={'decay','lr'}
opt= optimizers.Nadam(learning_rate=0.00020, beta_1=0.9, beta_2=0.899)

model_final.compile(optimizer=opt, loss=keras.losses.categorical_crossentropy, metrics=['accuracy'])
	
model_final.summary()

from keras.callbacks import ModelCheckpoint, EarlyStopping

#save the model with the best validation accuracy
checkpoint = ModelCheckpoint("weights3class.h5", monitor='val_accuracy', verbose=1, save_best_only=False, save_weights_only=False, mode='auto', period=5)

#stop model training if accuracy doesnt change in 40 epochs
early = EarlyStopping(monitor='val_accuracy', min_delta=0, patience=300, verbose=1, mode='auto')
model_final.save_weights("weights3class.h5")

#TRAINIGN #####################################################
#pass data to the model (images) with number of epochs 100
hist = model_final.fit_generator(steps_per_epoch=40,generator=traindata, validation_data= testdata, validation_steps=1,epochs=300,callbacks=[checkpoint,early])

#visualize training and validation accuracy and loss
import matplotlib.pyplot as plt
plt.xlim(0,300)
plt.ylim(0,1)
plt.plot(hist.history["accuracy"])
plt.plot(hist.history['val_accuracy'])
plt.plot(hist.history['loss'])
plt.plot(hist.history['val_loss'])
plt.title("model accuracy")
plt.ylabel("Accuracy")
plt.xlabel("Epoch")
plt.legend(["Accuracy","Validation Accuracy","loss","Validation Loss"])
plt.show()
