import imghdr
import typing
from argon2 import _typing
from matplotlib import pyplot as plt
import tensorflow as tf
import os


import cv2
import numpy as np
# from tensorflow.keras.models import Sequential --> bug in pycharm
from tensorflow_estimator.python.estimator.api._v2 import estimator as estimator
import keras.api._v2.keras as keras
from keras.api._v2.keras import losses
from keras.api._v2.keras import metrics
from keras.api._v2.keras import optimizers
from keras.api._v2.keras import initializers
from keras.api._v2.keras import Sequential
from keras.api._v2.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout,BatchNormalization
from keras.api._v2.keras.metrics import Precision, Recall, BinaryAccuracy
from keras.api._v2.keras.models import load_model
from keras.api._v2.keras.callbacks import LearningRateScheduler,TensorBoard, ModelCheckpoint
from keras.api._v2.keras.optimizers import Adam

data_dir = 'Data'
image_exts = ['png','jpg']
#print(os.listdir(data_dir))# return every folder of the specidifed folder
#print(os.listdir(os.path.join(data_dir,'Normal'))) # return every photo of the specified folder
# in case there is a dodgy photo, we gonna eliminate it ^^
"""
for image_class in os.listdir(data_dir):
    class_path = os.path.join(data_dir,image_class)
    #check if it's a directory - to avoid NotADirectoryError
    if os.path.isdir(class_path):
        for image in os.listdir(os.path.join(data_dir,image_class)):
            image_path = os.path.join(data_dir,image_class,image)
            try:
                img = cv2.imread(image_path)
                tip = imghdr.what(image_path)
                if tip not in image_exts:
                    print('image not in ext list {}'.format(image_path))
                    os.remove(image_path)
            except Exception as e:
                print('Issue with image {}'.format(image_path))
"""
# how to load Data ??
# it allows us to build data pipeline.Rather than loading everything into memory
#it actually allows us to build a data pipeline which one allows us to scale out
#to much larger datasets . More repetable
tf.data.Dataset
#in order to use it
data_set = tf.keras.utils.image_dataset_from_directory(data_dir)
#it allows us to access the generator from data pipeline
data_iterator = data_set.as_numpy_iterator()
batch = data_iterator.next()
#print(len(batch)) -- it is going to print out 2. Which means  there is 2 parts to this dataset;
#ther is the images from our directory loaded into memory as a set of numpy arrays and their labels
#print(batch[0].shape)#-- images represented as numpy arrays
#print(batch[1]) #-> print out flags of 2 or 1 or 0. They represents the labels( meningioma, glioma, pituitary tumor)
# in order to check which labels is assigned to which type of images
"""
fig, ax = plt.subplots(ncols=4, figsize=(20, 20))
for idx, img in enumerate(batch[0][:4]):
    ax[idx].imshow(img.astype(int))
    ax[idx].title.set_text(batch[1][idx])

plt.show() # correct
"""

## how to pre-process the data
# in order to optimize our program, we want rgb values smallest as possible
#we tend to pre-process by scaling the images values to between 0 to 1
# instead of rgb of three values to around 0 to 255. This helps our deep learning model to generalize
# faster and produces better result. We are also going to split up our data
#into training, testing and validation partition to ensure that we don't overfit.
scaled = batch[0] / 255
#print(scaled.max()) -> print 1
data_set = data_set.map(lambda x,y: (x/255, tf.one_hot(y, depth=4))) # data_set.map allows us to perform that transformation in pipeline
## one-hot encode labels for multi-class classification
# x represent our images, y is our labels
#print(data_set.as_numpy_iterator().next()[0]) #dat scaled(0 to 1)
# how to split our data into training and testing partition
#print(len(data_set)) #--> 678 batches -> every batch has 32 images
train_size = int(len(data_set)*0.7)
# training data is actually what is used to train our deep learning model
validation_size = int(len(data_set)* 0.2)+1
# validation data is actually what is used to evaluate our model while we are training
test_size = int(len(data_set)* 0.1) + 1
#print(len(data_set)) --> 96
#print(train_size + validation_size +test_size) #--> 678
# 3 different partition
train = data_set.take(train_size)
validation = data_set.skip(train_size).take(validation_size)
test = data_set.skip(train_size + validation_size).take(test_size)# everythin left over
#print(validation_size) #--> 136
#print(len(validation)) #--> 136
#print(len(train))# 474 # our data should already been shuffled
#print(train_size)# 474
# BUILD THE DEEP LEARNING MODEL
#1 - Building deep neuron network, 2 - train these deep learning model, 3- plot performance
#scans over the image and extract relevant information inside this image to make output classification, pixels by pixels

#Define a simple learning rate scheduler
def lr_scheduler(epoch, lr):
    if epoch % 10 == 0 and epoch > 0:
        lr = lr * 0.9
    return lr

model = Sequential()

# Add layers to the model
model.add(Conv2D(16, (3, 3), 1, activation='relu', input_shape=(256, 256, 3)))
model.add(BatchNormalization()) #Batch Normalization helps stabilize and accelerate the training process by normalizing the input to each layer.
model.add(MaxPooling2D()) # Max pooling is typically used to reduce the spatial resolution while retaining the most salient features.
model.add(Dropout(0.25)) # reduce overfitting

model.add(Conv2D(32, (3, 3), 1, activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D())
model.add(Dropout(0.25))

model.add(Conv2D(16, (3, 3), strides=1, activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D())
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(4, activation='softmax'))
#model.add(Dense(3, activation='softmax'))
# softmax is used for multi-class classification, and it provides probabilities for each class
#optimasirs
model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])
lr_callback = LearningRateScheduler(lr_scheduler)

model.summary()
"""
###
# TRAIN
#Tensorboard will store the logs and information about the training process in 'logs'
logdir = 'log1'
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir)
checkpoint_callback = ModelCheckpoint('model_weights.h5', save_best_only=True)
history = model.fit(train,epochs=20, validation_data=validation,callbacks=[tensorboard_callback,checkpoint_callback])
model.load_weights('model_weights.h5')
#loss vizualisation

fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 4))

axes[0].plot(history.history['loss'], color='teal', label='train_loss')
axes[0].plot(history.history['val_loss'], color='orange', label='val_loss')
axes[0].set_title('Loss', fontsize=20)
axes[0].legend(loc="upper left")

# visualize Accuracy
axes[1].plot(history.history['accuracy'], color='teal', label='train_accuracy')
axes[1].plot(history.history['val_accuracy'], color='orange', label='val_accuracy')
axes[1].set_title('Accuracy', fontsize=20)
axes[1].legend(loc="upper left")

plt.show()
"""
#########
# Initialize lists to collect true labels and predicted labels
true_labels = []
predicted_labels = []
loaded_model = load_model('model_weights.h5')
#Evaluate performance of our model with partition data_set that he hasn't seen yet
precision = Precision()
recall = Recall()
accuracy = BinaryAccuracy()
for batch in test.as_numpy_iterator():
    X, y = batch
    yhat = loaded_model.predict(X) #shape of our yhat will be "(batch_size, num_classes), where batch_size is the number of
    #samples in the batch. The element of the yhat[i,j] will represent the predicted probability of the j-th class
    #for the i-th sample in the batch.
    true_labels.extend(np.argmax(y, axis=1))
    predicted_labels.extend(np.argmax(yhat, axis=1))

    precision.update_state(y, yhat)
    recall.update_state(y,yhat)
    accuracy.update_state(y,yhat)
print(f'Precision:{precision.result().numpy()}, Recall:{recall.result().numpy()}, Accuracy:{accuracy.result().numpy()}')

class_labels = ["glioma","meningioma","normal","pituitary"]
num_classes = len(class_labels)
conf_matrix = np.zeros((num_classes, num_classes), dtype=int)

for true_label, predicted_label in zip(true_labels, predicted_labels):
    conf_matrix[true_label, predicted_label] += 1

plt.imshow(conf_matrix, interpolation='nearest', cmap=plt.cm.Blues)

plt.colorbar()

tick_marks = np.arange(num_classes)
plt.xticks(tick_marks, class_labels, rotation=45)
plt.yticks(tick_marks, class_labels)

for i in range(num_classes):
    for j in range(num_classes):
        plt.text(j, i, str(conf_matrix[i, j]), ha="center", va="center", color="w")

plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.title('Confusion Matrix')

plt.show()
