import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mt
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import models, layers, Model
from tensorflow.keras.applications import *
from tensorflow.keras.datasets import mnist
from tensorflow.keras.layers import *
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard
import shutil

def clear_tensorboard_logs(logs_dir='logs'):
    try:
        # Delete the entire contents of the logs directory
        shutil.rmtree(logs_dir)
        print(f"TensorBoard logs in '{logs_dir}' cleared successfully.")
    except FileNotFoundError:
        print(f"TensorBoard logs directory '{logs_dir}' not found.")
    except Exception as e:
        print(f"Error clearing TensorBoard logs: {e}")

logs_directory_to_clear = 'logs'
clear_tensorboard_logs(logs_directory_to_clear)


(x_train,y_train),(x_test,y_test)= mnist.load_data()
 
plt.subplot(431)
plt.imshow(x_train[0], cmap=plt.cm.binary)
plt.subplot(432)
plt.imshow(x_train[1], cmap=plt.cm.binary)
plt.subplot(433)
plt.imshow(x_train[2], cmap=plt.cm.binary)
plt.subplot(434)
plt.imshow(x_train[3], cmap=plt.cm.binary)
plt.subplot(435)
plt.imshow(x_train[4], cmap=plt.cm.binary)
plt.subplot(436)
plt.imshow(x_train[5], cmap=plt.cm.binary)


 
x_train=x_train/255
x_test=x_test/255

 
print(x_train.shape)
print(x_test.shape)

 
print("y_train Shape: %s and value: %s" % (y_train.shape, y_train))
print("y_test Shape: %s and value: %s" % (y_test.shape, y_test))



 
ytrain=to_categorical(y_train)
ytest=to_categorical(y_test)



 
# After one hot encoding
print("y_train Shape: %s and value: %s" % (y_train.shape, y_train[0]))
print("y_test Shape: %s and value: %s" % (y_test.shape, y_test[1]))

 
model=models.Sequential()

model.add(layers.Conv2D(32,(3,3),padding='same',input_shape=(28,28,1),activation='relu'))
model.add(layers.MaxPooling2D(pool_size=(2,2)))
# model.add(Dropout(0.5))

model.add(layers.Conv2D(64,(3,3),padding='same', activation='relu'))
# model.add(layers.MaxPooling2D(pool_size=(2,2)))
# model.add(Dropout(0.5))

model.add(layers.Conv2D(128,(3,3),padding='same', activation='relu'))
model.add(layers.MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.5))

model.add(layers.Conv2D(256,(3,3),padding='same', activation='relu'))
model.add(layers.MaxPooling2D(pool_size=(2,2)))
# model.add(Dropout(0.5))

model.add(layers.Conv2D(512,(3,3),padding='same', activation='relu'))
model.add(layers.MaxPooling2D(pool_size=(2,2)))
# model.add(Dropout(0.5))

model.add(layers.Flatten(input_shape=(28,28)))
model.add(layers.Dense(512, activation='relu')) 
model.add(layers.Dense(256, activation='relu')) 
model.add(layers.Dense(10, activation='softmax'))


 
model.summary()

 
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy']) 

 

xtrain2=x_train.reshape(60000,28,28,1)
xtest2=x_test.reshape(10000,28,28,1)

print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)


checkpoint=ModelCheckpoint('mnist_model_1.h5', monitor='val_loss', verbose=1, save_best_only=True)
tensorboard_callback = TensorBoard(log_dir='logs', histogram_freq=1, update_freq='epoch',)
history=model.fit(xtrain2,ytrain,epochs=40,batch_size=1000,verbose=True,validation_data=(xtest2,ytest), callbacks=[checkpoint, tensorboard_callback])

 
test_loss, test_acc = model.evaluate(xtest2, ytest)
print("acc:", test_acc*100,'%')

