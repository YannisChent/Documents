# -*- coding: utf-8 -*-
"""
Created on Tue Jan 30 16:16:01 2018

@author: Yannis
"""
import tensorflow as tf

# 只使用 30% 的 GPU 記憶體
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.3)
sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

# 設定 Keras 使用的 TensorFlow Session
tf.keras.backend.set_session(sess)

import numpy as np
import time
from keras.utils import np_utils

print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()), "Processing Start........")
np.random.seed(10)

from keras.datasets import mnist
(X_train_image, y_train_label), (X_test_image, y_test_label) = mnist.load_data()
print("training data size=", len(X_train_image))
print("testing data size=", len(X_test_image))
print("training data image shape=", X_train_image.shape)
print("training data label shape=", y_train_label.shape)
print("testing data image shape=", X_test_image.shape)
print("testing data label shape=", y_test_label.shape)


##### image preprocessing #####
## x_Train = X_train_image.reshape(60000,784).astype("float32")
## x_Test = X_test_image.reshape(10000,784).astype("float32")
## print("x_Train label reshape=", x_Train.shape)
## print("x_Test label reshape=", x_Test.shape)

x_Train4D = X_train_image.reshape(60000,28,28,1).astype("float32")
x_Test4D = X_test_image.reshape(10000,28,28,1).astype("float32")
print("x_Train4D label reshape=", x_Train4D.shape)
print("x_Test4D label reshape=", x_Test4D.shape)
x_Train4D_normalize = x_Train4D / 255
x_Test4D_normalize = x_Test4D / 255


##### label preprocessing #####
y_TrainOnehot = np_utils.to_categorical(y_train_label)
y_TestOnehot = np_utils.to_categorical(y_test_label)

##### Build training model #####
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from keras.callbacks import EarlyStopping

model = Sequential()
model.add(Conv2D(filters=32, kernel_size=(5,5), padding="same", input_shape=(28,28,1), activation="relu"))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.4))
model.add(Conv2D(filters=64, kernel_size=(5,5), padding="same", input_shape=(28,28,1), activation="relu"))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.4))
model.add(Flatten())
model.add(Dense(1024, activation="relu"))
model.add(Dropout(0.5))
model.add(Dense(10, activation="softmax"))

print(model.summary())

##### start to train model #####
model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["acc"])
early_stop = EarlyStopping(monitor='val_loss', min_delta=0, patience=3, verbose=0, mode='min')
train_history=model.fit(x=x_Train4D_normalize, y=y_TrainOnehot, callbacks=[early_stop], validation_split=0.2, epochs=20, batch_size=200, verbose=2)


##### accuracy and loss #####
import matplotlib.pyplot as plt
def show_train_history(train_history, train, validation):
    plt.plot(train_history.history[train])
    plt.plot(train_history.history[validation])
    plt.title("Train History")
    plt.ylabel("train")
    plt.xlabel("Epoch")
    plt.legend(["train", "validation"], loc="upper left")
    plt.show()
show_train_history(train_history, "acc", "val_acc")
show_train_history(train_history, "loss", "val_loss")
print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()), "Processing End........")

##### Evaluate Model #####
scores = model.evaluate(x_Test4D_normalize, y_TestOnehot)
print("\nScores=",scores[1])

##### Save the Model #####
model.save(time.strftime("%Y%m%d%H%M%S", time.localtime()) + "_Best_Cnn_Mnist_model_" + str(scores[1]).split(".")[1] + ".h5")

##### Predict Test Data #####
prediction=model.predict_classes(x_Test4D_normalize)

##### Confusion matrix #####
import pandas as pd
ConfusionMatrix = pd.crosstab(y_test_label, prediction, rownames=["lables"], colnames=["predict"])
print(ConfusionMatrix)