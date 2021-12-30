import numpy as np
from keras.utils import np_utils
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Dropout, Conv2D, Flatten, MaxPooling2D

(x_train, y_train) , (x_test , y_test) = tf.keras.datasets.cifar10.load_data()
# print(x_train[0])
# print(x_test[0])

for i in range(150, 155):
    plt.subplot(120+1+i)
    img = x_train[i]
    # plt.imshow(img)     #this will show image
    # plt.show()

x_train = x_train.reshape(x_train.shape[0], 32,32,3)
x_test = x_test.reshape(x_test.shape[0], 32, 32, 3)
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
# converting the scale of pixels 0-255 between 0-1
x_train/= 255
x_test/= 255
n_classes = 10

print("Shape before one-hot encoding :", y_train.shape)
y_train = np_utils.to_categorical(y_train, n_classes)
y_test = np_utils.to_categorical(y_test, n_classes)
print('Shape after One-Hot encoding :', y_train.shape)

model = Sequential()
model.add(Conv2D(50,kernel_size=(3,3), strides=(1,1), padding='same', activation='relu', input_shape=(32,32,3)))

model.add(Conv2D(75, kernel_size=(3,3), strides=(1,1), padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))

model.add(Conv2D(125, kernel_size=(3,3), strides=(1,1), padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))

model.add(Flatten())

#Hidden Layer
model.add(Dense(500, activation='relu'))
model.add(Dropout(0.4))
model.add(Dense(250, activation='relu'))
model.add(Dropout(0.3))

#output layer
model.add(Dense(10,activation='softmax'))
model.compile(loss='categorical_crossentropy',metrics= ['accuracy'], optimizer='adam')
model.fit(x_train, y_train, batch_size=128, epochs=2, validation_data=(x_test, y_test))
# model.predict(x_test, verbose=0)

classes = range(0,10)

names = [
    'airplane',
    'automobile',
    'bird',
    'cat',
    'deer',
    'dog',
    'frog',
    'horse',
    'ship',
    'truck']

class_labels = dict(zip(classes, names))
print(class_labels)
batch = x_test[100:109]
labels = np.argmax(y_test[100:109], axis=1)

predictions = model.predict(batch, verbose = 2)
print(predictions)
# for img in predictions:
    # print(np.sum(img))

class_result = np.argmax(predictions, axis= 1)
print(class_result)

fig, axs = plt.subplots(3,3,figsize=(19,6))
fig.subplots_adjust(hspace=1)
axs = axs.flatten()

for i, img in enumerate(batch):
    for key,values in class_labels.items():
        if class_result[i] == key:
            title = f'Prediction : {class_labels[key]}\nActual : {class_labels[labels[i]]}'
            axs[i].set_title(title)
            axs[i].axes.get_xaxis().set_visible(False)
            axs[i].axes.get_yaxis().set_visible(False)
    axs[i].imshow(img)
plt.show()