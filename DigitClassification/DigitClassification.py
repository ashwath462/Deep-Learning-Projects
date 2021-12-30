import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Dropout, Flatten, MaxPooling2D

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
print(x_train.shape)

fig, axs = plt.subplots(5,5,figsize= (10,10))
plt.gray()
for i, ax in enumerate(axs.flat):
    ax.matshow(x_train[i])
    ax.axis("off")
    ax.set_title(f'Number {y_train[i]}')
plt.show()

x_train = x_train.reshape(x_train.shape[0], 28,28,1)
x_test = x_test.reshape(x_test.shape[0], 28,28,1) # (*,*,1) denotes that i have converted the color to gray scale
input_shape = (28,28,1)
# print(x_train.shape)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

x_train /= 255
x_test /= 255

print('x_train shape', x_train.shape)
print('Number of images in x_train:', x_train.shape[0])
print('Number of images in x_test:', x_test.shape[0])

model = Sequential()
model.add(Conv2D(28, kernel_size= (3,3), input_shape= input_shape))
model.add(MaxPooling2D(pool_size=(2,2)))                #Max pooling is a pooling operation that selects the maximum element from the region of the feature map covered by the filter. Thus, the output after max-pooling layer would be a feature map containing the most prominent features of the previous feature map

model.add(Flatten())   #Flattening involves transforming the entire pooled feature map matrix into a single column which is then fed to the neural network for processing.
model.add(Dense(128, activation=tf.nn.relu)) #DENSE LAYER , A linear operation in which every input is connected to every output by a weight . It connects neurons in one layer to neurons in another layer. It is used to classify images between different category by training.
model.add(Dropout(0.2)) #DROPOUT : A Simple Way to Prevent Neural Networks from Overfitting
model.add(Dense(10, activation=tf.nn.softmax))

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(x = x_train, y = y_train, epochs=2)
model.evaluate(x_test, y_test)
