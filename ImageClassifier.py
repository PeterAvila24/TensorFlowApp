import tensorflow as tf
import numpy as np
from tensorflow import keras

import matplotlib.pyplot as plt

#load a predefined dataset
fashion_mnist = keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()


# Normalize images to values between 0 and 1 
train_images = train_images / 255.0
test_images = test_images / 255.0


model = keras.Sequential([
    # input is a 28x28 image ("Flatten" flattens the 28x28 into a single 784x1 input layer)
    keras.layers.Flatten(input_shape = (28, 28)),

    # Hidden layer is 128 deep 
    keras.layers.Dense(units = 128, activation = tf.nn.relu),

    # output is 0-10. Retun max
    keras.layers.Dense(10, activation =  tf.nn.softmax)
])

#complie model
model.compile(optimizer = tf.optimizers.Adam(), loss = 'sparse_categorical_crossentropy', metrics = ['accuracy'])

model.fit(train_images, train_labels, epochs = 5)

test_loss, test_acc = model.evaluate(test_images, test_labels)
print(f'\nTest Accuracy: {test_acc}')


plt.imshow(train_images[0], cmap = 'gray', vmin = 0, vmax = 255)
plt.show()

print(f'Actual label: {test_labels[0]}')

predictions = model.predict(test_images)

print(predictions[0])

print(f'Predicted label: {np.argmax(predictions[0])}')



print("Code Complete")
