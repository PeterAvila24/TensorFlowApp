import tensorflow as tf 
from tensorflow import keras
import numpy as np



model = keras.Sequential([keras.layers.Dense(units = 1, input_shape = [1])])

model.complie(optimizer = 'sgd', loss = 'mean_squared_error')


x_inputs = np.array([-1.0, 0.0, 1.0, 2.0, 3.0, 4.0], dtype = float)
y_outputs = np.array([-3.0, -1.0, 1.0, 3.0, 5.0, 7.0], dtype = float)

model.fit(x_inputs, y_outputs, epochs = 500)


print(model.predict([100.0]))

                         

