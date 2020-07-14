from data_prep import *
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import regularizers
from tensorflow.keras.callbacks import LambdaCallback
import os
import numpy as np
import tensorflow as tf

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

x_train = np.asarray(x_train, np.float32)
y_train = np.asarray(y_train, np.float32)
x_tf = tf.convert_to_tensor(x_train, np.float32)
y_tf = tf.convert_to_tensor(y_train, np.float32)

# Build an ANN
# The initial learning rate is quite large, when the loss starts oscillating
# I will save the model and run refine_model.py but set the learning rate to
# a smaller value and continue learning.
model = Sequential()
model.add(Dense(10, input_dim=n, activation='sigmoid', kernel_initializer='random_uniform'))
model.add(Dense(10, activation='sigmoid', kernel_initializer='random_uniform'))
model.add(Dense(5, activation='sigmoid', kernel_initializer='random_uniform',))
model.compile(optimizer=Adam(learning_rate=0.1), loss='mean_squared_error', metrics=['accuracy'])

# Train
eval_acc = LambdaCallback(on_epoch_end=lambda batch, logs: print(model.evaluate(x_test, y_test)[1]))
model.fit(x_tf, y_tf, epochs=300, batch_size=32, verbose=2, class_weight=None, callbacks=[eval_acc])

# Evaluate
print(model.evaluate(x_test, y_test)[1])

# Save the model
model.save('model2.h5')