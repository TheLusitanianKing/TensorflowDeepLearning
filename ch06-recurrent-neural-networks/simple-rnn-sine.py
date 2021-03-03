import tensorflow as tf
import numpy as np
import pandas as pnd
import matplotlib.pyplot as plt

from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.python.keras.layers.recurrent import SimpleRNN

# make the original data
series = np.sin(0.1 * np.arange(200)) + np.random.randn(200) * 0.1
# the right side of + is basically noise
plt.plot(series)
plt.show()

# build dataset
T = 10
X = []
Y = []
for t in range(len(series) - T):
    x = series[t:t+T]
    X.append(x)
    y = series[t+T]
    Y.append(y)

X = np.array(X).reshape(-1, T, 1) # now the data should be N x T x D
Y = np.array(Y)
N = len(X)
print('X.shape', X.shape, 'Y.shape', Y.shape)

# try autoregressive linear model
i = Input(shape=(T, 1))
x = SimpleRNN(5, activation='relu')(i)
x = Dense(1)(x)
model = Model(i, x)

model.compile(loss='mse', optimizer=Adam(lr=0.1))

# train the RNN
r = model.fit(
    X[:-N//2], Y[:-N//2],
    epochs=80, validation_data=(X[-N//2:], Y[-N//2:])
)

# plot loss per iteration
plt.plot(r.history['loss'], label='loss')
plt.plot(r.history['val_loss'], label='val_loss')
plt.legend()
plt.show()

# "wrong" forecast using true targets
validation_target = Y[-N//2:]
validation_predictions = []

# index of the first validation input
i = -N//2

while len(validation_predictions) < len(validation_target):
    p = model.predict(X[i].reshape(1, -1))[0, 0] # 1x1 array -> scalar
    i += 1
    validation_predictions.append(p)

plt.plot(validation_target, label='forecast target')
plt.plot(validation_predictions, label='forecast predictions')
plt.legend()
plt.show()

# forecast future values (use only-self predictions for making future predictions)
validation_target = Y[-N//2:]
validation_predictions = []

# last train input
last_x = X[-N//2] # 1-D array of length T

while len(validation_predictions) < len(validation_target):
    p = model.predict(last_x.reshape(1, -1))[0, 0] # 1x1 array -> scalar
    validation_predictions.append(p)
    last_x = np.roll(last_x, -1)
    last_x[-1] = p

plt.plot(validation_target, label='forecast target')
plt.plot(validation_predictions, label='forecast predictions')
plt.legend()
plt.show()