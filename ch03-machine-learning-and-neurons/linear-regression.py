import tensorflow as tf
import pandas as pnd
import numpy as np
import matplotlib.pyplot as plt

# numpy.ndarray of shape (162, 2) -> 162 lines, 2 values
data = pnd.read_csv('../datasets/moore.csv', header=None).values

# reshaping from a 1-D array to a 2-D array of shape (N, D)
# where N = 162, D = 1
X = data[:,0].reshape(-1, 1)
Y = data[:,1]
plt.scatter(X, Y)
plt.title('Initial raw data')
plt.show()

# since we want a linear model, we'll take the log of Y
Y = np.log(Y)
plt.scatter(X, Y)
plt.title('After applying log to Y')
plt.show()

# we can center X too
X = X - X.mean()
plt.scatter(X, Y)
plt.title('After centering X')
plt.show()

# creating Tensorflow model
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Input(shape=(1,)))
model.add(tf.keras.layers.Dense(1))

model.compile(optimizer=tf.keras.optimizers.SGD(0.001, 0.9), loss='mse')

# learning rate scheduler
def schedule(epoch, lr):
    if epoch >= 50:
        return 0.0001
    return 0.001
scheduler = tf.keras.callbacks.LearningRateScheduler(schedule)

# train the model
r = model.fit(X, Y, epochs=200, callbacks=[scheduler])

# plot the loss
plt.plot(r.history['loss'], label='loss')
plt.show()

# get the slope of the line
a = model.layers[0].get_weights()[0][0,0]
print('Time to double:', np.log(2) / a) # should be near 2 to prove Moore's law

# making predictions
Yhat = model.predict(X).flatten()
plt.scatter(X, Y)
plt.plot(X, Yhat) # to make sure the line fits the data
plt.show()

# manual calculation
# get the weighs
w, b = model.layers[0].get_weights()
X = X.reshape(-1, 1)
Yhat2 = (X.dot(w) + b).flatten()

# comparing manual calculation
# do not use == with floating points
print(np.allclose(Yhat, Yhat2)) # -> True

# saving model
model.save('linearregression.h5')