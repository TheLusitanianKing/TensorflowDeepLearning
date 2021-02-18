import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# make the dataset
N = 1000
X = np.random.random((N, 2)) * 6 - 3 # uniformly distributed between -3 and +3
Y = np.cos(2*X[:,0]) + np.cos(2*X[:,1])

# plot it
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X[:,0], X[:,1], Y)
plt.show()

# build the model
model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(128, input_shape=(2,), activation='relu'),
    tf.keras.layers.Dense(1)
])

# compile and fit
opt = tf.keras.optimizers.Adam(0.01)
model.compile(optimizer=opt, loss='mse')
r = model.fit(X, Y, epochs=100)

# plot the loss
plt.plot(r.history['loss'], label='loss')
plt.show()