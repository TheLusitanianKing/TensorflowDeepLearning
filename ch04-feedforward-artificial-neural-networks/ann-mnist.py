import tensorflow as tf
import matplotlib.pyplot as plt

# load in the data
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# scaling, values are between 0 and 255, will be between 0 and 1
x_train, x_test = x_train / 255.0, x_test / 255.0
print('x_train.shape', x_train.shape) # 60000 x 28 x 28

# build the model
model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(10, activation='softmax')
])

# compile the model
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# train the model
r = model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=10)

# plot loss per iteration
plt.plot(r.history['loss'], label='loss')
plt.plot(r.history['val_loss'], label='val_loss')
plt.legend()
plt.show()

# plot accuracy
plt.plot(r.history['accuracy'], label='accuracy')
plt.plot(r.history['val_accuracy'], label='val_accuracy')
plt.legend()
plt.show()

# evaluate the model
print(model.evaluate(x_test, y_test))