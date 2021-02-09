import tensorflow as tf
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

data = load_breast_cancer() # sklearn.utils.Bunch

print('Keys', data.keys()) # the keys of the object of type sklearn.utils.Bunch
print('Shape of the data', data.data.shape) # (178, 13) means 178 lines, 13 features
print('Shape of the target', data.target.shape) # (178,)
print('Target names', data.target_names) # there are 3 classes
print('Features names', data.feature_names) # the 13 features' names

# dividing the data
X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.33)
N, D = X_train.shape
print('N', N, 'D', D)

# scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
# once fit_transform has been called once
# you can then call transform as it has internally kept the values needed to scale
X_test = scaler.transform(X_test)

# creating the model
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Input(shape=(D,)))
model.add(tf.keras.layers.Dense(1, activation='sigmoid'))

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# train the model
r = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=100)

print('Train score:', model.evaluate(X_train, y_train)) # accuracy: 0.9790
print('Test score:', model.evaluate(X_test, y_test))    # accuracy: 0.9628

plt.plot(r.history['loss'], label='loss')
plt.plot(r.history['val_loss'], label='val_loss')
plt.legend()
plt.show()