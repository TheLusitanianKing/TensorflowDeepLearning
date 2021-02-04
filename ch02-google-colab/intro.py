import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

# get version of any imported module
tensorflow_version = tf.__version__

x = np.linspace(0, 10*np.pi, 1000)
y = np.sin(x)

plt.plot(x, y)
plt.show()