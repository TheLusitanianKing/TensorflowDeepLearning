import numpy as np
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model
from tensorflow.python.keras.layers.recurrent import SimpleRNN

# things to know and memorize
# N = number of samples
# T = sequence length
# D = number of input features
# M = number of hidden units
# K = number of output units

# make some data
N = 1
T = 10
D = 3
K = 2
X = np.random.randn(N, T, D) # random array of size N x T x D

# make an RNN
M = 5
i = Input(shape=(T, D))
x = SimpleRNN(M)(i)
x = Dense(K)(x)

model = Model(i, x)

# get the output
Yhat = model.predict(X)
print(Yhat) # => 1 sample, 2 output nodes, therefore N x K (1 x 2)

# see if we can replicate this output
# get the weights first
print(model.summary())

# check the shape of the arrays and deduce:
# first is input > hidden
# second is hidden > hidden
# third is the bias term (vector of length M)
Wx, Wh, bh = model.layers[1].get_weights()
Wo, bo = model.layers[2].get_weights()

# RNN
h_last = np.zeros(M) # initial hidden state
x = X[0] # the one and only sample
Yhats = [] # where we store the output

for t in range(T):
    h = np.tanh(x[t].dot(Wx) + h_last.dot(Wh) + bh)
    y = h.dot(Wo) + bo # we only about this value in the last iteration
    Yhats.append(y)

    h_last = h

print(Yhats[-1]) # final output (same as the model.predict above)
