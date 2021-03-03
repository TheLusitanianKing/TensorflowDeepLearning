import tensorflow as tf
from sklearn.utils import shuffle
import numpy as np
import pandas as pnd
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Input, Dense, Embedding, Flatten, Concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam, SGD

df = pnd.read_csv('ml-25m/ratings.csv')
print(df.head())

# we can't trust the userId and movieId to be numbered 0...N-1
# so let's use our own IDs
df.userId = pnd.Categorical(df.userId)
df['new_user_id'] = df.userId.cat.codes

df.movieId = pnd.Categorical(df.movieId)
df['new_movie_id'] = df.movieId.cat.codes

# get the user IDs, movie IDs and ratings as separate arrays
user_ids = df['new_user_id'].values
movie_ids = df['new_movie_id'].values
ratings = df['rating'].values

# get the number of users and number of movies
N = len(set(user_ids))
M = len(set(movie_ids))

# set embedding dimensions
K = 10 # hyperparameter

# make the neural network
u = Input(shape = (1,)) # user input
m = Input(shape = (1,)) # movie input

# user embedding
u_emb = Embedding(N, K)(u) # output is (num_samples, 1, K)

# movie embedding
m_emb = Embedding(M, K)(m) # output is (num_samples, 1, K)

# flatten both embeddings
u_emb = Flatten()(u_emb) # now both are (num_samples, K)
m_emb = Flatten()(m_emb)

# concatenate user-movie embeddings into a feature vector x
x = Concatenate()([u_emb, m_emb]) # now it is (num_samples, 2K)

# now that we have a feature vector, it is just a regular ANN
x = Dense(1024, activation = 'relu')(x)
# x = Dense(400, activation='relu')(x) # instead a deep layer, you can try many not-so-deep layers
# x = Dense(400, activation='relu')(x)
x = Dense(1)(x)

# build the model and compile
model = Model(inputs = [u, m], outputs = x)
model.compile(
    loss = 'mse',
    optimizer = SGD(lr = 0.08, momentum = 0.9)
)

# split the data
user_ids, movie_ids, ratings = shuffle(user_ids, movie_ids, ratings)
Ntrain = int(0.8 * len(ratings))
train_user = user_ids[:Ntrain]
train_movie = movie_ids[:Ntrain]
train_ratings = ratings[:Ntrain]

test_user = user_ids[Ntrain:]
test_movie = movie_ids[Ntrain:]
test_ratings = ratings[Ntrain:]

# center the ratings
avg_rating = train_ratings.mean()
train_ratings = train_ratings - avg_rating
test_ratings = test_ratings - avg_rating

r = model.fit(
    x = [train_user, train_movie],
    y = train_ratings,
    epochs = 25,
    batch_size = 1024,
    verbose = 2, # goes a little faster when you don't print the progress bar
    validation_data = ([test_user, test_movie], test_ratings)
)

# plot losses
plt.plot(r.history['loss'], label='train loss')
plt.plot(r.history['val_loss'], label='validation loss')
plt.legend()
plt.show()

# by the end, we get 0.6259 val_loss
# is this on par with other approaches?
np.sqrt(0.6259)