import tensorflow as tf
import pandas # data analysis library and manipulation tool

url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data'
path = tf.keras.utils.get_file('auto-mpg.data', url) # read file from external URL
df = pandas.read_csv(path, header=None, delim_whitespace=True) # here we read a local file (previously downloaded or any local file)
print(df.head())