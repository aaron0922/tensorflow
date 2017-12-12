import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np
import tensorflow as tf

#https://ml-with-tensorflow.info/2017/03/11/clustering/amp/
#https://github.com/sentimos/kmeans/blob/master/KMeans.ipynb

def ScatterPlot(X, Y, assignments=None, centers=None):
  if assignments is None:
    assignments = [0] * len(X)
  fig = plt.figure(figsize=(14,8))
  cmap = ListedColormap(['red', 'green', 'blue', 'magenta'])
  plt.scatter(X, Y, c=assignments, cmap=cmap)
  if centers is not None:
    plt.scatter(centers[:, 0], centers[:, 1], c=range(len(centers)), 
                marker='+', s=400, cmap=cmap)  
  plt.xlabel('Height (in)')
  plt.ylabel('Weight (lbs)')
  plt.show()

def input_fn():
  return tf.constant(hw_frame.as_matrix(), tf.float32, hw_frame.shape), None


hw_frame = pd.read_csv(
  'D:/workspace/python/tensorflow/examples/data/human_hw.csv',
  delim_whitespace=True,
  header=1, names=['Index', 'Height', 'Weight'])
'''
hw_frame = pd.read_csv(
  'D:/workspace/python/tensorflow/examples/data/hw-data.txt',
  delim_whitespace=True,
  header=None, names=['Index', 'Height', 'Weight'])
'''  
hw_frame.drop('Index', 1, inplace=True)
print(hw_frame.head(5))
#ScatterPlot(hw_frame.Height, hw_frame.Weight)

tf.logging.set_verbosity(tf.logging.ERROR)
print(111)
kmeans = tf.contrib.learn.KMeansClustering(num_clusters=4, relative_tolerance=0.0001)
_ = kmeans.fit(input_fn=input_fn)
print(222)
clusters = kmeans.clusters()
assignments = list(kmeans.predict_cluster_idx(input_fn=input_fn))
print(333)
ScatterPlot(hw_frame.Height, hw_frame.Weight, assignments, clusters)

print('END')