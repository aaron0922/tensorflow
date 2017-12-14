import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np
import tensorflow as tf
import time
#https://ml-with-tensorflow.info/2017/03/11/clustering/amp/
#https://github.com/sentimos/kmeans/blob/master/KMeans.ipynb

def ScatterPlot(X, Y, assignments=None, centers=None):
  if assignments is None:
    assignments = [0] * len(X)
  fig = plt.figure(figsize=(14,8))
  #cmap = ListedColormap(['red', 'green', 'blue', 'magenta'])
  cmap = plt.cm.get_cmap('RdYlBu')
  plt.scatter(X, Y, c=assignments, cmap=cmap)
  if centers is not None:
    plt.scatter(centers[:, 0], centers[:, 1], c=range(len(centers)), 
                marker='+', s=400, cmap=cmap)  
  plt.xlabel('extcredits2')
  plt.ylabel('posts')
  plt.savefig("D:/workspace/python/tensorflow/examples/data/test.png", dpi=120)
  plt.show()


def input_fn():
  return tf.constant(hw_frame.as_matrix(), tf.float32, hw_frame.shape), None
a = time.time()


hw_frame = pd.read_csv(
  #'D:/workspace/python/tensorflow/examples/data/20171213-protw_ thread_vr.csv',
  #'D:/workspace/python/tensorflow/examples/data/20171213-protw_ thread_v0_r0_all.csv',
  'D:/workspace/python/tensorflow/examples/data/20171213-protw_ member_credit_post_all.csv',
  sep=',',
  header=1, names=['uid', 'extcredits2', 'posts'])


#hw_frame.drop('Index', 1, inplace=True)
hw_frame.drop('uid', 1, inplace=True)
print(hw_frame.head(5))

count = hw_frame.extcredits2.count()
print(count)
mean1 = hw_frame.extcredits2.mean()
std1 = hw_frame.extcredits2.std()
mean2 = hw_frame.posts.mean()
std2 = hw_frame.posts.std()
hw_frame = hw_frame[(abs(hw_frame.extcredits2 - mean1) < std1*3) & (abs(hw_frame.posts - mean2) < std2*3) & (hw_frame.extcredits2 >=0)]
print(hw_frame.extcredits2.count())
#ScatterPlot(hw_frame.extcredits2, hw_frame.posts)



b = time.time()
print('b-a:'+str(b-a))
tf.logging.set_verbosity(tf.logging.ERROR)

kmeans = tf.contrib.learn.KMeansClustering(num_clusters=4, relative_tolerance=0.0001)
c = time.time()
print('c-b:'+str(c-b))
_ = kmeans.fit(input_fn=input_fn)
d = time.time()
print('d-c:'+str(d-c))
clusters = kmeans.clusters()
print(clusters)
print(kmeans.ALL_SCORES)
assignments = list(kmeans.predict_cluster_idx(input_fn=input_fn))
#score = kmeans.score(input_fn=input_fn)
#print(score)

e = time.time()
print('e-d:'+str(e-d))
#ScatterPlot(hw_frame.views, hw_frame.replies, assignments, clusters)
ScatterPlot(hw_frame.extcredits2, hw_frame.posts, assignments, clusters)
f = time.time()
print('f-e:'+str(f-e))

print('END')