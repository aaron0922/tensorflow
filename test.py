import time
a = time.time()
import tensorflow as tf
b = time.time()
hello = tf.constant('Hello, TensorFlow!')
sess = tf.Session()
print(sess.run(hello))
c = time.time()
print(b - a)
print(c - b)
print('end')