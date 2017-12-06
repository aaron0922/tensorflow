import tensorflow as tf

#3
a = tf.add(1, 2,)
#9
b = tf.multiply(a, 3)
#9
c = tf.add(4, 5,)
#54
d = tf.multiply(c, 6,)
#20
e = tf.multiply(4, 5,)
#1
f = tf.div(c, 6,)
#63
g = tf.add(b, d)
#63
h = tf.multiply(g, f)

with tf.Session() as sess:
    '''
    print(sess.run(a))
    print(sess.run(b))
    print(sess.run(c))
    print(sess.run(d))
    print(sess.run(e))
    print(sess.run(f))
    print(sess.run(g))
    print(sess.run(h))
    '''
    writer = tf.summary.FileWriter("output", sess.graph)
    print(sess.run(h))
    writer.close()

