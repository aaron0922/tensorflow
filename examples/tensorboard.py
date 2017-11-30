import tensorflow as tf
a = tf.constant(10, name="a")
b = tf.constant(90, name="b")
y = tf.Variable(a+b*2, name="y")
model = tf.global_variables_initializer()
with tf.Session() as sess:
    merge = tf.summary.merge_all()
    write = tf.summary.FileWriter \
        ("D:/workspace/python/tensorflow/examples/tensorflowlogs", sess.graph)
    sess.run(model)
    print(sess.run(y))
print("end")

#cmd
#d:
#tensorboard --logdir=D:/workspace/python/tensorflow/examples/tensorflowlogs/