import tensorflow as tf
import numpy as np

x_data = np.random.rand(200)
y_data = x_data * 0.2 + 0.5

k = tf.Variable(1000.0)
b = tf.Variable(1000.0)
y = k * x_data + b

loss = tf.reduce_mean(tf.square(y - y_data))
optimizer = tf.train.GradientDescentOptimizer(0.1)
train = optimizer.minimize(loss)

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    for step in range(1001):
        sess.run(train)
        if step % 20 == 0:
            print(step, sess.run([k,b]))


