import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

def add_layer(inputs, in_size, out_size, activation_func):
    Weights = tf.Variable(tf.random.normal([in_size,out_size]))
    biases = tf.Variable(tf.zeros([1,out_size]) + 0.1)
    Wx_plus_biases = tf.matmul(inputs, Weights) + biases
    if activation_func is None:
        result = Wx_plus_biases
    else:
        result = activation_func(Wx_plus_biases)
    return result

x_data = np.linspace(-1, 1, 100)[:, np.newaxis]
# print(x_data)
noise = np.random.normal(0,0.05,x_data.shape)
# print(noise)
y_data = np.square(x_data) - 0.5 + noise

# plt.figure()
# plt.scatter(x_data, y_data)
# plt.show()


xs = tf.placeholder(tf.float32, [None, 1])
ys = tf.placeholder(tf.float32, [None, 1])

l1 = add_layer(xs, 1, 10, activation_func=tf.nn.tanh)
prediction = add_layer(l1, 10, 1, activation_func=None)

loss = tf.reduce_mean(tf.square(ys - prediction))
train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for step in range(101):
        sess.run(train_step, feed_dict={xs:x_data, ys:y_data})
        if step % 20 == 0:
            print(step, sess.run(loss, feed_dict={xs:x_data, ys:y_data}))
    plt.plot(x_data, sess.run(prediction, feed_dict={xs: x_data}), 'r-')
    plt.show()