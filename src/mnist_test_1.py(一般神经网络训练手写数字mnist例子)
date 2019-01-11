import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("MNIST_data", one_hot=True)
batch_size = 50
n_batch = mnist.train.num_examples // batch_size


def add_layer(inputs, in_size, out_size, activation_func=None):
    Weight = tf.Variable(tf.random.normal([in_size, out_size]))
    biases = tf.Variable(tf.zeros([out_size]))
    Wx_plus_b = tf.matmul(inputs, Weight) + biases
    if activation_func is None:
        result = Wx_plus_b
    else:
        result = activation_func(Wx_plus_b)
    return result


xs = tf.placeholder(tf.float32, [None, 784])
ys = tf.placeholder(tf.float32, [None, 10])
# prediction1 = add_layer(xs, 784, 100, activation_func=tf.nn.tanh)
# prediction2 = add_layer(prediction1, 1000, 500, activation_func=tf.nn.softmax)
prediction = add_layer(xs, 784, 10, activation_func=tf.nn.softmax)


def compute_accuracy(sess, v_xs, v_ys):
    global prediction
    y_pre = sess.run(prediction, feed_dict={xs: v_xs})
    correct_pre = tf.equal(tf.argmax(y_pre, 1), tf.argmax(v_ys, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pre, tf.float32))
    result = sess.run(accuracy, feed_dict={xs: v_xs, ys: v_ys})
    return result


# loss = tf.reduce_mean(tf.square(ys - prediction))
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=ys, logits=prediction))
train = tf.train.GradientDescentOptimizer(1.3).minimize(loss)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for step in range(21):
        for batch in range(n_batch):
            batch_nx, batch_ny = mnist.train.next_batch(batch_size)
            sess.run(train, feed_dict={xs: batch_nx, ys: batch_ny})
        acc = compute_accuracy(sess, mnist.test.images, mnist.test.labels)
        # if step % 20 == 0:
        print(str(acc))
