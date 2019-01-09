import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("MNIST_data", one_hot=True)

'''
定义输入输出
'''
# 输入为N * (28*28) 二维矩阵
xs = tf.placeholder(tf.float32, [None, 784])
# 输出为N * 10 二维矩阵， 10的含义是0-9的概率分布
ys = tf.placeholder(tf.float32, [None, 10])

'''
计算准确率函数
    传入参数：v_xs输入图像
              v_ys计算结果
'''
def compute_accuracy(v_xs, v_ys):
    global prediction
    #计算预测结果
    y_pre = sess.run(prediction,
                     feed_dict={xs: v_xs})
    #判断预测结果是否与真是值相等，方法是取1的坐标是否相等
    correct_pre = tf.equal(tf.argmax(y_pre, 1), tf.argmax(v_ys, 1))
    #计算平均准确率reduce_mean平均值
    accuracy = tf.reduce_mean(tf.cast(correct_pre, tf.float32))
    result = sess.run(accuracy, feed_dict={xs: v_xs, ys: v_ys})
    return result


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def biases_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


'''
# stride [1, x_movement, y_movement, 1]
# Must have strides[0] = strides[3] = 1
x 与 W进行卷积操作
'''
def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding="SAME")


'''
# stride [1, x_movement, y_movement, 1]
'''
def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1],
                          padding="SAME")

#把xs的形状变成[-1,28,28,1]，-1代表先不考虑输入的图片例子多少这个维度，后面的1是channel的数量
#因为我们输入的图片是黑白的，因此channel是1，例如如果是RGB图像，那么channel就是3。
x_image = tf.reshape(xs, [-1, 28, 28, 1])

'''卷积层'''
#本层我们的卷积核patch的大小是5x5，因为黑白图片channel是1所以输入是1，输出是32个featuremap,下一层的输入参数之一
W_conv1 = weight_variable([5, 5, 1, 32])
b_conv1 = biases_variable([32])
# output size 28x28x32
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
# output size 14x14x32
h_pool1 = max_pool_2x2(h_conv1)

W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = biases_variable([64])
# output size 14x14x64
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
# output size 7x7x32
h_pool2 = max_pool_2x2(h_conv2)


'''全连接层'''
W_fc1 = weight_variable([7 * 7 * 64, 128])
b_fc1 = biases_variable([128])
h_pool_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool_flat, W_fc1) + b_fc1)

W_fc2 = weight_variable([128, 10])
b_fc2 = biases_variable([10])
prediction = tf.nn.softmax(tf.matmul(h_fc1, W_fc2) + b_fc2)

'''交叉熵损失函数'''
cross_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=ys,
                                                                    logits=prediction))
train = tf.train.AdamOptimizer(1e-3).minimize(cross_loss)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for step in range(100):
        #给输入值
        batch_xs, batch_ys = mnist.train.next_batch(100)
        #计算训练结果
        sess.run(train, feed_dict={xs: batch_xs, ys: batch_ys})
        if step % 20 == 0:
            #打印准确率
            print(compute_accuracy(mnist.test.images[:1000], mnist.test.labels[:1000]))
