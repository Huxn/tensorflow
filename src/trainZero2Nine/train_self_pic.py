import os
import tensorflow as tf
import numpy as np
from PIL import Image

'''
获取样本数据一共多少文件
'''
input_count = 0

for i in range(0, 10):
    dir = "./mnist_digits_images/%s/" % i
    for rt, dirs, files in os.walk(dir):
        for filename in files:
            input_count += 1

print(str(input_count))
# 得到一个1000行, 784列的二位数组, 其值皆为0
input_images = np.array([[0] * 784 for i in range(input_count)])
# 得到一个1000行, 10列的二位数组, 其值皆为0
input_labels = np.array([[0] * 10 for i in range(input_count)])


index = 0
print("开始载入训练图片")
for i in range(0, 10):
    dir = "./mnist_digits_images/%s/" %i
    for rt, dirs, files in os.walk(dir):
        for filename in files:
            file = dir + filename
            image = Image.open(file)
            width = image.size[0]
            height = image.size[1]
            for h in range(0, height):
                for w in range(0, width):
                    if image.getpixel((w, h)) > 230:
                        input_images[index][h * width + w] = 0
                    else:
                        input_images[index][h * width + w] = 1
            # 这一行代码：第二个参数意思是该张图片是数字几
            # 就将对应的列（即Y轴）置为1
            # X轴为第几张图片
            input_labels[index][i] = 1
            index += 1
print("训练图片载入完成")

# 输入为N * (28*28) 二维矩阵
xs = tf.placeholder(tf.float32, [None, 784])
# 输出为N * 10 二维矩阵， 10的含义是0-9的概率分布
ys = tf.placeholder(tf.float32, [None, 10])
'''
计算准确率函数
    传入参数：v_xs输入图像
              v_ys计算结果
'''
# def compute_accuracy(v_xs, v_ys):
#     global prediction
#     #计算预测结果
#     y_pre = sess.run(prediction,
#                      feed_dict={xs: v_xs})
#     #判断预测结果是否与真是值相等，方法是取1的坐标是否相等
#     correct_pre = tf.equal(tf.argmax(y_pre, 1), tf.argmax(v_ys, 1))
#     #计算平均准确率reduce_mean平均值
#     accuracy = tf.reduce_mean(tf.cast(correct_pre, tf.float32))
#     result = sess.run(accuracy, feed_dict={xs: v_xs, ys: v_ys})
#     return result
#
#
# def weight_variable(shape):
#     initial = tf.truncated_normal(shape, stddev=0.1)
#     return tf.Variable(initial)
#
#
# def biases_variable(shape):
#     initial = tf.constant(0.1, shape=shape)
#     return tf.Variable(initial)
#
#
# '''
# # stride [1, x_movement, y_movement, 1]
# # Must have strides[0] = strides[3] = 1
# x 与 W进行卷积操作
# '''
# def conv2d(x, W):
#     return tf.nn.conv2d(x, W,
#                         strides=[1, 1, 1, 1],
#                         padding="SAME")
#
#
# '''
# # stride [1, x_movement, y_movement, 1]
# '''
# def max_pool_2x2(x):
#     return tf.nn.max_pool(x,
#                           ksize=[1, 2, 2, 1],
#                           strides=[1, 2, 2, 1],
#                           padding="SAME")
#
# #把xs的形状变成[-1,28,28,1]，-1代表先不考虑输入的图片例子多少这个维度，后面的1是channel的数量
# #因为我们输入的图片是黑白的，因此channel是1，例如如果是RGB图像，那么channel就是3。
# x_image = tf.reshape(xs, [-1, 28, 28, 1])
#
# '''卷积层'''
# #本层我们的卷积核patch的大小是5x5，因为黑白图片channel是1所以输入是1，输出是32个featuremap,下一层的输入参数之一
# W_conv1 = weight_variable([5, 5, 1, 32])
# b_conv1 = biases_variable([32])
# # output size 28x28x32
# h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
# # output size 14x14x32
# h_pool1 = max_pool_2x2(h_conv1)
#
# W_conv2 = weight_variable([5, 5, 32, 64])
# b_conv2 = biases_variable([64])
# # output size 14x14x64
# h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
# # output size 7x7x32
# h_pool2 = max_pool_2x2(h_conv2)
#
#
# '''全连接层'''
# W_fc1 = weight_variable([7 * 7 * 64, 1024])
# b_fc1 = biases_variable([1024])
# h_pool_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])
# h_fc1 = tf.nn.relu(tf.matmul(h_pool_flat, W_fc1) + b_fc1)
#
# W_fc2 = weight_variable([1024, 10])
# b_fc2 = biases_variable([10])
# prediction = tf.matmul(h_fc1, W_fc2) + b_fc2
#
# '''交叉熵损失函数'''
# cross_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=ys,
#                                                                     logits=prediction))
# train = tf.train.AdamOptimizer(1e-4).minimize(cross_loss)


'''------------------------------------------------------------------'''
def add_layer(inputs, in_size, out_size, activation_func=None):
    Weight = tf.Variable(tf.truncated_normal([in_size, out_size]))
    biases = tf.Variable(tf.zeros([out_size]))
    Wx_plus_b = tf.matmul(inputs, Weight) + biases
    if activation_func is None:
        result = Wx_plus_b
    else:
        result = activation_func(Wx_plus_b)
    return result

xs = tf.placeholder(tf.float32, [None, 784])
ys = tf.placeholder(tf.float32, [None, 10])
layer1 = add_layer(xs, 784, 2048,  activation_func=tf.nn.tanh)
layer2 = add_layer(layer1, 2048, 2048,  activation_func=tf.nn.tanh)
prediction = add_layer(layer2, 2048, 10, activation_func=tf.nn.softmax)

def compute_accuracy(v_xs, v_ys):
    global prediction
    y_pre = sess.run(prediction, feed_dict={xs: v_xs})
    correct_pre = tf.equal(tf.argmax(y_pre, 1), tf.argmax(v_ys, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pre, tf.float32))
    result = sess.run(accuracy, feed_dict={xs: v_xs, ys: v_ys})
    return result
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=ys, logits=prediction))
train = tf.train.AdamOptimizer(1e-4).minimize(loss)


saver = tf.train.Saver()
iterate_accuracy = 0
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    print("一共读取了 %s 个输入图像， %s 个标签" % (input_count, input_count))

    # 设置每次训练op的输入个数和迭代次数，这里为了支持任意图片总数，定义了一个余数remainder，譬如，如果每次训练op的输入个数为60，图片总数为150张，则前面两次各输入60张，最后一次输入30张（余数30）
    batch_size = 60
    batches_count = int(input_count / batch_size)
    remainder = input_count % batch_size
    iterations = 100
    print("数据集分成 %s 批, 前面每批 %s 个数据，最后一批 %s 个数据" % (batches_count + 1, batch_size, remainder))

    # 执行训练迭代
    for it in range(iterations):
        writer = tf.summary.FileWriter("F://python//tensorflow//test", sess.graph)
        # 这里的关键是要把输入数组转为np.array
        for n in range(batches_count):
            sess.run(train, feed_dict={xs: input_images[n * batch_size:(n + 1) * batch_size],
                                       ys: input_labels[n * batch_size:(n + 1) * batch_size]})
            # train.run(feed_dict={xs: input_images[n * batch_size:(n + 1) * batch_size],
            #                           ys: input_labels[n * batch_size:(n + 1) * batch_size]})
        if remainder > 0:
            start_index = batches_count * batch_size
            sess.run(train, feed_dict={xs: input_images[start_index:input_count - 1],
                                       ys: input_labels[start_index:input_count - 1]})
            # train.run(
            #     feed_dict={xs: input_images[start_index:input_count - 1],
            #                ys: input_labels[start_index:input_count - 1]})

        # 每完成五次迭代，判断准确度是否已达到100%，达到则退出迭代循环
        if it % 5 == 0:
            iterate_accuracy = compute_accuracy(input_images, input_labels)
            print('iteration %d: accuracy %s' % (it, iterate_accuracy))
        if iterate_accuracy > 0.99:
            break
    saver.save(sess, "pic_train/pic_train.ckpt")
    print('完成训练!')
    writer.close()


