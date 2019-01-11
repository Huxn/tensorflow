import tensorflow as tf
import numpy as np
from PIL import Image
from tensorflow.python import pywrap_tensorflow

model_reader = pywrap_tensorflow.NewCheckpointReader(r"model/mymodel.ckpt-1")

#然后，使reader变换成类似于dict形式的数据
var_dict = model_reader.get_variable_to_shape_map()

#最后，循环打印输出
# for key in var_dict:
#     print("variable name: ", key)
#     print(model_reader.get_tensor(key))


# 输入为N * (28*28) 二维矩阵
x = tf.placeholder(tf.float32, [None, 784])
# 输出为N * 10 二维矩阵， 10的含义是0-9的概率分布
y_ = tf.placeholder(tf.float32, [None, 10])
x_image = tf.reshape(x, [-1, 28, 28, 1])

# 定义第一个卷积层的variables和ops
W_conv1 = tf.Variable(tf.truncated_normal([7, 7, 1, 32], stddev=0.1))
b_conv1 = tf.Variable(tf.constant(0.1, shape=[32]))

L1_conv = tf.nn.conv2d(x_image, W_conv1, strides=[1, 1, 1, 1], padding='SAME')
L1_relu = tf.nn.relu(L1_conv + b_conv1)
L1_pool = tf.nn.max_pool(L1_relu, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

# 定义第二个卷积层的variables和ops
W_conv2 = tf.Variable(tf.truncated_normal([3, 3, 32, 64], stddev=0.1))
b_conv2 = tf.Variable(tf.constant(0.1, shape=[64]))

L2_conv = tf.nn.conv2d(L1_pool, W_conv2, strides=[1, 1, 1, 1], padding='SAME')
L2_relu = tf.nn.relu(L2_conv + b_conv2)
L2_pool = tf.nn.max_pool(L2_relu, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

# 全连接层
W_fc1 = tf.Variable(tf.truncated_normal([7 * 7 * 64, 1024], stddev=0.1))
b_fc1 = tf.Variable(tf.constant(0.1, shape=[1024]))

h_pool2_flat = tf.reshape(L2_pool, [-1, 7 * 7 * 64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

# dropout
keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

# readout层
W_fc2 = tf.Variable(tf.truncated_normal([1024, 10], stddev=0.1))
b_fc2 = tf.Variable(tf.constant(0.1, shape=[10]))

y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

# # 定义优化器和训练op
# cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=y_, logits=y_conv))
# train_step = tf.train.AdamOptimizer((1e-4)).minimize(cross_entropy)
#
# # 计算准确率
# correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
# accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
license_num = []
with tf.Session() as sess:
    for n in range(0, 4):
        path = "result/%s.bmp" % n
        img = Image.open(path)
        width = img.size[0]
        height = img.size[1]

        img_data = [[0] * 784 for i in range(1)]
        for h in range(0, height):
            for w in range(0, width):
                if img.getpixel((w, h)) > 230:
                    img_data[0][h * width + w] = 0
                else:
                    img_data[0][h * width + w] = 1
        soft_max = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

        save = tf.train.Saver()

        sess.run(tf.global_variables_initializer())
        save.restore(sess, 'model/mymodel.ckpt-1')
        # tf.train.import_meta_graph('model/mymodel.ckpt-1')
        # save.restore(sess, tf.train.latest_checkpoint("model/"))
        result = sess.run(soft_max, feed_dict={x: np.array(img_data), keep_prob: 1.0})
        # result2 = result.argsort()[::-1]
        for i in range(10):
            print(i, result[0][i])
        result1 = np.squeeze(result)
        top_3 = result1.argsort()[::-1]
        for i in range(10):
            print(top_3[i])
        max1 = 0
        max2 = 0
        max3 = 0
        max1_index = 0
        max2_index = 0
        max3_index = 0
        for j in range(10):
            if result[0][j] > max1:
                max1 = result[0][j]
                max1_index = j
                continue
            if (result[0][j] > max2) and (result[0][j] <= max1):
                max2 = result[0][j]
                max2_index = j
                continue
            if (result[0][j] > max3) and (result[0][j] <= max2):
                max3 = result[0][j]
                max3_index = j
                continue
        print("softmax结果前三位概率：%s: %.2f%%    %s: %.2f%%   %s: %.2f%%"
            % (max1_index, max1 * 100, max2_index, max2 * 100, max3_index, max3 * 100))
        license_num.append(max1_index)
    print("result: ", license_num)