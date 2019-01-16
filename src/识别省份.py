import sys
import os
import time

import numpy as np
import tensorflow as tf

from PIL import Image

SIZE = 1280
WIDTH = 32
HEIGHT = 40
NUM_CLASS = 6
iterator = 300

SAVE_DIR = './train-saver/province/'

PROVINCES = ("京","闽","粤","苏","沪","浙")

# 用于测试结果的index，从PROVINCES取inde下标的值
nProvinceIndex = 0

# 起始时间
time_begin = time.time()

# 定义输入节点，对应于图片像素值矩阵集合和图片标签(即所代表的数字)
with tf.name_scope("input"):
    x = tf.placeholder(tf.float32, shape=[None, SIZE], name="x_input")
    y_ = tf.placeholder(tf.float32, shape=[None, NUM_CLASS], name="y_input")

x_image = tf.reshape(x, [-1, WIDTH, HEIGHT, 1])


def conv_layer(inputs, W, b, conv_strides, kernel_size, pools_strides, padding, name=None):
    with tf.name_scope(name):
        L1_conv = tf.nn.conv2d(inputs, W,
                               strides=conv_strides,
                               padding=padding, name="conv")
        L1_relu = tf.nn.relu(L1_conv + b, name="relu")
        return tf.nn.max_pool(L1_relu, ksize=kernel_size,
                              strides=pools_strides,
                              padding="SAME", name="pool")


def full_connect(inputs, W, b, name=None):
    with tf.name_scope(name):
        return tf.nn.relu(tf.matmul(inputs, W) + b, name="full")


if __name__ == '__main__' and sys.argv[1] == "train":
    input_count = 0
    # 计算训练数据的个数
    for i in range(0, NUM_CLASS):
        dir = "./tf_car_license_dataset/train_images/training-set/chinese-characters/%s/" % i
        for tr, dirs, files in os.walk(dir):
            for file in files:
                input_count += 1

    # 定义训练集的输入images 与 labels
    input_images = np.array([[0] * SIZE for i in range(input_count)])
    input_labels = np.array([[0] * NUM_CLASS for i in range(input_count)])

    index = 0
    for i in range(0, NUM_CLASS):
        dir  = "./tf_car_license_dataset/train_images/training-set/chinese-characters/%s/" % i
        for rt, dirs, files in os.walk(dir):
            for file in files:
                path = dir + file
                image = Image.open(path)
                width = image.size[0]
                height = image.size[1]
                for h in range(0, height):
                    for w in range(0, width):
                        # 通过这样的处理，使数字的线条变细，有利于提高识别准确率
                        # input_images[index][w + h * width] = image.getpixel((w, h)) / 255.0
                        if image.getpixel((w, h)) > 230:
                            input_images[index][w + h * width] = 0
                        else:
                            input_images[index][w + h * width] = 1
                input_labels[index][i] = 1
                index += 1

    val_count = 0
    for i in range(0, NUM_CLASS):
        dir = "./tf_car_license_dataset/train_images/validation-set/chinese-characters/%s/" % i
        for rt, dirs, files in os.walk(dir):
            for file in files:
                val_count += 1
    val_images = np.array([[0] * SIZE for i in range(val_count)])
    val_labels = np.array([[0] * NUM_CLASS for i in range(val_count)])
    index = 0
    for i in range(0, NUM_CLASS):
        dir = "./tf_car_license_dataset/train_images/validation-set/chinese-characters/%s/" % i
        for rt, dirs, files in os.walk(dir):
            for file in files:
                path = dir + file
                image = Image.open(path)
                width = image.size[0]
                height = image.size[0]
                for h in range(0, height):
                    for w in range(0, width):
                        # 通过这样的处理，使数字的线条变细，有利于提高识别准确率
                        # input_images[index][w + h * width] = image.getpixel((w, h)) / 255.0
                        if image.getpixel((w, h)) > 230:
                            val_images[index][w + h * width] = 0
                        else:
                            val_images[index][w + h * width] = 1
                val_labels[index][i] = 1
                index += 1

    # 括号引发的血案!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    with tf.Session() as sess:
        W_conv1 = tf.Variable(tf.truncated_normal([8, 8, 1, 16], stddev=0.1), name="W_conv1")
        b_conv1 = tf.Variable(tf.constant(0.1, shape=[16]), name="b_conv1")
        conv_strides = [1, 1, 1, 1]
        kernel_size = [1, 2, 2, 1]
        pool_strides = [1, 2, 2, 1]
        padding = "SAME"
        L1_pool = conv_layer(x_image, W_conv1, b_conv1,
                             conv_strides, kernel_size, pool_strides, padding, name="conv1")
        W_conv2 = tf.Variable(tf.truncated_normal([5, 5, 16, 32], stddev=0.1), name="W_conv2")
        b_conv2 = tf.Variable(tf.constant(0.1, shape=[32]), name="b_conv2")
        conv_strides = [1, 1, 1, 1]
        kernel_size = [1, 1, 1, 1]
        pool_strides = [1, 1, 1, 1]
        padding = "SAME"
        L2_pool = conv_layer(L1_pool, W_conv2, b_conv2,
                             conv_strides, kernel_size, pool_strides, padding, name="conv2")

        W_fc1 = tf.Variable(tf.truncated_normal([16 * 20 * 32, 512], stddev=0.1), name="W_fc1")
        b_fc1 = tf.Variable(tf.constant(0.1, shape=[512]), name="b_fc1")
        h_pool2_flat = tf.reshape(L2_pool, [-1, 16 * 20 * 32])
        h_fc1 = full_connect(h_pool2_flat, W_fc1, b_fc1, name="full")

        # keep_dropout = tf.placeholder(tf.float32)
        # h_fc1_drop = tf.nn.dropout(h_fc1, keep_dropout)
        # # dropout
        # keep_prob = tf.placeholder(tf.float32)
        #
        # h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

        W_fc2 = tf.Variable(tf.truncated_normal([512, NUM_CLASS], stddev=0.1), name="W_fc2")
        b_fc2 = tf.Variable(tf.constant(0.1, shape=[NUM_CLASS]), name="b_fc2")

        # y_conv = tf.nn.softmax(tf.matmul(h_fc1, W_fc2) + b_fc2)
        y_conv = tf.matmul(h_fc1, W_fc2) + b_fc2
        # cross_loss = -tf.reduce_sum(y_ * tf.log(y_conv))
        cross_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))
        train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_loss)

        correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        saver = tf.train.Saver()
        sess.run(tf.global_variables_initializer())

        time_elapsed = time.time() - time_begin
        print("读取图片文件耗时%s秒" % time_elapsed)

        time_begin = time.time()
        print("一共读取了%s个训练图像， %s 个标签图像" % (input_count,input_count))

        batch_size = 60
        iterators = iterator
        batch_count =int(input_count / batch_size)
        remainder = input_count % batch_size
        print("训练数据集分%s 批, 前面每批%s个数据，最后一批%s个数据" %
              (batch_count + 1, batch_size, remainder))

        for i in range(iterator):
            for n in range(batch_count):
                train_step.run(feed_dict={
                    x: input_images[n * batch_size: (n+1) * batch_size],
                    y_: input_labels[n * batch_size: (n+1) * batch_size]
                })
            if remainder > 0:
                train_step.run(feed_dict={
                    x: input_images[batch_count * batch_size: input_count - 1],
                    y_: input_labels[batch_count * batch_size: input_count - 1]
                })
            iterators_accuracy = 0
            if i % 5 == 0:
                iterators_accuracy = accuracy.eval(feed_dict={
                    x: val_images,
                    y_:val_labels
                })
                print("第%s次训练迭代准确率为%0.5f%%" % (i, iterators_accuracy * 100))
                if iterators_accuracy > 0.95:
                    break
        print("训练完成")
        time_elapsed = time.time() - time_begin
        print("训练耗费时间%s秒" % time_elapsed)
        if not os.path.exists(SAVE_DIR):
            print("不存在训练数据保存目录，现在创建目录")
            os.mkdir(SAVE_DIR)
        save_path = saver.save(sess, "%smodel.ckpt" % SAVE_DIR)


if __name__ == "__main__" and sys.argv[1] == "predict":
    saver = tf.train.import_meta_graph("%smodel.ckpt.meta" % SAVE_DIR)
    with tf.Session() as sess:
        model_file = tf.train.latest_checkpoint(SAVE_DIR)
        saver.restore(sess, model_file)
        W_conv1 = sess.graph.get_tensor_by_name("W_conv1:0")
        b_conv1 = sess.graph.get_tensor_by_name("b_conv1:0")
        conv_strides = [1, 1, 1, 1]
        kernel_size = [1, 2, 2, 1]
        pool_strides = [1, 2, 2, 1]
        padding = "SAME"
        L1_pool = conv_layer(x_image, W_conv1, b_conv1,
                             conv_strides, kernel_size, pool_strides, padding, name="conv1")
        W_conv2 = sess.graph.get_tensor_by_name("W_conv2:0")
        b_conv2 = sess.graph.get_tensor_by_name("b_conv2:0")
        conv_strides = [1, 1, 1, 1]
        kernel_size = [1, 1, 1, 1]
        pool_strides = [1, 1, 1, 1]
        L2_pool = conv_layer(L1_pool, W_conv2, b_conv2,
                             conv_strides, kernel_size, pool_strides, padding, name="conv2")
        W_fc1 = sess.graph.get_tensor_by_name("W_fc1:0")
        b_fc1 = sess.graph.get_tensor_by_name("b_fc1:0")
        h_pool2_flat = tf.reshape(L2_pool, [-1, 16 * 20 * 32])
        h_fc1 = full_connect(h_pool2_flat, W_fc1, b_fc1, name="full")

        # keep_dropout = tf.placeholder(tf.float32)
        # h_fc1_drop = tf.nn.dropout(h_fc1, keep_dropout)

        W_fc2 = sess.graph.get_tensor_by_name("W_fc2:0")
        b_fc2 = sess.graph.get_tensor_by_name("b_fc2:0")
        conv = tf.argmax(tf.nn.softmax(tf.matmul(h_fc1, W_fc2) + b_fc2), 1)

        for it in range(1,2):
            path = "./tf_car_license_dataset/test_images/%s.bmp" % it
            print (path)
            image = Image.open(path)
            width = image.size[0]
            height = image.size[1]
            image_data = [[0] * SIZE for i in range(1)]
            for h in range(0, height):
                for w in range(0, width):
                    # input_images[index][w + h * width] = image.getpixel((w, h)) / 255.0
                    if image.getpixel((w, h)) > 230:
                        image_data[0][w + h * width] = 0
                    else:
                        image_data[0][w + h * width] = 1
            result = conv.eval(feed_dict={
                x: np.array(image_data)
            })
            nProvinceIndex = result
            print("结果为%s" % np.array(PROVINCES)[nProvinceIndex])



