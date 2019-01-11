import tensorflow as tf

#定义网络中间层
'''
inputs: 输入矩阵
in_size：输入矩阵的列，中间层的行
out_size：中间层的列，结果矩阵的列
activation_function：激活函数
这里可以联想矩阵的乘法，输入矩阵*中间矩阵=输出矩阵
根据矩阵乘法满足的条件，又已知输入矩阵的行列，输出矩阵的行列
则中间矩阵的行列是可以求得的
'''

def add_layer(inputs, in_size, out_size, activation_function = None):
    Weight = tf.Variable(tf.random.normal([in_size,out_size]))
    biases = tf.Variable(tf.zeros([1, out_size]))
    Wx_plus_b = tf.matmul(inputs,Weight) + biases
    if activation_function is None:
        outputs = Wx_plus_b
    else:
        outputs = activation_function(Wx_plus_b)
    return outputs
