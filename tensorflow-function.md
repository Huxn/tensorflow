# Tesnorflow常见函数

### tf.random_normal(
	shape, #一个一维整数张量或Python数组。代表张量的形状。
	mean=0.0, #数据类型为dtype的张量值或Python值。是正态分布的均值。
	stddev=1.0, #数据类型为dtype的张量值或Python值。是正态分布的标准差。
	dtype=tf.float32, #输出的数据类型。
	seed=None, #一个Python整数。是随机种子。
	name=None #操作的名称(可选)
) #从正态分布中输出随机值。


### tf.random_uniform(
    shape, #生成的张量的形状
    minval=0, #最小值
    maxval=None, #最大值
    dtype=tf.float32, #输出的数据类型
    seed=None, #随机种子
    name=None #操作的名称(可选)
) #从均匀分布中返回随机值。

### tf.truncated_normal(
	shape,#一个一维整数张量或Python数组。代表张量的形状。
	mean=0.0,#数据类型为dtype的张量值或Python值。是正态分布的均值。
	stddev=1.0,#数据类型为dtype的张量值或Python值。是正态分布的标准差
	dtype=tf.float32,#输出的数据类型。
	seed=None,#一个Python整数。是随机种子。
	name=None#操作的名称(可选)
)#截断的正态分布函数。生成的值遵循一个正态分布，但不会大于平均值2个标准差。

### tf.random_shuffle(
	value,# 要被洗牌的张量
    seed=None,
    name=None

)#沿着要被洗牌的张量的第一个维度，随机打乱。

### tf.argmax(
	input, #传入的array或者matrix
	axis #该参数能指定按照哪个维度计算。
		 #如 在矩阵的结构中，axis可被设置为0或1，分别表示
		 #0：按列计算，1：行计算
)#某一维上的其数据最大值所在的索引值

### tf.softmax_cross_entropy_with_logits(
	logits, #就是神经网络最后一层的输出，如果有batch的话，它的大小就是[batchsize，num_classes]，单样本的话，大小就是num_classes
	labels  #实际的标签，大小同上
)
	

	
	