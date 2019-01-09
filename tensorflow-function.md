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

### tf.nn.conv2d(
	input, #指需要做卷积的输入图像，它要求是一个Tensor，具有[batch, in_height, in_width, in_channels]这样的shape，具体含义是[训练时一个batch的图片数量, 图片高度, 图片宽度, 图像通道数]，注意这是一个4维的Tensor，要求类型为float32和float64其中之一
	filter, #相当于CNN中的卷积核，它要求是一个Tensor，具有[filter_height, filter_width, in_channels, out_channels]这样的shape，具体含义是[卷积核的高度，卷积核的宽度，图像通道数，卷积核个数]，要求类型与参数input相同，有一个地方需要注意，第三维in_channels，就是参数input的第四维
	strides, #卷积时在图像每一维的步长，这是一个一维的向量，长度4
	padding #string类型的量，只能是"SAME","VALID"其中之一，这个值决定了不同的卷积方式
)

### tf.nn.max_pool(
	value, #需要池化的输入，一般池化层接在卷积层后面，所以输入通常是feature map，依然是[batch, height, width, channels]这样的shape
	ksize, #池化窗口的大小，取一个四维向量，一般是[1, height, width, 1]，因为我们不想在batch和channels上做池化，所以这两个维度设为了1
	strides, #和卷积类似，窗口在每一个维度上滑动的步长，一般也是[1, stride,stride, 1]
	padding #和卷积类似，可以取'VALID' 或者'SAME'
)
	

	
	