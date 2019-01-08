#numpy常见函数

###np.linspace(
	start, #队列的起始值
	stop, #队列的最大值
	num=50, #可选要生成的样本数。默认是50，必须要非负。
	endpoint=True, #可选。如果是True，’stop'是最后的样本。否则，'stop'将不会被包含。默认为true
	retstep=False, #If True, return (`samples`, `step`), where `step` is the spacing between samples.
	dtype=None
) #返回固定间隔的数据


###np.random.normal(
	loc=0.0, #此概率分布的均值（对应着整个分布的中心centre）
	scale=1.0, #此概率分布的标准差（对应于分布的宽度，scale越大越矮胖，scale越小，越瘦高）
	size=None #输出的shape，默认为None，只输出一个值
)