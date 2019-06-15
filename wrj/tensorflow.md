## tf.layers.conv1d（一维卷积）

函数定义如下：

~~~python
tf.layers.conv1d(
inputs,
filters,
kernel_size,
strides=1,
padding='valid',
data_format='channels_last',
dilation_rate=1,
activation=None,
use_bias=True,
kernel_initializer=None,
bias_initializer=tf.zeros_initializer(),
kernel_regularizer=None,
bias_regularizer=None,
activity_regularizer=None,
kernel_constraint=None,
bias_constraint=None,
trainable=True,
name=None,
reuse=None
)
~~~

**inputs** :  输入tensor。例如，(None,  a, b) ——三维tensor。

- None： 样本个数，batch_size
- 句子中的词或字数
- 字或词向量

**filters** :  过滤器个数。

**kernel_size** : 卷积核的大小。卷积核其实应是二维的，这里只需指定一维。因为卷积核的第二维与输入词向量维度一致。

 ![1559271871759](C:\Users\Ruijia Wang\AppData\Roaming\Typora\typora-user-images\1559271871759.png)



## tf.nn.dropout

为了防止或减轻过拟合而使用的函数，一般用在全连接层。

在不同训练过程中随机扔掉部分神经元，也就是让某个神经元的激活值以一定概率p停止工作。从而在这次训练过程中不更新权值，也不参与神经网络计算。但它的权重需要保留下来（只是暂时不更新而已），因为下次样本输入时可能又需要工作了。

~~~python
tf.nn.dropout( 
x,
keep_prob,
noise_shape=None,
seed=None,
name=None
)
~~~

**x**：输入tensor。

**keep_prob**: float类型，每个元素被保留下来的概率。初始化时，keep_prob是一个占位符, keep_prob = tf.placeholder(tf.float32) 。run时，设置keep_prob的具体值，如keep_prob: 0.5。

train的时候dropout起作用，test的时候dropout不起作用。



## tf.transpose

~~~python
tf.transpose(
input, 
[dimension_1, dimenaion_2,..,dimension_n]
)
~~~

适用于交换输入张量的不同维度。如果输入张量是二维，相当于转置。

dimension_n是整数，如果张量是三维，就用0,1,2表示。这个列表里的每个数对应相应的维度。如果是[2,1,0]，就把输入张量的第三维度和第一维度交换。

![1559273583408](C:\Users\Ruijia Wang\AppData\Roaming\Typora\typora-user-images\1559273583408.png)



## 数组加法

![1559300547581](C:\Users\Ruijia Wang\AppData\Roaming\Typora\typora-user-images\1559300547581.png)

![1559300612484](C:\Users\Ruijia Wang\AppData\Roaming\Typora\typora-user-images\1559300612484.png)



## tf.nn.softmax

~~~python
tf.nn.softmax(
logits,
axis=None,
name=None,
dim=None
)

softmax = tf.exp(logits) / tf.reduce_sum(tf.exp(logits), axis)
~~~

**logits** : 非空张量。必须是以下类型之一： float32, float64。

**axis** : 将被执行的softmax维度，默认值-1，表示最后一个维度。



## tf.nn.leaky_relu

![1559303057396](C:\Users\Ruijia Wang\AppData\Roaming\Typora\typora-user-images\1559303057396.png)

~~~python
tf.nn.leaky_relu(
features,
alpha=0.2,
name=None
)
~~~

**features**：一个Tensor，表示预激活值，必须是下列类型之一：float16,float32,float64,int32,int64。

**alpha**：x <0时激活函数的斜率。



## tf.SparseTensor

代表稀疏张量。

TensorFlow表示稀疏张量，作为三个独立的稠密张量：indices,values和dense_shape。在Python中，三个张量被集合到一个SparseTensor类中，以方便使用。

具体来说，该稀疏张量SparseTensor(indices, values, dense_shape)包括以下组件，其中N和ndims分别是在SparseTensor中的值的数目和维度：

- indices：density_shape[N, ndims]的2-D int64张量，指定稀疏张量中包含非零值的元素的索引。例如，indices=[[1,3], [2,4]]指定索引为[1,3]和[2,4]的元素具有非零值。
- values：任何类型和dense_shape [N]的一维张量，它提供indices中每个元素的值。例如，给定indices=[[1,3], [2,4]]的参数values=[18, 3.6]，则指定稀疏张量元素[1,3]的值为18，[2,4]的值为3.6。
- dense_shape：density_shape[ndims]的1-D int64张量，指定稀疏张量的dense_shape。获取一个列表，指出每个维度中元素的数量。例如，dense_shape=[3,6]指定二维3x6张量，dense_shape=[2,3,4]指定三维2x3x4张量，dense_shape=[9]指定具有9个元素的一维张量。

相应的稠密张量满足：

~~~python
dense.shape = dense_shape
dense[tuple(indices[i])] = values[i]
~~~

例如，稀疏张量表示：

~~~python
SparseTensor(indices=[[0, 0], [1, 2]], values=[1, 2], dense_shape=[3, 4])
~~~

表示稠密张量：

~~~python
[[1, 0, 0, 0]
 [0, 0, 2, 0]
 [0, 0, 0, 0]]
~~~



## tf.squeeze

~~~python
squeeze(
input,
axis=None,
name=None,
squeeze_dims=None
)
~~~

该函数返回一个张量，这个张量是将原始input中所有维度为1的那些维都删掉的结果。
`axis`可以用来指定要删掉的为1的维度，此处要注意指定的维度必须确保其是1，否则会报错。

例子：

~~~python
#  't' 是一个维度是[1, 2, 1, 3, 1, 1]的张量
tf.shape(tf.squeeze(t))   # [2, 3]， 默认删除所有为1的维度

# 't' 是一个维度[1, 2, 1, 3, 1, 1]的张量
tf.shape(tf.squeeze(t, [2, 4]))  # [1, 2, 3, 1]，标号从零开始，只删掉了2和4维的1
~~~



## tf.expand_dims

~~~python
tf.expand_dims(
input,
axis=None,
name=None,
dim=None
)
~~~

所实现的功能是给定input，在axis轴处给input增加一个为1的维度。

例如：

~~~python
# 't2' is a tensor of shape [2, 3, 5]
tf.shape(tf.expand_dims(t2, 0))  # [1, 2, 3, 5]
~~~

axis=0，所以矩阵维度变成1\*2\*3*5；同理，如果axis=2，矩阵就会变为2\*3\*5\*1。



## isinstance

isinstance() 函数判断一个对象是否是一个已知类型，类似 type()。

isinstance() 与 type() 区别：

- type() 不会认为子类是一种父类类型，不考虑继承关系。
- isinstance() 会认为子类是一种父类类型，考虑继承关系。

如果要判断两个类型是否相同推荐使用 isinstance()。

~~~python
isinstance(object, classinfo)
~~~

- object —— 实例对象。
- classinfo —— 可以是直接或间接类名、基本类型或由它们组成的元组。

如果对象类型与参数类型（classinfo）相同则返回 True，否则返回 False。

例如：

~~~python
>>>a = 2
>>> isinstance (a,int)
True
>>> isinstance (a,str)
False
>>> isinstance (a,(str,int,list))    # 是元组中的一个返回 True
True
~~~

type() 与 isinstance()区别：

~~~python
class A:
    pass
 
class B(A):
    pass
 
isinstance(A(), A)    # returns True
type(A()) == A        # returns True
isinstance(B(), A)    # returns True
type(B()) == A        # returns False
~~~



## tf.concat

~~~python
tf.concat(
values,
axis,
name='concat'
)
~~~

- values，tensor的list或tuple。
- axis，连接维度。

tf.concat返回连接后的tensor。

如果list中tensor的shape都是（2，2，2），如果此时的axis为2，即连接第三个维度，那么连接后的shape是（2，2，4），具体表现为对应维度的堆砌。

例如：

~~~python
t1 = [[[1, 2], [2, 3]], [[4, 4], [5, 3]]]
t2 = [[[7, 4], [8, 4]], [[2, 10], [15, 11]]]
tf.concat([t1, t2], axis=-1)
~~~

输出结果为：

~~~python
<tf.Tensor 'concat_2:0' shape=(2, 2, 4) dtype=int32>

[[[ 1,  2,  7,  4],
  [ 2,  3,  8,  4]],
 [[ 4,  4,  2, 10],
  [ 5,  3, 15, 11]]]
~~~



##  list v.s. tuple

list：

- 赋值类似字符数组，访问索引也类似数组

classmates = ['Michael', 'Bob', 'Tracy']，可以访问classmates[0]、classmates[1]等元素。classmates[-1]表示最后一个元素。list可以从后索引，索引越界会报错。

- 使用append追加元素，使用insert插入元素到指定位置，使用pop删除末尾元素，还可以用pop(i)删除索引i位置的元素

例如，

~~~python
classmates.append('Susan')

classmates.insert(1,'David') # 1代表插入元素后元素所在索引为1

classmates.pop()

classmates.pop(2)
~~~

- 修改list中某项的值，直接索引访问赋值

例如，

~~~python
classmates[2]='LiLei'
~~~

- list中包含的元素数据类型可以不同，且list里面可以嵌套list

tuple和list的区别在于：tuple在定义初始化后就不能修改了，而list可以修改。因此，tuple更安全。tuple是圆括号，list是方括号

例如，

~~~python
t=（1,2）# 初始化后不能再修改
~~~

- 如果定义空的tuple，写成 t=()；空的list，写成 s=[]。

- 如果定义只含有一个元素的tuple，应写成 t=(1,)；否则，如果定义成 t=(1)，圆括号会被解释成数学里的小括号，从而变成算式 t=1，而并非定义了一个只含有1个元素的tuple，所以需要在一个元素后面加逗号区别。

- tuple的元素不变是指tuple的元素的指向不变，但如果某个元素指向了list，那么list本身是可以变化的，访问tuple中的元素时可以使用list的访问方式。



## tf.tensordot

~~~python
tf.tensordot(
a,
b,
axes,
name=None
)
~~~

a 和 b 沿特定轴的张量收缩。

Tensordot(也称张量收缩)，对从 a 和 b 所指定的索引 a_axes 和 b_axes 的元素的乘积进行求和。列表 a_axes 和 b_axes 指定收缩张量的轴对。对于所有 range(0, len(a_axes)) 中的 i，a 的轴 a_axes[i] 必须与 b 的轴 b_axes[i] 具有相同的维度。列表 a_axes 和 b_axes 必须具有相同的长度，并由唯一的整数组成，用于为每个张量指定有效的坐标轴。

示例1：当 a 和 b 是矩阵(2阶)时，axes = 1 相当于矩阵乘法。

示例2：当 a 和 b 是矩阵(2阶)时，axes = [[1], [0]] 相当于矩阵乘法。

- a：float32 或 float64 类型的 Tensor。
- b：Tensor，与 a 具有相同的类型。
- axes：标量 N，也可以是具有形状 [2,k] 的 int32 Tensor 列表。如果轴是标量，则按顺序对 a 的最后 N 个轴和 b 的前 N 个轴进行求和；如果轴是一个列表或 Tensor，则分别对于轴 a 和 b计算，a 和 b 的坐标轴数必须相等。



## tf.reduce_sum

求和函数，可以通过调整 axis =0,1 来控制求和维度。

![1559475676687](C:\Users\Ruijia Wang\AppData\Roaming\Typora\typora-user-images\1559475676687.png)



## tensorflow多维张量计算

把前面的维度当成batch，对最后两维进行普通矩阵乘法。也就是说，最后两维之前的维度，需要相同。



## tf.one_hot

将input转化为one-hot类型数据输出，相当于将多个数值联合放在一起作为多个相同类型的向量，可用于表示各自的概率分布，通常用于分类任务中作为最后FC层的输出。

~~~python
one_hot(
indices,#输入，这里是一维的
depth,# one hot dimension.
on_value=None,#output 默认1
off_value=None,#output 默认0
axis=None,
dtype=None,
name=None
)
~~~

indices表示输入的多个数值，通常是矩阵形式；depth表示输出维度。
由于one-hot类型数据长度为depth位，其中，只用一位数字表示原输入数据，这里的on_value就是这个数字，默认值为1；one-hot数据的其他位用off_value表示，默认值为0。

tf.one_hot()函数规定输入元素indices从0开始，最大不能超过（depth - 1）。若输入的元素值超出范围，输出编码均为 [0, 0 … 0, 0]。

indices = 0 对应输出是[1, 0 … 0, 0]，indices = 1 对应的输出是[0, 1 … 0, 0]，依次类推，最大可能值输出是[0, 0 … 0, 1]。

例如，

~~~python
import tensorflow as tf  
      
classes = 3
labels = tf.constant([0,1,2]) # 输入的元素值最小为0，最大为2
output = tf.one_hot(labels,classes)

sess = tf.Session()
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    output = sess.run(output)
    print("output of one-hot is : ",output)

# ('output of one-hot is : ', array([[ 1.,  0.,  0.],
#       [ 0.,  1.,  0.],
#       [ 0.,  0.,  1.]], dtype=float32))
~~~



## tf.nn.sparse_softmax_cross_entropy_with_logits

传入的logits为神经网络输出层输出，shape为[batch_size，num_classes]，传入的label为一维向量，长度等于batch_size，每一个值的取值区间必须是[0，num_classes)，代表batch中对应的样本类别。

这个函数的具体实现分为两个步骤。

第一步：Softmax，将每个类别所对应的输出分量归一化，使各分量的和为1，将output vector的输出分量值，转化为将input data分类为每个类别的概率。

第二步：计算Cross-Entropy

需要将labels转化为one-hot格式编码，例如分量为3，代表该样本属于第三类，其对应的one-hot编码为[0，0，0，1，.......0]，若已经是one-hot格式，可以使用tf.nn.softmax_cross_entropy_with_logits()函数来进行softmax和loss的计算。



## tf.add_n

~~~python
tf.add_n([p1, p2, p3....])
~~~

实现列表元素的相加，输入对象是一个列表，列表里的元素可以是向量、矩阵等。



## tf.confusion_matrix

![1559969648112](C:\Users\Ruijia Wang\AppData\Roaming\Typora\typora-user-images\1559969648112.png)

~~~python
tf.confusion_matrix(
labels,
predictions,
num_classes=None,
dtype=tf.int32,
name=None,
weights=None
)
~~~

- labels：分类任务的1-D 真实标签。
- predictions：分类的1-D 预测。
- num_classes：分类任务可能具有的标签数量。如果未提供此值，则将使用预测和标签数组计算该值。
- dtype：混淆矩阵数据类型。
- weights：形状匹配`predictions`。

```python
tf.confusion_matrix([1, 2, 4], [2, 2, 4]) ==>
      [[0 0 0 0 0]
       [0 0 1 0 0]
       [0 0 1 0 0]
       [0 0 0 0 0]
       [0 0 0 0 1]]
```



## tf.cast

~~~python
cast(
x,
dtype,
name=None
)
~~~

将 x 的数据格式转化成 dtype。例如，原来 x 的数据格式是 bool，那么将其转化成 float 后，能将其转化成 0 和 1 的序列。

例如，

~~~python
a = tf.Variable([1,0,0,1,1])
b = tf.cast(a,dtype=tf.bool)
sess = tf.Session()
sess.run(tf.initialize_all_variables())
print(sess.run(b)) 

[ True False False  True  True]
~~~



## tf.nn.sigmoid_cross_entropy_with_logits

这个函数的输入是logits和labels，logits是神经网络模型中的 W * X矩阵，注意不需要经过sigmoid。而labels的shape和logits相同，是正确label值。

对W * X得到的值进行sigmoid激活，保证取值在0到1之间，然后放在交叉熵函数中计算Loss。

![1560000591956](C:\Users\Ruijia Wang\AppData\Roaming\Typora\typora-user-images\1560000591956.png)



## tf.argmax

~~~python
tf.argmax(input,axis)
~~~

根据axis取值的不同返回每行或每列最大值的索引。

例如，

~~~python
test = np.array([[1, 2, 3], [2, 3, 4], [5, 4, 3], [8, 7, 2]])
np.argmax(test, 0)　　　＃输出：array([3, 3, 1])
np.argmax(test, 1)　　　＃输出：array([2, 2, 0, 0])
~~~



## tf.round

~~~python
round(
x,
name=None
)
~~~

将张量的值四舍五入为最接近的整数。



## zip

将可迭代的对象作为参数，将对象中对应的元素打包成元组，然后返回由这些元组组成的列表。如果各迭代器元素个数不一致，则返回列表长度与最短对象相同。利用 * 号操作符，可以将元组解压为列表。

~~~python
zip([iterable, ...])
~~~

- iterable —— 一或多个迭代器。

~~~python
>>>a = [1,2,3]
>>> b = [4,5,6]
>>> c = [4,5,6,7,8]
>>> zipped = zip(a,b)     # 打包为元组的列表
[(1, 4), (2, 5), (3, 6)]
>>> zip(a,c)              # 元素个数与最短的列表一致
[(1, 4), (2, 5), (3, 6)]
>>> zip(*zipped)          # 与 zip 相反，*zipped 可理解为解压，返回二维矩阵式
[(1, 2, 3), (4, 5, 6)]
~~~



## tf.layers.dense

全连接层。

~~~python
tf.layers.dense(
inputs,
units,
activation=None,
use_bias=True,
kernel_initializer=None,
bias_initializer=tf.zeros_initializer(),
kernel_regularizer=None,
bias_regularizer=None,
activity_regularizer=None,
kernel_constraint=None,
bias_constraint=None,
trainable=True,
name=None,
reuse=None
)
~~~

- inputs：输入。

-  units：输出大小（维数），整数或long。

-  activation：使用什么激活函数（神经网络的非线性层），默认为None，不使用激活函数。

-  use_bias：是否使用bias，默认为True。

-  kernel_initializer：权重矩阵初始化函数。如果为None（默认值），则使用tf.get_variable默认初始化程序初始化权重。

-  bias_initializer：bias初始化函数。

-  kernel_regularizer：权重矩阵正则函数。

-  bias_regularizer：bias正则函数。

-  activity_regularizer：输出的正则函数。

- trainable：Boolean。若为True，则将变量添加到图集。

* reuse：Boolean，是否以同一名称重用前一层权重。



## numpy.eye

生成对角矩阵。

~~~python
numpy.eye(
N,
M=None, 
k=0, 
dtype=<type 'float'>
)
~~~

- N：输出方阵的规模，即行数或列数。
- k：默认情况下输出对角线全“1”，其余全“0”的方阵。如果k为正整数，则在右上方第k条对角线全“1”其余全“0”，k为负整数，则在左下方第k条对角线全“1”其余全“0”。

~~~python
>>> np.eye(2, dtype=int)
array([[1, 0],
       [0, 1]])
>>> np.eye(3, k=1)
array([[ 0.,  1.,  0.],
       [ 0.,  0.,  1.],
       [ 0.,  0.,  0.]])
~~~



## numpy切片

对于一维数组来说，list和numpy的array切片操作相同。

arr_name[start: end: step]

例如，

![1560052480182](C:\Users\Ruijia Wang\AppData\Roaming\Typora\typora-user-images\1560052480182.png)

[:]表示复制源列表，负的index表示，从后往前。-1表示最后一个元素。

![1560052558949](C:\Users\Ruijia Wang\AppData\Roaming\Typora\typora-user-images\1560052558949.png)

二维（多维）数组一般语法是arr_name[行操作, 列操作]。

~~~python
in:arr = np.arange(12).reshape((3, 4)) 

out:
array([[ 0,  1,  2,  3],
       [ 4,  5,  6,  7],
       [ 8,  9, 10, 11]])

arr[i, :] #取第i行数据
arr[i:j, :] #取第i行到第j行的数据

in:arr[:,0] # 取第0列的数据，以行的形式返回
out:
array([0, 4, 8])

in:arr[:,:1] # 取第0列的数据，以列的形式返回
out:
array([[0],
       [4],
       [8]])

# 取第一维的索引1到索引2之间的元素，也就是第二行 
# 取第二维的索引1到索引3之间的元素，也就是第二列和第三列
in:arr[1:2, 1:3] 

out: 
array([[5, 6]])


 # 取第一维的全部 
 # 按步长为2取第二维的索引0到末尾之间的元素，也就是第一列和第三列
in: arr[:, ::2]

out: 
array([[ 0,  2],
       [ 4,  6],
       [ 8, 10]])
~~~



## np.newaxis

在np.newaxis所在的位置增加一维。

例如，

~~~python
x1 = np.array([1, 2, 3, 4, 5])
# the shape of x1 is (5,)
x1_new = x1[:, np.newaxis]
# now, the shape of x1_new is (5, 1)
# array([[1],
#        [2],
#        [3],
#        [4],
#        [5]])
x1_new = x1[np.newaxis,:]
# now, the shape of x1_new is (1, 5)
# array([[1, 2, 3, 4, 5]])
~~~

~~~python
In [124]: arr = np.arange(5*5).reshape(5,5)

In [125]: arr.shape
Out[125]: (5, 5)

# promoting 2D array to a 5D array
In [126]: arr_5D = arr[np.newaxis, ..., np.newaxis, np.newaxis]

In [127]: arr_5D.shape
Out[127]: (1, 5, 5, 1, 1)
~~~



## np.empty

返回一个随机元素的矩阵，大小按照参数定义。



## size

用来统计矩阵元素个数，或矩阵某一维上的元素个数。

~~~python
numpy.size(
a, 
axis=None
) 
~~~

- a：输入矩阵。 
- axis：int型可选参数，指定返回哪一维元素个数。当没有指定时，返回整个矩阵的元素个数。

~~~python
>>> a = np.array([[1,2,3],[4,5,6]])
>>> np.size(a)
6
>>> np.size(a,1)
3
>>> np.size(a,0)
2
~~~



## 科学计数法

浮点型(floating point real values)：由整数部分与小数部分组成，可以使用科学计数法表示（2.5e2 = 2.5 x 102 = 250）



## mask softmax

![1560071132061](C:\Users\Ruijia Wang\AppData\Roaming\Typora\typora-user-images\1560071132061.png)



## tf.train.Saver

将训练好的模型参数保存起来，以便以后进行验证或测试。

模型保存，先要创建一个Saver对象：

~~~python
saver=tf.train.Saver()
~~~

创建Saver对象时，经常会用到 max_to_keep 参数，用来设置保存模型的个数，默认为5，即 max_to_keep=5，保存最近的5个模型。如果想每训练一代（epoch)就保存一次模型，则可以将 max_to_keep设置为None或0，如：

~~~python
saver=tf.train.Saver(max_to_keep=0)
~~~

但是这样做除了多占用硬盘，并没有实际多大用处，因此不推荐。如果只想保存最后一代模型，则只需要将max_to_keep设置为1，即

~~~python
saver=tf.train.Saver(max_to_keep=1)
~~~

创建完saver对象后，就可以保存训练好的模型了，如：

~~~python
saver.save(sess,'ckpt/mnist.ckpt',global_step=step)
~~~

第二个参数设定保存路径和名字，第三个参数将训练次数作为后缀加入模型名字中：

~~~python
saver.save(sess, 'my-model', global_step=0) ==> filename: 'my-model-0'
...
saver.save(sess, 'my-model', global_step=1000) ==> filename: 'my-model-1000'
~~~



## tf.group

用于创造一个操作，可将传入参数的所有操作进行分组。

~~~python
tf.group(
*inputs,
**kwargs
)
~~~

ops = tf.group(tensor1, tensor2,...) 

其中*inputs是0个或多个，用于组合tensor，一旦ops完成，传入的tensor1，tensor2，……等等都会完成。

经常用于组合训练节点，如在Cycle GAN中的多个训练节点：

~~~python
generator_train_op = tf.train.AdamOptimizer(g_loss, ...)
discriminator_train_op = tf.train.AdamOptimizer(d_loss,...)
train_ops = tf.groups(generator_train_op ,discriminator_train_op)

with tf.Session() as sess:
  sess.run(train_ops) 
  # 一旦运行了train_ops,那么里面的generator_train_op和discriminator_train_op都将被调用
~~~

注意的是，tf.group()返回的是个操作，而不是值。



## numpy.inf

表示一个无限大的正数。

~~~python
import numpy
x =  numpy.inf
x>9999999999999999999

True
~~~



## Dictionary.update

~~~python
dict.update(dict2)
~~~

- dict2 -- 添加到指定字典dict里的字典。

该方法没有任何返回值。

~~~python
dict = {'Name': 'Zara', 'Age': 7}
dict2 = {'Sex': 'female' }

dict.update(dict2)
print "Value : %s" %  dict

Value : {'Age': 7, 'Name': 'Zara', 'Sex': 'female'}
~~~

有相同的键会直接替换成 update 的值：

~~~python
>>> a = {1: 2, 2: 2}
>>> b = {1: 1, 3: 3}
>>> b.update(a)
>>> print b
{1: 2, 2: 2, 3: 3}
~~~



## 

