## 全连接层 / 线性层 Linear

~~~python
import torch
import torch.nn as nn
import torch.nn.functional as F


# torch.nn.Linear()
# torch.nn.Bilinear()   # 双线性变换

# torch.nn.Linear()
m = nn.Linear(in_features=20, out_features=30)
input = torch.randn(128, 20)
output = m(input)
print(output.size())    # (128, 30)

# torch.nn.Bilinear()
# Applies a bilinear transformation to the incoming data
m = nn.Bilinear(in1_features=20, in2_features=30, out_features=40)
input1 = torch.randn(128, 20)
input2 = torch.randn(128, 30)
output = m(input1, input2)
print(output.size())    # (128, 40)
~~~

测试：

~~~python
input = torch.arange(0, 9.0).view(3, -1)
weight = torch.linspace(0, 2, 9).view(3, -1)
bias = torch.arange(0, 9.0).view(3, -1)
out = F.linear(input, weight)
print(out)
~~~

![img](https://upload-images.jianshu.io/upload_images/11478104-f2e2b05a86911028.png?imageMogr2/auto-orient/)



## pytorch.range() v.s. pytorch.arange()

~~~python
>>> y=torch.range(1,6)
>>> y
tensor([1., 2., 3., 4., 5., 6.])
>>> y.dtype
torch.float32

>>> z=torch.arange(1,6)
>>> z
tensor([1, 2, 3, 4, 5])
>>> z.dtype
torch.int64
~~~

torch.range(start=1, end=6)的结果包含end，而torch.arange(start=1, end=6)的结果不包含end。

两者创建的tensor类型不一样。



## pytorch.linspace()

linear space的缩写，线性等分向量。

~~~python
torch.linspace(start, end, steps=100, dtype=None) → Tensor
~~~

返回一维tensor，包含从start到end，分成steps个线段得到的向量。

- start：开始值

- end：结束值

- steps：分割的点数，默认100

- dtype：张量数据类型

例如，

~~~python
import torch
print(torch.linspace(3,10,5))
#tensor([ 3.0000,  4.7500,  6.5000,  8.2500, 10.0000])


type=torch.float
print(torch.linspace(-10,10,steps=6,dtype=type))
#tensor([-10.,  -6.,  -2.,   2.,   6.,  10.])
~~~



## BatchNorm

批规范化层，分为1D、2D和3D。

![1560829082473](C:\Users\Ruijia Wang\AppData\Roaming\Typora\typora-user-images\1560829082473.png)



