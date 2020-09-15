## TFRecord详解

### TFRecord简介

TFRecord 是Google官方推荐的一种数据格式，是Google专门为TensorFlow设计的一种数据格式。

实际上，TFRecord是一种二进制文件，其能更好的利用内存，其内部包含了多个**tf.train.Example**， 而Example是protocol buffer数据标准的实现，在一个Example消息体中包含了一系列的**tf.train.feature**属性，而 每一个feature 是一个key-value的键值对，其中，key 是string类型，而value 的取值有三种：

- bytes_list: 可以存储string 和byte两种数据类型。

- float_list: 可以存储float(float32)与double(float64) 两种数据类型 。

- int64_list: 可以存储：bool, enum, int32, uint32, int64, uint64 。

值的一提的是，TensorFlow 源码中到处可见 .proto 的文件，且这些文件定义了TensorFlow重要的数据结构部分，且多种语言可直接使用这类数据，很强大。



### 为什么用TFRecord？

TFRecord 并非是TensorFlow唯一支持的数据格式，你也可以使用CSV或文本等格式，但是对于TensorFlow来说，TFRecord 是最友好的，也是最方便的。前面提到，TFRecord内部是一系列实现了protocol buffer数据标准的Example。对于大型数据，相比其余数据格式，protocol buffer类型的数据优势很明显。



在数据集较小时，我们会把数据全部加载到内存里方便快速导入，但当数据量超过内存大小时，就只能放在硬盘上来一点点读取，这时就不得不考虑数据的移动、读取、处理等速度。使用TFRecord就是为了**提速和节约空间**的。

参考：https://halfrost.com/protobuf_encode/

https://zhuanlan.zhihu.com/p/50808597



### TFRecord格式

TFRecord 可以理解为一系列序列化的 tf.train.Example 元素所组成的列表文件，而每一个 tf.train.Example 又由若干个 tf.train.Feature 的字典组成。

```python
[
    {   # example 1 (tf.train.Example)
        'feature_1': tf.train.Feature,
        ...
        'feature_k': tf.train.Feature
    },
    ...
    {   # example N (tf.train.Example)
        'feature_1': tf.train.Feature,
        ...
        'feature_k': tf.train.Feature
    }
]

```

- bytes_list: 可以存储string 和byte两种数据类型。

- float_list: 可以存储float(float32)与double(float64) 两种数据类型 。

- int64_list: 可以存储：bool, enum, int32, uint32, int64, uint64 。



- int64_list: tf.train.Feature(int64_list = tf.train.Int64List(value=输入))

- float_list: tf.train.Feature(float_list = tf.train.FloatList(value=输入))

- bytes_list ：tf.train.Feature(bytes_list=tf.train.BytesList(value=输入))

​    注：输入必须是list(向量)

### 写入TFRecord文件

为了将形式各样的数据集整理为 TFRecord 格式，我们可以对数据集中的每个元素进行以下步骤：

1. 读取该数据元素到内存；
2. 建立 Feature 的字典；
3. 将该元素转换为 tf.train.Example 对象（每一个 tf.train.Example 由若干个 tf.train.Feature 的字典组成）；
4. 将该 tf.train.Example 对象序列化为字符串，并通过一个预先定义的 tf.io.TFRecordWriter 写入 TFRecord 文件。

注意: tensorflow feature类型只接受list数据，但如果数据类型是矩阵或者张量该如何处理？

- 转成list类型：将张量flatten成list(也就是向量)，再用写入list的方式写入。

- 转成string类型：将张量用.tostring()转换成string类型，再用tf.train.Feature(bytes_list=tf.train.BytesList(value=[input.tostring()]))来存储。

### 读取TFRecord文件

而读取 TFRecord 数据则可按照以下步骤：

1. 通过 tf.data.TFRecordDataset 读入原始的 TFRecord 文件（此时文件中的 tf.train.Example 对象尚未被反序列化），获得一个 tf.data.Dataset 数据集对象；
2. 定义Feature结构，告诉解码器每个Feature的类型是什么；
3. 通过 Dataset.map 方法，对该数据集对象中的每一个序列化的 tf.train.Example 字符串执行 **tf.io.parse_single_example** 函数，从而实现反序列化。