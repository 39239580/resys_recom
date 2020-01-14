import tensorflow as tf
import numpy as np
# from tensorflow.examples.tutorials.mnist import *
# from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.examples.tutorials.mnist import input_data

# from tensorflow.models.offic
# from tensorflow.


"""
TFrecord  支持的数据类型有三种
字符串，整数，浮点型， Int64List,
BytestList, FloatList
"""


#  生成整数型的属性
def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


# 生成字符串型的属性
def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


# 生成浮点型的属性
def _float_feature(value):
    return tf.train.Feature(floatt_list=tf.train.FloatList(value=[value]))


def creat_tf_example(params, data_type=None):
    # writer = tf.python_io.TFRecordWriter(params["filename"])  # 旧版本
    writer = tf.io.TFRecordWriter(params["filename"])  # 旧版本
    if not data_type:  # 图像数据
        for index in range(params["num_example"]):
            # 将图像矩阵转成一个字符串
            image_raw = params["data_obj"][index].tostring()
            # 将一个样例转化成 Example Protocol Buffer , 并将所有信息写入这个数据结构
            tf_example = creat_image_format(params["pixels"], params["label"], image_raw)
            writer.write(tf_example.SerializeToString())  # 将一个Example写入TFRecord文件
        writer.close()


def creat_image_format(pixels, labels, image_raw):  # 图片的
    tf_example = tf.train.Example(features=tf.train.Features(
        feature={"image_raw": _bytes_feature(image_raw),
                 "label": _int64_feature(np.argmax(labels)),
                 "pixels": _int64_feature(pixels)
                 }
    )
    )
    return tf_example




def TFrecord_image():
    mnist = input_data.read_data_sets("./mnist_data/data", dtype=tf.uint8, one_hot=True)
    images = mnist.train.images
    labels = mnist.train.labels
    # 图像分辨率
    pixels = images.shape[1]
    num_example = mnist.train.num_examples
    filename = "./mnist_data/output.tfrecords"   #  保存成 2进制文件
    params = {"filename": filename,
              "label": labels,
              "pixels": pixels,
              "num_example": num_example,
              "data_obj": images
              }
    creat_tf_example(params)


TFrecord_image()


