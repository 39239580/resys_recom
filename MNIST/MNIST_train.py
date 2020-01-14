# -*- coding: utf-8 -*-
"""
Created on Wed Dec 25 17:19:20 2019

@author: Administrator
"""
import tensorflow as tf 
import os
from tensorflow.examples.tutorials.mnist import input_data

INPUT_NODE = 784
OUTPUT_NODE = 10
LAYER1_NODE = 500
BATCH_SIZE = 100
LEARNING_RATE_BASE = 0.8
LEARNING_RATE_DECAY = 0.99
REGULARAZTION_RATE = 0.0001
TRANING_STEP = 30000
MOVING_AVERAGE_DECAY = 0.99
MODEL_SAVE_PATH = "./model/"
MODEL_NAME = "model.ckpt"


def get_weight_variable(shape, regularizer):
    weights = tf.compat.v1.get_variable("weight", shape, initializer=tf.truncated_normal_initializer(stddev=0.1))
    if regularizer!=None:
        tf.compat.v1.add_to_collection("losses", regularizer(weights))
    return weights


def inference(input_tensor,regularizer):
    with tf.compat.v1.variable_scope("layer1", reuse=tf.compat.v1.AUTO_REUSE):
        weights = get_weight_variable([INPUT_NODE, LAYER1_NODE], regularizer)
        biases = tf.compat.v1.get_variable("biases", shape=[LAYER1_NODE],initializer=tf.constant_initializer(0.0))
        layer1 = tf.nn.relu(tf.matmul(input_tensor, weights)+biases)
    
    with tf.compat.v1.variable_scope("layer2", reuse=tf.compat.v1.AUTO_REUSE):
        weights = get_weight_variable([LAYER1_NODE, OUTPUT_NODE], regularizer)
        biases = tf.compat.v1.get_variable("biases", shape=[OUTPUT_NODE], initializer=tf.constant_initializer(0.0))
        layer2 = tf.add(tf.matmul(layer1, weights), biases, name="out")
        logits = tf.compat.v1.nn.softmax(layer2, name="logits")
    return layer2, logits


def train(mnist):
    # self.graph = tf.Graph()  可写也可以不写
    # with self.graph.as_default():
    x = tf.compat.v1.placeholder(tf.float32, [None, INPUT_NODE], name="x_input")
    y_ = tf.compat.v1.placeholder(tf.float32, [None, OUTPUT_NODE], name="y_input")
    regularizer = tf.contrib.layers.l2_regularizer(REGULARAZTION_RATE)
    y, y_porb = inference(x, regularizer)  # porb 为  输出概率值， y 为未进行操作的值

    global_step = tf.compat.v1.Variable(0, trainable=False, name="step")  # 初始话变量为0， 不可训练
    
    variable_averages = tf.train.ExponentialMovingAverage(
            MOVING_AVERAGE_DECAY, global_step)
    variables_averages_op = variable_averages.apply(tf.compat.v1.trainable_variables())
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y, labels=tf.math.argmax(y_, 1))
    cross_entropy_mean = tf.reduce_mean(cross_entropy)
    loss = cross_entropy_mean + tf.add_n(
        tf.get_collection("losses"))
    tf.add_to_collection("loss", loss)
    learning_rate = tf.compat.v1.train.exponential_decay(
            LEARNING_RATE_BASE,
            global_step,
            mnist.train.num_examples/BATCH_SIZE,
            LEARNING_RATE_DECAY)
    train_step = tf.compat.v1.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)
    # tf.add_to_collection("step", train_step)
    with tf.control_dependencies([train_step, variables_averages_op]):
        train_op = tf.no_op(name="train")
    
    # 初始化
    saver = tf.compat.v1.train.Saver(max_to_keep=5)  # 默认为5个模型。更改为保存最多100个模型
    with tf.Session() as sess:
        tf.global_variables_initializer().run()   
        # 等价于 init_opt = tf.global_variables_initialzer()
        # sess.run(init_opt)
        
        # 加入断点继续训练
        ckpt = tf.train.get_checkpoint_state(MODEL_SAVE_PATH)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
            print("Model restored...")
        else:
            print("模型不存在， 需要重新训练")
        for i in range(TRANING_STEP):
            XS, YS = mnist.train.next_batch(BATCH_SIZE)  # 训练集数据
            _, loss_value, step = sess.run([train_op, loss, global_step], feed_dict={x: XS, y_: YS})
            
            # 每一千轮保存一次模型
            if i % 1000 == 0:
                saver.save(sess, os.path.join(MODEL_SAVE_PATH, MODEL_NAME),
                           global_step=global_step)
                print(global_step)
                print(type(global_step))
                print("After %d traning step(s),loss on training batch is %g" % (step, loss_value))


def breakpoint_training_v1(mnist):
    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        ckpt = tf.compat.v1.train.get_checkpoint_state(MODEL_SAVE_PATH)  # 检查模型状态
        if ckpt and ckpt.model_checkpoint_path:  # 模型文件存在
            saver = tf.compat.v1.train.import_meta_graph(ckpt.model_checkpoint_path+".meta")   # 将文件中的网络加载到当前的图上
            graph = tf.get_default_graph()  # 获取整个计算图出来
            saver.restore(sess, ckpt.model_checkpoint_path)  # 模型加载完毕，  此方式不需要再进行计算图的定义
            name = graph.get_all_collection_keys()  # 查看所有集合名
            print(name)   # 输出 ['trainable_variables', 'cond_context', 
            # 'losses', 'moving_average_variables', 'train_op', 'variables']
            ops = graph.get_operations()   # 查看所有的操作
            # 测试使用，查看所有操作部分，
            for op in ops:  # 查看所有的操作部分
                print(op.name, op.values())
            print(ops)   # 很多变量  查找到 很多需要的参数
            print("//////")
            print(graph.get_collection("train_op"))  # optimizer
            print(graph.get_collection("losses"))  # 损失集合
            print(graph.get_collection("variables"))  # 所有变量
            print(graph.get_collection("trainable_variables"))  # 可训练变量集合
            x = graph.get_tensor_by_name("x_input:0")  # 这些事输入与输出
            y_ = graph.get_tensor_by_name("y_input:0")
            pred = graph.get_tensor_by_name("layer2/out:0")   # 预测的输出
            print(x)
            print(y_)
            print(pred)
            current_step = graph.get_tensor_by_name("step:0").eval()   # 检查断点代数
            print(type(current_step))
            print("当前训练代数:%d" % current_step)
            step = graph.get_tensor_by_name("step:0")
            train_op = graph.get_collection('train_op')  # 优化器放在一个集合中
            loss = graph.get_collection("loss")  # loss也放在一个集合中
            for i in range(5000):   # 继续训练
                XS, YS = mnist.train.next_batch(BATCH_SIZE)  # 训练集数据
                _, loss_value, steps = sess.run([train_op, loss, step], feed_dict={x: XS, y_: YS})
                # 每一千轮保存一次模型
                if i % 1000 == 0:
                    saver.save(sess, os.path.join(MODEL_SAVE_PATH, MODEL_NAME), global_step=steps)
                    print("After %d traning step(s),loss on training batch is %g" % (steps, loss_value[0]))

            # 进行预测  预测10批数据
            for i in range(10):
                x_test, _ = mnist.train.next_batch(BATCH_SIZE)  #
                out = sess.run([pred], feed_dict={x: x_test})
                print(out)

def main(argv=None):
    mnist = input_data.read_data_sets("./MNIST_data", one_hot=True)  # 下载的数据集
    tf.compat.v1.reset_default_graph()  # 重置计算图
    train(mnist)
#    breakpoint_training_v1(mnist)
    

if __name__ == "__main__":
    main()
    # 或者使用 tf.app.run()
    # 若使用tf.app.run(),则需要多main进行改造处理，即main 中需要加入mian(argv =None)，否则会报错
