import tensorflow as tf
import numpy as np
# import pandas as pd
from time import sleep, time
from yellowfin import YFOptimizer
from sklearn.metrics import roc_auc_score
from tensorflow.contrib.layers import batch_norm, l2_regularizer
import os
from code_tools.FM_util.XdeepFM_batch_next import BacthDataset


class XDeepFM(object):
    def __init__(self,
                 feature_size,   # one_hot 之后的维度， M
                 field_size,  # 未编码前的维度 F
                 embedding_size=8,  # 嵌入层  k embeding 之后的长度
                 dropout_cin=[1.0, 1.0],   # cin 第一层的链接部分
                 deep_layers=[32, 32],   # DNN部分的 节点数
                 dropout_deep=[0.5, 0.5, 0.5],  # DNN部分的 dropout层
                 cin_param=[12, 12],  # cin 部件中每个部件 的行数
                 deep_layers_activattion="relu",
                 batch_size=64,
                 learning_rate=0.001,
                 optimizer_type="adam",
                 batch_norms=True,   #采用BN 层进行操作
                 bact_norm_decay=0.995,
                 random_seed=22,
                 use_cin=True,
                 use_deep=True,
                 loss_type="logloss",
                 eval_metric=roc_auc_score,
                 l2_lambda=0.0,
                 direct=False,
                 greater_is_better=True,
                 ifregularizer=True,  # 开启正则化
                 movingaverage=False,  # 滑动平均关闭
                 exponentialdecay=False,  # 指数衰减关闭
                 traning_step=1000,  # 总共的训练代数
                 reduce_d=True,
                 cin_activation="sigmod",
                 moving_average_decay=0.99,  # 滑动平均系数
                 learning_rate_decay=0.99,  # 指数衰减系数
                 regularaztion_rate=0.0001,
                 num_example=None,
                 checkpoint_training=True,  # 自动加入继训练功能
                 model_save_path=None,
                 evaluate_type="all"
                 ):
        assert (use_deep or use_cin)  # 确保其中为真, 为真才会执行
        assert (loss_type in ["logloss", "mse"]),\
            "loss_type can be either 'logloss' for classification task or 'mse' for regression task"
        assert (model_save_path is not None)  # 保证不能为空
        self.feature_size = feature_size   # M  one_hot 之后的长
        self.field_size = field_size  # F  one_hot编码之前的长度
        self.embedding_size = embedding_size  # k维
        self.dropout_cin = dropout_cin  # ?
        self.deep_layers = deep_layers
        self.dropout_deep = dropout_deep
        self.deep_layers_activattion = deep_layers_activattion
        self.batch_size = batch_size
        self.learning_rate_base = learning_rate
        self.optimizer_type = optimizer_type
        self.batch_norm = batch_norms
        self.batch_norm_decay = bact_norm_decay
        self.random_seed = random_seed
        self.use_cin = use_cin
        self.use_deep = use_deep
        self.loss_type = loss_type
        self.eval_metric = eval_metric
        self.l2_lambda = l2_lambda
        self.greater_is_better = greater_is_better
        self.direct = direct
        self.movingaverage = movingaverage
        self.exponentialdecay = exponentialdecay
        self.traning_step = traning_step
        self.cin_param = cin_param
        self.reduce_D = reduce_d
        self.cin_activation = cin_activation
        self.moving_average_decay = moving_average_decay
        self.learning_rate_decay = learning_rate_decay
        self.ifregularizer = ifregularizer
        self.regularizer = l2_regularizer(regularaztion_rate)  # 正则化默认参数
        self.num_example = num_example
        self.checkpoint_training = checkpoint_training
        self.model_save_path = model_save_path
        self.evaluate_type = evaluate_type

    def _embeding_net(self):   # embedding 部分网络进行定义
        with tf.compat.v1.variable_scope("embeddings", reuse=tf.compat.v1.AUTO_REUSE):
            feature_embeddings_weights = tf.compat.v1.get_variable(name="embeddings_weights",
                                                                   shape=[self.feature_size, self.embedding_size],
                                                                   # F *k
                                                                   initializer=tf.compat.v1.random_normal_initializer(
                                                                       mean=1.0,
                                                                       stddev=0.01,
                                                                       dtype=tf.float32
                                                                   ))
            if self.ifregularizer:   # 使用正则化处理
                tf.compat.v1.add_to_collection("losses", self.regularizer(feature_embeddings_weights))

            feature_embeddings_bias = tf.compat.v1.get_variable(name="embeddings_bias",
                                                                shape=[self.embedding_size],
                                                                initializer=tf.compat.v1.random_uniform_initializer(
                                                                    minval=0.0,
                                                                    maxval=1.0,
                                                                    dtype=tf.float32
                                                                ))
        return feature_embeddings_weights, feature_embeddings_bias

    def _deep_net(self):   # 深度部分网络结构进行定义
        layer_num = len(self.deep_layers)  # 层数
        input_size = self.field_size * self.embedding_size   # 每个字段一个embedding

        self.y_deep = tf.reshape(self.embeddings, shape=[-1, self.field_size, self.embedding_size])
        self.y_deep = tf.compat.v1.nn.dropout(self.y_deep, self.dropout_deep[0])  # 即第一部分的

        for i in range(layer_num):  # 遍历所有的层
            if i == 0:
                glorot = np.sqrt(2.0 / (input_size + self.deep_layers[0]))  # 标准差
            else:
                glorot = np.sqrt(2.0 / (self.deep_layers[i-1] + self.deep_layers[i]))  # 标准差

            with tf.compat.v1.variable_scope("deep_layer_%d" % i, reuse=tf.compat.v1.AUTO_REUSE):
                deep_weights = tf.compat.v1.get_variable(name="deep_weights_%d" % i,
                                                         shape=[self.deep_layers[i-1], self.deep_layers[i]],
                                                         initializer=tf.compat.v1.random_normal_initializer(
                                                             mean=0.0,
                                                             stddev=glorot,
                                                             dtype=tf.float32
                                                         ))
                if self.ifregularizer:  # 如果正则化
                    tf.compat.v1.add_to_collection("losses", self.regularizer(deep_weights))  # 损失函数加载在集合内
                    # 正则化只处理  权重部分

                deep_bias = tf.compat.v1.get_variable(name="deep_bias_%d" % i,
                                                      shape=[self.deep_layers[i]],
                                                      initializer=tf.compat.v1.random_uniform_initializer(  # 均匀分布
                                                          minval=0.0,
                                                          maxval=1.0,
                                                          dtype=tf.float32
                                                      ))
                self.y_deep = tf.add(tf.matmul(self.y_deep, deep_weights), deep_bias)
                if self.batch_norm:  # 是否采用BN层
                    self.y_deep = self._batch_norm_layer(self.y_deep, train_phase=self.train_phase,
                                                         scope_bn="bn_%d" % i)

                self.y_deep = self._activate(self.y_deep, self.deep_layers_activattion)
                # self.y_deep = self.deep_layers_activattion(self.y_deep)
                self.y_deep = tf.compat.v1.nn.dropout(self.y_deep, self.dropout_deep[i+1])   # 加上dropout 层

        return self.y_deep

    def _cin_net(self, bias=False):  # cin 网络结构
        """
        :param bias  # 是否带上偏置
        :return:
        layer_params  为每层的size， 列表
        """
        hidden_nn_layers = []  # 存放每层的输入
        field_nums = []   # 用于存放每层的特征数，即Hk的行数
        final_len = 0  # 输出的长度
        field_num = self.field_size  # 字段个数,即H0,x0的行数
        field_nums.append(int(field_num))
        hidden_nn_layers.append(self.embeddings)
        final_result = []
        self.split_tensor_h0 = tf.split(hidden_nn_layers[0], self.embedding_size*[1], 2)
        # 进行拆分， 拆分成k个 batch*h0*1 小张量
        # self.layer_params   为每层需要确定的行， Hk
        for idx, layer_size in enumerate(self.cin_param):
            # 人工指定特征数 layer_size
            # idx为层索引
            split_tensor_hk = tf.split(hidden_nn_layers[-1], self.embedding_size*[1], 2)
            # x^k-1   #  张量变成k个  batch*H_(K)*1   i个小张量,即field个数
            dot_result_hkm = tf.matmul(split_tensor_hk, self.split_tensor_h0, transpose_b=True)
            # 得到D*batch*H_(K)*h0 矩阵 D*batch*H_(K)*m
            dot_result_hko = tf.reshape(dot_result_hkm, shape=[self.embedding_size, -1, field_num[-1]*field_num[0]])
            # 将上述尺寸变成D*batch*(H_(K)*m)

            dot_result = tf.transpose(dot_result_hko, perm=[1, 0, 2])
            #   转置 batch*D*(H_(K)*m)  小张量

            filter_param = {"idx": idx, "layer_size": layer_size, "field_nums": field_nums}

            cin_second = tf.nn.conv1d(dot_result, filters=self._filter_cin(filter_param), stride=1, padding="VALID")
            # 一维卷积操作，  # batch*D*(H_(K)*m)
            # tf.nn.conv1d(value,filters,stride,padding)   # value  的形状[batch, in_width, in_chanels]
            # 依次为[batch为样本数量,宽度，样本的通道数] batch*D
            # filter =[filter_width, in_channels, out_channels]
            # 输出为 [batch, out_width,out_channels]      batch *HK*D

            cin_unit = tf.transpose(cin_second, perm=[0, 2, 1])  # 转置得到 batch*(H_(K)*D

            if bias:  # 含有偏置
                with tf.compat.v1.variable_scope("cin_bias", reuse=tf.compat.v1.AUTO_REUSE):
                    cin_bias = tf.compat.v1.get_variable(name="f_b"+str(idx),
                                                         shape=[layer_size],
                                                         initializer=tf.compat.v1.zeros_initializer(),
                                                         dtype=tf.float32)
                    cin_unit = tf.compat.v1.nn.bias_add(cin_unit, cin_bias)  # 带上偏置 batch*(H_(K)*D

            cin_unit = self._activate(cin_unit, self.cin_activation)  # batch*(H_(K)*D

            cin_unit = tf.compat.v1.transpose(cin_unit, perm=[0, 2, 1])  # bacth*D*(H_(K))

            # 再进行sum_poooling 操作
            if self.direct:
                # direct 方式， e=[e1,e2......em]  将输出全部直接连接到最后的输出上。
                direct_connect = cin_unit
                next_hidden = cin_unit
                final_len += layer_size
                field_nums.append(int(layer_size))  # 添加特征数进去

            else:
                # 非direct 方式， 将输出结果按照layer_size 进行均分，前一半作为计算下一个隐含层向量的输入，后一般作为
                # 最后输出结果的一部分
                if idx != len(self.cin_param)-1:
                    # 非最后一层
                    next_hidden, direct_connect = tf.split(cin_unit, 2*[int(layer_size/2)], 1)  # 取一半值，将D进行分解
                    # 一半用于下一层输入，一半丢给输出
                    final_len += int(layer_size/2)
                else:
                    # 最后一层
                    direct_connect = cin_unit  # 最后一层
                    next_hidden = 0
                    final_len += layer_size
                field_nums.append(int(layer_size/2))  # 特征行进去
            final_result.append(direct_connect)   # 每个cin模块的输出
            hidden_nn_layers.append(next_hidden)  # 存放下一层的输入
        cin = tf.compat.v1.concat(final_result, axis=1)  # 按列 进行拼接
        cin_out = tf.compat.v1.reduce_sum(cin, -1)  # 维度内进行相加
        cin_out = tf.compat.v1.nn.dropout(cin_out, keep_prob=self.dropout_keep_cin[-1])
        return cin_out

    def _filter_cin(self, params, f_dim=2):
        """
        :param params:  参数  dict
        :param f_dim:
        :return:
        """
        idx = params["idx"]
        layer_size = params["layer_size"]
        field_nums = params["field_nums"]
        if self.reduce_D:
            with tf.compat.v1.variable_scope("reduce_D", reuse=tf.compat.v1.AUTO_REUSE):
                filters0 = tf.compat.v1.get_variable("f0_"+str(idx),
                                                     shape=[1, layer_size, field_nums[0], f_dim],
                                                     dtype=tf.float32
                                                     )  # 1*N*m*k
                filters_ = tf.compat.v1.get_variable("f_"+str(idx),
                                                     shape=[1, layer_size, f_dim, field_nums[-1]],
                                                     dtype=tf.float32
                                                     )  # 1*N*k*hm
                filters_m = tf.matmul(filters0, filters_)  # 得到1*N*m*hm

                filters_o = tf.reshape(filters_m, shape=[1, layer_size, field_nums[0]*field_nums[-1]])

                filters = tf.transpose(filters_o, perm=[0, 2, 1])   # 变成1 *(m*hm)*N

        else:
            with tf.compat.v1.variable_scope("None_reduce_D", reuse=tf.compat.v1.AUTO_REUSE):
                filters = tf.compat.v1.get_variable(name="f_"+str(idx),
                                                    shape=[1, field_nums[-1]*field_nums[0], layer_size],
                                                    dtype=tf.float32)
        return filters
        # self.filters 用于一维3D卷积  kernel_size(width),  in_channel, out_channel
        # 输入为batch, width, in_channel    卷积后  batch, out_width, out_channels

    def _concat_net(self):   # 拼接层

        with tf.compat.v1.variable_scope("concat_lay", reuse=tf.compat.v1.AUTO_REUSE):
            if self.use_cin and self.use_deep:  # 使用XDeepFM
                concat_input = tf.compat.v1.concat([self.embedding_out, self.cin_out, self.deep_out], axis=1)
                self.input_size = self.field_size + self.embedding_size+self.deep_layers[-1]
                # 按列进行拼接

            elif self.use_cin:  # 只使用CIN 部分
                concat_input = tf.compat.v1.concat([self.embedding_out, self.cin_out], axis=1)  # 按列进行拼接
                self.input_size = self.field_size + self.embedding_size

            elif self.use_deep:  # 使用Deep问题
                concat_input = tf.compat.v1.concat([self.embedding_out], axis=1)  #
                self.input_size = self.deep_layers[-1]

            else:
                raise ValueError("this network is undefined{0},{1}".format(self.use_deep, self.use_cin))

            glorot = np.sqrt((2.0 / self.input_size + 1))  # 标准差
            weights = tf.compat.v1.get_variable(name="weights",
                                                shape=[self.input_size],
                                                initializer=tf.compat.v1.random_normal_initializer(
                                                    mean=0.0,
                                                    stddev=glorot,
                                                    dtype=tf.float32
                                                    ))
            bias = tf.compat.v1.get_variable(name="bias",
                                             initializer=tf.compat.v1.constant_initializer(0.01),
                                             dtype=tf.float32
                                             )

            out = tf.compat.v1.add(tf.matmul(concat_input, weights), bias)

        return out

    def _get_predict(self, logits, task_class="classify"):  # 获取最后的输出类型
        if task_class == "classify":
            activate_fn = "sigmoid"
            self.pred = self._activate(logits, activate_fn)
            # 做的操作是对tensor logits  进行复制的操作，详细
            # 可见https://blog.csdn.net/qq_23981335/article/details/81361748

        elif task_class == "regression":
            activate_fn = "identity"
            self.pred = self._activate(logits, activate_fn)
            # 默认采用sigmoid 作为激活函数

        else:
            raise ValueError("task_class must be regression or classify")

    def _loss(self, logits):   # 损失函数
        self._get_predict(logits=logits)
        if self.loss_type == "logloss":
            self.loss = tf.losses.log_loss(labels=self.label, predictions=self.pred)
            self.loss = tf.compat.v1.reduce_mean(self.loss)  # 误差求均值求均值

        elif self.loss_type == "mse":
            self.loss = tf.compat.v1.nn.l2_loss(tf.subtract(self.label, self.pred))
            self.loss = tf.compat.v1.reduce_mean(self.loss)  # 误差求均值求均值

        elif self.loss_type == "cross_entropy":  # 交叉熵
            self.loss = tf.compat.v1.nn.softmax_cross_entropy_with_logits(labels=self.label, logits=self.pred)
            self.loss = tf.compat.v1.reduce_mean(self.loss)  # 误差求均值求均值

        elif self.loss_type == "square_loss":
            self.loss = tf.compat.v1.sqrt(tf.reduce_mean(
                tf.compat.v1.squared_difference(tf.reshape(self.pred, [-1]),
                                                tf.reshape(self.label, [-1]))
            ))

    def _batch_norm_layer(self, x, train_phase, scope_bn):   # BN层
        bn_train = batch_norm(x, decay=self.batch_norm_decay, center=True, scale=True,
                              updates_collections=None, is_training=True, reuse=None,
                              trainable=True, scope=scope_bn)

        bn_inference = batch_norm(x, decay=self.batch_norm_decay, center=True, scale=True,
                                  updates_collections=None, is_training=True, reuse=True,
                                  trainable=True, scope=scope_bn)
        return tf.cond(train_phase, lambda: bn_train, lambda: bn_inference)
    # tf.cond  使用
    # tf.cond(x， lambda :A, lambda :B)    x为真， 输出A， x 为假，输出B

    """
    tf.contrib.layers.batch_norm(input,   #输入
    decay=0.999, # 衰减系数
    center=True, # 如果为True,  有beta偏移量； 如果为False,无beta偏移量
    epsilon=0.001, # 避免被零除
    scale=False, #如果为True, 则乘以gamma。如果为False, gamma 则不使用。当下一层是线性时，缩放层可以在下一层完成。可以禁止该层
    param_initializers=None, #参数初始化
    activation_fn=None, #用于激活，默认为线性激活函数
    updates_colletions= tf.GraphKeys.UPDATE_OPS,
    param_regularizers=None,  #beta和gamma 正则化优化,is_training=True.  # 是否使用训练层
    outputs_collections =None,
    reuse =None,
    variables_collections =None,
    data_format= DATA_FORMAT_NHWC,
    trainable= True,batch_weights=None, fused=None,
    zero_debias_moving_mean= False,  提高稳定性，设为True,
    scope=None, 
    renorm=False,
    renorm_clipping= None, 
    renorm_decay=0.99,
    adjustment=None)
    """

    def _optimizer(self, global_step):   # 选择优化器
        if self.optimizer_type == "adam":
            optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=self.learning_rate,
                                                         beta1=0.9, beta2=0.999, epsilon=1e-8
                                                         ).minimize(self.loss, global_step=global_step)

        elif self.optimizer_type == "adadelta":
            optimizer = tf.compat.v1.train.AdadeltaOptimizer(learning_rate=self.learning_rate
                                                             ).minimize(self.loss, global_step=global_step)

        elif self.optimizer_type == "adagrad":
            optimizer = tf.compat.v1.train.AdagradOptimizer(learning_rate=self.learning_rate,
                                                            initial_accumulator_value=1e-8
                                                            ).minimize(self.loss, global_step=global_step)

        elif self.optimizer_type == "gd" or self.optimizer_type == "sgd":
            optimizer = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=self.learning_rate
                                                                    ).minimize(self.loss, global_step=global_step)

        elif self.optimizer_type == "ftrl":
            optimizer = tf.compat.v1.train.FtrlOptimizer(learning_rate=self.learning_rate
                                                         ).minimize(self.loss, global_step=global_step)

        elif self.optimizer_type == "padagrad":
            optimizer = tf.compat.v1.train.ProximalAdagradOptimizer(learning_rate=self.learning_rate
                                                                    ).minimize(self.loss, global_step=global_step)

        elif self.optimizer_type == "pgd":
            optimizer = tf.compat.v1.train.ProximalGradientDescentOptimizer(learning_rate=self.learning_rate
                                                                            ).minimize(self.loss,
                                                                                       global_step=global_step)

        elif self.optimizer_type == "rmsprop":
            optimizer = tf.compat.v1.train.RMSPropOptimizer(learning_rate=self.learning_rate
                                                            ).minimize(self.loss, global_step=global_step)

        elif self.optimizer_type == "momentum":
            optimizer = tf.compat.v1.train.MomentumOptimizer(learning_rate=self.learning_rate,
                                                             momentum=0.2
                                                             ).minimize(self.loss, global_step=global_step)

        elif self.optimizer_type == "yellowfin":
            optimizer = YFOptimizer(learning_rate=self.learning_rate,
                                    momentum=0.0).minimize(self.loss, global_step=global_step)

        else:
            raise ValueError("this optimizer is undefined{0}".format(self.optimizer_type))

        return optimizer

    def _skill(self, global_step, decay_step):  # 开启滑动平均
        if self.movingaverage:  # 开启滑动平均
            variable_averages = tf.compat.v1.train.ExponentialMovingAverage(
                self.moving_average_decay, global_step)
            self.variable_averages_op = variable_averages.apply(tf.compat.v1.trainable_variables())  # 确保

        if self.exponentialdecay:  # 开启指数衰减
            self.learning_rate = tf.compat.v1.train.exponential_decay(
                learning_rate=self.learning_rate_base,
                global_step=global_step,
                decay_steps=decay_step,
                decay_rate=self.learning_rate_decay)
        else:
            self.learning_rate = self.learning_rate_base

    @staticmethod  # 静态方法
    def _activate(logit, activation):  # 激活函数
        if activation == "sigmoid":
            return tf.compat.v1.nn.sigmoid(logit)
        elif activation == "softmax":
            return tf.compat.v1.nn.softmax(logit)
        elif activation == "relu":
            return tf.compat.v1.nn.relu(logit)
        elif activation == "tanh":
            return tf.compat.v1.nn.tanh(logit)
        elif activation == "elu":
            return tf.compat.v1.nn.elu(logit)
        elif activation == "identity":
            return tf.compat.v1.identity(logit)
        elif activation == "leaky_relu":
            return tf.compat.v1.nn.leaky_relu(logit)
        else:
            raise ValueError("this activations not defined {0}".format(activation))

    def _inference(self, prob=False):   # 定义前向推断
            feature_embeddings_weights, feature_embeddings_bias = self._embeding_net()

            # embedding part
            with tf.compat.v1.variable_scope("embedding_part", reuse=tf.compat.v1.AUTO_REUSE):
                # model feat_index  为特征嵌入的索引值， F为field个数，k为每个field转成长度为k的嵌入
                self.embeddings = tf.compat.v1.nn.embedding_lookup(feature_embeddings_weights, self.feat_index)
                # None *F*k
                # 特征嵌入矩阵 M*k, feat_index 为[None, F个]
                self.feat_value = tf.reshape(self.feat_value, shape=[-1, self.field_size, self.embedding_size])
                # 变成None *F *k 2维变成三维
                self.embeddings = tf.multiply(self.embeddings, self.feat_value)
                # 两个矩阵中对应元素相乘, 得到嵌入层的输出   None *F*k,   None 页。 F行， 1列。
                # 每行为一个filed的嵌入向量

            # linear part
            with tf.compat.v1.variable_scope("linear_part", reuse=tf.compat.v1.AUTO_REUSE):
                self.line_out = tf.compat.v1.nn.embedding_lookup(feature_embeddings_bias, self.feat_index)  # None*F*1,
                # 从M*1的矩阵中去除F行，组成None*F*1 的矩阵/张量
                # 做的是FM 中的<wx> 操作，无偏置
                self.embedding_out = tf.reduce_sum(tf.multiply(self.line_out, self.feat_value), 2)  # 页内列相加，None*F*1
                self.embedding_out = tf.compat.v1.nn.dropout(self.embedding_out, self.dropout_keep_cin[0])

            # cin component
            with tf.compat.v1.variable_scope("cin_component", reuse=tf.compat.v1.AUTO_REUSE):
                self.cin_out = self._cin_net()

            # Dnn part
            with tf.compat.v1.variable_scope("DNN_component", reuse=tf.compat.v1.AUTO_REUSE):
                self.deep_out = self._deep_net()

            # 进行拼接
            logits = self._concat_net()      # 直接得到输出值

            if prob:   # 如果需要输出概率值
                logits = tf.compat.v1.nn.softmax(logits)

            return logits

    def early_termination(self, valid_result, threshold_val=10, auto=True):  # 提前终止
        """
        有验证集才能开启提前终止，否则进行报错
        :param valid_result:  # 验证集
        :param threshold_val:   # 阈值
        :param auto:  #默认 为手动
        :return:
        """
        if auto:   # 使用的是loss 来进行操作
            if len(valid_result) > threshold_val:  # 必须保证有效数据大于10条，否则不必开启

                if (abs(valid_result[-1] - valid_result[-2]) < 0.001) and \
                    (abs(valid_result[-2] - valid_result[-3]) < 0.001) and \
                    (abs(valid_result[-3] - valid_result[-4]) < 0.001) and \
                    (abs(valid_result[-4] - valid_result[-5]) < 0.001) and \
                        (valid_result[-1] < valid_result[0]):
                    return True
                else:
                    return False
        else:
            if len(valid_result) > threshold_val:  # 必须保证有效数据大于10条，否则不必开启
                if self.greater_is_better:  # 越大越好  适合 acc, auc
                    if (valid_result[-1] < valid_result[-2]) and \
                        (valid_result[-2] < valid_result[-3]) and \
                            (valid_result[-3] < valid_result[-4]) and \
                            (valid_result[-4] < valid_result[-5]):
                        return True
                else:   # 越小越好
                    if (valid_result[-1] > valid_result[-2]) and \
                        (valid_result[-2] > valid_result[-3]) and \
                            (valid_result[-3] > valid_result[-4]) and \
                            (valid_result[-4] > valid_result[-5]):
                        return True
            else:
                return False

    def set_on_batch(self, xi, xv, y, train_phase):   # 组装数据
        feed_dict = {self.feat_index: xi,
                     self.feat_value: xv,
                     self.label: y,
                     # self.dropout_keep_cin: self.dropout_cin,
                     # self.dropout_keep_deep: self.dropout_keep_deep,
                     self.train_phase: train_phase}
        return feed_dict

    def evaluate(self, real_label, evaluate_type="acc"):   # 评估模块
        """
        :param real_label:
        :param evaluate_type:
        :return:
        """
        if evaluate_type == "acc" or evaluate_type == "all":
            cor_pre = tf.compat.v1.equal(tf.argmax(self.pred, 1), tf.argmax(real_label, 1),
                                         name="correct_predict")
            acc = tf.reduce_mean(tf.cast(cor_pre, tf.float32), name="acc")  # 强制转换类型
        else:
            cor_pre = None
            acc = None

        if evaluate_type == "auc" or evaluate_type == "all":
            auc = self.eval_metric(y_true=real_label, y_score=self.pred)
        else:
            auc = None
        if self.evaluate_type == "all":
            cal_op = [self.loss, cor_pre, acc, auc]
        elif self.evaluate_type == "acc":
            cal_op = [self.loss, cor_pre, acc]
        elif self.evaluate_type == "auc":
            cal_op = [self.loss, auc]
        else:
            cal_op = [self.loss]
        return cal_op

    def _train(self, data, save_params, every_save=1000, every_eval=1, early_stop=False,
               train_test_split=False, autoevaluate=True):
        # 默认关闭操作
        """
        :param data:  为字典   {x:XS, y_:YS}
        :param save_params: 参数  dict
        :param every_save: 默认值为1000， 1000轮保存一次
        :param every_eval: 默认值为1轮，一轮评估一次
        :param early_stop: 是否开启提前终止
        :param train_test_split:  是否需要开启 验证集划分
        :param autoevaluate:  # 是否自动开启评估
        :return:
        """
        # 确保 提前终止开启的条件是有验证集
        assert not ((early_stop is True) and (train_test_split is False))
        self.graph = tf.Graph()   # 定义计算图
        with self.graph.as_default():
            tf.set_random_seed(self.random_seed)  # 设定随机种子

            self.feat_index = tf.placeholder(tf.int32, shape=[None, None],
                                             name="feat_index")
            # None*F 样本数为None， F为field的个数，原始特征的数目
            # 为索引，查找随机矩阵中的嵌入向量
            self.feat_value = tf.placeholder(tf.int32, shape=[None, None],  # None*F
                                             name="feat_value")
            self.label = tf.placeholder(tf.float32, shape=[None, 1], name="label")  # None*1
            self.dropout_keep_cin = tf.placeholder(tf.float32, shape=[None], name="dropout_keep_cin")  # None
            self.dropout_keep_deep = tf.placeholder(tf.float32, shape=[None], name="dropout_keep_deep")  # None
            self.train_phase = tf.placeholder(tf.bool, name="train_phase")

            predict = self._inference()  # 前向推断
            global_step = tf.Variable(0, trainable=False, name="step")  # 初始化变量为0 ，不可训练

            self.decay_step = int(self.num_example/self.batch_size)   # 衰减步系数， 必须为整形
            self._skill(global_step, self.decay_step)  # 使用 滑动平均与  指数衰减

            # 计算损失函数
            self._loss(predict)

            # 加上正则化的损失操作， 通过集合的形式，来获取所有的损失值
            self.loss = self.loss + tf.compat.v1.add_n(tf.compat.v1.get_collection("losses"))

            tf.add_to_collection("loss", self.loss)  # 便于离线断点操作

            updata_ops = tf.compat.v1.get_collection(tf.GraphKeys.UPDATE_OPS)  # 网络层中所有的变量，训练之前完成

            # 如有需要打印
            # for i in updata_ops:
            #     print(i)

            train_step = self._optimizer(global_step=global_step)  # 当前步数， 优化器，即反向传播

            with tf.compat.v1.control_dependencies(train_step, self.variable_averages_op):
                self.train_op = tf.compat.v1.no_op(name="train")  # 集合操作
                # tf.no_op啥都不进行操作，仅仅作为占位符来控制边界， 起到占位符的作用
                self.train_op = tf.group([self.train_op, updata_ops])   # tf.group  返回的是个操作，不是值， 输入为tensor

            if train_test_split:  # 需要训练集与验证集分离操作
                xi_train, xi_valid, xv_train, xv_valiad, label_train, label_valid = data.train_test_split(test_size=0.3)
                # 进行数据分离操作
                train_data = BacthDataset(xi_train, xv_train, label_train)
                valid_data = BacthDataset(xi_valid, xv_valiad, label_valid)

            else:   # 无需分离
                train_data = data
                valid_data = None

            evaluate_loss = []
            evaluate_auc = []
            evaluate_acc = []

            saver = tf.compat.v1.train.Saver(max_to_keep=5)  # 默认保存5个

            with tf.Session() as sess:
                tf.global_variables_initializer().run()

                if self.checkpoint_training:   # 加入断点继训练功能
                    ckpt = tf.train.get_checkpoint_state(self.model_save_path)   # 只加载参数不加载图的方式进行自动续训
                    if ckpt and ckpt.model_checkpoint_path:
                        saver.restore(sess, ckpt.model_checkpoint_path)
                        print("模型加载成功")
                    else:
                        print("模型不存在需要重新训练")

                for epoch in range(self.traning_step):
                    # 获取批量数据
                    xi, xv, label = train_data.next_batch(self.batch_size)
                    # 通过字典的形式喂入数据
                    _, loss_value, step = sess.run([self.train_op, self.loss, global_step],
                                                   feed_dict=self.set_on_batch(xi, xv, label, True)
                                                   )

                    # 每一千轮 保存一次模型
                    if epoch % every_save == 0:
                        print("After %d traning step(s),loss on training batch is %g" % (step, loss_value))
                        saver.save(sess, os.path.join(save_params["model_save_path"], save_params["model_name"]),
                                   global_step=global_step
                                   )

                    # 每隔一定轮数进行 评估
                    if train_test_split:    # 有验证集 则进行评估处理
                        valid_num = 0
                        if epoch % every_eval == 0:  # 每隔一定轮数，进行操作
                            valid_num += 1
                            xi_v, xv_v, label_v = valid_data.next_batch(valid_data.num_examples)  # 一次性全部导入

                            cal_op = self.evaluate(real_label=label_v, evaluate_type=self.evaluate_type)
                            if self.evaluate_type == "all":
                                valid_loss, _, valid_acc, valid_auc = sess.run(cal_op,
                                                                               feed_dict=self.set_on_batch(xi_v, xv_v,
                                                                                                           label_v,
                                                                                                           False))
                                evaluate_acc.append(valid_acc)
                                evaluate_auc.append(valid_auc)

                            elif self.evaluate_type == "acc":
                                valid_loss, _, valid_acc = sess.run(cal_op, feed_dict=self.set_on_batch(xi_v, xv_v,
                                                                                                        label_v,
                                                                                                        False))
                                evaluate_acc.append(valid_acc)

                            elif self.evaluate_type == "auc":
                                valid_loss, valid_auc = sess.run(cal_op, feed_dict=self.set_on_batch(xi_v, xv_v,
                                                                                                     label_v,
                                                                                                     False))
                                evaluate_auc.append(valid_auc)

                            else:
                                valid_loss = sess.run(cal_op, feed_dict=self.set_on_batch(xi_v, xv_v, label_v, False))

                            print("After %d validing step(s),loss on validata batch is %g" % (valid_num,
                                                                                              valid_loss))
                            evaluate_loss.append(valid_loss)

                            if early_stop:  # 开启 提前终止
                                if autoevaluate:  # 自动进行评估
                                    ifstop = self.early_termination(valid_result=evaluate_loss)

                                else:  # 手动进行评估
                                    if self.evaluate_type == "acc":   # 终止条件为  连续5次评估不再变化为止
                                        ifstop = self.early_termination(evaluate_acc, auto=False)

                                    elif self.evaluate_type == "auc":   # 终止条件为
                                        ifstop = self.early_termination(evaluate_auc, auto=False)

                                    elif self.evaluate_type == "all":   # 终止条件为
                                        ifstop = self.early_termination(evaluate_auc, auto=False) and\
                                                 self.early_termination(evaluate_acc, auto=False)

                                    else:
                                        ifstop = False

                                if ifstop:  #
                                    break


def online_evaluate(sess, loss, pred, real_label, evaluate_type, valid_num, feed_dict):
    """
    :param sess:   会话对象
    :param loss:   损失对象
    :param pred:   推断部分
    :param real_label:   真实标签部分
    :param evaluate_type:  评估函数部分
    :param valid_num: 验证轮数
    :param feed_dict: 数据集
    :return:
    """
    evaluate_dict = {}  # 用于存放评估结果
    if evaluate_type == "acc" or evaluate_type == "all":
        cor_pre = tf.compat.v1.equal(tf.argmax(pred, 1), tf.argmax(real_label, 1),
                                     name="correct_predict")
        acc = tf.reduce_mean(tf.cast(cor_pre, tf.float32), name="acc")  # 强制转换类型
    else:
        cor_pre = None
        acc = None

    if evaluate_type == "auc" or evaluate_type == "all":
        auc = roc_auc_score(y_true=real_label, y_score=pred)
    else:
        auc = None

    if evaluate_type == "all":
        cal_op = [loss, cor_pre, acc, auc]
        valid_loss, _, valid_acc, valid_auc = sess.run(cal_op, feed_dict=feed_dict)
        evaluate_dict["acc"] = valid_acc
        evaluate_dict["auc"] = valid_auc

    elif evaluate_type == "acc":
        cal_op = [loss, cor_pre, acc]
        valid_loss, _, valid_acc = sess.run(cal_op, feed_dict=feed_dict)
        evaluate_dict["acc"] = valid_acc

    elif evaluate_type == "auc":
        cal_op = [loss, auc]
        valid_loss, valid_auc = sess.run(cal_op, feed_dict=feed_dict)
        evaluate_dict["auc"] = valid_auc

    else:
        cal_op = [loss]
        valid_loss = sess.run(cal_op, feed_dict=feed_dict)

    evaluate_dict["loss"] = valid_loss

    print("After %d validing step(s),loss on validata batch is %g" % (valid_num,
                                                                      valid_loss))
    return evaluate_dict


def early_termination(valid_result, threshold_val=10, greater_is_better=True, auto=False):  # 提前终止
    """
    有验证集才能开启提前终止，否则进行报错
    :param valid_result:  # 验证集
    :param threshold_val:   # 阈值
    :param greater_is_better:
    :param auto:  #默认 为手动
    :return:
    """
    if auto:   # 使用的是loss 来进行操作
        if len(valid_result) > threshold_val:  # 必须保证有效数据大于10条，否则不必开启

            if (abs(valid_result[-1] - valid_result[-2]) < 0.001) and \
                (abs(valid_result[-2] - valid_result[-3]) < 0.001) and \
                (abs(valid_result[-3] - valid_result[-4]) < 0.001) and \
                (abs(valid_result[-4] - valid_result[-5]) < 0.001) and \
                    (valid_result[-1] < valid_result[0]):
                return True
            else:
                return False
    else:
        if len(valid_result) > threshold_val:  # 必须保证有效数据大于10条，否则不必开启
            if greater_is_better:  # 越大越好  适合 acc, auc
                if (valid_result[-1] < valid_result[-2]) and \
                    (valid_result[-2] < valid_result[-3]) and \
                        (valid_result[-3] < valid_result[-4]) and \
                        (valid_result[-4] < valid_result[-5]):
                    return True
            else:   # 越小越好
                if (valid_result[-1] > valid_result[-2]) and \
                    (valid_result[-2] > valid_result[-3]) and \
                        (valid_result[-3] > valid_result[-4]) and \
                        (valid_result[-4] > valid_result[-5]):
                    return True
        else:
            return False


# 模型的离线版的断点续训功能
def break_point_training(data, train_epoch, batch_size=64, max_epoch=100000, evaluate_type=None,
                         train_test_split=True, every_eval=2, model_save_path=None, model_name=None,
                         early_stop=False, autoevaluate=True):
    """
    :param data:  数据集对象
    :param train_epoch:   想要训练的代数
    :param batch_size:  批大小
    :param max_epoch: 训练代数的上限制 10万次
    :param evaluate_type: 评估类型
    :param train_test_split: 默认需要将训练集进行切分操作，默认为True
    :param every_eval:  默认两轮进行一次验证
    :param model_save_path:   模型路径
    :param model_name:  模型名字
    :param early_stop:  提前停止的条件
    :param autoevaluate:   自动评估是否启用
    :return:
    """
    # 确保 提前终止开启的条件是有验证集
    assert not ((early_stop is True) and (train_test_split is False))
    # 确保评估的前提为有验证集
    assert not ((train_test_split is True) and (evaluate_type is None))
    if train_test_split:  # 需要训练集与验证集分离操作
        xi_train, xi_valid, xv_train, xv_valid, label_train, label_valid = data.train_test_split(test_size=0.3)
        # 进行数据分离操作
        train_data = BacthDataset(xi_train, xv_train, label_train)
        valid_data = BacthDataset(xi_valid, xv_valid, label_valid)
    else:  # 无需分离
        train_data = data
        valid_data = None
    try:
        ckpt = tf.compat.v1.train.get_checkpoint_state(model_save_path)  # 检查模型状态
        if ckpt and ckpt.model_checkpoint_path:  # 模型文件存在
            print("断点文件存在，加载计算图")
            with tf.Session() as sess:
                tf.global_variables_initializer().run()
                # 将文件中的网络加载到当前图上
                graph = tf.get_default_graph()  # 获取整个计算图
                saver = tf.compat.v1.train.import_meta_graph(ckpt.model_checkpoint_path+".meta")
                collection_name = graph.get_all_collection_keys()  # 获取所有 集合的名字
                print(collection_name)  # 打印包含的集合名字
                ops = graph.get_operations()  # 获取所有操作名
                for op in ops:  # 查看所有的操作部分
                    print(op.name, op.values())

                # 获取关键 张量 与op
                step = graph.get_tensor_by_name("step:0")
                print("检查断点文件最新训练代数:%d" % step.eval())
                if step.eval() < max_epoch:
                    feat_index = graph.get_tensor_by_name("feat_index:0")
                    feat_value = graph.get_tensor_by_name("feat_value:0")
                    label = graph.get_tensor_by_name("label:0")
                    train_phase = graph.get_tensor_by_name("train_phase:0")
                    train_op = graph.get_collection("train_op")
                    loss = graph.get_collection("loss")

                    if step.eval()+train_epoch <= max_epoch:
                        left_epoch = train_epoch   # 剩下迭代次数
                    else:
                        left_epoch = step.eval() + train_epoch - max_epoch

                    valid_num = 0
                    for epoch in range(left_epoch):
                        # 获取批量数据
                        xi_trains, xv_trains, label_trains = train_data.next_batch(batch_size=batch_size)
                        _, loss_value, steps = sess.run([train_op, loss, step],
                                                        feed_dict={feat_index: xi_trains,
                                                                   feat_value: xv_trains,
                                                                   label: label_trains,
                                                                   train_phase: True})
                        #  同样1000轮保持一次模型
                        if epoch % 1000 == 0:
                            saver.save(sess, os.path.join(model_save_path, model_name), global_step=steps)
                            print("After %d traing step(s),loss on training batch is %g" % (steps, loss_value))

                        if train_test_split:  # 有验证集 则进行评估处理
                            if epoch % every_eval == 0:  # 每隔一定轮数，进行操作, 一次性全部打入数据进行操作
                                xi_valids, xv_valids, label_valids = valid_data.next_batch(valid_data.num_examples)
                                valid_num += 1
                                valid_feed_dict = {feat_index: xi_valids,
                                                   feat_value: xv_valids,
                                                   label: label_valids,
                                                   train_phase: True}

                                evaluate_res = online_evaluate(sess=sess, loss=loss, pred=label,
                                                               real_label=label_valids, evaluate_type="all",
                                                               valid_num=valid_num,
                                                               feed_dict=valid_feed_dict)

                                if early_stop:  # 开启 提前终止
                                    if autoevaluate:  # 自动进行评估
                                        ifstop = early_termination(valid_result=evaluate_res["loss"])

                                    else:  # 手动进行评估
                                        if evaluate_type == "acc":  # 终止条件为  连续5次评估不再变化为止
                                            ifstop = early_termination(valid_result=evaluate_res["acc"])

                                        elif evaluate_type == "auc":  # 终止条件为
                                            ifstop = early_termination(valid_result=evaluate_res["auc"],
                                                                       auto=False)

                                        elif evaluate_type == "all":  # 终止条件为
                                            ifstop = early_termination(valid_result=evaluate_res["acc"],
                                                                       auto=False) and\
                                                early_termination(valid_result=evaluate_res["auc"], auto=False)

                                        else:
                                            ifstop = False

                                    if ifstop:  #
                                        break

                else:
                    print("已经到达训练代数上限")
        else:
            print("未找到断点文件")

    except ValueError:
        print("Error:未读取到模型文件")

    finally:
        print("finally!")


def evaluate(data, model_save_path, eval_interval_secs=10, task_type="auc"):  # 在线评估,每10秒评估一次模型
    # 需要开启一个进程运行, 并行操作， 即一个进程进行训练，另一个进程进行评估
    """
    :param data:     验证集数据对象
    :param model_save_path:   模型文件
    :param eval_interval_secs:   间隔时长进行评估一次
    :param task_type:   任务类型， 输出值为
    :return:
    """
    valid_num = 0
    while True:
        with tf.Session() as sess:
            tf.global_variables_initializer().run()
            ckpt = tf.compat.v1.train.get_checkpoint_state(model_save_path)  # 检查模型状态
            if ckpt and ckpt.model_checkpoint_path:  # 模型文件存在
                print("断点文件存在，加载计算图")
                graph = tf.get_default_graph()  # 获取整个计算图
                feat_index = graph.get_tensor_by_name("feat_index:0")
                feat_value = graph.get_tensor_by_name("feat_value:0")
                label = graph.get_tensor_by_name("label:0")
                train_phase = graph.get_tensor_by_name("train_phase:0")
                loss = graph.get_collection("loss")
                xi, xv, label_ = data.next_batch(data.num_examples)

                valid_feed_dict = {feat_index: xi,
                                   feat_value: xv,
                                   label: label,
                                   train_phase: False}

                online_evaluate(sess=sess, loss=loss, pred=label,
                                real_label=label, evaluate_type=task_type,
                                valid_num=valid_num,
                                feed_dict=valid_feed_dict)
                valid_num += 1

            else:
                print("No checkpoint file found")
        sleep(eval_interval_secs)  # 定时评估
    # -----------------------------------------
    # 如果不使用变量重命名的方式加载，使用下面的标准方式进行重载
    # saver = tf.compat.v1.train.import_meta_graph("path/model/model.ckpt/model.ckpt.meta")
    # with tf.Session() as sess:
    #       saver.restore(sess,"path/model/model.ckpt")   #模型加载完毕，  此方式不需要再进行计算图的定义


# if __name__ == "__main__":
#
#     XDeeFM_model=XDeepFM(feature_size=12, field_size=8)
#     XDeeFM_model._train(data=train_data,
#                         save_params=,
#                         )
