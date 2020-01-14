import numpy as np
import tensorflow as tf
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import roc_auc_score  # auc 得分计算工具包
from time import time  # 计时需要
from tensorflow.contrib.layers.python.layers import batch_norm as batch_norm  # BN使用的模块
from yellowfin import YFOptimizer


class DeepFM(BaseEstimator, TransformerMixin):
    def __init__(self,   # one_hot 编码之后的长度为M， 编码之前的长度为F, embedding之后的长度为k
                 feature_size,   # feature_size  one_hot 之后的维度 M
                 field_size,   # 未编码之前的之前的维度   F
                 embedding_size=8,   # embedding 之后的长度
                 dropout_fm=[1.0, 1.0],
                 deep_layers=[32, 32],
                 dropout_deep=[0.5, 0.5, 0.5],
                 deep_layers_activation=tf.nn.relu,
                 epoch=10,
                 batch_size=64,
                 learning_rate=0.001,
                 optimizer_type="adam",
                 batch_norms=None,
                 batch_norm_decay=0.995,
                 verbose=False,
                 random_seed=2018,
                 use_fm=True,
                 use_deep=True,
                 loss_type="logloss",
                 eval_metric=roc_auc_score,
                 l2_reg=0.0,    # l2 lambda 系数
                 greater_is_better=True):
        assert (use_deep or use_fm)  # 假设  开了其中一个不会报错
        assert loss_type in ["logloss", "mse"],  \
            "loss_type can be either 'logloss' for classification task or 'mse' for regression task"

        self.feature_size = feature_size  # 标记为M   one_hot编码后的大小
        self.field_size = field_size   # 标记为F    one_hot编码前的大小 ，一个样本的编码长度
        self.embedding_size = embedding_size  # 标记为k  嵌入尺寸

        self.dropout_fm = dropout_fm   # dropout 概率值
        self.deep_layers = deep_layers   # 层数大小 ，每层大小
        self.dropout_deep = dropout_deep   # 链接deep部分的结构
        self.deep_layers_activation = deep_layers_activation  # 激活函数
        self.use_fm = use_fm  # 默认使用fm部分
        self.use_deep = use_deep  # 默认使用deep部分
        self.l2_reg = l2_reg  # l2  lambda 系数

        self.epoch = epoch   # 代数
        self.batch_size = batch_size   # 批量大小
        self.learning_rate = learning_rate
        self.optimizer_type = optimizer_type   # ADAM 优化器

        self.batch_norm = batch_norms  # BN中系数
        self.batch_norm_decay = batch_norm_decay   # BN 延迟系数

        self.verbose = verbose   # 默认关闭
        self.random_seed = random_seed   # 随机数种子
        self.loss_type = loss_type   # 损失类型
        self.eval_metric = eval_metric  # 评估函数
        self.greater_is_better = greater_is_better  # 越来越好， 默认开启
        self.train_result, self.valid_result = [], []

        self._init_graph()   # 初始化计算图

    def _init_graph(self):
        self.graph = tf.Graph()   # 生成新的计算图
        with self.graph.as_default():  # 作为默认的计算图

            tf.compat.v1.set_random_seed(self.random_seed)   # 设置随机数种子
            # 占位符部分
            self.feat_index = tf.compat.v1.placeholder(tf.int32, shape=[None, None],
                                                       name="feat_index")  # None *F
            self.feat_value = tf.compat.v1.placeholder(tf.float32, shape=[None, None],
                                                       name="feat_value")   # None *F
            self.label = tf.compat.v1.placeholder(tf.float32, shape=[None, 1], name="label")   # None*1
            self.dropout_keep_fm = tf.compat.v1.placeholder(tf.float32, shape=[None], name="dropout_keep_fm")
            self.dropout_keep_deep = tf.compat.v1.placeholder(tf.float32, shape=[None], name="dropout_keep_deep")

            self.train_phase = tf.compat.v1.placeholder(tf.bool, name="train_phase")  # 默认为shape= None

            #
            self.weights = self._initialize_weights()  # 初始化权重

            # model
            self.embeddings = tf.nn.embedding_lookup(self.weights["feature_embeddings"], self.feat_index)  # None *F *k

            feat_value = tf.reshape(self.feat_value, shape=[-1, self.field_size, 1])
            self.embeddings = tf.multiply(self.embeddings, feat_value)

            # -------------------first order term------------------------------
            self.y_first_order = tf.nn.embedding_lookup(self.weights["feature_bias"], self.feat_index)  # None*F*1
            self.y_first_order = tf.reduce_sum(tf.multiply(self.y_first_order, feat_value), 2)
            self.y_first_order = tf.compat.v1.nn.dropout(self.y_first_order, rate=1-self.dropout_keep_fm[0])  # None *F

            # --------------------second order term-----------------------------
            # sum_square part
            self.summed_features_emb = tf.reduce_sum(self.embeddings, 1)  # None *k
            self.summed_features_emb_square = tf.square(self.summed_features_emb)  # None *k

            # square_sum part
            self.squared_features_emb = tf.square(self.embeddings)  # None*k
            self.squared_sum_features_emb = tf.reduce_sum(self.squared_features_emb, 1)  # None *k

            # second order
            self.y_second_order = 0.5 * tf.subtract(self.summed_features_emb_square,
                                                    self.squared_sum_features_emb)
            # None*k
            self.y_second_order = tf.nn.dropout(self.y_second_order, self.dropout_keep_fm[1])   # None *k

            # -------------------------Deep component---------------------------------------
            self.y_deep = tf.reshape(self.embeddings, shape=[-1, self.field_size * self.embedding_size])  # None *(F*k)
            self.y_deep = tf.nn.dropout(self.y_deep, self.dropout_keep_deep[0])
            for i in range(0, len(self.deep_layers)):
                self.y_deep = tf.add(tf.matmul(self.y_deep, self.weights["layer_%d" % i]),
                                     self.weights["bias_%d" % i])
                # None * layer[i] * 1
                if self.batch_norm:
                    self.y_deep = self.batch_norm_layer(self.y_deep, train_phase=self.train_phase,
                                                        scope_bn="bn_%d" % i)
                    # None *layer[i] *1
                self.y_deep = self.deep_layers_activation(self.y_deep)
                self.y_deep = tf.nn.dropout(self.y_deep, self.dropout_keep_deep[1+i])  # dropout at each Deep layer

            # --------------------------------------DeepFM-----------------------------------------
            if self.use_fm and self.use_deep:
                concat_input = tf.concat([self.y_first_order, self.y_second_order, self.y_deep], axis=1)
            elif self.use_fm:
                concat_input = tf.concat([self.y_first_order, self.y_second_order], axis=1)
            elif self.use_deep:
                concat_input = self.y_deep
            else:
                raise AttributeError   # 没选择式，导致错误

            self.out = tf.add(tf.matmul(concat_input, self.weights["concat_projection"]),
                              self.weights["concat_bias"])

            # ---------------------------------------loss-----------------------------------------
            if self.loss_type == "logloss":
                self.out = tf.nn.sigmoid(self.out)
                self.loss = tf.compat.v1.losses.log_loss(self.label, self.out)
            elif self.loss_type == "mse":
                self.loss = tf.nn.l2_loss(tf.subtract(self.label, self.out))
            # l2 regularization on weights
            if self.l2_reg > 0:
                self.loss += tf.contrib.layers.l2_regularizer(self.l2_reg)(self.weights["concat_projection"])
                if self.use_deep:
                    for i in range(len(self.deep_layers)):
                        self.loss += tf.contrib.layers.l2_regularizer(self.l2_reg)(self.weights["layer_%d" % i])

            # -----------------------------------optimizer优化器----------------------------------
            if self.optimizer_type == "adam":
                self.optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=self.learning_rate, beta1=0.9,
                                                                  beta2=0.999, epsilon=1e-8).minimize(self.loss)
            elif self.optimizer_type == "adagrad":
                self.optimizer = tf.compat.v1.train.AdagradOptimizer(learning_rate=self.learning_rate,
                                                                     initial_accumulator_value=1e-8).minimize(self.loss)

            elif self.optimizer_type == "gd":
                self.optimizer = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=self.learning_rate
                                                                             ).minimize(self.loss)

            elif self.optimizer_type == "momentum":
                self.optimizer = tf.compat.v1.train.MomentumOptimizer(learning_rate=self.learning_rate,
                                                                      momentum=0.95).minimize(self.loss)

            elif self.optimizer_type == "yellowfin":
                self.optimizer = YFOptimizer(learning_rate=self.learning_rate, momentum=0.0).minimize(self.loss)

            # init
            self.save = tf.compat.v1.train.Saver()  # 实例化对象，进行声明
            init = tf.compat.v1.global_variables_initializer()
            self.sess = self._init_session()
            self.sess.run(init)

            # number of params
            total_parameters = 0
            for variable in self.weights.values():
                shape = variable.get_shape()
                variable_parameters = 1
                for dim in shape:
                    variable_parameters *= dim.value
                total_parameters += variable_parameters
            if self.verbose > 0:
                print("#params: %d" % total_parameters)

    @staticmethod
    def _init_session():   # 初始会话
        config = tf.compat.v1.ConfigProto(device_count={"gpu": 0})
        config.gpu_options.allow_growth = True
        return tf.compat.v1.Session(config=config)

    def _initialize_weights(self):
        weights = dict()  # 字典
        #  创建张量部分
        # embeddings   嵌入部分
        weights["feature_embeddings"] = tf.Variable(
            tf.compat.v1.random_normal([self.feature_size, self.embedding_size], 0.0, 0.01),  # M*k, 均值为0，标准差为0.01正态分布
            name="feature_embeddings")   # feature_size *K

        weights["feature_bias"] = tf.Variable(
            tf.compat.v1.random_uniform([self.feature_size, 1], 0.0, 1.0),  # 生成均匀分布 M*1  的
            name="feature_bias")    # feature_size *1

        # deep layers  深度部分
        num_layer = len(self.deep_layers)    # 层数  多少层
        input_size = self.field_size * self.embedding_size   # F*k     F 字段个数，k为embedding长度，embedding之后的长度
        glorot = np.sqrt(2.0 / (input_size + self.deep_layers[0]))
        weights["layer_0"] = tf.Variable(
            np.random.normal(loc=0, scale=glorot, size=(input_size, self.deep_layers[0])), dtype=tf.float32)
        # 正态分布  权重

        weights["bias_0"] = tf.Variable(
            np.random.normal(loc=0, scale=glorot, size=(1, self.deep_layers[0])), dtype=np.float32)  # 1*layers[0]
        # 正态分布  偏置

        for i in range(1, num_layer):  # 遍历第二层到最后一层
            glorot = np.sqrt(2.0/(self.deep_layers[i-1] + self.deep_layers[i]))
            weights["layer_%d" % i] = tf.Variable(
                np.random.normal(loc=0, scale=glorot, size=(self.deep_layers[i-1], self.deep_layers[i])),
                dtype=tf.float32)   # layers[i-1]*layers[i]
            weights["bias_%d" % i] = tf.Variable(
                np.random.normal(loc=0, scale=glorot, size=(1, self.deep_layers[i])),  # 标准正态分布
                dtype=tf.float32)
        #  每层中的数据进行正态分布

        # final concat projrction layer   #  最后拼接的部分
        if self.use_fm and self.use_deep:  # 使用DeepFM拼接
            input_size = self.field_size + self.embedding_size + self.deep_layers[-1]   # F+k +节点数
        elif self.use_fm:  # 使用FM部分   F+k
            input_size = self.field_size + self.embedding_size
        elif self.use_deep:  # 使用deep 部分  节点数
            input_size = self.deep_layers[-1]

        glorot = np.sqrt(2.0/(input_size+1))
        weights["concat_projection"] = tf.Variable(np.random.normal(loc=0, scale=glorot, size=(input_size, 1)),
                                                   dtype=tf.float32)   # n*1

        weights["concat_bias"] = tf.Variable(tf.constant(0.01), dtype=tf.float32)

        return weights

    def batch_norm_layer(self, x, train_phase, scope_bn):   # BN 层
        bn_train = batch_norm(x, decay=self.batch_norm_decay, center=True, scale=True, updates_collections=None,
                              is_training=True, reuse=tf.compat.v1.AUTO_REUSE, trainable=True, scope=scope_bn)   # 训练阶段
        bn_inderence = batch_norm(x, decay=self.batch_norm_decay, center=True, scale=True, updates_collections=None,
                                  is_training=False, reuse=tf.compat.v1.AUTO_REUSE, trainable=True, scope=scope_bn)  # 预测阶段， 只有钱箱
        z = tf.cond(train_phase, lambda: bn_train, lambda: bn_inderence)   # train_phase为真，执行第一个
        return z

    @staticmethod
    def get_batch(xi, xv, y, batch_size, index):  # 获取批量数据
        """
        :param xi:   数据集中样本特征索引
        :param xv:   数据集中样本特征值
        :param y:   数据集样本标签
        :param batch_size:   批量大小
        :param index:   开始索引
        :return:
        """
        start = index * batch_size  # 第一批从index=0开始
        end = (index+1) * batch_size
        end = end if end < len(y) else len(y)
        return xi[start:end], xv[start:end], [[y_] for y_ in y[start:end]]  # 组成一列数据

    # shuffle three lists simutaneously  同时随机 排列三个列表
    @staticmethod
    def shuffle_in_unison_scary(a, b, c):
        """
        :param a:   list
        :param b:   list
        :param c:   list
        :return:
        """
        rng_state = np.random.get_state()  # 获取种子
        np.random.shuffle(a)
        np.random.set_state(rng_state)
        np.random.shuffle(b)
        np.random.set_state(rng_state)
        np.random.shuffle(c)

    def fit_on_batch(self, xi, xv, y):  # 批量上的拟合
        feed_dict = {self.feat_index: xi,
                     self.feat_value: xv,
                     self.label: y,
                     self.dropout_keep_fm: self.dropout_fm,
                     self.dropout_keep_deep: self.dropout_deep,
                     self.train_phase: True}
        loss, opt = self.sess.run((self.loss, self.optimizer), feed_dict=feed_dict)
        return loss

    def fit(self, xi_train, xv_train, y_train,
            xi_valid=None, xv_valid=None, y_valid=None,
            early_stopping=False, refit=False):
        """
        :param xi_train: [[ind1_1,inde1_2,...],[inde2_1, inde2-2,...],...]  indei_j  第i个样本中的feature_field j索引
        :param xv_train: [[val1_1,val1_2,...],[val2_1,val2_2,...],...] indei_j  第i个样本中的feature_field j 值
        :param y_train:  每个训练样本的标签
        :param xi_valid:
        :param xv_valid:
        :param y_valid:
        :param early_stopping: bool  是否执行过早停止
        :param refit: 是否在训练和测试集上使用重训练
        :return:
        """
        has_valid = xv_valid is not None
        for epoch in range(self.epoch):  # 训练轮数
            t1 = time()
            self.shuffle_in_unison_scary(xi_train, xv_valid, y_train)  # 打乱数据集
            total_batch = int(len(y_train)/self.batch_size)  # 每轮中含有的批次
            for i in range(total_batch):
                xi_batch, xv_batch, y_batch = self.get_batch(xi_train, xv_train, y_train, self.batch_size, i)
                self.fit_on_batch(xi_batch, xv_batch, y_batch)   # 训练

            # 评估
            train_result = self.evaluate(xi_train, xv_train, y_train)
            self.train_result.append(train_result)
            if has_valid:
                valid_result = self.evaluate(xi_valid, xv_valid, y_valid)
                self.valid_result.append(valid_result)
            else:
                valid_result = None

            if self.verbose > 0 and epoch % self.verbose == 0:  # 每个一定代数进行评估
                if has_valid:  # 有验证集
                    print("[%d]train-result =%.4f,valid-result =%.4f [%.1f s]"
                          % (epoch+1, train_result, valid_result, time()-t1))
                else:
                    print("[%d]train-result =%.4f[%.1f s]" % (
                        epoch+1, train_result, time()-t1))

            if has_valid and early_stopping and self.training_termination(self.valid_result):
                #  有验证集，且开了过早停止， 加上 训练终止条件打开
                break

        if has_valid and refit:
            # 训练集+测试集一起训练
            if self.greater_is_better:
                best_valid_score = max(self.valid_result)
            else:
                best_valid_score = min(self.valid_result)
            best_epoch = self.valid_result.index(best_valid_score)
            best_train_score = self.train_result[best_epoch]
            xi_train = xi_train + xi_valid
            xv_train = xv_train + xv_valid
            y_train = y_train + y_valid
            for epoch in range(100):
                self.shuffle_in_unison_scary(xi_train, xv_train, y_train)
                total_batch = int(len(y_train)/self.batch_size)
                for i in range(total_batch):
                    xi_batch, xv_batch, y_batch = self.get_batch(xi_train, xv_train, y_train,
                                                                 self.batch_size, i)
                    self.fit_on_batch(xi_batch, xv_batch, y_batch)  # 批量进行训练
                # check

                train_result = self.evaluate(xi_train, xv_train, y_train)
                if abs(train_result - best_train_score) < 0.001 or \
                        (self.greater_is_better and train_result > best_train_score) or \
                        ((not self.greater_is_better) and train_result < best_train_score):
                    break

    def training_termination(self, valid_result):      # 是否开启提前停止
        """
        :param valid_result:   验证结果的
        :return:
        """
        if len(valid_result) > 5:   # 连续5次验证结果
            if self.greater_is_better:  # 打开
                # 结果越大越好
                if (valid_result[-1] < valid_result[-2]) and \
                        (valid_result[-2] < valid_result[-3]) and \
                        (valid_result[-3] < valid_result[-4]) and \
                        (valid_result[-4] < valid_result[-5]):
                    return True
            else:
                if (valid_result[-1] > valid_result[-2]) and \
                        (valid_result[-2] > valid_result[-3]) and \
                        (valid_result[-3] > valid_result[-4]) and \
                        (valid_result[-4] > valid_result[-5]):
                    return True
        return False

    def predict(self, xi, xv):
        """
        :param xi:   数据集中每个样本的特征索引列表
        :param xv:   数据集中每个样本的特征值列表
        :return: 每个样本的预测概率
        """
        dummy_y = [1]*len(xi)   # 得到  1*len(xi)的列表
        batch_index = 0
        xi_batch, xv_batch, y_batch = self.get_batch(xi, xv, dummy_y, self.batch_size, batch_index)  # 对数据进行分批处理
        y_pred = None
        while len(xi_batch) > 0:  # 遍历
            num_batch = len(y_batch)
            feed_dict = {self.feat_index: xi_batch,
                         self.feat_value: xv_batch,
                         self.label: y_batch,
                         self.dropout_keep_deep: [1.0] * len(self.dropout_deep),
                         self.dropout_keep_fm: [1.0]*len(self.dropout_fm),
                         self.train_phase: False}
            batch_out = self.sess.run(self.out, feed_dict=feed_dict)

            if batch_index == 0:
                y_pred = np.reshape(batch_out, (num_batch,))
            else:
                y_pred = np.concatenate((y_pred, np.reshape(batch_out, (num_batch,))))

            batch_index += 1
            xi_batch, xv_batch, y_batch = self.get_batch(xi, xv, dummy_y, self.batch_size, batch_index)
        return y_pred

    def evaluate(self, xi, xv, y):
        """
        :param xi:   数据集中每个样本的特征索引列表
        :param xv:   数据集中每个样本的特征值列表
        :param y:   数据集中每个样本的标签
        :return:   评估的度量
        """
        y_pred = self.predict(xi, xv)
        return self.eval_metric(y, y_pred)   # 评估auc得分


if __name__ == "__main__":
    dfm_params = {
        "use_fm": True,
        "use_deep": True,
        "embedding_size": 8,
        "dropout_fm": [1.0, 1.0],
        "deep_layers": [32, 32],
        "dropout_deep": [0.5, 0.5, 0.5],
        "deep_layers_activation": tf.nn.relu,
        "epoch": 30,
        "batch_size": 1024,
        "learning_rate": 0.001,
        "optimizer_type": "adam",
        "batch_norm": 1,
        "batch_norm_decay": 0.995,
        "l2_reg": 0.01,
        "verbose": True,
        "eval_metric": roc_auc_score,
        "random_seed": 2017
    }
    # 加载数据进来
    DFM_model = DeepFM(**dfm_params)  # 初始化对象

    # 训练一个模型
    DFM_model.fit(Xi_train, Xv_train, y_train)

    # 做预测
    DFM_model.predict(Xi_valid, Xv_valid)

    # 评估一个训练模型
    DFM_model.evaluate(Xi_valid, Xv_valid, y_valid)

    # 使用过早停止
    DFM_model.fit(Xi_train, Xv_train, y_train, Xi_valid, Xv_valid, y_valid, early_stopping=True)

    # 重新训练
    DFM_model.fit(Xi_train, Xv_train, y_train, Xi_valid, Xv_valid, y_valid, early_stopping=True, refit=True)

    #