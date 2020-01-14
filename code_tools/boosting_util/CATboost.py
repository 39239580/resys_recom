from sklearn.metrics import accuracy_score, roc_auc_score
from catboost import CatBoostClassifier
import matplotlib.pyplot as plt
# from catboost import
"""
相关算法模块的接口地址  https://catboost.ai/docs/concepts/python-reference_catboostclassifier_predict.html
"""


class Catboost(object):
    def __init__(self, **params):
        self.model = CatBoostClassifier(learning_rate=params.get("learning_rate", None),   # 学习率
                                        l2_leaf_reg=params.get("l2_leaf_reg", None),  # l2正则化系数
                                        depth=params.get("depth", None),  # 树深度
                                        max_depth=params.get("max_depth", None),    # 最大深度
                                        loss_function=params.get("loss_function", None),   # RMSE /logloss
                                        # /MAE/CrossEntropy  #默认使用二分类logloss
                                        custom_metric=params.get("custom_metric", None),   # RMSE/logloss
                                        # /MAE/CrossEntropy/Recall/Accuracy
                                        # Precision /F1 /Accuracy/AUC/R2
                                        eval_metric=params.get("eval_metric", None),   # RMSE/logloss/MAE
                                        # /CrossEntropy/Recall/Accuracy
                                        # Precision /F1 /Accuracy/AUC/R2
                                        iterations=params.get("iterations", None),
                                        nan_mode=None,  # NAN 的处理方法  RMSE/logloss/MAE/CrossEntropy/Recall
                                        # Precision /F1 /Accuracy/AUC/R2
                                        leaf_estimation_method=params.get("leaf_estimation_method", None),
                                        # 迭代求解的方法，梯度和牛顿Newton/Gradient
                                        random_seed=params.get("random_seed", None),  # 随机数种子
                                        thread_count=params.get("thread_count", None),   # 训练时使用的cpu/gpu核数
                                        used_ram_limit=params.get("ram_limit", None),  # CTR 问题， 计算时的内存限制
                                        gpu_ram_part=params.get("gpu_ram", None),    # GPUn内存限制
                                        task_type=params.get("task", None),  # 设备运行
                                        use_best_model=params.get("use_best_model", None),   # 默认使用最好模型
                                        logging_level=params.get("logging_level", None)
                                        )

    def train(self, x_train, y_train, **params):
        """
        :param x_train:  训练集
        :param y_train:  训练集样本
        :param params:  参数
        :return:
        有分类特征，加上 cat_features参数
        """
        self.model.fit(x_train, y_train,
                       cat_features=params.get("cat_features", None),
                       early_stopping_rounds=params.get("early_stopping_rounds", None),  # 过早停止的条件
                       verbose=params.get("verbose", None),  # 是否开启冗余，
                       eval_set=params.get("eval_set", None),  # 验证集(x_val, y_val)
                       plot=params.get("plot", False)  # 开启训练过程中的绘图  bool型数据
                       )
        """
        fit 方法使用，x_train数据可为list, array,df,pandas.Series ,为2维的矩阵也可以为Pool， 效率最高
        y_train  为label， 与x_train 数据类型对应，必须是个一维的数组
        -Logloss
        -CrossEntropy [0; 1] 中间的一个概率值  交叉熵
        Multiclassification  整形或字符串新数据，代表标签的分类 
        cat_features  是否包含分类特征list  array
        sample_weight :采样权重， list,array,df,Series  默认设置为1
        """

    def predict(self, x_test, **params):
        """
        :param x_test:  测试使用数据
        :param predict_type:  参数为  想要输出结果的类型
        "Class/Probability/RawFormulaVal"
        :return:
        """
        self.my_pred = self.model.predict(x_test,
                                          prediction_type=params.get("prediction_type","Class"),
                                          thread_count=params.get("thread_count", -1),
                                          verbose=params.get("verbose", None)
                                          )
        return self.my_pred

    def plt_importance(self):
        # 绘制特征重要性
        feat_, feat_name_ = self.get_feature_importances()
        plt.figure(figsize=(15, 15))
        plt.barh(feat_name_, feat_, height=0.5)
        plt.show()  # 显示图像

    def evaluate(self, evalue_fun, y_test):  # 进行评估
        if evalue_fun == "acc":
            result = accuracy_score(y_test, self.my_pred)
            print("accuarcy:%.2f%%" % (result * 100.0))
        elif evalue_fun == "auc":
            result = roc_auc_score(y_test, self.my_pred)
            print("accuarcy:%.2f%%" % (result * 100.0))
        return result

    def get_feature_importances(self):
        # feature_dict={}
        importan_feature = self.model.get_feature_importance()  # 获取重要的参数
        original_feature_name = self.model.feature_names_
        # feat_name =[feature_dict.get(i,i) for i in original_feature_name]
        return importan_feature, original_feature_name

    def save_model(self, save_params):  # 模型保存
        self.model.save_model(fname=save_params.get("fname", "./model/Catboostmodel"),   # 保存的文件路径名字
                              format=save_params.get("format", "cbm"),  # 保存的数据格式
                              pool=save_params.get("pool", None)  #  训练使用的数据   模型保存成json格式，无需使用pool
                              )

    def load_model(self,load_param):   # 加载模型
        models = self.model.load_model(fname=load_param.get("fname", "./model/Catboostmodel"),
                                       format=load_param.get("format", 'cbm'))
        return models
