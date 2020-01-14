import lightgbm as lgb
from lightgbm import plot_importance  # 画出重要型
# from sklearn import metrics
from sklearn.metrics import accuracy_score, roc_auc_score
import matplotlib.pyplot as plt
# from catboost import
"""
相关算法模块的接口地址  https://lightgbm.readthedocs.io/en/latest/Parameters.html
"""
class Lightgbm():
    def __init__(self, **params):
        """
        :param isdefault:
        :param params:
        "max_depth": -1,\   # 最大深度

                          "min_data_in_leaf": 20, # 叶子可能具有的最小记录数，过拟合使用,默认使用20
                "feature_fraction":1, #  随机森林使用
                "lambda_l1":0.0,  # 正则化参数
                "lambda_l2":0.0,  # 正则化参数
                "early_stopping_round ":0  # 提前停止的条件
                # 核心参数
                "Task":"train",    # 算法的用途  train/predict/convert_model/refit refit为将存在的模型重新训练
                # convert_model 将模型文件转成 if-else的格式
                "application": "multiclassova", # 默认为objective
                # regression, regression_l1, huber, fair, poisson, quantile, mape, gamma, tweedie, binary, multiclass,
                # multiclassova, cross_entropy, cross_entropy_lambda, lambdarank
                "boosting": "gbdt", # gbdtm,rf,dart,goss
                "device_type":"cpu",   # 默认值为cpu，可供选择GPU
                "num_threads":0,  # x线程数
                "seed"=2,   # 随机种子
                "learning_rate": 0.1,  # 学习
                "num_leaves":31,  # 最大叶子树
                "num_iterations":100, # 默认代数100
                "verbose_eval" = True,
                "num_class":  # 指定要分类的数目，做多分类时
        """
        self.params = params

    def packpag_data(self, x_train, y_train):  # 组装数据
        d_train = lgb.Dataset(x_train, label=y_train)  # 组装好数据
        return d_train

    def train(self, isCatFeatures, x_train, y_train, cate_features_name=[]):
        """
        :param isCatFeatures:  是否带 种类特征数据， bool
        :param x_train:  训练数据
        :param y_train:  训练数据标签
        :param cate_features_name: list  存放是种类特征的名称
        :return:
        """
        if isCatFeatures:  # 带种类特征的数据
            self.model = lgb.train(self.params, self.packpag_data(x_train, y_train),
                                   categorical_feature=cate_features_name)
        else:
            self.model = lgb.train(self.params, self.packpag_data(x_train, y_train))

    def predict(self, x_test):
        """
        :param x_test:  测试使用数据
        :return:
        """
        self.my_pred = self.model.predict(x_test)
        return self.my_pred

    def plt_importance(self):
        # 绘制特征重要性
        fig, ax = plt.subplots(figsize=(15, 15))
        plot_importance(self.model,
                        height=0.5,
                        ax=ax,
                        max_num_features=64)  # 最多绘制64个特征
        plt.show()  # 显示图片

    def evaluate(self, evalue_fun, y_test):  # 进行评估
        if evalue_fun == "acc":
            result = accuracy_score(y_test, self.my_pred)
            print("accuarcy:%.2f%%" % (result * 100.0))
        elif evalue_fun == "auc":
            result = roc_auc_score(y_test, self.my_pred)
            print("auc:%.2f%%" % (result))
        return result

    def get_feature_importance(self):
        # feature_dict={}
        importan_feature = self.model.feature_importance()  # 获取重要的参数
        original_feature_name = self.model.feature_name()
        # feat_name =[feature_dict.get(i,i) for i in original_feature_name]
        return importan_feature, original_feature_name

    def save_model(self, save_params):  # 模型保存
        self.model.save_model(filename=save_params.get("filename", "./model/LightGbmmodel.pkl"),   # 保存的文件路径名字
                              num_iteration=save_params.get("num_iteration", None),  # 迭代数
                              start_iteration=save_params.get("start_iteration ", 0)  # 开始迭代的索引
                              )

    def load_model(self, load_param):   # 加载模型
        models = lgb.Booster(model_file=load_param.get("fname", "./model/LightGbmmodel.pkl"),
                             params=self.params)
        return models


