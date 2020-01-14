import matplotlib.pyplot as plt
# from sklearn import datasets  #  数据集
# from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score
from xgboost import XGBClassifier
from xgboost import plot_importance
# import xgboost as xgb
# from sklearn.preprocessing import OneHotEncoder  # 独热编码的包

#  需要多数据进行拆分
# 数据切割
# x_train,x_test,y_train,y_test =train_test_split(
#         data, target,test_size=0.3,random_sate =33  # 对数据进行分割
#         )
"""
xgboost 算法接口  https://xgboost.readthedocs.io/en/latest/parameter.html
"""
class Xgboost():
    def __init__(self,**params):
        self.model = XGBClassifier(learning_rate=params.get("learning_rate", 0.3),
                                   n_estimators=params.get("n_estimators", 100),  # 树的个数100,即代数
                                   max_depth=params.get("max_depth", 6),  # 树的深度
                                   min_child_weight=params.get("min_child_weight", 1),  # 叶子节点最小权重
                                   gamma=params.get("gamma", 0),  # 惩罚项中叶子节点个数前的参数
                                   reg_lambda=params.get("lambda", 1),  # lambda
                                   reg_alpha=params.get("alpha", 0),
                                   tree_method=params.get("tree_method", "auto"),
                                   subsample=params.get("subsample", 1),  # 随机选择100%样本建立决策树
                                   colsample_bytree=1,  # 随机选择80%特征建立决策树
                                   objective=params.get("objective", "multi:softmax"),  # 指定损失函数
                                   # num_class=params.get("num_class", 2),  # 不指定即为2分类
                                   scale_pos_weight=1,  # 解决样本不平衡问题
                                   random_state=27,  # 随机数
                                   )
        """
        目标函数类型
        具体查看  https://xgboost.readthedocs.io/en/latest/parameter.html
        obejctive:  默认  reg:squarederror:
        reg:squarederror:  #回归平方误差
        reg:squaredlogerror  # 上述误差上取对数
        reg:logistic logistic regression
        reg:logistic    逻辑回归
        binary:logistic    逻辑回归二分类， 输出为概率值
        binary:logitraw    逻辑回归 2分类，输出为logits之前的得分
        binary:hinge   用于二元分类的铰链损失。这使得预测为0或1，而不是产生概率。
        multi:softmax:  多分类，需要指定num_class的类别
        multi:softprob:  输出为概率  ndata*nclass 的矩阵，即，每行数据为分属类别的概率
        """

    def train(self, x_train, y_train):
        # print(self.model)
        self.model.fit(x_train, y_train)
                       # early_stopping_rounds=10  过早停止的条件
                       # verbose=True  # 是否开启冗余


    def predict(self, x_test):
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

    def evaluate(self, evalue_fun, y_test):
        if evalue_fun == "acc":
            result = accuracy_score(y_test, self.my_pred)
            print("accuarcy:%.2f%%" % (result * 100.0))
        elif evalue_fun == "auc":
            result = roc_auc_score(y_test, self.my_pred)
            print("auc:%.2f" %(result))
        return result

    def save_model(self, save_params):  # 模型保存
        self.model.save_model(fname=save_params.get("fname","./model/Catboostmodel"),   # 保存的文件路径名字
                              format=save_params.get("format","cbm"),  # 保存的数据格式
                              pool=save_params.get("pool",None)  #  训练使用的数据   模型保存成json格式，无需使用pool
                              )

    def load_model(self,load_param):   # 加载模型
        models = self.model.load_model(fname=load_param.get("fname", "./model/Catboostmodel"),
                                       format=load_param.get("format", 'cbm'))
        return models
