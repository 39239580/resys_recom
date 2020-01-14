from code_tools.boosting_util.XGboost import Xgboost
from sklearn.model_selection import train_test_split
import pandas as pd
# from xgboost import XGBClassifier
# import numpy as np
import logging
# from sklearn.preprocessing import OneHotEncoder  # 独热编码的包
# import time
from sklearn.datasets import load_iris  # 加载鸢尾花数据集的包


def data_process(df_train):
    x = df_train.values[:, :-1]  # 构建 x
    y = df_train.values[:, -1]  # 构建标签
    x_train, x_test, y_train, y_test= train_test_split(x, y, test_size=0.3, random_state=25, shuffle=True)

    return x_train, x_test, y_train, y_test


logging.basicConfig(format="%(asctime)s:%(levelname)s:%(message)s", level=logging.INFO)

df = load_iris()  # 数据
print(df)
print(type(df))  # <class 'sklearn.utils.Bunch'>  数据格式
df_data = pd.concat([pd.DataFrame(df.data), pd.DataFrame(df.target)], axis=1)  # 变成df格式数据
df_data.columns = ["x1", "x2", "x3", "x4", "y"]
print(df_data.head())
df_data = df_data[df_data["y"] < 2]  # 做二分类， 做三分类不需要进行此操作
print(df_data.head(2))
x_train, x_test, y_train, y_test = data_process(df_data)

print(x_train)
print(x_test)
print(y_train)
print(y_test)

"""
通过字典来控制哪些参数使用默认参数
"""
params = {"learning_rate": 0.1, "max_depth": 3, "objective": "binary:logitraw",
          "n_estimators": 100}
# params= {} # 使用默认参数

# model =XGBClassifier(**params)
xg_model = Xgboost(**params)  # 实例化模型, 不使用  默认参数
print(xg_model.model)  # 打印算法参数

# xg_model.model.fit(x_train, y_train)

# model.fit(x_train, y_train)
# r=model.predict(x_test)
# print(r)
xg_model.train(x_train, y_train)  # 训练
xg_model.predict(x_test)   # 预测
print(y_test,xg_model.my_pred)
xg_model.evaluate("auc", y_test=y_test)  # 进行评估
xg_model.plt_importance()  # 绘制重要性
r = xg_model.model.feature_importances_  # 打印特征

print(r)

save_params = {}
xg_model.save_model(save_params)  # 保存模型
load_params = {}
xg_model.load_model(load_params)  # 加载模型


