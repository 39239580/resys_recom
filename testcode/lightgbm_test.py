from code_tools.boosting_util.LightGbm import Lightgbm
from sklearn.model_selection import train_test_split
import pandas as pd
# import lightgbm as lgb
from sklearn.datasets import load_iris  # 加载鸢尾花数据集的包


def data_process(df_train):
    x = df_train.values[:, :-1]  # 构建 x
    y = df_train.values[:, -1]  # 构建标签
    x_train, x_test, y_train, y_test= train_test_split(x, y, test_size=0.3, random_state=25, shuffle=True)

    return x_train, x_test, y_train, y_test

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
params = {"learning_rate": 0.1, "max_depth": 3, "objective": "binary",
          "n_estimators": 100,"num_class": None}
# params = {}  # 空字典   使用默认参数

lg_model = Lightgbm(**params)  # 实例化模型  使用默认参数

lg_model.train(False, x_train, y_train)  # 训练
print(lg_model.model.params)  # 查看参数
lg_model.predict(x_test)   # 预测
print(y_test, lg_model.my_pred)
# lg_model.evaluate("acc", y_test=y_test)  # 进行评估,   # 需要调整？
lg_model.plt_importance()  # 绘制重要性
# r = lg_model.model.feature_importances_  # 打印特征
# print(r)
feat, name = lg_model.get_feature_importance()
print(feat, name)
save_params = {}
lg_model.save_model(save_params)
load_params = {}   # 加载模型
model = lg_model.load_model(load_params)  # 加载模型
