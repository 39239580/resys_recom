from code_tools.boosting_util.CATboost import Catboost
from sklearn.model_selection import train_test_split
import pandas as pd
# import lightgbm as lgb
from sklearn.datasets import load_iris  # 加载鸢尾花数据集的包
# from catboost import CatBoostClassifier

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
#  通过字典来控制是哪些使用默认参数，
# params = {}  # 使用默认参数
params = {"learning_rate": 0.1, "iterations": 1000,
          "max_depth": 3}  # 使用自定义参数
cat_model = Catboost(**params)
print(type(cat_model.model))  # 打印算法参数
train_add_param = {"plot": True}
cat_model.train(x_train, y_train, **train_add_param)
predict_add_param ={}
cat_model.predict(x_test, **predict_add_param)
print(cat_model.my_pred)  # 输出预测值   #

# cat_model.evaluate("acc", x_test)  # 需要调整？
# #ValueError: Classification metrics can't handle a mix of continuous-multioutput and binary targets
cat_model.plt_importance()  # 画图
feat, name = cat_model.get_feature_importances()  #
print(feat, name)
save_params = {}
cat_model.save_model(save_params)  # 模型保存

model = cat_model.load_model(save_params)  # 加载模型。
