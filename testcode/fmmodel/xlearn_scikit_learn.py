# import numpy as  np
import xlearn as xl
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

# load dataset
iris_data = load_iris()
X = iris_data["data"]
y = (iris_data["target"] == 2)

x_train, x_val, y_train, y_val = train_test_split(X, y, test_size= 0.3, random_state=0)

# 创建模型
linear_model = xl.LRModel(task="binary", init=0.1,
                          epoch=10, lr=0.1, reg_lambda=1.0,
                          opt="sgd")
linear_model.fit(x_train, y_train,
                 eval_set=[x_val, y_val],
                 is_lock_free=False)

# 生成预测
y_pred = linear_model.predict(x_val)   # 生成概率值
print(y_pred)
