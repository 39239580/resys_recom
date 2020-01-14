from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import OrdinalEncoder
import numpy as np
"""
三种编码方式，输入均可以是数组array, 也可以是列表，也可以是矩阵， 输出为数组
"""


class SklearnEncoder(object):
    def __init__(self, encoder_type):
        self.encoder_type = encoder_type
        if self.encoder_type == "Label":
            self.encoder_module = LabelEncoder()

        elif self.encoder_type == "OneHot":
            self.encoder_module = OneHotEncoder()

        elif self.encoder_type == "Ordinal":   # 序数编码
            self.encoder_module = OrdinalEncoder()

    def _fit(self, x, y=None):
        if self.encoder_type == "Label":
            self.encoder_module.fit(y=x)
        else:
            self.encoder_module.fit(X=x, y=y)

    def _transform(self, x):
        if self.encoder_type == "Label":
            return self.encoder_module.transform(y=x)
        else:
            return self.encoder_module.transform(X=x)

    def _fit_transform(self, x, y=None):
        if self.encoder_type == "Label":
            return self.encoder_module.fit_transform(y=x)
        else:
            return self.encoder_module.fit_transform(X=x, y=y)

    def _reversal(self, x):   # 与transform的操作刚好相反
        return self.encoder_module.inverse_transform(X=x)


if __name__ == "__main__":
    # # --------------------------------------------onehotEncoder ----------------------------------------
    # # 用法1
    # data1 = np.array([[0, 0, 3], [1, 1, 0], [0, 2, 1], [1, 0, 2]])
    # print(data1)
    # print(SklearnEncoder("OneHot")._fit_transform(data1).toarray())  # 更加高效
    # # onehot 编码的操作为  首先看那特征的取值范围，第一位有0，1 两种，第二位有0,1,2 三种，第三种有0,1,2,3 四种
    # # 总计2+3+4= 9 长度为9
    # # 用法2
    #
    # onehotencode = SklearnEncoder("OneHot")
    # onehotencode._fit(data1)
    # print(onehotencode._transform(data1).toarray())  # data1 可以换成其他的

    # # --------------------------------------------labelEncoder----------------------------------------
    # # ---------------可用于标准标签， 可以用非数字标签
    # # 用法1
    # data2 = [1, 2, 2, 6]
    # labelencode = SklearnEncoder("Label")
    # labelencode._fit(data2)
    # print(data2)
    # print(labelencode._transform(data2))  # 输出数组
    #
    # # 用法2
    # print(SklearnEncoder("Label")._fit_transform(data2))

    #  对于非数字标签
    # 用法1
    # data2_x = ["paris", "paris", "tokyo", "amsterdam"]  # 列表
    # print(data2_x)
    # print(SklearnEncoder("Label")._fit_transform(data2_x))

    data2_xx = np.array(["paris", "paris", "tokyo", "amsterdam"])
    print(data2_xx)
    print(SklearnEncoder("Label")._fit_transform(data2_xx))

    # ------------------------------------------- OrdinalEncoder---------------------------------------
    # 将编码进行反转与Labelencoder 刚好相反, 将编码转成序列
    data3 = [['Male', 1], ['Female', 3], ['Female', 2]]
    ordinalencode = SklearnEncoder("Ordinal")
    ordinalencode._fit(data3)
    print(data3)
    print(ordinalencode._transform(data3))
