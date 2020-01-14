import category_encoders as ce
import pandas as pd
"""
将分类特征转成数字编码的工作
"""


class CeEncoder(object):
    def __init__(self, encoder_type, columns_name=None):
        """
        :param encoder_type:
        :param columns_name: list， 特征名组成的列表名
        """
        if encoder_type == "BackwardDe":  # 反向差分编码
            self.encoder = ce.BackwardDifferenceEncoder(cols=columns_name)

        elif encoder_type == "BaseN":  # BaseN编码
            self.encoder = ce.BaseNEncoder(cols=columns_name)

        elif encoder_type == "Binary":  # 二值编码
            self.encoder = ce.BinaryEncoder(cols=columns_name)

        elif encoder_type == "Catboost":
            self.encoder = ce.CatBoostEncoder(cols=columns_name)

        elif encoder_type == "Hash":
            self.encoder = ce.HashingEncoder(cols=columns_name)

        elif encoder_type == "Helmert":
            self.encoder = ce.HelmertEncoder(cols=columns_name)

        elif encoder_type == "JamesStein":
            self.encoder = ce.JamesSteinEncoder(cols=columns_name)

        elif encoder_type == "LOO":    # LeaveOneOutEncoder  编码
            self.encoder = ce.LeaveOneOutEncoder(cols=columns_name)

        elif encoder_type == "ME":
            self.encoder = ce.MEstimateEncoder(cols=columns_name)   # M估计编码器

        elif encoder_type == "OneHot":
            self.encoder = ce.OneHotEncoder(cols=columns_name)

        elif encoder_type == "OridinalEncoder":   # 原始编码
            self.encoder = ce.OrdinalEncoder(cols=columns_name)

        elif encoder_type == "Sum":  # 求和编码
            self.encoder = ce.SumEncoder(cols=columns_name)

        elif encoder_type == "Polynomial":  # 多项式编码
            self.encoder = ce.PolynomialEncoder(cols=columns_name)

        elif encoder_type == "Target":  # 目标编码
            self.encoder = ce.TargetEncoder(cols=columns_name)

        elif encoder_type == "WOE":  # WOE 编码器
            self.encoder = ce.WOEEncoder(cols=columns_name)

        else:
            raise ValueError("请选择正确的编码方式")

    def _fit(self, x, y=None):
        self.encoder.fit(X=x, y=y)

    def _transform(self, x):
        return self.encoder.transform(X=x)

    def _fit_transform(self, x, y=None):
        return self.encoder.fit_transform(X=x, y=y)


if __name__ == "__main__":
    df = pd.DataFrame({'ID': [1, 2, 3, 4, 5, 6],
                       'RATING': ['G', 'B', 'G', 'B', 'B', 'G'],
                       'score': ["A", "A", "B", "C", "C", "A"]})

    columns = ["RATING"]
    df_encode = CeEncoder("Binary", columns)._fit_transform(df)  # 指定点进行操作
    df_encode_new = CeEncoder("Binary")._fit_transform(df)   # 不指定列，将对所有非数值型列进行种类编码操作
    print(df)
    print(df_encode)
    print(df_encode_new)

    # -------- 分步操作 -------------

    cr = CeEncoder("Binary", columns)
    cr._fit(df)
    print(cr._transform(df))



