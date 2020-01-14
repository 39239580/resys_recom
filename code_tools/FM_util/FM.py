import xlearn as xl
from code_tools.dataformat_util.csv2libffm import get_libffm
"""
xlearn 工具的API接口
https://xlearn-doc-cn.readthedocs.io/en/latest/python_api/index.html
"""


#  数据格式
#  LR 与FM 算法， 输入数据格式必须是CSV或libsvm， FFM 算法必须是libffm 格式
"""
libsvm format:

   y index_1:value_1 index_2:value_2 ... index_n:value_n

   0   0:0.1   1:0.5   3:0.2   ...
   0   0:0.2   2:0.3   5:0.1   ...
   1   0:0.2   2:0.3   5:0.1   ...

CSV format:

   y value_1 value_2 .. value_n

   0      0.1     0.2     0.2   ...
   1      0.2     0.3     0.1   ...
   0      0.1     0.2     0.4   ...

libffm format:

   y field_1:index_1:value_1 field_2:index_2:value_2   ...

   0   0:0:0.1   1:1:0.5   2:3:0.2   ...
   0   0:0:0.2   1:2:0.3   2:5:0.1   ...
   1   0:0:0.2   1:2:0.3   2:5:0.1   ...
   
xLearn 还可以使用","作为数据的分隔符，例如:

libsvm format:

   label,index_1:value_1,index_2:value_2 ... index_n:value_n

CSV format:

   label,value_1,value_2 .. value_n

libffm format:

   label,field_1:index_1:value_1,field_2:index_2:value_2 ...
注意，如果输入的 csv 文件里不含 y 值，用户必须手动向其每一行数据添加一个占位符 (同样针对测试数据)。否则，xLearn 会将第一个元素视为 y.

LR 和 FM 算法的输入可以是 libffm 格式，xLearn 会忽略其中的 field 项并将其视为 libsvm 格式。
"""


def model_train(input_params, model_type="FM", learning_way="offline", isDMatrix=False, isreturn=False):  # 创建模型并进行训练与继续训练
    """
    :param input_params:  存放相关参数
    :param model_type: 选择机器学习算法
    :param learning_way: 选择 学习方式， 提供在线与离线学习方式
    :param isreturn: 是否返回 对象
    :return:

    params["param"] 参数

    param={"task":"reg","lr"：0.2，"lambda":0.002}   # 回归任务
    param={"task":"binary","lr":0.2,"lambda":0.002}  # 分类任务

    """
    params = {"task": input_params.get("task", "binary"),
              "lr": input_params.get("lr", 0.2),
              "lambda": input_params.get("lambda", 0.002),  # L2 的正则化参数
              # "L2": input_params.get("L2", 0.00002),
              "k": input_params.get("k", 4),  # 隐向量长度
              "epoch": input_params.get("epoch", 1000),  # 迭代1000次，根据自己需要设置
              "fold": input_params.get("fold", 3),  # 默认情况下使用 3折交叉验证
              "opt": input_params.get("fold", "adagrad")   # 默认使用adgrad, sgd, ftrl
              }

    if input_params.get("nthread", None):  # 手动设置了核数，进行参数的传递
        params["nthread"] = input_params["nthread"]

    if input_params.get("metric", None):  # 判断是否有 手动输入，有进行下一步判断   评价指标
        if params["task"] == "reg":  # 回归任务而言
            if input_params["metric"] not in ["mae", "mape", "rmse"]:  # 若不在列表中
                raise AttributeError
            else:
                params["metric"] = input_params["metric"]

        elif params["task"] == "binary":  # 分类任务
            if input_params["metric"] not in ["acc", "prec", "f1", "auc"]:  # 若不在列表中
                raise AttributeError
            else:
                params["metric"] = input_params["metric"]

    if params["opt"] == "ftrl":
        params["alpha"] = input_params.get("alpha", 0.002)
        params["beta"] = input_params.get("beta", 0.8)
        params["lambda_1"] = input_params.get("lambda_1", 0.001)
        params["lambda_2"] = input_params.get("lambda_2", 1.0)

    model = creat_model(model_type)  # 创建模型对象

    if isDMatrix:  # 需要转成libffm
        model.setValidate(get_libffm(input_params["trainPath"]))
    else:
        model.setTrain(train_path=input_params["trainPath"])  # 训练集一定要有

    if input_params.get("valPath", None):  # 有验证集
        if isDMatrix:  # 需要转成libffm
            model.setValidate(get_libffm(input_params["valPath"]))
        else:
            model.setValidate(val_path=input_params["valPath"])   # 验证集可有可无， 自动过早停止

    else:   # 无验证集的情况下
        if not input_params.get("stop_window", None):  # 无验证集情况下，没有设过早停止的话，自动禁止过早停止
            model.disableEarlyStop()
        else:
            params["stop_window"] = input_params["stop_window"]

    if input_params.get("disableLockFree", False):  # 禁止多核无锁学习
        model.disableLockFree()  # 此功能，计算的结果是确定性的。

    if input_params.get("disableNorm", False):   # 禁止 归一化操作
        model.disableNorm()

    if input_params.get("Quiet_Model", False):   # 开启安静模式, 不计算任何评价指标， 提高训练速度
        model.setQuiet()

    if input_params.get("Cross-Val", False):  # 开启交叉功能
        model.cv(param=params)  # 开启交叉验证程式

    if input_params.get("OnDisk", False):  # 仅仅使用磁盘训练  默认关闭
        model.setOnDisk()

    if input_params.get("Nobin", False):   # 开启不产生bin 文件， 默认关闭，即产生bin文件
        model.setNoBin()

    if learning_way == "offline":  # 离线学习
        if input_params.get("modelFormat", None) == "txt":  # 保存成txt文件  模型保存成txt文件
            model.setTXTModel(input_params["modelTxtPath"])
        model.fit(param=params, model_path=input_params["modelPath"])   # 模型保存

    elif learning_way == "online":  # 在线学习
        if input_params.get("modelFormat", None) == "txt":
            model.setTXTModel(input_params["modelTxtPath"])   # 保存成txt 文件

        model.setPreModel(pre_model_path=input_params["modelPath"])
        model.fit(param=params, model_path=input_params["modelPath"])
    if isreturn:  # 是否返回模型对象
        return model


def model_predict(model_type, params, isDMatrix):
    """
    :param model_type:  model 对象
    :param params:   预测中将使用的参数
    :return:
    params={"outType":"sigmoid", "testPath":"./data/testdata",
            "modelPath":"", ""
    """
    model = creat_model(model_type)  # 模型对象
    if params.get("outType", None) == "sigmoid":
        model.setSigmoid()   # 将预测结果的分数，直接转成(0-1) 之间的数
    elif params.get("outType", None) == "sign":
        model.setSign()   # 将预测结果的分数， 直接转成0或者1 的数
    if isDMatrix:
        model.setTest(get_libffm(params["testPath"]))
    else:
        model.setTest(test_path=params["testPath"])
    model.predict(params["modelPath"], params["ouputPath"])  # 将输出结果进行保存


def creat_model(model_type):  # 创建模型对象
    if model_type == "FM":
        model = xl.create_fm()

    elif model_type == "FFM":
        model = xl.create_ffm()

    else:  # 采用默认的线性模型
        model = xl.create_linear()

    return model


if __name__ == "__main__":
    params = {"task": "binary", "lr": 0.2, "lambda": 0.002, 'metric': 'acc',
              "trainPath": "F:/kanshancup/def/FMdata/",
              "modelPath": "F:/kanshancup/def/testcode/fmmodel/FFM/FFMmodel.out"}
    model_train(params=params, isreturn=False)  # 训练过程
