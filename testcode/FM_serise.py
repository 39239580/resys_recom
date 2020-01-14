from code_tools.FM_util.FM import model_train, model_predict
# ---------------------------------------criteo_ctr数据集(在线广告预测)----------------------------------------#
#   分类 使用的数据集为  criteo_ctr 数据集格式为  txt 文件， 使用FFM模型进行 分解  FFM
# train_params = {"task": "binary", "lr": 0.2, "lambda": 0.002, 'metric': 'acc',
#                 "trainPath": "F:/kanshancup/def/FMdata/data/criteo_ctr/small_train.txt",
#                 # "valPath": "F:/kanshancup/def/FMdata/data/criteo_ctr/small_test.txt",
#                 "modelPath": "F:/kanshancup/def/testcode/fmmodel/FFM/criteo_ctr/FFMmodel.out"}
# model_train(input_params=train_params, model_type="FFM", isreturn=False)  # 训练过程
#
# test_params = {"testPath": "F:/kanshancup/def/FMdata/data/criteo_ctr/small_test.txt",
#                "modelPath": "F:/kanshancup/def/testcode/fmmodel/FFM/criteo_ctr/FFMmodel.out",
#                "ouputPath": "F:/kanshancup/def/testcode/fmmodel/FFM/criteo_ctr/FFMoutput.txt",
#                "outType": "sign"}
# model_predict("FFM", params=test_params)

#    分类  同样使用数据集criteo_crt   FM
# train_params = {"task": "binary", "lr": 0.2, "lambda": 0.002, 'metric': 'acc',
#                 "trainPath": "F:/kanshancup/def/FMdata/data/criteo_ctr/small_train.txt",
#                 "modelPath": "F:/kanshancup/def/testcode/fmmodel/FM/criteo_ctr/FMmodel.out"}
# model_train(input_params=train_params, isreturn=False)  # 训练过程
#
# test_params = {"testPath": "F:/kanshancup/def/FMdata/data/criteo_ctr/small_test.txt",
#                "modelPath": "F:/kanshancup/def/testcode/fmmodel/FM/criteo_ctr/FMmodel.out",
#                "ouputPath": "F:/kanshancup/def/testcode/fmmodel/FM/criteo_ctr/FMoutput.txt",
#                "outType": "sign"}
# model_predict("FM", params=test_params)

#   分类   同样使用数据集criteo_crt  linear
# train_params = {"task": "binary", "lr": 0.2, "lambda": 0.002, 'metric': 'acc',
#                 "trainPath": "F:/kanshancup/def/FMdata/data/criteo_ctr/small_train.txt",
#                 "modelPath": "F:/kanshancup/def/testcode/fmmodel/LN/criteo_ctr/FMmodel.out"}
# model_train(input_params=train_params, model_type="linear", isreturn=False)  # 训练过程
#
# test_params = {"testPath": "F:/kanshancup/def/FMdata/data/criteo_ctr/small_test.txt",
#                "modelPath": "F:/kanshancup/def/testcode/fmmodel/LN/criteo_ctr/FMmodel.out",
#                "ouputPath": "F:/kanshancup/def/testcode/fmmodel/LN/criteo_ctr/FMoutput.txt",
#                "outType": "sign"}
# model_predict("LN", params=test_params)

#  ---------------------------------------------------------------------------------------------------


#  ----------------------------------------house_price数据集------------------------------------------
#   house_price 数据集  数据格式为csv格式，不能用于FFM操作， 需要进行格式转换
#   分类 使用的数据集为  house_price 数据集格式为  txt 文件， 使用FFM模型进行 分解  FFM
train_params = {"task": "binary", "lr": 0.2, "lambda": 0.002, 'metric': 'acc',
                "trainPath": "F:/kanshancup/def/FMdata/data/house_price/house_price_train.txt",
                # "valPath": "F:/kanshancup/def/FMdata/data/criteo_ctr/small_test.txt",
                "modelPath": "F:/kanshancup/def/testcode/fmmodel/FFM/house_price/FFMmodel.out"}
model_train(input_params=train_params, model_type="FFM", isDMatrix=False, isreturn=False)  # 训练过程
#
test_params = {"testPath": "F:/kanshancup/def/FMdata/data/house_price/house_price_test.txt",
               "modelPath": "F:/kanshancup/def/testcode/fmmodel/FFM/house_price/FFMmodel.out",
               "ouputPath": "F:/kanshancup/def/testcode/fmmodel/FFM/house_price/FFMoutput.txt",
               "outType": "sign"}
model_predict("FFM", params=test_params, isDMatrix=False)    # 使用在线转libffm模块，将isDMatrix设置为True


#  分类 使用的数据集为  house_price 数据集格式为  txt 文件， 使用FFM模型进行 分解  FM
# train_params = {"task": "binary", "lr": 0.2, "lambda": 0.002, 'metric': 'acc',
#                 "trainPath": "F:/kanshancup/def/FMdata/data/house_price/house_price_train.txt",
#                 # "valPath": "F:/kanshancup/def/FMdata/data/criteo_ctr/small_test.txt",
#                 "modelPath": "F:/kanshancup/def/testcode/fmmodel/FM/house_price/FMmodel.out"}
# model_train(input_params=train_params, model_type="FM", isDMatrix=False, isreturn=False)  # 训练过程
#
# test_params = {"testPath": "F:/kanshancup/def/FMdata/data/house_price/house_price_test.txt",
#                "modelPath": "F:/kanshancup/def/testcode/fmmodel/FM/house_price/FMmodel.out",
#                "ouputPath": "F:/kanshancup/def/testcode/fmmodel/FM/house_price/FMoutput.txt",
#                "outType": "sign"}
# model_predict("FM", params=test_params, isDMatrix=False)

#  分类 使用的数据集为  house_price 数据集格式为  txt 文件， 使用FFM模型进行 分解  FM
# train_params = {"task": "binary", "lr": 0.2, "lambda": 0.002, 'metric': 'acc',
#                 "trainPath": "F:/kanshancup/def/FMdata/data/house_price/house_price_train.txt",
#                 # "valPath": "F:/kanshancup/def/FMdata/data/criteo_ctr/small_test.txt",
#                 "modelPath": "F:/kanshancup/def/testcode/fmmodel/LN/house_price/LNmodel.out"}
# model_train(input_params=train_params, model_type="linear", isDMatrix=False, isreturn=False)  # 训练过程
#
# test_params = {"testPath": "F:/kanshancup/def/FMdata/data/house_price/house_price_test.txt",
#                "modelPath": "F:/kanshancup/def/testcode/fmmodel/LN/house_price/LNmodel.out",
#                "ouputPath": "F:/kanshancup/def/testcode/fmmodel/LN/house_price/LNoutput.txt",
#                "outType": "sign"}
# model_predict("linear", params=test_params, isDMatrix=False)
