from code_tools.FM_util import DeepFM
import tensorflow as tf
from sklearn.metrics import roc_auc_score
from code_tools.dataformat_util.Deepfm_DataReader import ffmasvm2deepfm_v1

# dfm_params = {
#     "use_fm": True,
#     "use_deep": True,
#     "embedding_size": 8,
#     "dropout_fm": [1.0, 1.0],
#     "deep_layers": [32, 32],
#     "dropout_deep": [0.5, 0.5, 0.5],
#     "deep_layers_activation": tf.nn.relu,
#     "epoch": 30,
#     "batch_size": 256,
#     "learning_rate": 0.001,
#     "optimizer_type": "adam",
#     "batch_norm": 1,
#     "batch_norm_decay": 0.995,
#     "l2_reg": 0.01,
#     "verbose": False,
#     "eval_metric": roc_auc_score,
#     "random_seed": 2017
# }

# 加载数据进来
filepath = "F:/kanshancup/def/FMdata/data/house_price/libffm.txt"

xi, xv, label = ffmasvm2deepfm_v1(filepath=filepath, feat_len=444)
print()


# 初始化模型对象
DFM_model = DeepFM()  # 初始化对象

DFM_model.fit(xi, xv, label)
