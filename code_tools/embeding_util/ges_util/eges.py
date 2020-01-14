from gensim.models import Word2Vec
from code_tools.embeding_util.ges_util.Bge import BGE
import tensorflow as tf


class EGES(BGE):
    def __init__(self, num_walk: int = 0, walk_length: int = 0, wdinows_size: int =0,
                 num_negativates_samples: int =10000,  embed_dimension: int = 128,
                 graph_type:str ="no_di", input_file=None, model_type=False,
                 cal_model_type="Auto",p:float=1.0, q:float=1.0,workers=1,
                 sampling_strategy=None, temp_folder=None):
        """
        :param num_walk:
        :param walk_length:
        :param wdinows_size:
        :param num_negativates_samples:
        :param embed_dimension:
        :param graph_type:
        :param input_file:
        :param model_type:
        :param cal_model_type:
        :param p:
        :param q:
        :param workers:
        :param sampling_strategy:
        :param temp_folder:
        """
        super().__init__(graph_type=graph_type, input_file= input_file, model_type=model_type,
                         cal_model_type=cal_model_type, embed_dim=embed_dimension, walk_length=walk_length,
                         walk_step=num_walk, p=p, q=q, workers=workers,
                         sampling_strategy=sampling_strategy, temp_folder=temp_folder)
        self.wdinows_size = wdinows_size
        self.num_negativates_samples = num_negativates_samples
        self.walkses = super().walks











# def _inference(inpu_tensor, regularizer, ifregularizer=True):  # 前向推断
#     """
#     :param inpu_tensor:  [None, n+1, dim]  即batch_size, 1个自身embed, n个附加信息embed,
#     dim为嵌入向量维度信息
#     :param regularizer:
#     :param ifregularizer:
#     :return:
#     """
#     with tf.compat.v1.variable_scope("embedding_part", reuse=tf.compat.v1.AUTO_REUSE):
#
#         feature_embedding_weights = tf.compat.v1.get_variable(name="embedding_weights", shape=[n_feat, 1],
#                                                               initializer=tf.compat.v1.random_normal_initializer(
#                                                               mean=1.0,
#                                                               stddev=0.01,
#                                                               dtype=tf.float32
#                                                               ))
#
#         log_feature_embed_weights = tf.compat.v1.math.log(feature_embedding_weights)  # 取对数的 权重
#
#         if ifregularizer:
#             tf.compat.v1.add_to_collection("losses", regularizer(feature_embedding_weights))
#
#         feature_embedding_bias = tf.compat.v1.get_variable(name="embedding_weights", shape=[n_feat, 1],
#                                                            initializer=tf.compat.v1.constant_initializer(0.0))
#
#         weighted_denominator = tf.compat.v1.reduce_sum(log_feature_embed_weights)  # 加权分母
#         out = tf.compat.v1.add(tf.compat.v1.multiply(inpu_tensor, log_feature_embed_weights),
#                                feature_embedding_bias) # 得到的还是[None, n+1, dim]
#
#         #
#     return out
















