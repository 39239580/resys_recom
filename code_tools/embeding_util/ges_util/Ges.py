from code_tools.embeding_util.ges_util.Bge import BGE
import numpy as np
import tensorflow as tf
import random
import pickle
from code_tools.embeding_util.ges_util.BGE_api import get_similarnode

"""
Graph Emebedding with Side information(GES)
side information  辅助信息
论文地址  https://arxiv.org/pdf/1803.02349.pdf  淘宝中使用的千亿数据嵌入 EGES算法
"""


class GES(BGE):
    def __init__(self, graph_type, input_file, dimenson, walk_length, num_walks, feat_sequence,
                 side_info, window=10, min_count=5,
                 batch_words=1000,
                 embedding_path=None, embedding_model_path=None, embedder_type=None, seed=20,
                 model_type=False, cal_model_type="Auto", p=1, q=1, weight_key="weight", workers=1,
                 sampling_strategy=None, quiet=False, temp_folder=None):
        """
        :param graph_type:
        :param input_file:
        :param dimenson:
        :param walk_length:
        :param num_walks:
        :param feat_sequence  dict {feat_name: feat_index}  feat_index 为对特征名字下，所有节点附加信息的序号
        :param side_info:  dict 统计各特征的取值范围
        :param window
        :param:min_count
        :param batch_words:
        :param embedding_path:
        :param embedding_model_path:
        :param embedder_type:
        :param seed:
        :param model_type:
        :param cal_model_type:
        :param p:
        :param q:
        :param weight_key:
        :param workers:
        :param sampling_strategy:
        :param quiet:
        :param temp_folder:
        """
        super().__init__(graph_type=graph_type, input_file=input_file, model_type=model_type,   # 执行父类构造方法
                         cal_model_type=cal_model_type, embed_dim=dimenson, walk_length=walk_length,
                         walk_step=num_walks, p=p, q=q, weight_key=weight_key, workers=workers,
                         sampling_strategy=sampling_strategy, quiet=quiet, temp_folder=temp_folder)
        self.feat_sequence = feat_sequence
        self.side_info = side_info
        self.window = window
        self.min_counts = min_count
        self.batch_words = batch_words
        self.embedding_path = embedding_path
        self.embedding_model_path = embedding_model_path
        self.embedder_type = embedder_type
        self.seed = seed
        self.cal_type = "mean"
        self.dimenson = dimenson
        self.nodes, self.n_nodes = super().get_node_info()
        print("模型训练中...")
        self.model = super().train()  # 父类的训练方法
        print("保存训练模型...")
        self.save_model()

    def search_similar_node(self, node_name):  # 寻找相似节点
        """
        :param node_name: str
        :return:  str
        """
        return self.model.wv.most_similar(node_name)

    def save_embeddings(self):  # 保存训练好的嵌入向量
        self.model.wv.save_word2vec_format(self.embedding_path)

    def save_model(self):  # 保存训练好的模型
        self.model.save(self.embedding_model_path)

    def load_model(self):  # 加载模型
        return self.model.load(self.embedding_model_path)

    def load_embedding(self):  # 加载训练好的嵌入向量
        return self.model.wv.load_word2vec_format(self.embedding_path)

    def get_embeding(self, node_list):  # 加载部分item的嵌入向量
        return self.model.wv[node_list]

    # ------------------------------单线程版------------------------------
    def create_matrix_tensor(self):
        """
        side_info dict  {feat_name, side_size}  如{"name": 80}  取值范围
        :return:
        """
        print("创建所有side_feat的embedding矩阵...")
        side_embed_dict = dict()
        for feat_name, side_size in self.side_info.items():
            shape = [side_size, self.dimenson]
            matrix = tf.compat.v1.Variable(tf.compat.v1.random_uniform(shape=shape, minval=-1.0,
                                                                       maxval=1.0), dtype=tf.float32)
            side_embed_dict[feat_name] = matrix
        return side_embed_dict

    def create_idx_tensor(self):
        """
        side_name : list [feat1,feat2.....]
        feat_idx_dict : dict {feat_name1:idx1,....}  idx1  --->list
        idx：tensor,
        side_info:
        feat_sequence  {feat_name: fea_index_list}
        :return:
        """
        side_idx_dict = dict()
        for feat_name in [side_name for side_name in self.side_info.keys()]:
            # print(feat_name)
            idx = tf.compat.v1.Variable(self.feat_sequence[feat_name], dtype=tf.int32)
            side_idx_dict[feat_name] = idx
        return side_idx_dict

    def merge_embed(self, cal_type):
        """
        :param cal_type:  # 计算方式
        :return:
        self.n_nodes  节点数
        node_list: [node_name1,node_name2....]
        """
        print("SGE算法聚合")
        embeds_tensor_list = list()
        matrix_dict = self.create_matrix_tensor()
        idx_dict = self.create_idx_tensor()

        # 获取item 本身的嵌入向量
        item_embed_tensor = tf.compat.v1.reshape(tf.compat.v1.convert_to_tensor(self.get_embeding(self.nodes)),
                                                 shape=[self.n_nodes, 1, self.embed_dim])
        embeds_tensor_list.append(item_embed_tensor)
        for feat_name in [side_name for side_name in self.side_info.keys()]:
            embeds_tensor = tf.compat.v1.nn.embedding_lookup(matrix_dict[feat_name], ids=idx_dict[feat_name])
            embeds_tensor_list.append(tf.compat.v1.split(embeds_tensor, self.n_nodes*[1], axis=0))
        embedding_tensor = tf.compat.v1.concat(embeds_tensor_list, axis=1)

        if cal_type == "mean":
            new_embed_vec = tf.compat.v1.reduce_mean(embedding_tensor, axis=1)

        elif cal_type == "max":
            new_embed_vec = tf.compat.v1.reduce_max(embedding_tensor, axis=1)

        elif cal_type == "min":
            new_embed_vec = tf.compat.v1.reduce_min(embedding_tensor, axis=1)

        else:
            new_embed_vec = tf.compat.v1.reduce_sum(embedding_tensor, axis=1)

        with tf.compat.v1.Session() as sess:
            init_op = tf.compat.v1.global_variables_initializer()
            sess.run(init_op)
            new_embed_vec_arr = new_embed_vec.eval()  # 张量转成array

        new_embed_vec_arr = np.concatenate(new_embed_vec_arr)
        new_embed_vec_list = np.split(new_embed_vec_arr, self.n_nodes, axis=0)

        new_embed_vec_dict = dict(zip(self.nodes, new_embed_vec_list))

        return new_embed_vec_dict

    @staticmethod
    def emebed_save(dicts, pklfilename):
        with open(pklfilename, "wb") as f:  # 保存成pkl 文件
            pickle.dump(dicts, f)


def get_embeds(pkfilename, model_path, item_name_list):
    """
    :param pkfilename:  训练好的字典所在位置，pkl文件
    :param model_path:  获取模型中的相似点
    :param item_name_list:
    :return:
    """
    with open(pkfilename, "rb") as f:  # 读取pkl 字典文件
        embeds_dict = pickle.load(f)

    non_embeds = []
    out_dict = {}
    for item_name in item_name_list:
        embed = embeds_dict.get(item_name, None)
        if len(embed):
            out_dict[item_name] = embed
        else:
            non_embeds.append(item_name)
    similar_node = get_similarnode(model_path, non_embeds)
    return out_dict, similar_node


def gen_side_data():
    print("生成伪数据")
    col_name = ["name", "age", "hight", "weight", "price", "click"]
    data_size = 6301
    data = {}
    name = [i for i in range(data_size)]
    random.shuffle(name)
    age = list()
    hight = list()
    weight = list()
    price = list()
    click = list()
    for i in range(data_size):
        age.append(random.randint(1, 80))
        hight.append(random.randint(1, 200))
        weight.append(random.randint(1, 80))
        price.append(random.randint(1, 10000))
        click.append(random.randint(1, 2000))
    data["name"] = name
    data["age"] = age
    data["hight"] = hight
    data["weight"] = weight
    data["price"] = price
    data["click"] = click
    side_info = {"name": 6400, "age": 90, "hight": 201, "weight": 81, "price": 10001, "click": 2001}
    return col_name, data, side_info


if __name__ == "__main__":
    work_env = "com"
    if work_env == "home":
        path = "J:/data_set_0926/program/"
    else:
        path = "F:/kanshancup/def/"
    col_names, datas, side_infos = gen_side_data()
    graph_types = "no_di"
    input_files = path + "deepwalkdata/testdata/p2p-Gnutella08.edgelist"
    num_walkss = 10
    walk_lengths = 5
    dimensons = 64
    windows = 10
    min_counts = 5
    embedding_paths = path + "testcode/gestest/ges_model/ges_new_test_p2p.embeddings"
    embedding_new_path = path + "testcode/gestest/ges_model/ges_new_test_p2p.pkl"
    embedding_model_paths = path + "testcode/gestest/ges_model/ges_test_p2p.model"
    ges_oop = GES(graph_type=graph_types, input_file=input_files, dimenson=dimensons,
                  walk_length=walk_lengths, num_walks=num_walkss,
                  side_info=side_infos, feat_sequence=datas,
                  embedding_path=embedding_paths, window=windows, min_count=min_counts,
                  embedding_model_path=embedding_model_paths)
    embedss = ges_oop.merge_embed("mean")
    ges_oop.emebed_save(embedss, embedding_new_path)
    print("训练完成")
    search_node = ["3197", "445", "888"]
    print("加载嵌入向量")
    embedsx, non_nodes = get_embeds(embedding_new_path, embedding_model_paths, search_node)  # 加载向量
    print(embedsx)
    print(non_nodes)
