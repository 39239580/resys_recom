import networkx as nx  # 构建图的包
from node2vec import Node2Vec
from node2vec.edges import EdgeEmbedder, HadamardEmbedder, WeightedL1Embedder, WeightedL2Embedder
import pickle as pkl
from code_tools.embeding_util.deepwalkembedding import embedding2dict
from code_tools.embeding_util.node2vec_graph import Node2VecData
from gensim.models import Word2Vec


class Node2VecEmbedding(object):
    def __init__(self, formats=None, n=2000, prob=0.5, dimensions=128, walk_length=80, num_walks=10,
                 p=1.0, q=1.0, graph_type="no_di",
                 weight_key="weight", works=1, sampling_strategy=None,
                 quiet=False, temp_folder=None, windows=5, min_counts=5, batch_words=10000,
                 embedding_path=None, embedding_model_path=None, embedder_type=None,
                 model_type=False, input_file=None, matfile_variable_name="network"):
        """
        :param formats: 手动传入文件的格式，若不传入，则使用demo 数据测试
        :param n: # demo 中创建的节点数量
        :param prob:  # demo 中创建边的概率
        :param dimensions:  int 默认128 嵌入维度
        :param walk_length: int 默认80 游走步长，每次游走的节点数
        :param num_walks: int 默认10 每个节点的游走步数
        :param p: 返回的超参数  默认为1.0
        :param q:  输入输出参数， 默认为1.0
        :param graph_type: 初始化图类型， 默认为None, 即构建无向图
        :param weight_key: str 默认"weight"  加权图中，权重关键字
        :param works: int 默认1  并行数
        :param sampling_strategy: dict None  节点特定的采样策略，支持设置节点特定的“q”、“p”、“num_walks”
        和“walk_length”。准确地使用这些键。如果未设置，将使用在对象初始化上传递的全局初始化
        :param quiet: bool False
        :param temp_folder:  str None   临时文件
        :param windows:   # 窗口长度即当前词到预测词之间的最大距离
        :param min_counts:   # 词频低于min_counts将进行过滤
        :param batch_words:  #  样本批大小
        :param embedding_path:   # 嵌入向量保存的位置
        :param embedding_model_path:  # 嵌入模型保存的位置
        :param embedder_type:  # 边嵌入类型
        :param model_type: 模式类型， boolean  True时， 使用demo 数据。False 时，使用的手动模式数据
        :param input_file: # 手动 数据文件路径
        :param matfile_variable_name:
        p=q=1时，等价于deepwalk
        """
        self.n = n
        self.prob = prob   # 创建边的概率
        self.formats = formats   # 文件格式
        self.dimensions = dimensions   # 嵌入维度
        self.walk_length = walk_length  # 游走步长，每次游走的节点数
        self.num_walks = num_walks   # 每个节点的游走步数
        self.p = p  # 返回超参数
        self.q = q
        self.graph_type = graph_type
        self.weight_key = weight_key
        self.works = works
        self.sampling_strategy = sampling_strategy
        self.quiet = quiet
        self.temp_folder = temp_folder
        self.windows = windows
        self.min_counts = min_counts
        self.batch_words = batch_words
        self.embedding_path = embedding_path
        self.embedding_model_path = embedding_model_path
        self.embedder_type = embedder_type
        self.model_type = model_type
        self.input_file = input_file
        self.matfile_variable_name = matfile_variable_name
        print("加载数据中，并创建计算图...")
        self.data_size, self.graph = self.load_data()  # 加载数据并创建计算图
        print("随机游走中...")
        self.node2vec = self.gen_walks()
        print("训练嵌入节点...")
        self.model = self.embed_nodes()
        print("保存嵌入向量...")
        self.save_embeddings()  # 保存嵌入向量、
        print("保存模型...")
        self.save_model()   # 保存模型
        print("完成！")

    def load_data(self):
        if self.model_type:
            graph_oop = self.creat_graph()
            print("启动demo测试")

        else:
            mygraph = Node2VecData(self.formats, "r", self.input_file, None, None, graph_type=self.graph_type)  # 加载图对象
            if self.formats == "adjlist":
                graph_oop = mygraph.load_adjlist()
            elif self.formats == "edgelist":
                graph_oop = mygraph.load_edgelist()
            # elif self.formats == "mat":
            #     graph_oop = graph.load_matfile(self.input_file, variable_name=self.matfile_variable_name,
            #                                    undirected=self.undirected)
            else:
                print("输入格式有误，当前输入格式为 %s" % self.formats)
                raise Exception("Unknown file format: '%s'.  Valid formats: 'adjlist', 'edgelist', "
                                "mat" % self.formats)

        print("Number of node :{}".format(len(graph_oop.nodes())))
        num_walks = len(graph_oop.nodes()) * self.num_walks

        print("Number of walks:{}".format(self.num_walks))

        data_size = num_walks * self.walk_length
        print("Data size (walks*length):{}".format(data_size))

        return data_size, graph_oop

    def creat_graph(self):  # 创建图
        graph_oop = nx.fast_gnp_random_graph(n=self.n, p=self.prob)
        return graph_oop

    def gen_walks(self):  # 产生随机游走与预计算概率
        node2vec = Node2Vec(self.graph,
                            dimensions=self.dimensions,
                            walk_length=self.walk_length,
                            num_walks=self.num_walks,
                            p=self.p,
                            q=self.q,
                            weight_key=self.weight_key,
                            workers=self.works,
                            sampling_strategy=self.sampling_strategy,
                            quiet=self.quiet,
                            temp_folder=self.temp_folder
                            )

        return node2vec

    def embed_nodes(self):  # 训练嵌入节点
        #  使用的word2vec 中的参数训练
        model = self.node2vec.fit(window=self.windows, min_count=self.min_counts,
                                  batch_words=self.batch_words)
        # model = self.node2vec.fit(window=self.windows, min_count=self.min_counts,
        #                           batch_words=self.batch_words)
        """ word2vec 
        windows  窗口长度，当前词到预测位置的最大距离， 默认为5
        min_counts  忽略词频低于5 的词
        batch_words  样本批次大小
        """
        return model

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

    def edges(self):   # 接下来几部一般不使用  ，边嵌入  获取边嵌入
        if self.embedder_type == "Edge":
            edges_embs = EdgeEmbedder(keyed_vectors=self.model.wv)

        elif self.embedder_type == "Wl1":
            edges_embs = WeightedL1Embedder(keyed_vectors=self.model.wv)

        elif self.embedder_type == "Wl2":
            edges_embs = WeightedL2Embedder(keyed_vectors=self.model.wv)

        else:
            edges_embs = HadamardEmbedder(keyed_vectors=self.model.wv)
        return edges_embs

    def get_edges_embes(self, tuple_params):
        return self.edges()[tuple_params]

    def save_edges_embed(self, tupe_params, edges_embedding_path):
        edges_kv = self.edges().as_keyed_vectors()
        edges_kv.most_similar(tupe_params)
        edges_kv.save_word2vec_format(edges_embedding_path)


def get_similarnode(model_path, node_name):   # 获取相似节点
    node2vec_model = Word2Vec.load(model_path)
    similar_node = node2vec_model.most_similar(node_name)  # 查找相似节点
    return similar_node


def get_node2vecembedding(pkfilename, key_list):  # 转成 字典的形式获取出来
    """
    :param pkfilename:   字典所在位置
    :param key_list:    需要查询的embedding  所在位置
    :return:
    """
    embededdict = {}
    with open(pkfilename, "rb") as f:  # 读取pkl 字典文件
        embeddingdict = pkl.load(f)

    for key in key_list:
        print(key)
        embededdict[key] = embeddingdict[key]
    return embededdict


if __name__ == "__main__":
    format_s = "edgelist"
    n_s = 20
    prob_s = 0.5
    dimensions_s = 64
    walk_length_s = 10
    num_walk_s = 10
    window_s = 10
    min_count_s = 5
    batch_word_s = 4
    model_type_s = False    # 使用demo ,则使用 True, 否则，使用False
    input_file_s = "F:/kanshancup/def/deepwalkdata/testdata/p2p-Gnutella08.edgelist"
    embedding_path_s = "F:/kanshancup/def/testcode/node2vecmodel/p2p.embeddings"
    embedding_model_path_s = "F:/kanshancup/def/testcode/node2vecmodel/p2p.model"
    # 训练过程
    Node2VecEmbedding(formats=format_s, n=n_s, prob=prob_s, dimensions=dimensions_s, walk_length=walk_length_s,
                      num_walks=num_walk_s, windows=window_s, min_counts=min_count_s,
                      batch_words=batch_word_s,
                      model_type=model_type_s, input_file=input_file_s, embedding_path=embedding_path_s,
                      embedding_model_path=embedding_model_path_s)

    embedding_pkl_path = "F:/kanshancup/def/testcode/node2vecmodel/p2p.pkl"
    embedding2dict(filename=embedding_path_s, pklfilename=embedding_pkl_path)

    key_list = ["3197", "445"]
    # 获取嵌入向量值
    result = get_node2vecembedding(embedding_pkl_path, set(key_list))
    print(result)

    # print(get_similarnode(embedding_model_path_s, "8888"))
