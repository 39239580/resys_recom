from gensim.models import Word2Vec
import random
from joblib.parallel import Parallel, delayed
import networkx as nx
import numpy as np
from tqdm import tqdm
from collections import defaultdict
from code_tools.embeding_util.node2vec_graph import Node2VecData
import os
from multiprocessing import cpu_count

"""
Base Graph Emebedding
side information  辅助信息
论文地址  https://arxiv.org/pdf/1803.02349.pdf  淘宝中使用的千亿数据嵌入 EGES算法
"""


class BGE(object):
    def __init__(self, graph_type, input_file, model_type: bool = False, cal_model_type: str = "Auto",
                 embed_dim: int =128,
                 walk_length: int =80, walk_step: int=10,
                 p: float = 1, q: float = 1, weight_key: str = "weight", workers: int =1,
                 sampling_strategy: dict =None, quiet: bool = False, temp_folder: str =None,
                 first_travel_key: str = "first_travel_key", probabilities_key: str ="probabilities",
                 neighbors_key: str ="neighbors", walk_step_key: str="num_walks", walk_length_key: str="walk_length",
                 p_key: str="p", q_key: str="q"):
        """
        :param graph_type:
        :param input_file:
        :param model_type:
        :param cal_model_type:
        :param embed_dim:
        :param walk_length:
        :param walk_step:
        :param p:
        :param q:
        :param weight_key:
        :param workers:  # 进程数
        :param sampling_strategy:
        :param quiet:
        :param temp_folder:
        :param first_travel_key:
        :param probabilities_key:
        :param neighbors_key:
        :param walk_step_key:
        :param walk_length_key:
        :param p_key:
        :param q_key:
        p与q 共同控制随机游走的倾向性，p为返回参数， q 为进出参数，
        p 越小，随机游走会节点t 的可能越大， 网络就更加注重表达网络的同质性，
        q 越小，游走到远方节点的可能性越大， 网络就更加注重表达网络的结构性。
        """
        self.cal_model_type = cal_model_type
        print("性能模型已设定:%s" % self.cal_model_type)
        self._model_set(p, q, workers)  # 选择模型
        self.model_type = model_type
        self.input_file = input_file
        self.graph_type = graph_type
        self.embed_dim = embed_dim
        self.walk_length = walk_length
        self.walk_step = walk_step
        self.weight_key = weight_key
        self.sampling_strategy = sampling_strategy
        self.quiet = quiet
        self.first_travel_key = first_travel_key
        self.probabilities_key = probabilities_key
        self.neighbors_key = neighbors_key
        self.walk_step_key = walk_step_key
        self.walk_length_key = walk_length_key
        self.p_key = p_key
        self.q_key = q_key
        self.fileformat = self.auto_disc_format()  # 自动识别文件格式
        print("文件格式已自动识别:%s" % self.fileformat)
        self.data_graph = defaultdict(dict)
        self.temp_folder, self.require = self.condition_config()  # 配置参数
        print("参数配置完毕")
        if temp_folder:
            if not os.path.isdir(temp_folder):
                raise NotADirectoryError("temp_folder does not exist or is not a directory. ({})".format(
                    temp_folder))
            self.temp_folder = temp_folder
            self.require = "sharedmem"
        print("创建图完毕,数据加载中...")
        self.data_size, self.graph = self.auto_disc_run_model()  # 加载数据
        print("预计算概率中...")
        self._precompute_prob()
        print("随机游走中...")
        self.walks = self._random_walk()
        """
        defaultdict 当字典key不存在时，返回一个默认值，list-->[] str-->空字符串，set -->set()  int -->0
        """

    def _model_set(self, p, q, workers):
        if self.cal_model_type == "dw":   # deepwalk 模型
            self.p = 1.0
            self.q = 1.0
            self.workers = workers
        elif self.cal_model_type == "n2v":  # 采用n2v模型 手动调整参数
            self.p = p
            self.q = q
            self.workers = workers
        else:     # 自动模型"auto"
            self.p = 1.0
            self.q = 1.0
            self.workers = cpu_count()  # 自动获取cpu 数据

    def auto_disc_format(self):  # 自动识别 文件格式
        return self.input_file.split(".")[-1]

    def auto_disc_run_model(self):
        if self.model_type:
            data_size, graph_oop = self._creat_graph()
        else:
            data_size, graph_oop = self._load_data()
        return data_size, graph_oop

    def _creat_graph(self):
        graph_oop = self._create_demo_graph()
        print("启动demo 测试")
        print("Number of node :{}".format(len(graph_oop.nodes())))
        num_walks = len(graph_oop.nodes()) * self.walk_step

        print("Number of walks:{}".format(self.walk_step))

        data_size = num_walks * self.walk_length
        print("Data size (walks*length):{}".format(data_size))

        return data_size, graph_oop

    @staticmethod
    def _create_demo_graph():  # 快速生成节点
        print("启动demo测试")
        return nx.fast_gnp_random_graph(n=2000, p=0.5)

    def condition_config(self):
        if self.sampling_strategy is None:
            self.sampling_strategy = {}

        temp_folder, require = None, None
        return temp_folder, require

    def _load_data(self):
        mygraph = Node2VecData(self.fileformat, "r", self.input_file, None, None, graph_type=self.graph_type)
        if self.fileformat == "adjlist":
            graph_oop = mygraph.load_adjlist()
        elif self.fileformat == "edgelist":
            graph_oop = mygraph.load_edgelist()
        else:
            print("输入格式有误，当前输入格式为 %s" % self.fileformat)
            raise ("Unknown file format: '%s'.  Valid formats: 'adjlist', 'edgelist', "
                   "mat" % self.fileformat)
        print("Number of node :{}".format(len(graph_oop.nodes())))
        num_walks = len(graph_oop.nodes()) * self.walk_step

        print("Number of walks:{}".format(self.walk_step))

        data_size = num_walks * self.walk_length
        print("Data size (walks*length):{}".format(data_size))

        return data_size, graph_oop

    def _precompute_prob(self):
        """
        预计算每个节点的转移概率
        """
        data_graph = self.data_graph
        first_travel_done = set()
        if self.quiet:   # 若结束
            nodes_generator, _ = self.graph.get_all_node()
        else:  # 没结束
            nodes_generator = tqdm(self.graph.nodes(), desc='Computing transition probabilities')

        for source in nodes_generator:   # 遍历所有节点
            # 初始化概率字典，用于first_travel
            if self.probabilities_key not in data_graph[source]:  # 若图中不存在
                data_graph[source][self.probabilities_key] = dict()
            for current_node in self.graph.neighbors(source):  # 遍历当前节点的邻居
                # 初始化 概率字典：
                if self.probabilities_key not in data_graph[current_node]:
                    data_graph[current_node][self.probabilities_key] = dict()

                unnormalized_weights = list()
                first_travel_weights = list()
                data_neighbors = list()

                # 计算 unnormalized_weights
                for destination in self.graph.neighbors(current_node):
                    if current_node in self.sampling_strategy:
                        p = self.sampling_strategy[current_node].get(self.p_key, self.p)
                        q = self.sampling_strategy[current_node].get(self.q_key, self.q)
                    else:
                        p = self.p
                        q = self.q

                    if destination == source:   # 反向概率
                        ss_weight = self.graph[current_node][destination].get(self.weight_key, 1) * 1 / p
                    elif destination in self.graph[source]:  # 邻居连ji起点
                        ss_weight = self.graph[current_node][destination].get(self.weight_key, 1)
                    else:
                        ss_weight = self.graph[current_node][destination].get(self.weight_key, 1) * 1 / q

                    # 在随机游走中，分配unnormalized 采样策略 weight, normalize
                    unnormalized_weights.append(ss_weight)
                    if current_node not in first_travel_done:
                        first_travel_weights.append(self.graph[current_node][destination].get(self.weight_key, 1))
                    data_neighbors.append(destination)

                # Normalize
                unnormalized_weights = np.array(unnormalized_weights)
                data_graph[current_node][self.probabilities_key][source] = \
                    unnormalized_weights / unnormalized_weights.sum()

                if current_node not in first_travel_done:
                    unnormalized_weights = np.array(first_travel_weights)
                    data_graph[current_node][self.first_travel_key] = unnormalized_weights/unnormalized_weights.sum()

                # Save neighbors
                data_graph[current_node][self.neighbors_key] = data_neighbors

    @staticmethod
    def _loop_fn(inputs):
        return [item for sublist in inputs for item in sublist]

    def _random_walk(self):
        """
        生成随机游走，用于 skip-gram input 输入
        :return:  游走的list, 每次游走是一个序列
        """
        # 对于每个进程，分割walk_steps
        walk_steps_list = np.array_split(range(self.walk_step), self.workers)  # 分割成 self.workers个小数组，使用list包起来

        # 进行并行计算
        walk_results = Parallel(n_jobs=self.workers, temp_folder=self.temp_folder, require=self.require)(
            delayed(self._parallel_generrate_walks)(len(num_walks), idx) for
            idx, num_walks in enumerate(walk_steps_list, 1))

        walks = self._loop_fn(walk_results)  # 遍历每个值

        return walks

    # 并行计算模块
    def _parallel_generrate_walks(self, num_walks, cpu_num: int):
        """
        :param num_walks  步长数
        :param cpu_num:  # cpu 数量，核心数
        :return:
        data_graph  为dict
        """
        walks = list()
        print(self.quiet, type(self.quiet))

        if not self.quiet:
            # 进度条
            pbar = tqdm(total=num_walks, desc='Generating walks (CPU: {})'.format(cpu_num))
        for n_walk in range(num_walks):  # 游走的次数
            # 更新进度条
            if not self.quiet:

                pbar.update(1)

            # 随机排放节点
            shuffled_node = list(self.data_graph.keys())  # 所有节点列表
            random.shuffle(shuffled_node)  # 打乱

            # 每个节点开始随机游走
            for source in shuffled_node:  # 遍历每个节点
                # 指定的 walks_step 中跳过部分节点
                if source in self.sampling_strategy and \
                        self.walk_step_key in self.sampling_strategy[source] and \
                        self.sampling_strategy[source][self.walk_step_key] <= n_walk:
                    continue

                # 开始游走
                walk = [source]

                # 计算游走长度
                if source in self.sampling_strategy:
                    walk_length = self.sampling_strategy[source].get(self.walk_length_key, self.walk_length)
                else:
                    walk_length = self.walk_length

                # 执行游走
                while len(walk) < walk_length:

                    walk_options = self.data_graph[walk[-1]].get(self.neighbors_key, None)  # 字典数据

                    # 跳过死亡节点
                    if not walk_options:
                        break

                    if len(walk) == 1:  # 第一步
                        prob = self.data_graph[walk[-1]][self.first_travel_key]

                    else:
                        prob = self.data_graph[walk[-1]][self.probabilities_key][walk[-2]]
                    walk_to = np.random.choice(walk_options, size=1, p=prob)[0]

                    walk.append(walk_to)

                walk = list(map(str, walk))  # 转成[str]

                walks.append(walk)

        if not self.quiet:
            pbar.close()

        return walks

    def train(self, **skip_gram_params):
        """
        使用gensim 的word2Vec进行训练
        :param skip_gram_params:
        :return:  A gensim word2vec model
        """
        if "workers" not in skip_gram_params:
            skip_gram_params["workers"] = self.workers

        if "size" not in skip_gram_params:
            skip_gram_params["size"] = self.embed_dim

        print(skip_gram_params["size"])

        return Word2Vec(self.walks, **skip_gram_params)

    def get_node_info(self):  # 获取节点信息
        return self.graph.nodes(), self.graph.number_of_nodes()
