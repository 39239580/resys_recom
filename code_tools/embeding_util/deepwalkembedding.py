from deepwalk.skipgram import Skipgram
from gensim.models import Word2Vec
from six.moves import range
import os
import random
from deepwalk import walks as wk
from deepwalk import graph
import sys
import logging
import traceback
import pdb
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import psutil
# import pandas as pd
import csv  # csv  文件读取数据
import pickle
import numpy as np
from multiprocessing import cpu_count

p = psutil.Process(os.getpid())  # 获取进程

try:
    p.set_cpu_affinity(list(range(cpu_count())))
except AttributeError:
    try:
        p.cpu_affinity(list(range(cpu_count())))
    except AttributeError:
        pass

logger = logging.getLogger(__name__)
LOGFORMAT = "%(asctime).19s %(levelname)s %(filename)s: %(lineno)s %(message)s"


def debug(type_, value, tb):
    if hasattr(sys, 'ps1') or not sys.stderr.isatty():
        sys.__excepthook__(type_, value, tb)
    else:
        traceback.print_exception(type_, value, tb)
    print(u"\n")
    pdb.pm()


def process(params, save=True):
    """
    :param params:  传入参数用于训练
    :param save:   是否保存 训练的数据
    :return:
    """
    if params["format"] == "adjlist":
        G = graph.load_adjacencylist(params["input"], undirected=params["undirected"])
    elif params["format"] == "edgelist":
        G = graph.load_edgelist(params["input"], undirected=params["undirected"])
    elif params["format"] == "mat":
        G = graph.load_matfile(params["input"], variable_name=params["matfile_variable_name"],
                               undirected=params["undirected"])
    else:
        print("输入格式有误，当前输入格式为 %s" % (params["format"]))
        raise Exception("Unknown file format: '%s'.  Valid formats: 'adjlist', 'edgelist', "
                        "mat" % params["format"])
    print("Number of node :{}".format(len(G.nodes())))

    num_walks = len(G.nodes()) * params["number_walks"]

    print("Number of walks:{}".format(num_walks))

    data_size = num_walks * params["walk_length"]

    print("Data size (walks*length):{}".format(data_size))

    if data_size < params["max_memory_data_size"]:
        print("Walking...")
        walks = graph.build_deepwalk_corpus(G, num_paths=params.get("number_walks", 10),
                                            path_length=params.get("walk_length", 40),
                                            alpha=params.get("alpha", 0),
                                            rand=random.Random(params.get("seed", 0)))

        print("Training...")
        model = Word2Vec(walks, size=params.get("representation_size", 64),
                         window=params.get("window_siz", 5),
                         min_count=params.get("min_count", 0),
                         sg=params.get("sg", 1),
                         hs=params.get("hs", 1),
                         workers=params.get("workers", 1))
    else:
        print("Data size{} is larger than limit(max-memory-data-size:{}).Dumping walks t disk.".format(
            data_size, params.get("max_memory_data_size")))

        print("walking...")

        walks_filebase = params["output"]+".walks"
        walks_files = wk.write_walks_to_disk(G, walks_filebase, num_paths=params.get("number_walks", 10),
                                             path_length=params.get("walk_length", 40),
                                             alpha=params.get("alpha", 0),
                                             rand=random.Random(params.get("seed", 0)),
                                             num_workers=params.get("workers", 1))

        print("Counting vertex frequecy...")  # 统计节点频次

        if params["vertex_freq_degree"]:
            vertex_counts = wk.count_textfiles(walks_files, params["workers"])

        else:
            vertex_counts = G.degree(nodes=G.iterkeys())

        print("Training...")

        walks_corpus = wk.WalksCorpus(walks_files)  # walk 语料

        model = Skipgram(sentences=walks_corpus, vocabulary_counts=vertex_counts,
                         size=params.get("representation_size"),
                         window=params.get("windows_size", 80),
                         min_count=params.get("min_count", 0),
                         trim_rule=params.get("trim_rule", None),
                         workers=params.get("workers", 8)
                         )
    if save == True:
        model.wv.save_word2vec_format(params["output"])   # 对模型进行保存
    else:
        models = model.wv.load_word2vec_format(params["output"])   # 加载模型.
        return models


def deepwalk_api(params):  # 使用API 方式处理
    paramsdata = {"debug": params.get("debug", False),
                  "format": params.get("format", "adjlist"),
                  "input": params["input"],
                  "log": params.get("log", "INFO"),
                  "max_memory_data_size": params.get("max_memory_data_size", 10e+9),
                  "matfile_variable_name": params.get("matfile_variable_name", "network"),
                  "number_walks": params.get("number_walks", 10),
                  "output": params["output"],
                  "representation_size": params.get("representation_size", 64),
                  "seed": params.get("seed", 0),
                  "undirected": params.get("undirected", True),
                  "vertex_freq_degree": params.get("vertex_freq_degree", False),
                  "walk_length": params.get("walk_length", 40),
                  "window_size": params.get("window_size", 5),
                  "workers": params.get("workers", 1)
                  }

    if paramsdata["debug"]:
        sys.excepthook = debug  # 捕获 没有捕获到的异常

    process(params=paramsdata)  # 运行


def param2dict(args):  # 命令参数转成字典 方便 进行统一调用接口
    new_params = {"debug": args.debug,
                  "format": args.format,
                  "input": args.input,
                  "log": args.log,
                  "max-memory-data-size": args.max_memory_data_size,
                  "matfile-variable-name": args.matfile_variable_name,
                  "number_walks": args.number_walks,
                  "output": args.output,
                  "representation-size": args.representation_size,
                  "seed": args.seed,
                  "undirected": args.undirected,
                  "vertex-freq-degree": args.verte_freq_degree,
                  "walk-length": args.walk_length,
                  "window-size": args.window_size,
                  "workers": args.workers
                  }
    return new_params


def deepwalk_command():  # 使用命令行操作
    parser = ArgumentParser("deepwalk",
                            formatter_class=ArgumentDefaultsHelpFormatter,
                            conflict_handler='resolve')
    parser.add_argument("--debug", dest="debug", action='store_true', default=False,
                        help="drop a debugger if an exception is raised.")

    parser.add_argument('--format', default='adjlist',
                        help='File format of input file')

    parser.add_argument('--input', nargs='?', required=True,
                        help='Input graph file')

    parser.add_argument("-l", "--log", dest="log", default="INFO",
                        help="log verbosity level")

    parser.add_argument('--matfile-variable-name', default='network',
                        help='variable name of adjacency matrix inside a .mat file.')

    parser.add_argument('--max-memory-data-size', default=1000000000, type=int,
                        help='Size to start dumping walks to disk, instead of keeping them in memory.')

    parser.add_argument('--number-walks', default=10, type=int,
                        help='Number of random walks to start at each node')

    parser.add_argument('--output', required=True,
                        help='Output representation file')

    parser.add_argument('--representation-size', default=64, type=int,
                        help='Number of latent dimensions to learn for each node.')

    parser.add_argument('--seed', default=0, type=int,
                        help='Seed for random walk generator.')

    parser.add_argument('--undirected', default=True, type=bool,
                        help='Treat graph as undirected.')

    parser.add_argument('--vertex-freq-degree', default=False, action='store_true',
                        help='Use vertex degree to estimate the frequency of nodes '
                             'in the random walks. This option is faster than '
                             'calculating the vocabulary.')

    parser.add_argument('--walk-length', default=40, type=int,
                        help='Length of the random walk started at each node')

    parser.add_argument('--window-size', default=5, type=int,
                        help='Window size of skipgram model.')

    parser.add_argument('--workers', default=1, type=int,
                        help='Number of parallel processes.')

    args = parser.parse_args()

    numeric_level = getattr(logging, args.log.upper(), None)

    logging.basicConfig(format=LOGFORMAT)

    logger.setLevel(numeric_level)

    if args.debug:
        sys.excepthook = debug

    process(param2dict(args))


def embedding2dict(filename, pklfilename):  # 将 编码好的数据转成dict, 保存成pkl 对象数据
    """
    :param filename:  加载的embedding 文件数据
    :return:
    """
    random_seed = random.randint(1, 3)  # pandas 方式没有处理好，故前面三种
    dicts = {}
    # 通过四种方式对 embedding 文件进行加载，任意选择一种就好
    # 1. csv 模块直接读取
    if random_seed == 1:
        with open(filename, "r", encoding="utf-8") as file:
            csv_reader = csv.reader(file)  # 得到的是一个对象
            for line in csv_reader:
                # print(line)  # 读取每行文件
                lines = line[0].split(" ")
                if len(lines) == 2:
                    continue
                # print(lines)  # ["1","2","3"]
                dicts[lines[0]] = np.array(list(map(np.float32, lines[1:])))  # 将字符列表转成 数组，再转成字典进行保存

    # 2.直接读取
    elif random_seed == 2:
        with open(filename, "r", encoding="utf-8") as file:
            for line in file:
                # print(line)
                lines = line.split(" ")
                if len(lines) == 2:
                    continue
                # print(lines)  # ["1","2","3"]
                dicts[lines[0]] = np.array(list(map(np.float32, lines[1:])))  # 将字符列表转成 数组，再转成字典进行保存

    # 3.整行读入
    # dicts = {}
    elif random_seed == 3:
        with open(filename, "r", encoding="utf-8") as file:
            reader = file.readlines()  # 一次性读入    缺点是 将数据一次性读取到内存中， 速度块，数据量大占用内存
            for line in reader:
                lines = line.split(" ")
                if len(lines) == 2:
                    continue
                # print(lines)  # ["1","2","3"]
                dicts[lines[0]] = np.array(list(map(np.float32, lines[1:])))  # 将字符列表转成 数组，再转成字典进行保存

    # 4.pandas 读入
    # df=pd.read_csv(filename,"rb")
    # 写入数据
    with open(pklfilename, "wb") as f:  # 保存成pkl 文件
        pickle.dump(dicts, f)


def get_deepwalkembedding(pklfiname, key_list):
    """
    :param pklfiname:   字典所在位置
    :param key_list:   需要查询的embedding  所在位置
    :return:
    """
    embededdict = {}
    with open(pklfiname, "rb") as f:  # 读取pkl 字典文件
        embeddingdict = pickle.load(f)
        # print(embeddingdict)

    # print(embeddingdict)
    for key in key_list:
        print(key)
        embededdict[key] = embeddingdict[key]

    return embededdict


if __name__ == "__main__":
    params = {"format": "mat",
              "input": "F:/kanshancup/def/deepwalkdata/blogcatalog.mat",
              "output": "F:/kanshancup/def/testcode/deepwalkmodel/blogcatalog.embeddings",
              "number_walks": 80,  # 每个节点游走的次数
              "representation_size": 64,   # 输出的嵌入长度为  128
              "walk_length": 40,  # 每次游走的长度
              "window_size": 10   # skgram  中进行编码的窗口长度
              }
    deepwalk_api(params)  # API 方式操作

    # sys.exit( deepwalk_command())  # 使用命令行操作,  使用命令行时，打开
    # 读取数据进来

    embedding2dict("F:/kanshancup/def/testcode/deepwalkmodel/blogcatalog.embeddings",
                   "F:/kanshancup/def/testcode/deepwalkmodel/blogcatalog.pkl")

    key_list = ["3197", "445"]  #
    result = get_deepwalkembedding("F:/kanshancup/def/testcode/deepwalkmodel/blogcatalog.pkl", set(key_list))
    print(result)
