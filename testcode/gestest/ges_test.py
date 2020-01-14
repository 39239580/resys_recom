from code_tools.embeding_util.ges_util.Ges import GES
import random


def gen_side_data():
    col_name = ["name", "age", "hight", "weight", "price", "click"]
    data_size = 6301
    data = {}
    name = [i for i in range(data_size)]
    random.shuffle(name)
    for i in range(data_size):
        data[i] = []
        data[i].append(name[i])
        data[i].append(random.randint(1, 91))
        data[i].append(random.randint(1, 201))
        data[i].append(random.randint(1, 81))
        data[i].append(random.randint(1, 10000))
        data[i].append(random.randint(1, 2000))
    side_infos = {"name": 6301, "age": 90, "hight": 200, "weight": 80, "price": 10000, "click": 2000}
    return col_name, data, side_infos


def process(env):
    if env == "com":
        file_path = "F:/kanshancup/def/testcode/"
    else:
        file_path = "F:"
    col_name,  data, side_info = gen_side_data()

    new_data = {"feat_name": col_name, "index": data}
    num_walks_s = 10
    walk_length_s = 10
    dimenson_s = 64
    n_negtive_sample_s = 100
    n_side_info_s = len(col_name)
    windows_s = 10
    min_counts_s = 5
    batch_words_s = 1000
    graph_type_s = "no_di",
    input_file_s = "F:/kanshancup/def/deepwalkdata/testdata/p2p-Gnutella08.edgelist"
    all_node_side_info_s = new_data
    embedding_path_s = file_path + "gestest/test_p2p.embeddings"
    embedding_model_path_s = file_path + "gestest/test_p2p.model"

    GES(graph_type_s, )

process("com")
