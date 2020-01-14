from code_tools.embeding_util.node2vecembedding import Node2VecEmbedding, get_node2vecembedding, embedding2dict


def fn(file_type, key_list):
    if file_type == "edge":
        format_s = "edgelist"
        input_file_s = "F:/kanshancup/def/deepwalkdata/testdata/p2p-Gnutella08.edgelist"
        embedding_path_s = "F:/kanshancup/def/testcode/node2vecmodel/p2p.embeddings"
        embedding_model_path_s = "F:/kanshancup/def/testcode/node2vecmodel/p2p.model"
        embedding_pkl_path = "F:/kanshancup/def/testcode/node2vecmodel/p2p.pkl"

    elif file_type == "adj":
        format_s = "adjlist"
        input_file_s = "F:/kanshancup/def/deepwalkdata/testdata/karate.adjlist"
        embedding_path_s = "F:/kanshancup/def/testcode/node2vecmodel/karate.embeddings"
        embedding_model_path_s = "F:/kanshancup/def/testcode/node2vecmodel/karate.model"
        embedding_pkl_path = "F:/kanshancup/def/testcode/node2vecmodel/karate.pkl"
    n_s = 20
    prob_s = 0.5
    dimensions_s = 64
    walk_length_s = 10
    num_walk_s = 10
    window_s = 10
    min_count_s = 5
    batch_word_s = 4
    model_type_s = False  # 使用demo ,则使用 True, 否则，使用False

    # 训练过程
    Node2VecEmbedding(formats=format_s, n=n_s, prob=prob_s, dimensions=dimensions_s, walk_length=walk_length_s,
                      num_walks=num_walk_s, windows=window_s, min_counts=min_count_s,
                      batch_words=batch_word_s,
                      model_type=model_type_s, input_file=input_file_s, embedding_path=embedding_path_s,
                      embedding_model_path=embedding_model_path_s)

    embedding2dict(filename=embedding_path_s, pklfilename=embedding_pkl_path)

    # 获取嵌入向量值
    result = get_node2vecembedding(embedding_pkl_path, set(key_list))
    print(result)


if __name__ == "__main__":
    file_type = "edge"
    fn(file_type, ["3197", "445"])
    file_type = "adj"
    fn(file_type, ["4", "5"])
