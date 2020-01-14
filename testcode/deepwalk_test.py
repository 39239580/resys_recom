from code_tools.embeding_util.deepwalkembedding import deepwalk_api, embedding2dict, get_deepwalkembedding
import time

params = {"format": "mat",
          "input": "F:/kanshancup/def/deepwalkdata/blogcatalog.mat",
          "output": "F:/kanshancup/def/testcode/deepwalkmodel/blogcatalog.embeddings",
          "number_walks": 10,  # 每个节点游走的次数
          "representation_size": 64,  # 输出的嵌入长度为  128
          "walk_length": 5,  # 每次游走的长度
          "window_size": 10  # skgram  中进行编码的窗口长度
          }
start = time.time()
deepwalk_api(params)  # API 方式操作  进行编码处理

embedding2dict("F:/kanshancup/def/testcode/deepwalkmodel/blogcatalog.embeddings",
               "F:/kanshancup/def/testcode/deepwalkmodel/blogcatalog.pkl")

end = time.time()
print("整个训练与保存编码耗时为：%s" % (end-start))
key_list = ["3197", "445"]
result = get_deepwalkembedding("F:/kanshancup/def/testcode/deepwalkmodel/blogcatalog.pkl", set(key_list))
print(result)
