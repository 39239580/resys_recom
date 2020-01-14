import networkx as nx
"""
目前仅支持 adjlist文件与edgelist 文件
--format adjlist 每行的代表有关系的序列
1 2 3 4 5 6 7 8 9 11 12 13 14 1 20 22 32  
2 1 3 4 8 14 18 20 22 31
3 1 2 4 8 9 10 14 28 29 33 

--format edgelist   每行代表一种关系  
1 2
1 3
1 4
...

--format weiget-edgelist   每行代表一种关系
1 2  0.12 
1 3  0.13
1 4  0.14
"""


class Node2VecData(object):
    def __init__(self, file_type, method_type, path, nodetype=None, edgetype=None, graph_type=None):
        """
        :param file_type:  文件类型  edgelist  weighted_edge, adjlist  str 字符串类型数据
        :param method_type:  "r" or "w"  读或写
        :param path:   文件路径
        :param nodetype:  节点类型，可为 int, str, float, python ，转换到指定的数据类型上，默认为None,自动识别
        :param edgetype:  边类型， 可为int str, float, python 等类型数据
        :param graph_type:  # 用于判断  以何种方式的图读取数据
        """
        self.file_type = file_type
        self.method_type = method_type
        self.path = path
        self.nodetype = nodetype
        self.edgetype = edgetype
        self.graph_type = graph_type
        self.create_using = self.create_graph()

    def create_graph(self):  # 创建一个空图
        if self.graph_type == "no_di":  # 创建 无向图
            create_using = nx.Graph()
        elif self.graph_type == "di":  # 创建 有向图
            create_using = nx.DiGraph()
        elif self.graph_type == "mul_no_di":   # 创建多重无向图
            create_using = nx.MultiGraph()
        # elif self.graph_type == "mul_di":   # 创建多重有向图
        elif self.graph_type == "mul_di":
            create_using = nx.MultiDiGraph()
        else:
            create_using = nx.Graph()
        return create_using

    def process(self):
        if self.method_type == "r":
            return self.loaddata()
        else:
            self.savedata()

    def loaddata(self):
        if self.file_type == "edgelist":
            graph = self.load_edgelist()
        elif self.file_type == "weighted_edge":
            graph = self.load_weighted_edgelist()
        elif self.file_type == "adjlist":
            graph = self.load_adjlist()
        else:
            raise ValueError("load_adjlist is error!")
        return graph

    def savedata(self):
        if self.file_type == "edge":
            self.save_edgelist()
        elif self.file_type == "weighted_edge":
            self.save_weighted_edgelist()
        elif self.file_type == "adjlist":
            self.save_adjlist()

    def load_edgelist(self):
        graph_oop = nx.read_edgelist(path=self.path,
                                     comments="#",
                                     delimiter=None,
                                     create_using=self.create_using,
                                     nodetype=self.nodetype,
                                     data=True,
                                     edgetype=self.edgetype,
                                     encoding="utf-8")

        """
        path 读取的文件名, 使用rb模式打开
        comments  字符串，可选， 表示注释开始的字符
        delimiter 字符串，可选项， 表示分隔符
        create_using 可选项，网络X图构造函数，可选（default=nx）。图）要创建的图形类型。
        如果图形实例，则在填充之前清除。、
        nodetype int, float, str, python类型，可选，将字符串转成指定类型
        data 布尔， 或者元组列表（label,type）,元组指定字典key和类型用于边数据，
        edgetype: int float str， python 类型，将string类型edge数据转成指定类型， 
        使用“weight”
        encoding, string 可选项，
        当读取文件时，指定想要编码的类型
        """
        return graph_oop

    def load_adjlist(self):
        graph_oop = nx.read_adjlist(path=self.path,
                                    comments="#",
                                    delimiter=None,
                                    create_using=self.create_using,
                                    nodetype=self.nodetype,
                                    encoding="utf-8")

        """
        path 读取的文件名, 使用rb模式打开
        comments  字符串，可选， 表示注释开始的字符
        delimiter 字符串，可选项， 表示分隔符
        create_using 可选项，网络X图构造函数，可选（default=nx）。图）要创建的图形类型。
        如果图形实例，则在填充之前清除。、
        nodetype int, float, str, python类型，可选，将字符串转成指定类型 
        使用“weight”
        encoding, string 可选项，
        当读取文件时，指定想要编码的类型
        """
        return graph_oop

    def load_weighted_edgelist(self):  # 加载加权的edgelist文件
        graph_oop = nx.read_weighted_edgelist(path=self.path,
                                              comments="#",
                                              delimiter=None,
                                              create_using=self.create_using,
                                              nodetype=self.nodetype,
                                              encoding="utf-8")
        return graph_oop

    def save_adjlist(self):
        nx.write_adjlist(G=nx.path_graph(4),
                         path=self.path,
                         comments="#",
                         delimiter=" ",
                         encoding="utf-8")

    def save_edgelist(self):
        nx.write_edgelist(G=nx.path_graph(4),
                          path=self.path,
                          comments="#",
                          delimiter=" ",
                          data=True,
                          encoding="utf-8")

    def save_weighted_edgelist(self):
        nx.write_weighted_edgelist(G=nx.path_graph(4),
                                   path=self.path,
                                   comments="#",
                                   delimiter=" ",
                                   encoding="utf-8")


if __name__ == "__main__":
    # file_path = "F:/kanshancup/def/deepwalkdata/testdata/p2p-Gnutella08.edgelist"
    # mygraph = Node2VecData("edgelist", "r", file_path, None, None, "di")
    # mygraph = mygraph.load_edgelist()
    # print(type(mygraph))
    # print("Done!")

    # 读取带加权的
    file_path = "F:/kanshancup/def/deepwalkdata/testdata/p2p-x.edgelist"
    mygraph = Node2VecData("edgelist", "r", file_path, None, None, "di")
    mygraph = mygraph.load_weighted_edgelist()
    print(type(mygraph))
    print("Done!")
