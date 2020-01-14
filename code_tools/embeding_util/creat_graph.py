import networkx as nx
import matplotlib.pyplot as plt
"""
networkx  包对应的文档说明, 创建图
https://networkx.github.io/documentation/stable/tutorial.html，
https://www.cnblogs.com/minglex/p/9205160.html
得到的数据，最简单方式，使用list化对象，得到结果
"""


class Graph(object):
    def __init__(self, graph_type):
        self.graph_type = graph_type
        self.create_graph()

    def create_graph(self):  # 创建一个空图
        if self.graph_type == "no_di":  # 创建 无向图
            self.G = nx.Graph()
        elif self.graph_type == "di":  # 创建 有向图
            self.G = nx.DiGraph()
        elif self.graph_type == "mul_no_di":   # 创建多重无向图
            self.G = nx.MultiGraph()
        # elif self.graph_type == "mul_di":   # 创建多重有向图
        elif self.graph_type == "mul_di":
            self.G = nx.MultiDiGraph()
        else:
            raise ValueError("error graph_type")
        """
        多重无向， 两个节点之间的边多于一条，又允许顶点通过同一条边和自己关联
        """

    def set_graph_attributes(self, attributes_key, attributes_value):  # 添加图属性
        self.G.graph[attributes_key] = attributes_value

    def get_graph_attributes(self):  # 获取图属性
        return self.G.graph

    def add_nodes(self, add_type, node_name):   # 添加节点仅仅添加节点
        """
        :param add_type:  只能为 single, set， cycle 三种操作
        :param node_name: str, list[str] nodename 若果为字符串， set下，"spam"，
        会创建“s”,"p","a","m"四个节点， 在single 下，创建“spam”一个节点
        :return:
        """
        if add_type == "single":  # 添加单个节点或图,如果使用子图node_name=H，那么H作为新图的一个节点,
            self.G.add_node(node_name)  # 1，time ="5pm" 为添加接电视，带进属性
        elif add_type == "set":  # 添加list,子图,如添加无向图对象node_name =H
            self.G.add_nodes_from(node_name)   # ["1","2"]
        elif add_type == "cycle":
            self.G.add_cycle(node_name)  # 添加环 list
        else:
            raise ValueError("error add_type!")

    def set_node_attributes(self, node_name, attributes_key, attributes_value):
        self.G.nodes[node_name][attributes_key] = attributes_value

    def get_all_node_attributes(self):  # 返回所有节点属性
        return self.G.nodes.data()

    def get_single_nodel_attributes(self, node_name):  # 返回单个节点属性
        return self.G.nodes[node_name]

    @staticmethod
    def return_graph(node_num):  # 返回带node_num节点连接的无向图
        return nx.path_graph(node_num)

    def draw(self, filename,with_labels=True):  # 进行画图操作
        nx.draw(self.G, with_labels=with_labels)
        plt.savefig(filename, bbox_inches='tight')

    def get_all_node(self, types="A"):  # 获取所有节点信息
        all_node = self.G.nodes()
        all_num_node = self.G.number_of_nodes()
        if types == "A":
            return all_node, all_num_node
        elif types == "a":
            return all_node
        else:
            return all_num_node

    def get_single_node(self, node_name):  # 获取单个节点信息
        return self.G.nodes[node_name]

    def delet_node(self, node):
        if isinstance(node, list):
            self.G.remove_nodes_from(nodes=node)  # 删除集合中的节点
        else:
            self.G.remove_node(n=node)  # 删除指定节点

    def add_edges(self, edge, datatype):
        """
        当 添加边时，带进属性，在使用设置属性时，会进行覆盖。也可以先不带，后面添加进去
        :param edge:
        :param datatype: ("mutil_edge", "single_edge", "sub_graph", "weights_multi_edge")  只有四种模式
        :return:
        """
        if datatype == "mutil_edge":
            self.G.add_edges_from(edge)  # 添加多条边，[(1,2),(3,4)]，或 [(1,2，{"color":"red"}),(3,4,{"corlor":"blue"})]
            # 或使用 [(1,2),(3,4)], color="red"
        elif datatype == "single_edge":
            self.G.add_edge(edge)  # 添加单条边 （1,2） 或使用（1,2，weight = 3.1415}）  weights为属性，
        elif datatype == "sub_graph":
            self.G.add_edges_from(edge.edges())   # 子图的边
        elif datatype == "weight_multi_edge":  # 增加带权重的边
            self.G.add_weighted_edges_from(edge)  # [(1,2,0.125),(1,3,0.75)]
        else:
            print("error datatype")

    def get_edge(self):  # 获取边相关信息
        all_edges = self.G.edges()
        all_num_edge = self.G.number_of_edges()

        return all_edges, all_num_edge

    def del_edge(self, edge, del_type):   # 删除边
        if del_type == "multi_edges":
            self.G.remove_nodes_from(edge)   # 删除多条边
        elif del_type == "single_deges":   # 删除单条边
            self.G.remove_edge(edge)
        else:
            raise ValueError("Incorrect edge")

    def undi2di(self):  # 无向图转有向图
        return self.G.to_directed()

    def di2undi(self):  # 有向图转无向图
        return self.G.to_undirected()

    def search_each_deges_info(self, params="weight", search_type=None, threshold=0):
        result = []
        # 访问每条边信息
        if not search_type:
            for n, nbrs in self.G.adj.items():
                for nbr, eattr in nbrs.items():
                    data = eattr[params]   # 获取的权重信息
                    print("%d,%d,%0.3f" % (n, nbr, data))
                    result.append((n, nbr, data))
        elif search_type == "rt":
            for n, nbrs in self.G.adj.items():
                for nbr, eattr in nbrs.items():
                    data = eattr[params]   # 获取的权重信息
                    if data > threshold:
                        print("(%d,%d,%0.3f)" % (n, nbr, data))
                        result.append((n, nbr, data))
        elif search_type == "lt":
            for n, nbrs in self.G.adj.items():
                for nbr, eattr in nbrs.items():
                    data = eattr[params]   # 获取的权重信息
                    if data < threshold:
                        print("%d,%d,%0.3f" % (n, nbr, data))
                        result.append((n, nbr, data))
        elif search_type == "eq":
            for n, nbrs in self.G.adj.items():
                for nbr, eattr in nbrs.items():
                    data = eattr[params]   # 获取的权重信息
                    if data == threshold:
                        print("%d,%d,%0.3f" % (n, nbr, data))
                        result.append((n, nbr, data))
        else:
            print("search_type is wrong!")
        return result

    def fastsearcheach(self, params="weight", search_type=None, threshold=0):
        # 快速查找所有边,快速查找需要的边

        result = []
        if not search_type:
            for (u, v, wt) in self.G.edges.data(params):
                print(u, v, wt)
                result.append((u, v, wt))
        elif search_type == "rt":
            for (u, v, wt) in self.G.edges.data(params):
                if wt > threshold:
                    print(u, v, wt)
                    result.append((u, v, wt))
        elif search_type == "lt":
            for (u, v, wt) in self.G.edges.data(params):
                if wt < threshold:
                    print(u, v, wt)
                    result.append((u, v, wt))
        elif search_type == "eq":
            for (u, v, wt) in self.G.edges.data(params):
                if wt == threshold:
                    print(u, v, wt)
                    result.append((u, v, wt))
        else:
            print("search_type is wrong!")
        return result

    def get_basic_graph_properties(self, view_type="list"):
        all_nodes_info = self.G.nodes  # 所有的节点信息
        all_edges_info = self.G.edges  # 所有的边信息
        all_adj = self.G.adj.items()   # 所有的邻居  或者使用self.G.neighbors
        all_degree = self.G.degree   # 节点的度
        if view_type == "list":
            return list(all_nodes_info), list(all_edges_info), all_adj, all_degree
        elif view_type == "set":
            return set(all_nodes_info), set(all_edges_info), all_adj, all_degree
        elif view_type == "tuple":
            return tuple(all_edges_info), tuple(all_nodes_info), all_adj, all_degree
        else:
            return all_edges_info, all_edges_info, all_adj, all_degree

    def set_edge_attributes(self, node1, node2, attributes_key, attributes_value):   # 修改、设置已存在的属性值，
        self.G[node1][node2][attributes_key] = attributes_value

    def get_edge_attributes(self, node1, node2, attributes_key):  # 获取存在的属性值
        return self.G[node1][node2][attributes_key]

    def search_sortest_path(self, node1, node2):  # 获取两点之间最近的距离
        """
        两个节点之间最近的路径
        :param node1:
        :param node2:
        :return:
        """
        return nx.shortest_path(self.G, node1, node2)

    def search_sortest_len(self, node1, node2):  # 两点之间最近距离的长度
        return nx.shortest_path_length(self.G, node1, node2)

    def clear_graph(self):  # 清除图
        self.G.clear()

    # ------------------------------有向图部分属性-------------------------------
    # degree = in_degree + out_degree
    def get_digraph_single_degree(self, node_name, attributes):  # 获取有向图中的单个节点的度
        """
        :param node_name: 节点名字
        :param attributes:
        :return:
        """
        if self.graph_type == "di":
            return self.G.degree(node_name, weight=attributes)
        else:
            raise ValueError("有向图才能使用")

    def get_digraph_single_indegree(self, node_nam, attributes):  # 获取有向图中单个节点入度
        if self.graph_type == "id":
            return self.G.in_degree(node_nam, weight=attributes)
        else:
            raise ValueError("有向图才能使用")

    def get_diggraph_single_outdegree(self, node_name, attributes):  # 获取有向图中单个节点出度
        if self.graph_type == "id":
            return self.G.out_degree(node_name, weight=attributes)
        else:
            raise ValueError("有向图才能使用")

    def get_neighbors(self, node_name, types="None"):  # 获取邻居节点,有向无向均可
        if types == "list":
            return list(self.G.neighbors(node_name))
        elif types == "set":
            return set(self.G.neighbors(node_name))
        elif types == "tuple":
            return tuple(self.G.neighbors(node_name))
        else:
            return self.G.neighbors(node_name)  # dict_keyiterator object

    # ----------------------------多重图部分---------------------------------------------
    # 如[(1, 2, 0.5), (1, 2, 0.75), (2, 3, 0.5)]， 两个节点之间链接的权重不一致，
    def recal_edge(self, cal_type="min"):
        if cal_type == "min":
            for n, nbrs in self.G.adj.items():
                for nbr, edict in nbrs.items():
                    minvalue = min([d["weight"] for d in edict.values()])
                    self.G.add_edge(n, nbr, weight=minvalue)

        elif cal_type == "max":
            for n, nbrs in self.G.adj.items():
                for nbr, edict in nbrs.items():
                    minvalue = max([d["weight"] for d in edict.values()])
                    self.G.add_edge(n, nbr, weight=minvalue)

        else:
            raise ValueError("error cal_type")


def merge_graph(g1, g2, merge_type):  # 聚合图
    """
    :param g1: 第一个图对象
    :param g2:  第二个图对象
    :param merge_type:  聚合方式（"subgraph","union","dis_union", "cartesian", "compose"）
    :return:
    """
    if merge_type == "subgraph":
        new_g = nx.subgraph(g1, g2)  # g2为 list,
    elif merge_type == "union":   # 不相交的拼接
        new_g = nx.union(g1, g2)
    elif merge_type == "dis_union":  # 所有节点都不同的不相交拼接
        new_g = nx.disjoint_union(g1, g2)
    elif merge_type == "cartesian":   # 笛卡尔乘积图
        new_g = nx.cartesian_product(g1, g2)
    elif merge_type == "compose":
        new_g = nx.compose(g1, g2)  # 与g1 一样新的图
    else:
        raise ValueError("error merge_type")
    return new_g


# ---------------gml数据格式------------------------
def load_gml_graph(file_path):
    """
    :param file_path:  如"path.to.file"
    :return:   图对象
    """
    return nx.read_gml(file_path)


def save_gml_graph(graph, file_path):
    """
    :param graph: 图对象
    :param file_path:  保存的路径  如"path.to.file"
    :return:
    """
    nx.write_gml(graph, file_path)

#


