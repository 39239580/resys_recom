from code_tools.embeding_util.creat_graph import Graph
import time


#  创建无向图
def fn_1():
    graph1 = Graph("no_di")
    graph1.add_nodes("single", 1)
    graph1.add_nodes("set", [2, 3])
    graph1.draw("fn_1.png")
    graph1.clear_graph()


def fn_2():
    graph2 = Graph("no_di")
    H = graph2.return_graph(10)
    graph2.add_nodes("set", H)
    graph2.draw("fn_2.png")
    graph2.clear_graph()


def fn_3():
    graph3 = Graph("no_di")
    graph3.add_nodes("single", "a")
    graph3.add_nodes("set", ["b", "c", "d", "e"])
    graph3.add_nodes("cycle", ["f", "g", "h", "j"])
    H = Graph("no_di").return_graph(10)
    graph3.add_nodes("set", H)
    graph3.add_nodes("cycle", H)
    graph3.draw("fn_3.png")
    graph3.clear_graph()


def fn_4():
    data = [(1, 2, 0.125), (1, 3, 0.75), (2, 4, 1.2), (3, 4, 0.375)]
    graph4 = Graph("no_di")
    graph4.add_edges(data, datatype="weight_multi_edge")
    graph4.search_each_deges_info(search_type="lt", threshold=0.5)
    graph4.clear_graph()


def fn_5():
    data = [(1, 2, 0.5), (3, 1, 0.75)]
    graph5 = Graph("di")
    graph5.add_edges(data, datatype="weight_multi_edge")
    print(graph5.get_neighbors(1, types="None"))
    print(graph5.get_basic_graph_properties(view_type="list"))
    graph5.draw("fn_5.png")
    graph5.clear_graph()


if __name__ == "__main__":
    # fn_1()
    # time.sleep(3)
    # fn_2()
    # time.sleep(3)
    # fn_3()
    # time.sleep(3)
    # fn_4()
    time.sleep(3)
    fn_5()
