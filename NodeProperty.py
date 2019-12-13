import networkx as nx
import SettingSimulationValue
import InterconnectedLayerModeling
import operator
import time


class NodeProperty:
    def __init__(self, setting, inter_layer, select_layer_number=0, select_method='0'):
        self.nodes_order = NodeProperty.integrated_centrality_order(setting, inter_layer, select_method)[select_layer_number]

    @staticmethod
    def integrated_centrality_order(setting, inter_layer, select_method):
        A_node_order = []
        B_node_order = []
        mixed_order = []
        node_prdic = {}
        node_dedic = {}
        node_eidic = {}
        node_bedic = {}
        node_mixdic = {}
        pagerank = nx.pagerank(inter_layer.two_layer_graph)
        pagerank_order = sorted(pagerank.items(), key=operator.itemgetter(1), reverse=True)
        eigenvector_centrality = nx.eigenvector_centrality_numpy(inter_layer.two_layer_graph)
        eigenvector_order = sorted(eigenvector_centrality.items(), key=operator.itemgetter(1), reverse=True)
        degree_centrality = nx.degree_centrality(inter_layer.two_layer_graph)
        degree_order = sorted(degree_centrality.items(), key=operator.itemgetter(1), reverse=True)
        betweenness_centrality = nx.betweenness_centrality(inter_layer.two_layer_graph)
        betweenness_order = sorted(betweenness_centrality.items(), key=operator.itemgetter(1), reverse=True)
        closeness_centrality = nx.closeness_centrality(inter_layer.two_layer_graph)
        closeness_order = sorted(closeness_centrality.items(), key=operator.itemgetter(1), reverse=True)
        if select_method == 'pagerank':
            for i in sorted(inter_layer.two_layer_graph.nodes):
                if pagerank_order[i][0] < setting.A_node:
                    A_node_order.append((pagerank_order[i][0], pagerank_order[i][1]))
                else:
                    B_node_order.append((pagerank_order[i][0], pagerank_order[i][1]))
            mixed_order = pagerank_order
        elif select_method == 'eigenvector':
            for i in sorted(inter_layer.two_layer_graph.nodes):
                if eigenvector_order[i][0] < setting.A_node:
                    A_node_order.append((eigenvector_order[i][0], eigenvector_order[i][1]))
                else:
                    B_node_order.append((eigenvector_order[i][0], eigenvector_order[i][1]))
            mixed_order = eigenvector_order
        elif select_method == 'degree':
            for i in sorted(inter_layer.two_layer_graph.nodes):
                if degree_order[i][0] < setting.A_node:
                    A_node_order.append((degree_order[i][0], degree_order[i][1]))
                else:
                    B_node_order.append((degree_order[i][0], degree_order[i][1]))
            mixed_order = degree_order
        elif select_method == 'PR+DE':
            for i in sorted(inter_layer.two_layer_graph.nodes):
                node_prdic[pagerank_order[i][0]] = i
            for i in sorted(inter_layer.two_layer_graph.nodes):
                node_dedic[degree_order[i][0]] = i
            for i in sorted(inter_layer.two_layer_graph.nodes):
                node_mixdic[i] = node_prdic[i] + node_dedic[i]
            mixed_order = sorted(node_mixdic.items(), key=operator.itemgetter(1), reverse=False)
            for i in sorted(inter_layer.two_layer_graph.nodes):
                if mixed_order[i][0] < setting.A_node:
                    A_node_order.append((mixed_order[i][0], mixed_order[i][1]))
                else:
                    B_node_order.append((mixed_order[i][0], mixed_order[i][1]))
        elif select_method == 'PR+EI':
            for i in sorted(inter_layer.two_layer_graph.nodes):
                node_prdic[pagerank_order[i][0]] = i
            for i in sorted(inter_layer.two_layer_graph.nodes):
                node_eidic[eigenvector_order[i][0]] = i
            for i in sorted(inter_layer.two_layer_graph.nodes):
                node_mixdic[i] = node_prdic[i] + node_eidic[i]
            mixed_order = sorted(node_mixdic.items(), key=operator.itemgetter(1), reverse=False)
            for i in sorted(inter_layer.two_layer_graph.nodes):
                if mixed_order[i][0] < setting.A_node:
                    A_node_order.append((mixed_order[i][0], mixed_order[i][1]))
                else:
                    B_node_order.append((mixed_order[i][0], mixed_order[i][1]))
        elif select_method == 'DE+EI':
            for i in sorted(inter_layer.two_layer_graph.nodes):
                node_dedic[degree_order[i][0]] = i
            for i in sorted(inter_layer.two_layer_graph.nodes):
                node_eidic[eigenvector_order[i][0]] = i
            for i in sorted(inter_layer.two_layer_graph.nodes):
                node_mixdic[i] = node_dedic[i] + node_eidic[i]
            mixed_order = sorted(node_mixdic.items(), key=operator.itemgetter(1), reverse=False)
            for i in sorted(inter_layer.two_layer_graph.nodes):
                if mixed_order[i][0] < setting.A_node:
                    A_node_order.append((mixed_order[i][0], mixed_order[i][1]))
                else:
                    B_node_order.append((mixed_order[i][0], mixed_order[i][1]))
        elif select_method == 'PR+DE+EI':
            for i in sorted(inter_layer.two_layer_graph.nodes):
                node_prdic[pagerank_order[i][0]] = i
            for i in sorted(inter_layer.two_layer_graph.nodes):
                node_dedic[degree_order[i][0]] = i
            for i in sorted(inter_layer.two_layer_graph.nodes):
                node_eidic[eigenvector_order[i][0]] = i
            for i in sorted(inter_layer.two_layer_graph.nodes):
                node_mixdic[i] = node_prdic[i] + node_dedic[i] + node_eidic[i]
            mixed_order = sorted(node_mixdic.items(), key=operator.itemgetter(1), reverse=False)
            for i in sorted(inter_layer.two_layer_graph.nodes):
                if mixed_order[i][0] < setting.A_node:
                    A_node_order.append((mixed_order[i][0], mixed_order[i][1]))
                else:
                    B_node_order.append((mixed_order[i][0], mixed_order[i][1]))
        elif select_method == 'betweenness':
            for i in sorted(inter_layer.two_layer_graph.nodes):
                if betweenness_order[i][0] < setting.A_node:
                    A_node_order.append((betweenness_order[i][0], betweenness_order[i][1]))
                else:
                    B_node_order.append((betweenness_order[i][0], betweenness_order[i][1]))
            mixed_order = betweenness_order
        elif select_method == 'closeness':
            for i in sorted(inter_layer.two_layer_graph.nodes):
                if closeness_order[i][0] < setting.A_node:
                    A_node_order.append((closeness_order[i][0], closeness_order[i][1]))
                else:
                    B_node_order.append((closeness_order[i][0], closeness_order[i][1]))
            mixed_order = closeness_order
        elif select_method == 'PR+BE':
            for i in sorted(inter_layer.two_layer_graph.nodes):
                node_prdic[pagerank_order[i][0]] = i
            for i in sorted(inter_layer.two_layer_graph.nodes):
                node_bedic[betweenness_order[i][0]] = i
            for i in sorted(inter_layer.two_layer_graph.nodes):
                node_mixdic[i] = node_prdic[i] + node_bedic[i]
            mixed_order = sorted(node_mixdic.items(), key=operator.itemgetter(1), reverse=False)
            for i in sorted(inter_layer.two_layer_graph.nodes):
                if mixed_order[i][0] < setting.A_node:
                    A_node_order.append((mixed_order[i][0], mixed_order[i][1]))
                else:
                    B_node_order.append((mixed_order[i][0], mixed_order[i][1]))
        elif select_method == 'DE+BE':
            for i in sorted(inter_layer.two_layer_graph.nodes):
                node_dedic[degree_order[i][0]] = i
            for i in sorted(inter_layer.two_layer_graph.nodes):
                node_bedic[betweenness_order[i][0]] = i
            for i in sorted(inter_layer.two_layer_graph.nodes):
                node_mixdic[i] = node_dedic[i] + node_bedic[i]
            mixed_order = sorted(node_mixdic.items(), key=operator.itemgetter(1), reverse=False)
            for i in sorted(inter_layer.two_layer_graph.nodes):
                if mixed_order[i][0] < setting.A_node:
                    A_node_order.append((mixed_order[i][0], mixed_order[i][1]))
                else:
                    B_node_order.append((mixed_order[i][0], mixed_order[i][1]))
        elif select_method == 'PR+DE+BE':
            for i in sorted(inter_layer.two_layer_graph.nodes):
                node_prdic[pagerank_order[i][0]] = i
            for i in sorted(inter_layer.two_layer_graph.nodes):
                node_dedic[degree_order[i][0]] = i
            for i in sorted(inter_layer.two_layer_graph.nodes):
                node_bedic[betweenness_order[i][0]] = i
            for i in sorted(inter_layer.two_layer_graph.nodes):
                node_mixdic[i] = node_prdic[i] + node_dedic[i] + node_bedic[i]
            mixed_order = sorted(node_mixdic.items(), key=operator.itemgetter(1), reverse=False)
            for i in sorted(inter_layer.two_layer_graph.nodes):
                if mixed_order[i][0] < setting.A_node:
                    A_node_order.append((mixed_order[i][0], mixed_order[i][1]))
                else:
                    B_node_order.append((mixed_order[i][0], mixed_order[i][1]))
        return A_node_order, B_node_order, mixed_order

    @staticmethod
    def order_load_centrality(setting, inter_layer):
        A_node_order = []
        B_node_order = []
        load_centrality = nx.load_centrality(inter_layer.two_layer_graph)
        load_order = sorted(load_centrality.items(), key=operator.itemgetter(1), reverse=True)
        for i in sorted(inter_layer.two_layer_graph.nodes):
            if load_order[i][0] < setting.A_node:
                A_node_order.append((load_order[i][0], load_order[i][1]))
            else:
                B_node_order.append((load_order[i][0], load_order[i][1]))
        return A_node_order, B_node_order, load_order

    @staticmethod
    def order_AB_degree(setting, inter_layer):
        AB_degree = {}
        dict = nx.degree_centrality(inter_layer.two_layer_graph)
        for node_i in inter_layer.A_nodes:
            connected_B_nodes_list = NodeProperty.finding_B_node(setting, inter_layer, node_i)
            integrated_degree = 0
            for connected_B_node in connected_B_nodes_list:
                integrated_degree += dict[connected_B_node]
            AB_degree[node_i] = dict[node_i] + integrated_degree
        AB_degree_order = sorted(AB_degree.items(), key=operator.itemgetter(1), reverse=True)
        return AB_degree_order

    @staticmethod
    def order_AB_pagerank(setting, inter_layer):
        AB_pagerank = {}
        dict = nx.pagerank(inter_layer.two_layer_graph)
        for node_i in inter_layer.A_nodes:
            connected_B_nodes_list = NodeProperty.finding_B_node(setting, inter_layer, node_i)
            integrated_pagerank = 0
            for connected_B_node in connected_B_nodes_list:
                integrated_pagerank += dict[connected_B_node]
            AB_pagerank[node_i] = dict[node_i] + integrated_pagerank
        AB_pagerank_order = sorted(AB_pagerank.items(), key=operator.itemgetter(1), reverse=True)
        return AB_pagerank_order

    @staticmethod
    def order_AB_eigenvector(setting, inter_layer):
        AB_eigenvector = {}
        dict = nx.eigenvector_centrality_numpy(inter_layer.two_layer_graph)
        for node_i in inter_layer.A_nodes:
            connected_B_nodes_list = NodeProperty.finding_B_node(setting, inter_layer, node_i)
            integrated_eigenvector = 0
            for connected_B_node in connected_B_nodes_list:
                integrated_eigenvector += dict[connected_B_node]
            AB_eigenvector[node_i] = dict[node_i] + integrated_eigenvector
        AB_eigenvector_order = sorted(AB_eigenvector.items(), key=operator.itemgetter(1), reverse=True)
        return AB_eigenvector_order

    @staticmethod
    def order_AB_betweenness(setting, inter_layer):
        AB_betweenness = {}
        dict = nx.betweenness_centrality(inter_layer.two_layer_graph)
        for node_i in inter_layer.A_nodes:
            connected_B_nodes_list = NodeProperty.finding_B_node(setting, inter_layer, node_i)
            integrated_betweenness = 0
            for connected_B_node in connected_B_nodes_list:
                integrated_betweenness += dict[connected_B_node]
            AB_betweenness[node_i] = dict[node_i] + integrated_betweenness
        AB_betweenness_order = sorted(AB_betweenness.items(), key=operator.itemgetter(1), reverse=True)
        return AB_betweenness_order

    @staticmethod
    def order_AB_closeness(setting, inter_layer):
        AB_closeness = {}
        dict = nx.closeness_centrality(inter_layer.two_layer_graph)
        for node_i in inter_layer.A_nodes:
            connected_B_nodes_list = NodeProperty.finding_B_node(setting, inter_layer, node_i)
            integrated_closeness = 0
            for connected_B_node in connected_B_nodes_list:
                integrated_closeness += dict[connected_B_node]
            AB_closeness[node_i] = dict[node_i] + integrated_closeness
        AB_closeness_order = sorted(AB_closeness.items(), key=operator.itemgetter(1), reverse=True)
        return AB_closeness_order

    @staticmethod
    def order_AB_load(setting, inter_layer):
        AB_load = {}
        dict = nx.load_centrality(inter_layer.two_layer_graph)
        for node_i in inter_layer.A_nodes:
            connected_B_nodes_list = NodeProperty.finding_B_node(setting, inter_layer, node_i)
            integrated_load = 0
            for connected_B_node in connected_B_nodes_list:
                integrated_load += dict[connected_B_node]
            AB_load[node_i] = dict[node_i] + integrated_load
        AB_load_order = sorted(AB_load.items(), key=operator.itemgetter(1), reverse=True)
        return AB_load_order

    @staticmethod
    def finding_B_node(setting, inter_layer, node_i):
        connected_B_nodes_list = []
        neighbors = sorted(nx.neighbors(inter_layer.two_layer_graph, node_i))
        for neighbor in neighbors:
            if neighbor >= setting.A_node:
                connected_B_nodes_list.append(neighbor)
        return connected_B_nodes_list

    @staticmethod
    def finding_A_node(setting, inter_layer, node_i):
        connected_A_nodes_list = []
        neighbors = sorted(nx.neighbors(inter_layer.two_layer_graph, node_i))
        for neighbor in neighbors:
            if neighbor < setting.A_node:
                connected_A_nodes_list.append(neighbor)
        return connected_A_nodes_list

    @staticmethod
    def order_pagerank_individual(setting, inter_layer):
        A_node_order = []
        B_node_order = []
        individual_pagerank = {}
        pagerank_A = nx.pagerank(inter_layer.A_layer_graph)
        pagerank_B = nx.pagerank(inter_layer.B_layer_graph)
        for node_i in inter_layer.A_nodes:
            connected_B_nodes_list = NodeProperty.finding_B_node(setting, inter_layer, node_i)
            A_integrated_pagerank = 0
            for connected_B_node in connected_B_nodes_list:
                A_integrated_pagerank += (pagerank_B[connected_B_node] / len(connected_B_nodes_list))
            individual_pagerank[node_i] = pagerank_A[node_i] + A_integrated_pagerank
        for node_j in inter_layer.B_nodes:
            connected_A_nodes_list = NodeProperty.finding_A_node(setting, inter_layer, node_j)
            B_integrated_pagerank = 0
            for connected_A_node in connected_A_nodes_list:
                B_integrated_pagerank += (pagerank_A[connected_A_node] / len(connected_A_nodes_list))
            individual_pagerank[node_j] = pagerank_B[node_j] + B_integrated_pagerank
        individual_pagerank_order = sorted(individual_pagerank.items(), key=operator.itemgetter(1), reverse=True)
        for i in sorted(inter_layer.two_layer_graph.nodes):
            if individual_pagerank_order[i][0] < setting.A_node:
                A_node_order.append((individual_pagerank_order[i][0], individual_pagerank_order[i][1]))
            else:
                B_node_order.append((individual_pagerank_order[i][0], individual_pagerank_order[i][1]))
        return A_node_order, B_node_order, individual_pagerank_order   # value = pagerank[node_number]


if __name__ == "__main__":
    print("CalculatingProperty")
    setting = SettingSimulationValue.SettingSimulationValue()
    inter_layer = InterconnectedLayerModeling.InterconnectedLayerModeling(setting)
    start = time.time()
    ordering_nodes1 = NodeProperty(setting, inter_layer, 0, 'pagerank')
    ordering_nodes2 = NodeProperty(setting, inter_layer, 0, 'PR+DE')
    ordering_nodes3 = NodeProperty(setting, inter_layer, 0, 'PR+DE+BE')

    # select = cal_property.cal_node_A_and_node_B_centrality(inter_layer)
    for i in range(100):
        print(ordering_nodes1.nodes_order[i][0], ordering_nodes2.nodes_order[i][0], ordering_nodes3.nodes_order[i][0])
    # select2 = cal_property.order_AB_pagerank(inter_layer)
    # print(select2[0:10])
    # select3 = nx.pagerank(inter_layer.two_layer_graph)
    # select3 = sorted(select3.items(), key=operator.itemgetter(1), reverse=True)
    # print(select3[0:10])
    end = time.time()
    print(end-start)



