import networkx as nx
import SettingSimulationValue
import InterconnectedLayerModeling
import operator
import time


class EdgeProperty:
    def __init__(self, setting, inter_layer, select_edges_number=0, select_edge_method='0'):
        self.edges_order = EdgeProperty.ordering_edge(setting, inter_layer, select_edges_number, select_edge_method)

    @staticmethod
    def ordering_edge(setting, inter_layer, select_edges_number, select_edge_method):
        ordering_edge = []
        if select_edge_method == 'edge_betweenness' or 'edge_betweenness_sequential':
            ordering_edge = EdgeProperty.ordering_edge_betweenness(setting, inter_layer)[select_edges_number]
        elif select_edge_method == 'edge_pagerank' or 'edge_pagerank_sequential':
            ordering_edge = EdgeProperty.ordering_edge_pagerank(setting, inter_layer)[select_edges_number]
        elif select_edge_method == 'edge_degree' or 'edge_degree_sequential':
            ordering_edge = EdgeProperty.ordering_edge_degree(setting, inter_layer)[select_edges_number]
        elif select_edge_method == 'edge_eigenvector' or 'edge_eigenvector_sequential':
            ordering_edge = EdgeProperty.ordering_edge_eigenvector(setting, inter_layer)[select_edges_number]
        elif select_edge_method == 'edge_closeness' or 'edge_closeness_sequential':
            ordering_edge = EdgeProperty.ordering_edge_closeness(setting, inter_layer)[select_edges_number]
        elif select_edge_method == 'edge_load' or 'edge_load_sequential':
            ordering_edge = EdgeProperty.ordering_edge_load(setting, inter_layer)[select_edges_number]
        elif select_edge_method == 'edge_jaccard' or 'edge_jaccard_sequential':
            ordering_edge = EdgeProperty.ordering_edge_jaccard_coefficient(setting, inter_layer)[select_edges_number]
        return ordering_edge

    @staticmethod
    def ordering_edge_betweenness(setting, inter_layer):
        A_internal_order = []
        A_mixed_order = []
        B_internal_order = []
        B_mixed_order = []
        external_order = []
        edge_betweenness = nx.edge_betweenness_centrality(inter_layer.two_layer_graph)
        edge_betweenness_order = sorted(edge_betweenness.items(), key=operator.itemgetter(1), reverse=True)
        mixed_order = edge_betweenness_order
        for i in range(len(edge_betweenness_order)):
            edge = edge_betweenness_order[i][0]
            if (edge[0] < setting.A_node) and (edge[1] < setting.A_node):
                A_internal_order.append(edge_betweenness_order[i])
            if edge[0] < setting.A_node:
                A_mixed_order.append(edge_betweenness_order[i])
            if (edge[0] >= setting.A_node) and (edge[1] >= setting.A_node):
                B_internal_order.append(edge_betweenness_order[i])
            if edge[0] >= setting.A_node:
                B_mixed_order.append(edge_betweenness_order[i])
            if (edge[0] < setting.A_node) and (edge[1] >= setting.A_node):
                external_order.append(edge_betweenness_order[i])
        return A_internal_order, A_mixed_order, B_internal_order, B_mixed_order, external_order, mixed_order
    
    @staticmethod
    def ordering_edge_pagerank(setting, inter_layer):
        A_internal_order = []
        A_mixed_order = []
        B_internal_order = []
        B_mixed_order = []
        external_order = []
        edge_pagerank = {}
        dict = nx.pagerank(inter_layer.two_layer_graph)
        for edge in sorted(inter_layer.two_layer_graph.edges):
            edge_pagerank[edge] = dict[edge[0]] + dict[edge[1]]
        edge_pagerank_order = sorted(edge_pagerank.items(), key=operator.itemgetter(1), reverse=True)
        mixed_order = edge_pagerank_order
        for i in range(len(edge_pagerank_order)):
            edge = edge_pagerank_order[i][0]
            if (edge[0] < setting.A_node) and (edge[1] < setting.A_node):
                A_internal_order.append(edge_pagerank_order[i])
            if edge[0] < setting.A_node:
                A_mixed_order.append(edge_pagerank_order[i])
            if (edge[0] >= setting.A_node) and (edge[1] >= setting.A_node):
                B_internal_order.append(edge_pagerank_order[i])
            if edge[0] >= setting.A_node:
                B_mixed_order.append(edge_pagerank_order[i])
            if (edge[0] < setting.A_node) and (edge[1] >= setting.A_node):
                external_order.append(edge_pagerank_order[i])
        return A_internal_order, A_mixed_order, B_internal_order, B_mixed_order, external_order, mixed_order

    @staticmethod
    def ordering_edge_degree(setting, inter_layer):
        A_internal_order = []
        A_mixed_order = []
        B_internal_order = []
        B_mixed_order = []
        external_order = []
        edge_degree = {}
        dict = nx.degree_centrality(inter_layer.two_layer_graph)
        for edge in sorted(inter_layer.two_layer_graph.edges):
            edge_degree[edge] = dict[edge[0]] + dict[edge[1]]
        edge_degree_order = sorted(edge_degree.items(), key=operator.itemgetter(1), reverse=True)
        mixed_order = edge_degree_order
        for i in range(len(edge_degree_order)):
            edge = edge_degree_order[i][0]
            if (edge[0] < setting.A_node) and (edge[1] < setting.A_node):
                A_internal_order.append(edge_degree_order[i])
            if edge[0] < setting.A_node:
                A_mixed_order.append(edge_degree_order[i])
            if (edge[0] >= setting.A_node) and (edge[1] >= setting.A_node):
                B_internal_order.append(edge_degree_order[i])
            if edge[0] >= setting.A_node:
                B_mixed_order.append(edge_degree_order[i])
            if (edge[0] < setting.A_node) and (edge[1] >= setting.A_node):
                external_order.append(edge_degree_order[i])
        return A_internal_order, A_mixed_order, B_internal_order, B_mixed_order, external_order, mixed_order

    @staticmethod
    def ordering_edge_eigenvector(setting, inter_layer):
        A_internal_order = []
        A_mixed_order = []
        B_internal_order = []
        B_mixed_order = []
        external_order = []
        edge_eigenvector = {}
        dict = nx.eigenvector_centrality_numpy(inter_layer.two_layer_graph)
        for edge in sorted(inter_layer.two_layer_graph.edges):
            edge_eigenvector[edge] = dict[edge[0]] + dict[edge[1]]
        edge_eigenvector_order = sorted(edge_eigenvector.items(), key=operator.itemgetter(1), reverse=True)
        mixed_order = edge_eigenvector_order
        for i in range(len(edge_eigenvector_order)):
            edge = edge_eigenvector_order[i][0]
            if (edge[0] < setting.A_node) and (edge[1] < setting.A_node):
                A_internal_order.append(edge_eigenvector_order[i])
            if edge[0] < setting.A_node:
                A_mixed_order.append(edge_eigenvector_order[i])
            if (edge[0] >= setting.A_node) and (edge[1] >= setting.A_node):
                B_internal_order.append(edge_eigenvector_order[i])
            if edge[0] >= setting.A_node:
                B_mixed_order.append(edge_eigenvector_order[i])
            if (edge[0] < setting.A_node) and (edge[1] >= setting.A_node):
                external_order.append(edge_eigenvector_order[i])
        return A_internal_order, A_mixed_order, B_internal_order, B_mixed_order, external_order, mixed_order

    @staticmethod
    def ordering_edge_closeness(setting, inter_layer):
        A_internal_order = []
        A_mixed_order = []
        B_internal_order = []
        B_mixed_order = []
        external_order = []
        edge_closeness = {}
        dict = nx.closeness_centrality(inter_layer.two_layer_graph)
        for edge in sorted(inter_layer.two_layer_graph.edges):
            edge_closeness[edge] = dict[edge[0]] + dict[edge[1]]
        edge_closeness_order = sorted(edge_closeness.items(), key=operator.itemgetter(1), reverse=True)
        mixed_order = edge_closeness_order
        for i in range(len(edge_closeness_order)):
            edge = edge_closeness_order[i][0]
            if (edge[0] < setting.A_node) and (edge[1] < setting.A_node):
                A_internal_order.append(edge_closeness_order[i])
            if edge[0] < setting.A_node:
                A_mixed_order.append(edge_closeness_order[i])
            if (edge[0] >= setting.A_node) and (edge[1] >= setting.A_node):
                B_internal_order.append(edge_closeness_order[i])
            if edge[0] >= setting.A_node:
                B_mixed_order.append(edge_closeness_order[i])
            if (edge[0] < setting.A_node) and (edge[1] >= setting.A_node):
                external_order.append(edge_closeness_order[i])
        return A_internal_order, A_mixed_order, B_internal_order, B_mixed_order, external_order, mixed_order

    @staticmethod
    def ordering_edge_load(setting, inter_layer):
        A_internal_order = []
        A_mixed_order = []
        B_internal_order = []
        B_mixed_order = []
        external_order = []
        edge_load = {}
        dict = nx.load_centrality(inter_layer.two_layer_graph)
        for edge in sorted(inter_layer.two_layer_graph.edges):
            edge_load[edge] = dict[edge[0]] + dict[edge[1]]
        edge_load_order = sorted(edge_load.items(), key=operator.itemgetter(1), reverse=True)
        mixed_order = edge_load_order
        for i in range(len(edge_load_order)):
            edge = edge_load_order[i][0]
            if (edge[0] < setting.A_node) and (edge[1] < setting.A_node):
                A_internal_order.append(edge_load_order[i])
            if edge[0] < setting.A_node:
                A_mixed_order.append(edge_load_order[i])
            if (edge[0] >= setting.A_node) and (edge[1] >= setting.A_node):
                B_internal_order.append(edge_load_order[i])
            if edge[0] >= setting.A_node:
                B_mixed_order.append(edge_load_order[i])
            if (edge[0] < setting.A_node) and (edge[1] >= setting.A_node):
                external_order.append(edge_load_order[i])
        return A_internal_order, A_mixed_order, B_internal_order, B_mixed_order, external_order, mixed_order

    @staticmethod
    def ordering_edge_jaccard_coefficient(setting, inter_layer):
        A_internal_order = []
        A_mixed_order = []
        B_internal_order = []
        B_mixed_order = []
        external_order = []
        edge_jaccard = {}
        jaccard_list = list(nx.jaccard_coefficient(inter_layer.two_layer_graph, sorted(inter_layer.two_layer_graph.edges)))
        for jaccard in jaccard_list:
            edge_jaccard[(jaccard[0], jaccard[1])] = jaccard[2]
        edge_jaccard_order = sorted(edge_jaccard.items(), key=operator.itemgetter(1), reverse=False)
        mixed_order = edge_jaccard_order
        for i in range(len(edge_jaccard_order)):
            edge = edge_jaccard_order[i][0]
            if (edge[0] < setting.A_node) and (edge[1] < setting.A_node):
                A_internal_order.append(edge_jaccard_order[i])
            if edge[0] < setting.A_node:
                A_mixed_order.append(edge_jaccard_order[i])
            if (edge[0] >= setting.A_node) and (edge[1] >= setting.A_node):
                B_internal_order.append(edge_jaccard_order[i])
            if edge[0] >= setting.A_node:
                B_mixed_order.append(edge_jaccard_order[i])
            if (edge[0] < setting.A_node) and (edge[1] >= setting.A_node):
                external_order.append(edge_jaccard_order[i])
        return A_internal_order, A_mixed_order, B_internal_order, B_mixed_order, external_order, mixed_order

    @staticmethod
    def finding_B_node(setting, inter_layer, node_i):
        connected_B_nodes_list = []
        neighbors = sorted(nx.neighbors(inter_layer.two_layer_graph, node_i))
        for neighbor in neighbors:
            if neighbor >= setting.A_node:
                connected_B_nodes_list.append(neighbor)
        return connected_B_nodes_list


if __name__ == "__main__":
    print('EdgeProperty')
    setting = SettingSimulationValue.SettingSimulationValue()
    setting.A_node = 64
    setting.B_node = 64
    inter_layer = InterconnectedLayerModeling.InterconnectedLayerModeling(setting)
    start = time.time()
    edge_property1 = EdgeProperty(setting, inter_layer, 0, 'edge_pagerank')
    edge_property2 = EdgeProperty(setting, inter_layer, 0, 'edge_pagerank_sequential')
    for i in range(10):
        print(edge_property1.edges_order[i], edge_property2.edges_order[i])
    end = time.time()
    print(end - start)
    
    
    
    


