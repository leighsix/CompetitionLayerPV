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
        if select_edge_method == 'edge_betweenness':
            ordering_edge = EdgeProperty.ordering_edge_betweenness(setting, inter_layer)[select_edges_number]
        elif select_edge_method == 'edge_pagerank':
            ordering_edge = EdgeProperty.ordering_edge_pagerank(setting, inter_layer)[select_edges_number]
        elif select_edge_method == 'edge_degree':
            ordering_edge = EdgeProperty.ordering_edge_degree(setting, inter_layer)[select_edges_number]
        elif select_edge_method == 'edge_eigenvector':
            ordering_edge = EdgeProperty.ordering_edge_eigenvector(setting, inter_layer)[select_edges_number]
        elif select_edge_method == 'edge_closeness':
            ordering_edge = EdgeProperty.ordering_edge_closeness(setting, inter_layer)[select_edges_number]
        elif select_edge_method == 'edge_load':
            ordering_edge = EdgeProperty.ordering_edge_load(setting, inter_layer)[select_edges_number]
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
    inter_layer = InterconnectedLayerModeling.InterconnectedLayerModeling(setting)
    start = time.time()
    edge_property = EdgeProperty(setting, inter_layer, 0, 'edge_closeness')
    print(edge_property.edges_order[0:10])
    end = time.time()
    print(end - start)
    
    
    
    


