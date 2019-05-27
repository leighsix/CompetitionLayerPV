import networkx as nx
import SettingSimulationValue
import InterconnectedLayerModeling
import operator
import time


class EdgeProperty:
    def __init__(self, setting, inter_layer, select_edge_method='0', select_edges_number=0):
        self.edges_order = EdgeProperty.ordering_edge(setting, inter_layer, select_edge_method, select_edges_number)

    @staticmethod
    def ordering_edge(setting, inter_layer, select_edge_method, select_edges_number):
        ordering_edge = []
        if select_edge_method == 'edge_betweenness':
            ordering_edge = EdgeProperty.ordering_edge_betweenness(setting, inter_layer)[select_edges_number]
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
    
    
if __name__ == "__main__":
    setting = SettingSimulationValue.SettingSimulationValue()
    setting.A_node = 64
    setting.B_node = 64
    inter_layer = InterconnectedLayerModeling.InterconnectedLayerModeling(setting)
    edge_property = EdgeProperty(setting, inter_layer, select_edge_method='edge_betweenness', select_edges_number=5)
    print(edge_property.edges_order[0:10])
    
    
    
    


