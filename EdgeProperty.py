import networkx as nx
import SettingSimulationValue
import InterconnectedLayerModeling
import operator
import time

class EdgeProperty:
    def ordering_edge(self, setting, inter_layer, select_method, select_layer='A_layer', select_edges_number=0):
        ordering_edge = []
        if select_method == 'edge_betweenness':
            ordering_edge = EdgeProperty.ordering_edge_betweenness(setting, inter_layer)[select_edges_number]
        return ordering_edge

    @staticmethod
    def ordering_edge_betweenness(setting, inter_layer):
        A_internal_order = []
        B_internal_order = []
        external_order = []
        mixed_order = []
        edge_betweenness = nx.edge_betweenness_centrality(inter_layer.two_layer_graph)
        edge_betweenness_order = sorted(edge_betweenness.items(), key=operator.itemgetter(1), reverse=True)
        mixed_order = edge_betweenness_order
        for i in range(len(edge_betweenness_order)):
            edge = edge_betweenness_order[i][0]
            if (edge[0] < setting.A_node) and (edge[1] < setting.A_node):
                A_internal_order.append(edge_betweenness_order[i])
            elif (edge[0] >= setting.A_node) and (edge[1] >= setting.A_node):
                B_internal_order.append(edge_betweenness_order[i])
            elif (edge[0] < setting.A_node) and (edge[1] >= setting.A_node):
                external_order.append(edge_betweenness_order[i])
        return A_internal_order, B_internal_order, external_order, mixed_order
    
    
if __name__ == "__main__":
    setting = SettingSimulationValue.SettingSimulationValue()
    setting.A_node = 64
    setting.B_node = 64
    inter_layer = InterconnectedLayerModeling.InterconnectedLayerModeling(setting)
    edge_property = EdgeProperty()
    res = edge_property.ordering_edge(setting, inter_layer, 'edge_betweenness')
    print(res[0:10])
    
    
    
    


