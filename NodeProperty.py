import networkx as nx
import Setting_Simulation_Value
import InterconnectedLayerModeling
import operator
import time

class NodeProperty:
    def ordering_A_node(self, inter_layer, select_method):
        ordering_A_node = []
        if select_method == 'hub':
            ordering_A_node = self.order_hub_and_authority(inter_layer)[0]
        elif select_method == 'authority':
            ordering_A_node = self.order_hub_and_authority(inter_layer)[2]
        elif select_method == 'pagerank':
            ordering_A_node = self.order_pagerank(inter_layer)[0]
        elif select_method == 'eigenvector':
            ordering_A_node = self.order_eigenvector_centrality(inter_layer)[0]
        elif select_method == 'degree':
            ordering_A_node = self.order_degree_centrality(inter_layer)[0]
        elif select_method == 'betweenness':
            ordering_A_node = self.order_betweenness_centrality(inter_layer)[0]
        elif select_method == 'closeness':
            ordering_A_node = self.order_closeness_centrality(inter_layer)[0]
        elif select_method == 'load':
            ordering_A_node = self.order_closeness_centrality(inter_layer)[0]
        elif select_method == 'number_degree':
            ordering_A_node = self.order_number_of_degree(inter_layer)[0]
        elif select_method == 'AB_hub':
            ordering_A_node = self.order_AB_hub(inter_layer)
        elif select_method == 'AB_authority':
            ordering_A_node = self.order_AB_Authority(inter_layer)
        elif select_method == 'AB_pagerank':
            ordering_A_node = self.order_AB_pagerank(inter_layer)
        elif select_method == 'AB_eigenvector':
            ordering_A_node = self.order_AB_eigenvector(inter_layer)
        elif select_method == 'AB_degree':
            ordering_A_node = self.order_AB_degree(inter_layer)
        elif select_method == 'AB_betweenness':
            ordering_A_node = self.order_AB_betweenness(inter_layer)
        elif select_method == 'AB_closeness':
            ordering_A_node = self.order_AB_closeness(inter_layer)
        elif select_method == 'AB_load':
            ordering_A_node = self.order_AB_load(inter_layer)
        elif select_method == 'AB_number_degree':
            ordering_A_node = self.order_number_of_AB_degree(inter_layer)
        return ordering_A_node

    def ordering_B_node(self, inter_layer, select_method):
        ordering_B_node = []
        if select_method == 'hub':
            ordering_B_node = self.order_hub_and_authority(inter_layer)[1]
        elif select_method == 'authority':
            ordering_B_node = self.order_hub_and_authority(inter_layer)[3]
        elif select_method == 'pagerank':
            ordering_B_node = self.order_pagerank(inter_layer)[1]
        elif select_method == 'eigenvector':
            ordering_B_node = self.order_eigenvector_centrality(inter_layer)[1]
        elif select_method == 'degree':
            ordering_B_node = self.order_degree_centrality(inter_layer)[1]
        elif select_method == 'betweenness':
            ordering_B_node = self.order_betweenness_centrality(inter_layer)[1]
        elif select_method == 'closeness':
            ordering_B_node = self.order_closeness_centrality(inter_layer)[1]
        elif select_method == 'load':
            ordering_B_node = self.order_closeness_centrality(inter_layer)[1]
        elif select_method == 'number_degree':
            ordering_B_node = self.order_number_of_degree(inter_layer)[1]
        return ordering_B_node

    def finding_B_node(self, inter_layer, node_i):
        connected_B_node = 0
        neighbors = sorted(nx.neighbors(inter_layer.two_layer_graph, node_i))
        for neighbor in neighbors:
            if neighbor > (len(sorted(inter_layer.A_edges))-1):
                connected_B_node = neighbor
        return connected_B_node

    def order_hub_and_authority(self, inter_layer):
        A_node_order_h = []
        B_node_order_h = []
        A_node_order_a = []
        B_node_order_a = []
        hub, authority = nx.hits(inter_layer.two_layer_graph)
        hub_order = sorted(hub.items(), key=operator.itemgetter(1), reverse=True)
        authority_order = sorted(authority.items(), key=operator.itemgetter(1), reverse=True)
        for i in sorted(inter_layer.two_layer_graph.nodes):
            if hub_order[i][0] < len(sorted(inter_layer.A_edges)):
                A_node_order_h.append((hub_order[i][0], hub_order[i][1]))
            else:
                B_node_order_h.append((hub_order[i][0], hub_order[i][1]))
        for i in sorted(inter_layer.two_layer_graph.nodes):
            if authority_order[i][0] < len(sorted(inter_layer.A_edges)):
                A_node_order_a.append((authority_order[i][0], authority_order[i][1]))
            else:
                B_node_order_a.append((authority_order[i][0], authority_order[i][1]))
        return A_node_order_h, B_node_order_h, A_node_order_a, B_node_order_a

    def order_pagerank(self, inter_layer):
        A_node_order = []
        B_node_order = []
        pagerank = nx.pagerank(inter_layer.two_layer_graph)
        pagerank_order = sorted(pagerank.items(), key=operator.itemgetter(1), reverse=True)
        for i in sorted(inter_layer.two_layer_graph.nodes):
            if pagerank_order[i][0] < len(sorted(inter_layer.A_edges)):
                A_node_order.append((pagerank_order[i][0], pagerank_order[i][1]))
            else:
                B_node_order.append((pagerank_order[i][0], pagerank_order[i][1]))
        return A_node_order, B_node_order   # value = pagerank[node_number]

    def order_eigenvector_centrality(self, inter_layer):
        A_node_order = []
        B_node_order = []
        eigenvector_centrality = nx.eigenvector_centrality(inter_layer.two_layer_graph)
        eigenvector_order = sorted(eigenvector_centrality.items(), key=operator.itemgetter(1), reverse=True)
        for i in sorted(inter_layer.two_layer_graph.nodes):
            if eigenvector_order[i][0] < len(sorted(inter_layer.A_edges)):
                A_node_order.append((eigenvector_order[i][0], eigenvector_order[i][1]))
            else:
                B_node_order.append((eigenvector_order[i][0], eigenvector_order[i][1]))
        return A_node_order, B_node_order

    def order_degree_centrality(self, inter_layer):
        A_node_order = []
        B_node_order = []
        degree_centrality = nx.degree_centrality(inter_layer.two_layer_graph)
        degree_order = sorted(degree_centrality.items(), key=operator.itemgetter(1), reverse=True)
        for i in sorted(inter_layer.two_layer_graph.nodes):
            if degree_order[i][0] < len(sorted(inter_layer.A_edges)):
                A_node_order.append((degree_order[i][0], degree_order[i][1]))
            else:
                B_node_order.append((degree_order[i][0], degree_order[i][1]))
        return A_node_order, B_node_order

    def order_betweenness_centrality(self, inter_layer):
        A_node_order = []
        B_node_order = []
        betweenness_centrality = nx.betweenness_centrality(inter_layer.two_layer_graph)
        betweenness_order = sorted(betweenness_centrality.items(), key=operator.itemgetter(1), reverse=True)
        for i in sorted(inter_layer.two_layer_graph.nodes):
            if betweenness_order[i][0] < len(sorted(inter_layer.A_edges)):
                A_node_order.append((betweenness_order[i][0], betweenness_order[i][1]))
            else:
                B_node_order.append((betweenness_order[i][0], betweenness_order[i][1]))
        return A_node_order, B_node_order

    def order_closeness_centrality(self, inter_layer):
        A_node_order = []
        B_node_order = []
        closeness_centrality = nx.closeness_centrality(inter_layer.two_layer_graph)
        closeness_order = sorted(closeness_centrality.items(), key=operator.itemgetter(1), reverse=True)
        for i in sorted(inter_layer.two_layer_graph.nodes):
            if closeness_order[i][0] < len(sorted(inter_layer.A_edges)):
                A_node_order.append((closeness_order[i][0], closeness_order[i][1]))
            else:
                B_node_order.append((closeness_order[i][0], closeness_order[i][1]))
        return A_node_order, B_node_order

    def order_load_centrality(self, inter_layer):
        A_node_order = []
        B_node_order = []
        load_centrality = nx.load_centrality(inter_layer.two_layer_graph)
        load_order = sorted(load_centrality.items(), key=operator.itemgetter(1), reverse=True)
        for i in sorted(inter_layer.two_layer_graph.nodes):
            if load_order[i][0] < len(sorted(inter_layer.A_edges)):
                A_node_order.append((load_order[i][0], load_order[i][1]))
            else:
                B_node_order.append((load_order[i][0], load_order[i][1]))
        return A_node_order, B_node_order

    def order_number_of_degree(self, inter_layer):
        A_node_order = []
        B_node_order = []
        number_degree = {}
        for node_i in sorted(inter_layer.two_layer_graph.nodes):
            degree = len(sorted(nx.neighbors(inter_layer.two_layer_graph, node_i)))
            number_degree[node_i] = degree
        numberdegree_order = sorted(number_degree.items(), key=operator.itemgetter(1), reverse=True)
        for i in sorted(inter_layer.two_layer_graph.nodes):
            if numberdegree_order[i][0] < len(sorted(inter_layer.A_edges)):
                A_node_order.append((numberdegree_order[i][0], numberdegree_order[i][1]))
            else:
                B_node_order.append((numberdegree_order[i][0], numberdegree_order[i][1]))
        return A_node_order, B_node_order


    def order_AB_degree(self, inter_layer):
        AB_degree = {}
        dict = nx.degree_centrality(inter_layer.two_layer_graph)
        for node_i in sorted(inter_layer.A_edges):
            connected_B_node = self.finding_B_node(inter_layer, node_i)
            integrated_degree = dict[node_i] + dict[connected_B_node]
            AB_degree[node_i] = integrated_degree
        AB_degree_order = sorted(AB_degree.items(), key=operator.itemgetter(1), reverse=True)
        return AB_degree_order

    def order_number_of_AB_degree(self, inter_layer):
        AB_Number_degree = {}
        dict = {}
        for node_i in sorted(inter_layer.two_layer_graph.nodes):
            degree = len(sorted(nx.neighbors(inter_layer.two_layer_graph, node_i)))
            dict[node_i] = degree
        for node_i in sorted(inter_layer.A_edges):
            connected_B_node = self.finding_B_node(inter_layer, node_i)
            integrated_number_degree = dict[node_i] + dict[connected_B_node]
            AB_Number_degree[node_i] = integrated_number_degree
        AB_Number_degree_order = sorted(AB_Number_degree.items(), key=operator.itemgetter(1), reverse=True)
        return AB_Number_degree_order

    def order_AB_hub(self, inter_layer):
        AB_hub = {}
        dict = nx.hits(inter_layer.two_layer_graph)[0]
        for node_i in sorted(inter_layer.A_edges):
            connected_B_node = self.finding_B_node(inter_layer, node_i)
            integrated_hub = dict[node_i] + dict[connected_B_node]
            AB_hub[node_i] = integrated_hub
        AB_hub_order = sorted(AB_hub.items(), key=operator.itemgetter(1), reverse=True)
        return AB_hub_order

    def order_AB_Authority(self, inter_layer):
        AB_Authority = {}
        dict = nx.hits(inter_layer.two_layer_graph)[1]
        for node_i in sorted(inter_layer.A_edges):
            connected_B_node = self.finding_B_node(inter_layer, node_i)
            integrated_authority = dict[1][node_i] + dict[1][connected_B_node]
            AB_Authority[node_i] = integrated_authority
        AB_Authority_order = sorted(AB_Authority.items(), key=operator.itemgetter(1), reverse=True)
        return AB_Authority_order

    def order_AB_pagerank(self, inter_layer):
        AB_pagerank = {}
        dict = nx.pagerank(inter_layer.two_layer_graph)
        for node_i in sorted(inter_layer.A_edges):
            connected_B_node = self.finding_B_node(inter_layer, node_i)
            integrated_pagerank = dict[node_i] + dict[connected_B_node]
            AB_pagerank[node_i] = integrated_pagerank
        AB_pagerank_order = sorted(AB_pagerank.items(), key=operator.itemgetter(1), reverse=True)
        return AB_pagerank_order

    def order_AB_eigenvector(self, inter_layer):
        AB_eigenvector = {}
        dict = nx.eigenvector_centrality(inter_layer.two_layer_graph)
        for node_i in sorted(inter_layer.A_edges):
            connected_B_node = self.finding_B_node(inter_layer, node_i)
            integrated_eigenvector = dict[node_i] + dict[connected_B_node]
            AB_eigenvector[node_i] = integrated_eigenvector
        AB_eigenvector_order = sorted(AB_eigenvector.items(), key=operator.itemgetter(1), reverse=True)
        return AB_eigenvector_order

    def order_AB_betweenness(self, inter_layer):
        AB_betweenness = {}
        dict = nx.betweenness_centrality(inter_layer.two_layer_graph)
        for node_i in sorted(inter_layer.A_edges):
            connected_B_node = self.finding_B_node(inter_layer, node_i)
            integrated_betweenness = dict[node_i] + dict[connected_B_node]
            AB_betweenness[node_i] = integrated_betweenness
        AB_betweenness_order = sorted(AB_betweenness.items(), key=operator.itemgetter(1), reverse=True)
        return AB_betweenness_order

    def order_AB_closeness(self, inter_layer):
        AB_closeness = {}
        dict = nx.closeness_centrality(inter_layer.two_layer_graph)
        for node_i in sorted(inter_layer.A_edges):
            connected_B_node = self.finding_B_node(inter_layer, node_i)
            integrated_closeness = dict[node_i] + dict[connected_B_node]
            AB_closeness[node_i] = integrated_closeness
        AB_closeness_order = sorted(AB_closeness.items(), key=operator.itemgetter(1), reverse=True)
        return AB_closeness_order

    def order_AB_load(self, inter_layer):
        AB_load = {}
        dict = nx.load_centrality(inter_layer.two_layer_graph)
        for node_i in sorted(inter_layer.A_edges):
            connected_B_node = self.finding_B_node(inter_layer, node_i)
            integrated_load = dict[node_i] + dict[connected_B_node]
            AB_load[node_i] = integrated_load
        AB_load_order = sorted(AB_load.items(), key=operator.itemgetter(1), reverse=True)
        return AB_load_order

if __name__ == "__main__":
    print("CalculatingProperty")
    setting = Setting_Simulation_Value.Setting_Simulation_Value()
    inter_layer = InterconnectedLayerModeling.InterconnectedLayerModeling(setting)
    cal_property = NodeProperty()
    # select = cal_property.cal_node_A_and_node_B_centrality(inter_layer)
    start = time.time()
    select = cal_property.order_hub_and_authority(inter_layer)
    print(select[0])
    end = time.time()
    print(end-start)



