import networkx as nx
import Setting_Simulation_Value
import InterconnectedLayerModeling
import numpy as np
import math
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import operator


class CalculatingProperty:

    def making_df_property_for_100steps(self, df, inter_layer, node_i_name):
        df['Unchanged_A_Node'] = node_i_name
        node_i = node_i_name.split('_')[1]
        if node_i == 'N':
            df['Connected_B_node'] = 0
        else:
            node_i = int(node_i)
            connected_B_node = self.finding_B_node(inter_layer, node_i)
            df['Connected_B_node'] = 'B_%s' % (connected_B_node - len(sorted(inter_layer.A_edges)))
        return df

    def making_df_for_property(self, panda_df, inter_layer, node_i_name):
        node_i = node_i_name.split('_')[1]
        if node_i == 'N':
            panda_df['Unchanged_A_Node'] = node_i_name
            panda_df['Un_A_node_state'] = 0
            panda_df['A_Clustering'] = 0
            panda_df['A_Hub'] = 0
            panda_df['A_Authority'] = 0
            panda_df['A_Pagerank'] = 0
            panda_df['A_Eigenvector'] = 0
            panda_df['A_Degree'] = 0
            panda_df['A_Betweenness'] = 0
            panda_df['A_Closeness'] = 0
            panda_df['A_Load'] = 0
            panda_df['A_NumberofDegree'] = 0
            panda_df['Connected_B_node'] = 0
            panda_df['B_Clustering'] = 0
            panda_df['B_Hub'] = 0
            panda_df['B_Authority'] = 0
            panda_df['B_Pagerank'] = 0
            panda_df['B_Eigenvector'] = 0
            panda_df['B_Degree'] = 0
            panda_df['B_Betweenness'] = 0
            panda_df['B_Closeness'] = 0
            panda_df['B_Load'] = 0

        else:
            node_i = int(node_i)
            panda_df['Unchanged_A_Node'] = node_i_name
            panda_df['Un_A_node_state'] = inter_layer.two_layer_graph.node[node_i]['state']
            panda_df['A_Hub'] = self.cal_hub_and_authority(inter_layer)[0][node_i]
            panda_df['A_Authority'] = self.cal_hub_and_authority(inter_layer)[1][node_i]
            panda_df['A_Pagerank'] = self.cal_pagerank(inter_layer)[node_i]
            panda_df['A_Eigenvector'] = self.cal_eigenvector_centrality(inter_layer)[node_i]
            panda_df['A_Degree'] = self.cal_degree_centrality(inter_layer)[node_i]
            panda_df['A_Betweenness'] = self.cal_betweenness_centrality(inter_layer)[node_i]
            panda_df['A_Closeness'] = self.cal_closeness_centrality(inter_layer)[node_i]
            panda_df['A_Load'] = self.cal_load_centrality(inter_layer)[node_i]
            panda_df['A_NumberofDegree'] = self.cal_number_of_degree(inter_layer)[node_i]
            connected_B_node = self.finding_B_node(inter_layer, node_i)
            panda_df['Connected_B_node'] = 'B_%s' % (connected_B_node-len(sorted(inter_layer.A_edges)))
            panda_df['B_Hub'] = self.cal_hub_and_authority(inter_layer)[0][connected_B_node]
            panda_df['B_Authority'] = self.cal_hub_and_authority(inter_layer)[1][connected_B_node]
            panda_df['B_Pagerank'] = self.cal_pagerank(inter_layer)[connected_B_node]
            panda_df['B_Eigenvector'] = self.cal_eigenvector_centrality(inter_layer)[connected_B_node]
            panda_df['B_Degree'] = self.cal_degree_centrality(inter_layer)[connected_B_node]
            panda_df['B_Betweenness'] = self.cal_betweenness_centrality(inter_layer)[connected_B_node]
            panda_df['B_Closeness'] = self.cal_closeness_centrality(inter_layer)[connected_B_node]
            panda_df['B_Load'] = self.cal_load_centrality(inter_layer)[connected_B_node]
            panda_df['B_NumberofDegree'] = self.cal_number_of_degree(inter_layer)[connected_B_node]
        return panda_df

    def making_array_for_property(self, additional_array, inter_layer, node_i_name):
        node_i = node_i_name.split('_')[1]
        if node_i == 'N':
            additional_array2 = np.zeros(21)
            new_array = np.concatenate([additional_array, additional_array2])
        else:
            node_i = int(node_i)
            connected_B_node = self.finding_B_node(inter_layer, node_i)
            additional_array2 = np.array([inter_layer.two_layer_graph.node[node_i]['state'],
                                          self.cal_hub_and_authority(inter_layer)[0][node_i],
                                          self.cal_hub_and_authority(inter_layer)[1][node_i],
                                          self.cal_pagerank(inter_layer)[node_i],
                                          self.cal_eigenvector_centrality(inter_layer)[node_i],
                                          self.cal_degree_centrality(inter_layer)[node_i],
                                          self.cal_betweenness_centrality(inter_layer)[node_i],
                                          self.cal_closeness_centrality(inter_layer)[node_i],
                                          self.cal_load_centrality(inter_layer)[node_i],
                                          self.cal_number_of_degree(inter_layer)[node_i],
                                          self.cal_hub_and_authority(inter_layer)[0][connected_B_node],
                                          self.cal_hub_and_authority(inter_layer)[1][connected_B_node],
                                          self.cal_pagerank(inter_layer)[connected_B_node],
                                          self.cal_eigenvector_centrality(inter_layer)[connected_B_node],
                                          self.cal_degree_centrality(inter_layer)[connected_B_node],
                                          self.cal_betweenness_centrality(inter_layer)[connected_B_node],
                                          self.cal_closeness_centrality(inter_layer)[connected_B_node],
                                          self.cal_load_centrality(inter_layer)[connected_B_node],
                                          self.cal_number_of_degree(inter_layer)[connected_B_node]])
            new_array = np.concatenate([additional_array, additional_array2])
        return new_array

    def finding_B_node(self, inter_layer, node_i):
        connected_B_node = 0
        neighbors = sorted(nx.neighbors(inter_layer.two_layer_graph, node_i))
        for neighbor in neighbors:
            if neighbor > (len(sorted(inter_layer.A_edges))-1):
                connected_B_node = neighbor
        return connected_B_node

    def select_main_A_node(self, inter_layer):
        hub = self.cal_hub_and_authority(inter_layer)[0]
        hub_order = sorted(hub.items(), key=operator.itemgetter(1), reverse=True)[0][0]
        authority = self.cal_hub_and_authority(inter_layer)[1]
        authority_order = sorted(authority.items(), key=operator.itemgetter(1), reverse=True)[0][0]
        pagerank = self.cal_pagerank(inter_layer)
        pagerank_order = sorted(pagerank.items(), key=operator.itemgetter(1), reverse=True)[0][0]
        eigenvector = self.cal_eigenvector_centrality(inter_layer)
        eigenvector_order = sorted(eigenvector.items(), key=operator.itemgetter(1), reverse=True)[0][0]
        degree = self.cal_degree_centrality(inter_layer)
        degree_order = sorted(degree.items(), key=operator.itemgetter(1), reverse=True)[0][0]
        betweenness = self.cal_betweenness_centrality(inter_layer)
        betweenness_order = sorted(betweenness.items(), key=operator.itemgetter(1), reverse=True)[0][0]
        closeness = self.cal_closeness_centrality(inter_layer)
        closeness_order = sorted(closeness.items(), key=operator.itemgetter(1), reverse=True)[0][0]
        load = self.cal_load_centrality(inter_layer)
        load_order = sorted(load.items(), key=operator.itemgetter(1), reverse=True)[0][0]
        numberdegree = self.cal_number_of_degree(inter_layer)
        numberdegree_order = sorted(numberdegree.items(), key=operator.itemgetter(1), reverse=True)[0][0]
        main_A_node = [hub_order, authority_order, pagerank_order, eigenvector_order, degree_order,
                       betweenness_order, closeness_order, load_order, numberdegree_order]
        return main_A_node

    def select_main_AB_node(self, inter_layer):
        AB_numberdegree = self.cal_number_of_AB_degree(inter_layer)
        AB_numberdegree_order = sorted(AB_numberdegree.items(), key=operator.itemgetter(1), reverse=True)[0][0]
        AB_authority = self.cal_AB_Authority(inter_layer)
        AB_authority_order = sorted(AB_authority.items(), key=operator.itemgetter(1), reverse=True)[0][0]
        AB_betweenness = self.cal_AB_betweenness(inter_layer)
        AB_betweenness_order = sorted(AB_betweenness.items(), key=operator.itemgetter(1), reverse=True)[0][0]
        AB_degree= self.cal_AB_degree(inter_layer)
        AB_degree_order = sorted(AB_degree.items(), key=operator.itemgetter(1), reverse=True)[0][0]
        AB_closeness = self.cal_AB_closeness(inter_layer)
        AB_closeness_order = sorted(AB_closeness.items(), key=operator.itemgetter(1), reverse=True)[0][0]
        AB_eigenvector = self.cal_AB_eigenvector(inter_layer)
        AB_eigenvector_order = sorted(AB_eigenvector.items(), key=operator.itemgetter(1), reverse=True)[0][0]
        AB_hub = self.cal_AB_hub(inter_layer)
        AB_hub_order = sorted(AB_hub.items(), key=operator.itemgetter(1), reverse=True)[0][0]
        AB_load = self.cal_AB_load(inter_layer)
        AB_load_order = sorted(AB_load.items(), key=operator.itemgetter(1), reverse=True)[0][0]
        AB_pagerank = self.cal_AB_pagerank(inter_layer)
        AB_pagerank_order = sorted(AB_pagerank.items(), key=operator.itemgetter(1), reverse=True)[0][0]
        main_AB_node = [AB_hub_order, AB_authority_order, AB_pagerank_order, AB_eigenvector_order, AB_degree_order,
                       AB_betweenness_order, AB_closeness_order, AB_load_order, AB_numberdegree_order]
        return main_AB_node

    def cal_hub_and_authority(self, inter_layer):
        hub, authority = nx.hits(inter_layer.two_layer_graph)
        return hub, authority  # value = hub[node_number]
        # hub_order = sorted(h.items(), key=operator.itemgetter(1), reverse=True)
        # authority_order = sorted(a.items(), key=operator.itemgetter(1), reverse=True)

    def cal_pagerank(self, inter_layer):
        pagerank = nx.pagerank(inter_layer.two_layer_graph)
        return pagerank  # value = pagerank[node_number]

    def cal_eigenvector_centrality(self, inter_layer):
        eigenvector_centrality = nx.eigenvector_centrality(inter_layer.two_layer_graph)
        return eigenvector_centrality

    def cal_degree_centrality(self, inter_layer):
        degree_centrality = nx.degree_centrality(inter_layer.two_layer_graph)
        return degree_centrality

    def cal_betweenness_centrality(self, inter_layer):
        betweenness_centrality = nx.betweenness_centrality(inter_layer.two_layer_graph)
        return betweenness_centrality

    def cal_closeness_centrality(self, inter_layer):
        closeness_centrality = nx.closeness_centrality(inter_layer.two_layer_graph)
        return closeness_centrality

    def cal_load_centrality(self, inter_layer):
        load_centrality = nx.load_centrality(inter_layer.two_layer_graph)
        return load_centrality

    def cal_number_of_degree(self, inter_layer):
        number_degree = {}
        for node_i in sorted(inter_layer.two_layer_graph.nodes):
            degree = len(sorted(nx.neighbors(inter_layer.two_layer_graph, node_i)))
            number_degree[node_i] = degree
        return number_degree

    def cal_AB_degree(self, inter_layer):
        AB_degree = {}
        for node_i in sorted(inter_layer.A_edges):
            connected_B_node = self.finding_B_node(inter_layer, node_i)
            degree = self.cal_degree_centrality(inter_layer)[node_i] + self.cal_degree_centrality(inter_layer)[connected_B_node]
            AB_degree[node_i] = degree
        return AB_degree

    def cal_number_of_AB_degree(self, inter_layer):
        AB_Number_degree = {}
        for node_i in sorted(inter_layer.A_edges):
            connected_B_node = self.finding_B_node(inter_layer, node_i)
            Number_degree  = self.cal_number_of_degree(inter_layer)[node_i] + self.cal_number_of_degree(inter_layer)[connected_B_node]
            AB_Number_degree[node_i] = Number_degree
        return AB_Number_degree

    def cal_AB_hub(self, inter_layer):
        AB_hub = {}
        for node_i in sorted(inter_layer.A_edges):
            connected_B_node = self.finding_B_node(inter_layer, node_i)
            hub = self.cal_hub_and_authority(inter_layer)[0][node_i] + self.cal_hub_and_authority(inter_layer)[0][connected_B_node]
            AB_hub[node_i] = hub
        return AB_hub

    def cal_AB_Authority(self, inter_layer):
        AB_Authority = {}
        for node_i in sorted(inter_layer.A_edges):
            connected_B_node = self.finding_B_node(inter_layer, node_i)
            Authority = self.cal_hub_and_authority(inter_layer)[1][node_i] + self.cal_hub_and_authority(inter_layer)[1][connected_B_node]
            AB_Authority[node_i] = Authority
        return AB_Authority

    def cal_AB_pagerank(self, inter_layer):
        AB_pagerank = {}
        for node_i in sorted(inter_layer.A_edges):
            connected_B_node = self.finding_B_node(inter_layer, node_i)
            pagerank  = self.cal_pagerank(inter_layer)[node_i] + self.cal_pagerank(inter_layer)[connected_B_node]
            AB_pagerank[node_i] = pagerank
        return AB_pagerank

    def cal_AB_eigenvector(self, inter_layer):
        AB_eigenvector = {}
        for node_i in sorted(inter_layer.A_edges):
            connected_B_node = self.finding_B_node(inter_layer, node_i)
            eigenvector  = self.cal_eigenvector_centrality(inter_layer)[node_i] + self.cal_eigenvector_centrality(inter_layer)[connected_B_node]
            AB_eigenvector[node_i] = eigenvector
        return AB_eigenvector

    def cal_AB_betweenness(self, inter_layer):
        AB_betweenness = {}
        for node_i in sorted(inter_layer.A_edges):
            connected_B_node = self.finding_B_node(inter_layer, node_i)
            betweenness = self.cal_betweenness_centrality(inter_layer)[node_i] + self.cal_betweenness_centrality(inter_layer)[connected_B_node]
            AB_betweenness[node_i] = betweenness
        return AB_betweenness

    def cal_AB_closeness(self, inter_layer):
        AB_closeness = {}
        for node_i in sorted(inter_layer.A_edges):
            connected_B_node = self.finding_B_node(inter_layer, node_i)
            closeness = self.cal_closeness_centrality(inter_layer)[node_i] + self.cal_closeness_centrality(inter_layer)[connected_B_node]
            AB_closeness[node_i] = closeness
        return AB_closeness

    def cal_AB_load(self, inter_layer):
        AB_load = {}
        for node_i in sorted(inter_layer.A_edges):
            connected_B_node = self.finding_B_node(inter_layer, node_i)
            load = self.cal_load_centrality(inter_layer)[node_i] + self.cal_load_centrality(inter_layer)[connected_B_node]
            AB_load[node_i] = load
        return AB_load


if __name__ == "__main__":
    print("CalculatingProperty")
    setting = Setting_Simulation_Value.Setting_Simulation_Value()
    inter_layer = InterconnectedLayerModeling.InterconnectedLayerModeling(setting)
    cal_property = CalculatingProperty()
    # select = cal_property.cal_node_A_and_node_B_centrality(inter_layer)
    select = cal_property.select_main_A_node(inter_layer)
    select2 = cal_property.select_main_AB_node(inter_layer)
    print(select)
    print(select2)



