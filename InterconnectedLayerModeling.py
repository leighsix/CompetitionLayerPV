import networkx as nx
import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
import matplotlib
import Setting_Simulation_Value
matplotlib.use("TkAgg")

## A layer Modeling : A, A_edges, AB_edges, AB_neighbor


class InterconnectedLayerModeling:
    def __init__(self, setting):
        A_edges_array = self.A_layer_config(setting)
        self.A_edges = A_edges_array[0]
        self.AB_edges = A_edges_array[1]
        self.AB_neighbor = A_edges_array[2]
        self.B_edges = self.B_layer_config(setting)
        self.two_layer_graph = self.making_interconnected_graph(setting)

    def making_interconnected_graph(self, setting):
        self.two_layer_graph = nx.Graph()
        for i in range(setting.A_node):
            self.two_layer_graph.add_node(i, name='A_%s' % i, state=setting.A[i])
        for i in range(setting.B_node):
            self.two_layer_graph.add_node(i+setting.A_node, name='B_%s' % i, state=setting.B[i])
        A_edges_list = sorted(self.A_edges.edges)
        self.two_layer_graph.add_edges_from(A_edges_list)
        B_edges_list = self.B_edges
        self.two_layer_graph.add_edges_from(B_edges_list)
        AB_edges_list = self.AB_edges
        self.two_layer_graph.add_edges_from(AB_edges_list)
        return self.two_layer_graph

    def A_layer_config(self, setting):
        # A_layer 구성요소 A_layer_config(state = [1,2], node = 2048, edge = 5, Max = 2, Min = -2)
        self.select_layer_A_model(setting)
        self.making_interconnected_edges(setting)
        return self.A_edges, self.AB_edges, self.AB_neighbor
        # A : A의 각 노드의 상태, A_state : A 노드 상태의 종류(1, 2, -1, -2),
        # A_node : 노드의 수, A_edge : 내부연결선수, A_edges : 내부연결상태(튜플), MAX : 최대상태, MIN : 최소상태

    def select_layer_A_model(self, setting):
        if setting.Structure.split('-')[0] == 'RR':
            self.making_layer_A_random_regular(setting)
        elif setting.Structure.split('-')[0] == 'BA':
            self.making_layer_A_barabasi_albert(setting)
        return self.A_edges

    def making_layer_A_random_regular(self, setting):
        # A_layer random_regular network
        self.A_edges = nx.random_regular_graph(setting.A_edge, setting.A_node, seed=None)
        return self.A_edges

    def making_layer_A_barabasi_albert(self, setting):
        # A_layer 바바라시-알버트 네트워크
        self.A_edges = nx.barabasi_albert_graph(setting.A_node, setting.A_edge, seed=None)
        return self.A_edges

    def making_interconnected_edges(self, setting):
        self.AB_edges = []
        self.AB_neighbor = []
        for i in range(int(setting.A_node / setting.B_inter_edges)):
            for j in range(setting.B_inter_edges):
                connected_A_node = np.array(self.A_edges.nodes).reshape(-1, setting.B_inter_edges)[i][j]
                self.AB_neighbor.append(connected_A_node)
                self.AB_edges.append((i + setting.A_node, connected_A_node))
        self.AB_neighbor = np.array(self.AB_neighbor).reshape(-1, setting.B_inter_edges)
        return self.AB_edges, self.AB_neighbor
        # AB_neighbor은 B노드번호 기준으로 연결된 A노드번호  ex) AB_neighbor[0]= array([0, 1])
        # B 노드 0에 A노드 0번, 1번이 연결되어 있다는 뜻
        # AB_edges는 (0, 1)은 B 노드 0번과 A 노드 1번이 연결되어 있다는 뜻


    def B_layer_config(self, setting):  # B_layer 구성요소 B_layer_config(state = [-1], node = 2048, edge = 5, inter_edge= 1)
        self.select_layer_B_model(setting)
        return self.B_edges

    def select_layer_B_model(self, setting):
        if setting.Structure.split('-')[1] == 'RR':
            self.making_layer_B_random_regular(setting)
        elif setting.Structure.split('-')[1] == 'BA':
            self.making_layer_B_barabasi_albert(setting)
        return self.B_edges

    def making_layer_B_random_regular(self, setting):  # B_layer random_regular network
        b_edges = nx.random_regular_graph(setting.B_edge, setting.B_node, seed=None)
        self.B_edges = []
        for i in range(len(b_edges.edges)):
            self.B_edges.append((sorted(b_edges.edges)[i][0] + setting.A_node,
                            sorted(b_edges.edges)[i][1] + setting.A_node))
        return self.B_edges

    def making_layer_B_barabasi_albert(self, setting):  # B_layer 바바라시-알버트 네트워크
        b_edges = nx.barabasi_albert_graph(setting.B_node, setting.B_edge, seed=None)
        self.B_edges = []
        for i in range(len(b_edges.edges)):
            self.B_edges.append((sorted(b_edges.edges)[i][0] + setting.A_node,
                            sorted(b_edges.edges)[i][1] + setting.A_node))
        return self.B_edges


if __name__ == "__main__" :
    print("interconnectedlayer")
    setting = Setting_Simulation_Value.Setting_Simulation_Value()
    inter_layer = InterconnectedLayerModeling(setting)
    state = 0
    for i in range(len(inter_layer.two_layer_graph.nodes)):
        state += inter_layer.two_layer_graph.nodes[i]['state']
    print(state)
    print(inter_layer.A_edges.edges)
    external_edge_number = len(inter_layer.AB_neighbor[1])
    print(external_edge_number)
    inter_edges = len(sorted(nx.all_neighbors(inter_layer.two_layer_graph, 1 + setting.A_node))) - external_edge_number
    print(inter_edges)
    print(sorted(nx.all_neighbors(inter_layer.two_layer_graph, 1 + setting.A_node), reverse=True))

    #print(Layer_A.AB_edges)
    #print(Layer_A.AB_neighbor)
    #print(Layer_A.SS.A_node)



