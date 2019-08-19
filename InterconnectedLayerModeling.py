import networkx as nx
import numpy as np
import random
import time
from pymnet import *
import matplotlib.pyplot as plt
import SettingSimulationValue
from mpl_toolkits.mplot3d.axes3d import *
import matplotlib.animation as animation
import matplotlib
matplotlib.use("TkAgg")


class InterconnectedLayerModeling:
    def __init__(self, setting):
        self.A_nodes = [i for i in range(0, setting.A_node)]
        self.B_nodes = [i for i in range(setting.A_node, setting.A_node + setting.B_node)]
        graph = InterconnectedLayerModeling.making_interconnected_graph(setting, self.A_nodes, self.B_nodes)
        self.A_layer_graph = graph[0]
        self.B_layer_graph = graph[1]
        self.two_layer_graph = graph[2]
        self.unique_neighbor_dict = InterconnectedLayerModeling.making_unique_neighbor_dict(self.two_layer_graph)
        edges_tuple = InterconnectedLayerModeling.making_edges_tuple_on_layer(setting, self.two_layer_graph)
        self.edges_on_A = edges_tuple[0]
        self.edges_on_B = edges_tuple[1]
        self.edges_on_AB = edges_tuple[2]
        self.node_location = InterconnectedLayerModeling.making_node_location(self.A_nodes+self.B_nodes)

    @staticmethod
    def making_interconnected_graph(setting, A_nodes, B_nodes):
        A_states = random.sample(list(setting.A), setting.A_node)
        B_states = random.sample(list(setting.B), setting.B_node)
        two_layer_graph = nx.Graph()
        A_layer_graph = nx.Graph()
        B_layer_graph = nx.Graph()
        for i, a_node in enumerate(A_nodes):
            A_layer_graph.add_node(a_node, state=A_states[i])
            two_layer_graph.add_node(a_node, state=A_states[i])
        for i, b_node in enumerate(B_nodes):
            B_layer_graph.add_node(b_node, state=B_states[i])
            two_layer_graph.add_node(b_node, state=B_states[i])
        A_edges_list = InterconnectedLayerModeling.select_layer_A_model(setting)
        A_layer_graph.add_edges_from(A_edges_list)
        two_layer_graph.add_edges_from(A_edges_list)
        B_edges_list = InterconnectedLayerModeling.select_layer_B_model(setting)
        B_layer_graph.add_edges_from(B_edges_list)
        two_layer_graph.add_edges_from(B_edges_list)
        AB_edges_list = InterconnectedLayerModeling.making_interconnected_edges(setting, A_nodes, B_nodes)
        two_layer_graph.add_edges_from(AB_edges_list)
        return A_layer_graph, B_layer_graph, two_layer_graph

    @staticmethod
    def making_interconnected_edges(setting, A_nodes, B_nodes):
        AB_edges = []
        A_nodes_list = random.sample(A_nodes, len(A_nodes))
        B_nodes_list = random.sample(B_nodes, len(B_nodes))
        if setting.A_node >= setting.B_node:
            for i in range(setting.B_node):
                for j in range(int(setting.A_node / setting.B_node)):
                    connected_A_node = np.array(A_nodes_list).reshape(-1, int(setting.A_node / setting.B_node))[i][j]
                    AB_edges.append((connected_A_node, i + setting.A_node))
        else:
            for i in range(setting.A_node):
                for j in range(int(setting.B_node / setting.A_node)):
                    connected_B_node = np.array(B_nodes_list).reshape(-1, int(setting.B_node / setting.A_node))[i][j]
                    AB_edges.append((i, connected_B_node))
        return AB_edges

    @staticmethod
    def making_edges_tuple_on_layer(setting, two_layer_graph):
        edges_list = sorted(two_layer_graph.edges)
        edges_on_A = []
        edges_on_B = []
        edges_on_AB = []
        for i, j in edges_list:
            if (i < setting.A_node) and (j < setting.A_node):
                edges_on_A.append((i, j))
            elif (i >= setting.A_node) and (j >= setting.A_node):
                edges_on_B.append((i, j))
            else:
                edges_on_AB.append((i, j))
        return edges_on_A, edges_on_B, edges_on_AB

    @staticmethod
    def making_unique_neighbor_dict(two_layer_graph):
        unique_neighbor_dict = {}
        edges_list = sorted(two_layer_graph.edges)
        for node_i in two_layer_graph.nodes:
            unique_neighbor = []
            for edge in edges_list:
                if node_i == edge[0]:
                    unique_neighbor.append(edge[1])
            if len(unique_neighbor) != 0:
                unique_neighbor_dict[node_i] = unique_neighbor
        return unique_neighbor_dict

    @staticmethod
    def select_layer_A_model(setting):
        A_edges = []
        if setting.Structure.split('-')[0] == 'RR':
            a_edges = sorted(nx.random_regular_graph(setting.A_edge, setting.A_node, seed=None).edges)
            for i in range(len(a_edges)):
                A_edges.append(tuple(np.array(a_edges[i][0:2])))
        elif setting.Structure.split('-')[0] == 'BA':
            A_edges = sorted(nx.barabasi_albert_graph(setting.A_node, setting.A_edge, seed=None).edges)
        return A_edges

    @staticmethod
    def select_layer_B_model(setting):
        B_edges = []
        if setting.Structure.split('-')[1] == 'RR':
            b_edges = sorted(nx.random_regular_graph(setting.B_edge, setting.B_node, seed=None).edges)
            for i in range(len(b_edges)):
                B_edges.append(tuple(np.array(b_edges[i][0:2]) + setting.A_node))
        elif setting.Structure.split('-')[1] == 'BA':
            b_edges = sorted(nx.barabasi_albert_graph(setting.B_node, setting.B_edge, seed=None).edges)
            for i in range(len(b_edges)):
                B_edges.append(tuple(np.array(b_edges[i]) + setting.A_node))
        return B_edges

    @staticmethod
    def making_node_location(nodes):  # layer, node_number, location
        node_loc = {i: (random.random(), random.random()) for i in nodes}
        return node_loc

if __name__ == "__main__":
    print("interconnectedlayer")
    setting = SettingSimulationValue.SettingSimulationValue()
    setting.Structure = 'BA-RR'
    setting.A_edge = 3
    setting.B_edge = 6
    start = time.time()
    # print(InterconnectedLayerModeling.select_layer_A_model(setting))
    inter_layer = InterconnectedLayerModeling(setting)
    print(len(inter_layer.edges_on_A), len(inter_layer.edges_on_B))
    end = time.time()
    print(end - start)
    # print(inter_layer.two_layer_graph.nodes[1]['state'])
    # states = 0
    # for i in inter_layer.two_layer_graph.nodes:
    #     states += inter_layer.two_layer_graph.nodes[i]['state']
    # print(states)
    # print(sorted(inter_layer.two_layer_graph.nodes))
    # print(sorted(inter_layer.two_layer_graph.edges))


