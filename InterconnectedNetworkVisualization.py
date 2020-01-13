from pymnet import *
import SettingSimulationValue
import InterconnectedLayerModeling
import os
import random
from mpl_toolkits.mplot3d.axes3d import *
import matplotlib
import networkx as nx
import numpy as np
import copy
import matplotlib.pyplot as plt
import matplotlib.image as mgimg
from matplotlib import animation
matplotlib.use("TkAgg")


class InterconnectedNetworkVisualization:
    def __init__(self):
        self.n = 0

    def making_competition_movie(self, setting, inter_layer, p, v, unchanged_nodes):
        InterconnectedNetworkVisualization.draw_interconnected_network(setting, inter_layer, self.n)
        print('Initial : %s' % self.n)
        self.n += 1
        for step_number in range(1, setting.Limited_step+1):
            state = []
            for i in inter_layer.A_nodes + inter_layer.B_nodes:
                state.append(inter_layer.two_layer_graph.nodes[i]['state'])
            state = np.array(state)
            if (np.all(state < 0) != 1) and (np.all(state > 0) != 1):
                opinion = self.A_layer_sequential_dynamics(setting, inter_layer, p, unchanged_nodes)
                self.B_layer_sequential_dynamics(setting, opinion, v, unchanged_nodes)
        for i in range(10):
            InterconnectedNetworkVisualization.draw_interconnected_network(setting, inter_layer, self.n)
            print('final_state : %s' % self.n)
            self.n += 1
        print('final: %s' % self.n)
        self.func_animation()
        plt.show()
        self.n = 0

    def func_animation(self):
        fig = plt.figure()
        ax = plt.gca()

        def init():
            imobj.set_data(np.zeros((100, 100)))
            return imobj,

        def animate(i):
            fname = "dynamics%s.png" % i
            img = mgimg.imread(fname)[-1::-1]
            imobj.set_data(img)
            return imobj,
        imobj = ax.imshow(np.zeros((100, 100)), origin='lower', alpha=1.0, zorder=1, aspect=1)
        anim = animation.FuncAnimation(fig, animate, init_func=init, repeat=False,
                                       frames=self.n, interval=100, blit=True, save_count=self.n)
        anim.save('dynamic_images.mp4', dpi=800)
        for i in range(self.n):
            os.remove('dynamics%s.png' % i)

    def A_layer_sequential_dynamics(self, setting, inter_layer, p, unchanged_nodes):  # original_step
        if unchanged_nodes is None:
            unchanged_nodes = set()
        for node_i in inter_layer.A_nodes:
            neighbor_list = inter_layer.unique_neighbor_dict[node_i]
            random.shuffle(neighbor_list)
            for neighbor in neighbor_list:
                if neighbor < setting.A_node:
                    internal_result = self.two_node_in_layer_A(setting, inter_layer, p, unchanged_nodes, node_i, neighbor)
                    inter_layer.two_layer_graph.nodes[node_i]['state'] = internal_result[0]
                    inter_layer.two_layer_graph.nodes[neighbor]['state'] = internal_result[1]
                elif neighbor >= setting.A_node:
                    external_result = self.two_node_in_layer_AB(setting, inter_layer, p, unchanged_nodes, node_i, neighbor)
                    inter_layer.two_layer_graph.nodes[node_i]['state'] = external_result
        return inter_layer

    def B_layer_sequential_dynamics(self, setting, inter_layer, v, unchanged_nodes):  # B_layer 다이내믹스, 베타 적용 및 언어데스 알고리즘 적용
        if unchanged_nodes is None:
            unchanged_nodes = set()
        for node_i in inter_layer.B_nodes:
            if node_i not in unchanged_nodes:
                neighbors = np.array(sorted(nx.neighbors(inter_layer.two_layer_graph, node_i)))
                neighbor_state = []
                for neighbor in neighbors:
                    neighbor_state.append(inter_layer.two_layer_graph.nodes[neighbor]['state'])
                neighbor_array = np.array(neighbor_state)
                same_orientation = int(np.sum(neighbor_array * (inter_layer.two_layer_graph.nodes[node_i]['state']) > 0))
                opposite_orientation = len(neighbors) - same_orientation
                if opposite_orientation == 0:
                    prob_v = 0
                else:
                    if v == 0:
                        prob_v = 1
                    else:
                        prob_v = ((opposite_orientation / len(neighbors)) ** (1 / v)) * (len(neighbors) / opposite_orientation)
                z = random.random()
                if z < prob_v:
                    inter_layer.two_layer_graph.nodes[node_i]['state'] = -(inter_layer.two_layer_graph.nodes[node_i]['state'])
                    InterconnectedNetworkVisualization.draw_interconnected_network(setting, inter_layer, self.n)
                    print('layer B : %s' % self.n)
                    self.n += 1
        return inter_layer

    def two_node_in_layer_AB(self, setting, inter_layer, p, unchanged_nodes, node_i, neighbor):
        a = inter_layer.two_layer_graph.nodes[node_i]['state']
        b = inter_layer.two_layer_graph.nodes[neighbor]['state']
        result_a = a
        if a * b > 0:
            z = random.random()
            if z < p:
                if node_i not in unchanged_nodes:
                    result_a = self.one_node_persuasion_function(setting, inter_layer, node_i)
        elif a * b < 0:
            z = random.random()
            if z < (1 - p):
                if node_i not in unchanged_nodes:
                    result_a = self.one_node_compromise_function(setting, inter_layer, node_i)
        return result_a

    def two_node_in_layer_A(self, setting, inter_layer, p, unchanged_nodes, node_i, neighbor):
        a = inter_layer.two_layer_graph.nodes[node_i]['state']
        b = inter_layer.two_layer_graph.nodes[neighbor]['state']
        result_a = a
        result_b = b
        if a * b > 0:
            z = random.random()
            if z < p:
                persuasion_func = self.two_node_persuasion_function(setting, inter_layer, a, b)
                if node_i not in unchanged_nodes:
                    result_a = persuasion_func[0]
                    if neighbor not in unchanged_nodes:
                        result_b = persuasion_func[1]
                elif (node_i in unchanged_nodes) and (neighbor not in unchanged_nodes):
                    result_b = persuasion_func[1]
        elif a * b < 0:
            z = random.random()
            if z < (1 - p):
                compromise_func = self.two_node_compromise_function(setting, inter_layer, a, b, p, z)
                if node_i not in unchanged_nodes:
                    result_a = compromise_func[0]
                    if neighbor not in unchanged_nodes:
                        result_b = compromise_func[1]
                elif (node_i in unchanged_nodes) and (neighbor not in unchanged_nodes):
                    result_b = compromise_func[1]
        return result_a, result_b

    def one_node_persuasion_function(self, setting, inter_layer, node_i):
        a = inter_layer.two_layer_graph.nodes[node_i]['state']
        if a > 0:
            a = self.A_layer_node_right(setting, inter_layer, a, setting.MAX)
        elif a < 0:
            a = self.A_layer_node_left(setting, inter_layer, a, setting.MIN)
        return a

    def one_node_compromise_function(self, setting, inter_layer, node_i):
        a = inter_layer.two_layer_graph.nodes[node_i]['state']
        if a > 0:
            a = self.A_layer_node_left(setting, inter_layer, a, setting.MIN)
        elif a < 0:
            a = self.A_layer_node_right(setting, inter_layer, a, setting.MIN)
        return a

    def two_node_persuasion_function(self, setting, inter_layer, a, b):  # A layer 중에서 same orientation 에서 일어나는  변동 현상
        if a > 0 and b > 0:
            a = self.A_layer_node_right(setting, inter_layer, a, setting.MAX)
            b = self.A_layer_node_right(setting, inter_layer, b, setting.MAX)
        elif a < 0 and b < 0:
            a = self.A_layer_node_left(setting, inter_layer, a, setting.MIN)
            b = self.A_layer_node_left(setting, inter_layer, b, setting.MIN)
        return a, b

    def two_node_compromise_function(self, setting, inter_layer, a, b, p, z):  # A layer  중에서 opposite orientation 에서 일어나는 변동 현상
        if a * b == -1:
            if z < ((1 - p) / 2):
                a = 1
                b = 1
                InterconnectedNetworkVisualization.draw_interconnected_network(setting, inter_layer, self.n)
                print('layer A : %s' % self.n)
                self.n += 1
            elif z > ((1 - p) / 2):
                a = -1
                b = -1
                InterconnectedNetworkVisualization.draw_interconnected_network(setting, inter_layer, self.n)
                print('layer A : %s' % self.n)
                self.n += 1
        elif a > 0:
            a = self.A_layer_node_left(setting, inter_layer, a, setting.MIN)
            b = self.A_layer_node_right(setting, inter_layer, b, setting.MAX)
        elif a < 0:
            a = self.A_layer_node_right(setting, inter_layer, a, setting.MAX)
            b = self.A_layer_node_left(setting, inter_layer, b, setting.MIN)
        return a, b

    def A_layer_node_left(self, setting, inter_layer, a, Min):
        if a > Min:
            if a < 0 or a > 1:
                a = a - 1
                InterconnectedNetworkVisualization.draw_interconnected_network(setting, inter_layer, self.n)
                print('layer A : %s' % self.n)
                self.n += 1
            elif a == 1:
                a = -1
                InterconnectedNetworkVisualization.draw_interconnected_network(setting, inter_layer, self.n)
                print('layer A : %s' % self.n)
                self.n += 1
        elif a <= Min:
            a = Min
        return a

    def A_layer_node_right(self, setting, inter_layer, a, Max):
        if a < Max:
            if a > 0 or a < -1:
                a = a + 1
                InterconnectedNetworkVisualization.draw_interconnected_network(setting, inter_layer, self.n)
                print('layer A : %s' % self.n)
                self.n += 1
            elif a == -1:
                a = 1
                InterconnectedNetworkVisualization.draw_interconnected_network(setting, inter_layer, self.n)
                print('layer A : %s' % self.n)
                self.n += 1
        elif a >= Max:
            a = Max
        return a

    @staticmethod
    def making_node_color(setting, inter_layer):
        node_color_dic = {}
        for i in inter_layer.A_nodes:
            node_color_dic[(i, 'layer A')] = setting.NodeColorDict[inter_layer.two_layer_graph.nodes[i]['state']]
        for i in inter_layer.B_nodes:
            node_color_dic[(i-len(inter_layer.A_nodes), 'layer B')] = setting.NodeColorDict[inter_layer.two_layer_graph.nodes[i]['state']]
        return node_color_dic

    @staticmethod
    def making_edge_color(setting, inter_layer):
        edge_color_dic = {}
        for i, j in inter_layer.edges_on_A:
            a = inter_layer.two_layer_graph.nodes[i]['state']
            b = inter_layer.two_layer_graph.nodes[j]['state']
            edge_color_dic[(i, 'layer A'), (j, 'layer A')] = setting.EdgeColorDict[a * b]
        for i, j in inter_layer.edges_on_B:
            a = inter_layer.two_layer_graph.nodes[i]['state']
            b = inter_layer.two_layer_graph.nodes[j]['state']
            edge_color_dic[(i-len(inter_layer.A_nodes), 'layer B'), (j-len(inter_layer.A_nodes), 'layer B')] = setting.EdgeColorDict[a * b]
        for i, j in inter_layer.edges_on_AB:
            a = inter_layer.two_layer_graph.nodes[j]['state']
            b = inter_layer.two_layer_graph.nodes[i]['state']
            edge_color_dic[(i, 'layer A'), (j-len(inter_layer.A_nodes), 'layer B')] = setting.EdgeColorDict[a * b]
        return edge_color_dic

    @staticmethod
    def making_layer_A_graph(inter_layer, interconnected_network):
        interconnected_network.add_layer('layer A')
        for i in inter_layer.A_nodes:
            interconnected_network.add_node(i)
        for i, j in inter_layer.edges_on_A:
            interconnected_network[i, j, 'layer A'] = 1
        return interconnected_network

    @staticmethod
    def making_layer_B_graph(inter_layer, interconnected_network):
        interconnected_network.add_layer('layer B')
        for i in inter_layer.B_nodes:
            interconnected_network.add_node(i-len(inter_layer.A_nodes))
        for i, j in inter_layer.edges_on_B:
            interconnected_network[i-len(inter_layer.A_nodes), j-len(inter_layer.A_nodes), 'layer B'] = 1
        return interconnected_network

    @staticmethod
    def making_interconnected_layer(inter_layer):
        interconnected_network = MultilayerNetwork(aspects=1)
        InterconnectedNetworkVisualization.making_layer_A_graph(inter_layer, interconnected_network)
        InterconnectedNetworkVisualization.making_layer_B_graph(inter_layer, interconnected_network)
        for i, j in sorted(inter_layer.edges_on_AB):
            interconnected_network[i, 'layer A'][j-len(inter_layer.A_nodes), 'layer B'] = 1
        return interconnected_network

    @staticmethod
    def draw_interconnected_network(setting, inter_layer, n):
        ax = plt.axes(projection='3d')
        draw(InterconnectedNetworkVisualization.making_interconnected_layer(inter_layer), layout='circular',
             layergap=1.3,
             layershape='rectangle',
             nodeCoords={},
             # nodeCoords=inter_layer.node_location,
             nodelayerCoords={},
             layerPadding=0.02, alignedNodes=True, ax=ax, layerColorDict={'layer A': 'pink', 'layer B': 'steelblue'},
             layerColorRule={},
             edgeColorDict=InterconnectedNetworkVisualization.making_edge_color(setting, inter_layer),
             edgeColorRule={},
             edgeWidthDict={}, edgeWidthRule={}, defaultEdgeWidth=0.3, edgeStyleDict={},
             edgeStyleRule={'rule': 'edgetype', 'inter': ':', 'intra': '-'}, defaultEdgeStyle='-',
             nodeLabelDict={}, nodeLabelRule={}, defaultNodeLabel=None,
             nodeColorDict=InterconnectedNetworkVisualization.making_node_color(setting, inter_layer),
             nodeColorRule={},
             defaultNodeColor=None,
             nodeLabelColorDict={}, nodeLabelColorRule={}, defaultNodeLabelColor='k',
             nodeSizeDict={}, nodeSizeRule={"propscale": 0.07, 'rule': 'degree'}, defaultNodeSize=None)
        plt.savefig('dynamics%s.png' % n, dpi=200)


if __name__ == "__main__":
    print("Interconnected Layer Modeling")
    setting = SettingSimulationValue.SettingSimulationValue()
    setting.A_node = 40
    setting.B_node = 40
    setting.Structure = 'RR-RR'
    setting.A_edge = 3
    setting.B_edge = 3
    inter_layer = InterconnectedLayerModeling.InterconnectedLayerModeling(setting)
    IV = InterconnectedNetworkVisualization()
    IV.making_competition_movie(setting, inter_layer, p=0.2, v=0.4, unchanged_nodes=None)
    print("Operating finished")
