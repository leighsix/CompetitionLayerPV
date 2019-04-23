import random
import math
import numpy as np
import networkx as nx
import Setting_Simulation_Value
import InterconnectedLayerModeling
import time


class OpinionDynamics:
    def __init__(self):
        self.A_COUNT = 0

    def A_layer_simultaneous_dynamics(self, setting, inter_layer, p):
        temp_inter_layer = inter_layer
        for i, j in sorted(temp_inter_layer.A_edges.edges()):
            a = temp_inter_layer.two_layer_graph.nodes[i]['state']
            b = temp_inter_layer.two_layer_graph.nodes[j]['state']
            if a * b > 0:
                persuasion = self.A_layer_persuasion_function(setting, temp_inter_layer.two_layer_graph.nodes[i],
                                                              temp_inter_layer.two_layer_graph.nodes[j], p)
                inter_layer.two_layer_graph.nodes[i]['state'] = persuasion[0]
                inter_layer.two_layer_graph.nodes[j]['state'] = persuasion[1]
            elif a * b < 0:
                compromise = self.A_layer_compromise_function(setting, temp_inter_layer.two_layer_graph.nodes[i],
                                                              temp_inter_layer.two_layer_graph.nodes[j], p)
                inter_layer.two_layer_graph.nodes[i]['state'] = compromise[0]
                inter_layer.two_layer_graph.nodes[j]['state'] = compromise[1]
        for i, j in sorted(temp_inter_layer.AB_edges):
            a = temp_inter_layer.two_layer_graph.nodes[j]['state']
            b = temp_inter_layer.two_layer_graph.nodes[i]['state']
            if a * b > 0:
                inter_layer.two_layer_graph.nodes[j]['state'] \
                    = self.AB_layer_persuasion_function(setting, temp_inter_layer.two_layer_graph.nodes[j], p)
            elif a * b < 0:
                inter_layer.two_layer_graph.nodes[j]['state'] \
                    = self.AB_layer_compromise_function(setting, temp_inter_layer.two_layer_graph.nodes[j], p)
        return inter_layer


    def A_layer_simultaneous_dynamics2(self, setting, inter_layer, p):
        temp_inter_layer = inter_layer
        probability = self.A_state_change_probability_cal(temp_inter_layer, p)
        z = np.random.random((setting.A_node, 1))
        prob = np.sum(probability < z, axis=1)
        for node_i in sorted(temp_inter_layer.A_edges):
            if prob[node_i] == 1:
                inter_layer.two_layer_graph.nodes[node_i]['state'] \
                    = self.A_layer_persuasion_function2(setting, temp_inter_layer, node_i)
            elif prob[node_i] == 2:
                inter_layer.two_layer_graph.nodes[node_i]['state'] \
                    = self.A_layer_compromise_function2(setting, temp_inter_layer, node_i)
        return inter_layer

    def A_state_change_probability_cal(self, inter_layer, p):
        prob_list = []
        for node_i in sorted(inter_layer.A_edges):
            neighbors = np.array(sorted(nx.neighbors(inter_layer.two_layer_graph, node_i)))
            neighbor_state = []
            for neighbor in neighbors:
                neighbor_state.append(inter_layer.two_layer_graph.nodes[neighbor]['state'])
            neighbor_array = np.array(neighbor_state)
            same_orientation = int(np.sum(neighbor_array * inter_layer.two_layer_graph.nodes[node_i]['state'] > 0))
            opposite_orientation = len(neighbors) - same_orientation
            unchanging_prob = 0
            persuasion_prob = 0
            compromise_prob = 0
            for n in range(0, same_orientation + 1):
                for m in range(0, opposite_orientation + 1):
                    n_combi = self.nCr(same_orientation, n)
                    m_combi = self.nCr(opposite_orientation, m)
                    if n == m:
                        unchanging_prob += p ** (n + opposite_orientation - m) * (
                                    (1 - p) ** (same_orientation - n + m)) * n_combi * m_combi
                    elif n > m:
                        persuasion_prob += p ** (n + opposite_orientation - m) * (
                                    (1 - p) ** (same_orientation - n + m)) * n_combi * m_combi
                    elif n < m:
                        compromise_prob += p ** (n + opposite_orientation - m) * (
                                    (1 - p) ** (same_orientation - n + m)) * n_combi * m_combi
            prob_list.append((unchanging_prob, unchanging_prob+persuasion_prob,
                              unchanging_prob+persuasion_prob+compromise_prob))
        prob_array = np.array(prob_list)
        return prob_array

    def nCr(self, n, r):
        f = math.factorial
        return f(n) // f(r) // f(n - r)

    def A_layer_dynamics2(self, setting, inter_layer, p):  # A_layer 다이내믹스, 감마 적용 및 설득/타협 알고리즘 적용
        for node_i in sorted(inter_layer.A_edges):
            neighbors = np.array(sorted(nx.neighbors(inter_layer.two_layer_graph, node_i)))
            neighbor_state = []
            for neighbor in neighbors:
                neighbor_state.append(inter_layer.two_layer_graph.nodes[neighbor]['state'])
            neighbor_array = np.array(neighbor_state)
            same_orientation = int(np.sum(neighbor_array * inter_layer.two_layer_graph.nodes[node_i]['state'] > 0))
            opposite_orientation = len(neighbors) - same_orientation
            unchanging_prob = 0
            persuasion_prob = 0
            compromise_prob = 0
            for n in range(0, same_orientation + 1):
                for m in range(0, opposite_orientation + 1):
                    n_combi = self.nCr(same_orientation, n)
                    m_combi = self.nCr(opposite_orientation, m)
                    if n == m:
                        unchanging_prob += p ** (n + opposite_orientation - m) * (
                                    (1 - p) ** (same_orientation - n + m)) * n_combi * m_combi
                    elif n > m:
                        persuasion_prob += p ** (n + opposite_orientation - m) * (
                                    (1 - p) ** (same_orientation - n + m)) * n_combi * m_combi
                    elif n < m:
                        compromise_prob += p ** (n + opposite_orientation - m) * (
                                    (1 - p) ** (same_orientation - n + m)) * n_combi * m_combi
            z = random.random()
            if z < persuasion_prob:
                inter_layer.two_layer_graph.nodes[node_i]['state'] \
                    = self.A_layer_persuasion_function2(setting, inter_layer, node_i)
            elif z > persuasion_prob+unchanging_prob:
                inter_layer.two_layer_graph.nodes[node_i]['state'] \
                    = self.A_layer_compromise_function2(setting, inter_layer, node_i)
        return inter_layer

    def A_layer_dynamics(self, setting, inter_layer, p):  # A_layer 다이내믹스, 감마 적용 및 설득/타협 알고리즘 적용
        for i, j in sorted(inter_layer.A_edges.edges()):
            a = inter_layer.two_layer_graph.nodes[i]['state']
            b = inter_layer.two_layer_graph.nodes[j]['state']
            if a * b > 0:
                persuasion = self.A_layer_persuasion_function(setting, inter_layer.two_layer_graph.nodes[i],
                                                              inter_layer.two_layer_graph.nodes[j], p)
                inter_layer.two_layer_graph.nodes[i]['state'] = persuasion[0]
                inter_layer.two_layer_graph.nodes[j]['state'] = persuasion[1]
            elif a * b < 0:
                compromise = self.A_layer_compromise_function(setting, inter_layer.two_layer_graph.nodes[i],
                                                              inter_layer.two_layer_graph.nodes[j], p)
                inter_layer.two_layer_graph.nodes[i]['state'] = compromise[0]
                inter_layer.two_layer_graph.nodes[j]['state'] = compromise[1]
        for i, j in sorted(inter_layer.AB_edges):
            a = inter_layer.two_layer_graph.nodes[j]['state']
            b = inter_layer.two_layer_graph.nodes[i]['state']

            if a * b > 0:
                inter_layer.two_layer_graph.nodes[j]['state'] \
                    = self.AB_layer_persuasion_function(setting, inter_layer.two_layer_graph.nodes[j], p)
            elif a * b < 0:
                inter_layer.two_layer_graph.nodes[j]['state'] \
                    = self.AB_layer_compromise_function(setting, inter_layer.two_layer_graph.nodes[j], p)
        return inter_layer

    def A_layer_persuasion_function(self, setting, a, b, p):  # A layer 중에서 same orientation 에서 일어나는  변동 현상
        z = random.random()
        if z < p:
            if (a['state']) > 0 and (b['state']) > 0:
                a['state'] = self.A_layer_node_right(a, setting.MAX)
                b['state'] = self.A_layer_node_right(b, setting.MAX)
            elif (a['state']) < 0 and (b['state']) < 0:
                a['state'] = self.A_layer_node_left(a, setting.MIN)
                b['state'] = self.A_layer_node_left(b, setting.MIN)
        return a['state'], b['state']

    def A_layer_compromise_function(self, setting, a, b, p):  # A layer  중에서 opposite orientation 에서 일어나는 변동 현상
        z = random.random()
        if z < (1 - p):
            if (a['state']) * (b['state']) == -1:
                if z < ((1 - p) / 2):
                    (a['state']) = 1
                    (b['state']) = 1
                elif z > ((1 - p) / 2):
                    a['state'] = -1
                    b['state'] = -1
            elif (a['state']) > 0:
                a['state'] = self.A_layer_node_left(a, setting.MIN)
                b['state'] = self.A_layer_node_right(b, setting.MAX)
            elif (a['state']) < 0:
                a['state'] = self.A_layer_node_right(a, setting.MAX)
                b['state'] = self.A_layer_node_left(b, setting.MIN)
        return a['state'], b['state']

    def AB_layer_persuasion_function(self, setting, a, p):  # A-B layer 중에서 same orientation 에서 일어나는  변동 현상
        z = random.random()
        if z < p:
            if (a['state']) > 0:
                a['state'] = self.A_layer_node_right(a, setting.MAX)
            elif (a['state']) < 0:
                a['state'] = self.A_layer_node_left(a, setting.MIN)
        return a['state']

    def AB_layer_compromise_function(self, setting, a, p):  # A-B layer  중에서 opposite orientation 에서 일어나는 변동 현상
        z = random.random()
        if z < (1 - p):
            if (a['state']) > 0:
                a['state'] = self.A_layer_node_left(a, setting.MIN)
            elif (a['state']) < 0:
                a['state'] = self.A_layer_node_right(a, setting.MAX)
        elif z > (1 - p):
            a['state'] = a['state']
        return a['state']

    def A_layer_node_left(self, a, Min):
        if (a['state']) > Min:
            if (a['state']) < 0 or (a['state']) > 1:
                (a['state']) = (a['state']) - 1
                self.A_COUNT += 1
            elif (a['state']) == 1:
                a['state'] = -1
                self.A_COUNT += 1
        elif (a['state']) <= Min:
            (a['state']) = Min
        return a['state']

    def A_layer_node_right(self, a, Max):
        if (a['state']) < Max:
            if (a['state']) > 0 or (a['state']) < -1:
                a['state'] = (a['state']) + 1
                self.A_COUNT += 1
            elif (a['state']) == -1:
                a['state'] = 1
                self.A_COUNT += 1
        elif (a['state']) >= Max:
            a['state'] = Max
        return a['state']


    def A_layer_persuasion_function2(self, setting, inter_layer, node_i):  # A layer 중에서 same orientation 에서 일어나는  변동 현상
        node = inter_layer.two_layer_graph.nodes[node_i]
        if node['state'] > 0:
            node['state'] = self.A_layer_node_right(node, setting.MAX)
        elif node['state'] < 0:
            node['state'] = self.A_layer_node_left(node, setting.MIN)
        return node['state']

    def A_layer_compromise_function2(self, setting, inter_layer, node_i):  # A layer  중에서 opposite orientation 에서 일어나는 변동 현상
        node = inter_layer.two_layer_graph.nodes[node_i]
        if node['state'] > 0:
            node['state'] = self.A_layer_node_left(node, setting.MIN)
        elif node['state'] < 0:
            node['state'] = self.A_layer_node_right(node, setting.MIN)
        return node['state']


if __name__ == "__main__":
    print("OpinionDynamics")
    setting = Setting_Simulation_Value.Setting_Simulation_Value()
    inter_layer = InterconnectedLayerModeling.InterconnectedLayerModeling(setting)
    state = 0
    for i in range(setting.A_node):
        state += inter_layer.two_layer_graph.nodes[i]['state']
    print(state)
    opinion = OpinionDynamics()
    start = time.time()
    for i in range(10):
        inter_layer = opinion.A_layer_simultaneous_dynamics2(setting, inter_layer, 0.1)
    state = 0
    for i in range(setting.A_node):
        state += inter_layer.two_layer_graph.nodes[i]['state']
    print(state)
    end = time.time()
    print(end - start)

