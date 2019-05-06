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

    def AB_layer_dynamics(self, setting, inter_layer, p, v, node_i_names=None):
        persuasion_count = 0
        compromise_count = 0
        volatility_count = 0
        if node_i_names == None:
            node_i_names = set()
        for node_i in sorted(inter_layer.A_edges):
            if inter_layer.two_layer_graph.nodes[node_i]['name'] not in node_i_names:
                prob = self.three_probability_of_opinion_dynamics(inter_layer, p, node_i)
                z = random.random()
                if z < prob[1]:
                    inter_layer.two_layer_graph.nodes[node_i]['state'] = self.A_layer_persuasion_function2(setting, inter_layer, node_i)
                    persuasion_count += 1
                elif z > prob[1]+prob[0]:
                    inter_layer.two_layer_graph.nodes[node_i]['state'] = self.A_layer_compromise_function2(setting, inter_layer, node_i)
                    compromise_count += 1
                connected_B_node = self.finding_B_node(inter_layer, node_i)
                if inter_layer.two_layer_graph.nodes[connected_B_node]['name'] not in node_i_names:
                    B_node_neighbors = np.array(sorted(nx.neighbors(inter_layer.two_layer_graph, connected_B_node)))
                    B_node_neighbor_state = []
                    for B_node_neighbor in B_node_neighbors:
                        B_node_neighbor_state.append(inter_layer.two_layer_graph.nodes[B_node_neighbor]['state'])
                    B_node_neighbor_array = np.array(B_node_neighbor_state)
                    B_node_same_orientation = int(np.sum(B_node_neighbor_array * inter_layer.two_layer_graph.nodes[connected_B_node]['state'] > 0))
                    B_node_opposite_orientation = len(B_node_neighbors) - B_node_same_orientation
                    if B_node_opposite_orientation == 0:
                        prob_v = 0
                    else:
                        if v == 0:
                            prob_v = 0
                        else:
                            prob_v = (B_node_opposite_orientation / len(B_node_neighbors)) ** (1 / v) * (len(B_node_neighbors) / B_node_opposite_orientation)
                    z = random.random()
                    if z < prob_v:
                        inter_layer.two_layer_graph.nodes[connected_B_node]['state'] = -(inter_layer.two_layer_graph.nodes[connected_B_node]['state'])
                        volatility_count += 1
                        self.A_COUNT += 1
        volatility_prob = volatility_count / setting.A_node
        persuasion_prob = persuasion_count / setting.A_node
        compromise_prob = compromise_count / setting.A_node
        return inter_layer, persuasion_prob, compromise_prob, volatility_prob

    def A_layer_dynamics1(self, setting, inter_layer, p, node_i_names=None):  # original_step
        persuasion_count = 0
        compromise_count = 0
        if node_i_names == None : 
            node_i_names = set()
        total_edges = len(sorted(inter_layer.A_edges.edges())) + len(sorted(inter_layer.AB_edges))
        for i, j in sorted(inter_layer.A_edges.edges()):
            result = self.two_node_in_layer_A(setting, inter_layer, p, node_i_names, persuasion_count, compromise_count, i, j)
            inter_layer = result[0]
            persuasion_count = result[1]
            compromise_count = result[2]
        for i, j in sorted(inter_layer.AB_edges):
            result = self.two_node_in_layer_AB(setting, inter_layer, p, node_i_names, persuasion_count, compromise_count, i, j)
            inter_layer = result[0]
            persuasion_count = result[1]
            compromise_count = result[2]
        persuasion_prob = persuasion_count / total_edges
        compromise_prob = compromise_count / total_edges
        return inter_layer, persuasion_prob, compromise_prob

    def A_layer_dynamics2(self, setting, inter_layer, p, node_i_names=None):  # probability_step
        persuasion_count = 0
        compromise_count = 0
        if node_i_names == None : 
            node_i_names = set()
        for node_i in sorted(inter_layer.A_edges):
            if inter_layer.two_layer_graph.nodes[node_i]['name'] not in node_i_names:
                prob = self.three_probability_of_opinion_dynamics(inter_layer, p, node_i)
                z = random.random()
                if z < prob[1]:
                    inter_layer.two_layer_graph.nodes[node_i]['state'] = self.A_layer_persuasion_function2(setting, inter_layer, node_i)
                    persuasion_count += 1
                elif z > prob[1]+prob[0]:
                    inter_layer.two_layer_graph.nodes[node_i]['state'] = self.A_layer_compromise_function2(setting, inter_layer, node_i) 
                    compromise_count += 1
        persuasion_prob = persuasion_count / len(sorted(inter_layer.A_edges))
        compromise_prob = compromise_count / len(sorted(inter_layer.A_edges))
        return inter_layer, persuasion_prob, compromise_prob

    def A_layer_simultaneous_dynamics1(self, setting, inter_layer, p, node_i_names=None):  # original_same
        temp_inter_layer = inter_layer
        persuasion_count = 0
        compromise_count = 0
        if node_i_names == None : 
            node_i_names = set()
        total_edges = len(sorted(inter_layer.A_edges.edges())) + len(sorted(inter_layer.AB_edges))
        for i, j in sorted(temp_inter_layer.A_edges.edges()):
            result = self.two_node_in_layer_A(setting, inter_layer, p, node_i_names, persuasion_count, compromise_count, i, j)
            inter_layer = result[0]
            persuasion_count = result[1]
            compromise_count = result[2]
        for i, j in sorted(temp_inter_layer.AB_edges):
            result = self.two_node_in_layer_AB(setting, inter_layer, p, node_i_names, persuasion_count, compromise_count, i, j)
            inter_layer = result[0]
            persuasion_count = result[1]
            compromise_count = result[2]
        persuasion_prob = persuasion_count / total_edges
        compromise_prob = compromise_count / total_edges
        return inter_layer, persuasion_prob, compromise_prob

    def A_layer_simultaneous_dynamics2(self, setting, inter_layer, p, node_i_names=None):    # probability same
        temp_inter_layer = inter_layer
        persuasion_count = 0
        compromise_count = 0
        if node_i_names == None : 
            node_i_names = set()
        prob_array = self.A_state_change_probability_cal(temp_inter_layer, p)[0]
        z = np.random.random((setting.A_node, 1))
        prob = np.sum(prob_array < z, axis=1)
        for node_i in sorted(temp_inter_layer.A_edges):
            if inter_layer.two_layer_graph.nodes[node_i]['name'] not in node_i_names:  
                if prob[node_i] == 1:
                    inter_layer.two_layer_graph.nodes[node_i]['state'] = self.A_layer_persuasion_function2(setting, temp_inter_layer, node_i)
                    persuasion_count += 1
                elif prob[node_i] == 2:
                    inter_layer.two_layer_graph.nodes[node_i]['state'] = self.A_layer_compromise_function2(setting, temp_inter_layer, node_i)
                    compromise_count += 1
        persuasion_prob = persuasion_count / len(sorted(inter_layer.A_edges))
        compromise_prob = compromise_count / len(sorted(inter_layer.A_edges))              
        return inter_layer, persuasion_prob, compromise_prob

    def finding_B_node(self, inter_layer, node_i):
        connected_B_node = 0
        neighbors = sorted(nx.neighbors(inter_layer.two_layer_graph, node_i))
        for neighbor in neighbors:
            if neighbor > (len(sorted(inter_layer.A_edges))-1):
                connected_B_node = neighbor
        return connected_B_node

    def A_state_change_probability_cal(self, inter_layer, p):
        prob_list = []
        prob_per_list = []
        prob_com_list = []
        for node_i in sorted(inter_layer.A_edges):
            prob = self.three_probability_of_opinion_dynamics(inter_layer, p, node_i)
            prob_list.append((prob[0], prob[0]+prob[1], prob[0]+prob[1]+prob[2]))
            prob_per_list.append(prob[1])
            prob_com_list.append(prob[2])
        prob_array = np.array(prob_list)
        persuasion_prob = sum(prob_per_list) / len(prob_per_list)
        compromise_prob = sum(prob_com_list) / len(prob_com_list)
        return prob_array, persuasion_prob, compromise_prob

    def three_probability_of_opinion_dynamics(self, inter_layer, p, node_i):
        neighbors = np.array(sorted(nx.neighbors(inter_layer.two_layer_graph, node_i)))
        neighbor_state = []
        for neighbor in neighbors:
            neighbor_state.append(inter_layer.two_layer_graph.nodes[neighbor]['state'])
        neighbor_array = np.array(neighbor_state)
        same_orientation = int(np.sum(neighbor_array * inter_layer.two_layer_graph.nodes[node_i]['state'] > 0))
        opposite_orientation = len(neighbors) - same_orientation
        node_unchanging_prob = 0
        node_persuasion_prob = 0
        node_compromise_prob = 0
        for n in range(0, same_orientation + 1):
            for m in range(0, opposite_orientation + 1):
                n_combi = self.nCr(same_orientation, n)
                m_combi = self.nCr(opposite_orientation, m)
                if n == m:
                    node_unchanging_prob += p ** (n + opposite_orientation - m) * ((1 - p) ** (same_orientation - n + m)) * n_combi * m_combi
                elif n > m:
                    node_persuasion_prob += p ** (n + opposite_orientation - m) * ((1 - p) ** (same_orientation - n + m)) * n_combi * m_combi
                elif n < m:
                    node_compromise_prob += p ** (n + opposite_orientation - m) * ((1 - p) ** (same_orientation - n + m)) * n_combi * m_combi
        return node_unchanging_prob, node_persuasion_prob, node_compromise_prob

    def two_node_in_layer_A(self, setting, inter_layer, p, node_i_names, persuasion_count, compromise_count, i, j):
        temp_inter_layer = inter_layer
        a = temp_inter_layer.two_layer_graph.nodes[i]['state']
        b = temp_inter_layer.two_layer_graph.nodes[j]['state']
        if a * b > 0:
            z = random.random()
            if z < p:
                persuasion_func = self.A_layer_persuasion_function(setting, a, b)
                if temp_inter_layer.two_layer_graph.nodes[i]['name'] not in node_i_names:
                    inter_layer.two_layer_graph.nodes[i]['state'] = persuasion_func[0]
                    if temp_inter_layer.two_layer_graph.nodes[j]['name'] not in node_i_names:
                        inter_layer.two_layer_graph.nodes[j]['state'] = persuasion_func[1]
                    persuasion_count += 1
                elif (temp_inter_layer.two_layer_graph.nodes[i]['name'] in node_i_names) and (temp_inter_layer.two_layer_graph.nodes[j]['name'] not in node_i_names):
                    inter_layer.two_layer_graph.nodes[j]['state'] = persuasion_func[1]
                    persuasion_count += 1
        elif a * b < 0:
            z = random.random()
            if z < (1 - p):
                compromise_func = self.A_layer_compromise_function(setting, a, b, p, z)
                if temp_inter_layer.two_layer_graph.nodes[i]['name'] not in node_i_names:
                    inter_layer.two_layer_graph.nodes[i]['state'] = compromise_func[0]
                    if temp_inter_layer.two_layer_graph.nodes[j]['name'] not in node_i_names:
                        inter_layer.two_layer_graph.nodes[j]['state'] = compromise_func[1]
                    compromise_count += 1
                elif (temp_inter_layer.two_layer_graph.nodes[i]['name'] in node_i_names) and (temp_inter_layer.two_layer_graph.nodes[j]['name'] not in node_i_names):
                    inter_layer.two_layer_graph.nodes[j]['state'] = compromise_func[1]
                    compromise_count += 1
        return inter_layer, persuasion_count, compromise_count
    
    def two_node_in_layer_AB(self, setting, inter_layer, p, node_i_names, persuasion_count, compromise_count, i, j):
        temp_inter_layer = inter_layer
        a = temp_inter_layer.two_layer_graph.nodes[j]['state']
        b = temp_inter_layer.two_layer_graph.nodes[i]['state']
        if a * b > 0:
            z = random.random()
            if z < p:
                if temp_inter_layer.two_layer_graph.nodes[j]['name'] not in node_i_names :
                    inter_layer.two_layer_graph.nodes[j]['state'] = self.AB_layer_persuasion_function(setting, a)
                    persuasion_count += 1
        elif a * b < 0:
            z = random.random()
            if z < (1 - p):
                if temp_inter_layer.two_layer_graph.nodes[j]['name'] not in node_i_names :
                    inter_layer.two_layer_graph.nodes[j]['state'] = self.AB_layer_compromise_function(setting, a)
                    compromise_count += 1
        return inter_layer, persuasion_count, compromise_count
    
    def nCr(self, n, r):
        f = math.factorial
        return f(n) // f(r) // f(n - r)

    def A_layer_persuasion_function(self, setting, a, b):  # A layer 중에서 same orientation 에서 일어나는  변동 현상
        if a > 0 and b > 0:
            a = self.A_layer_node_right(a, setting.MAX)
            b = self.A_layer_node_right(b, setting.MAX)
        elif a < 0 and b < 0:
            a = self.A_layer_node_left(a, setting.MIN)
            b = self.A_layer_node_left(b, setting.MIN)
        return a, b

    def A_layer_compromise_function(self, setting, a, b, p, z):  # A layer  중에서 opposite orientation 에서 일어나는 변동 현상
        if a * b == -1:
            if z < ((1 - p) / 2):
                a = 1
                b = 1
            elif z > ((1 - p) / 2):
                a = -1
                b = -1
        elif a > 0:
            a = self.A_layer_node_left(a, setting.MIN)
            b = self.A_layer_node_right(b, setting.MAX)
        elif a < 0:
            a = self.A_layer_node_right(a, setting.MAX)
            b = self.A_layer_node_left(b, setting.MIN)
        return a, b

    def AB_layer_persuasion_function(self, setting, a):  # A-B layer 중에서 same orientation 에서 일어나는  변동 현상
        if a > 0:
            a = self.A_layer_node_right(a, setting.MAX)
        elif a < 0:
            a = self.A_layer_node_left(a, setting.MIN)
        return a
    def AB_layer_compromise_function(self, setting, a):  # A-B layer  중에서 opposite orientation 에서 일어나는 변동 현상
        if a > 0:
            a = self.A_layer_node_left(a, setting.MIN)
        elif a < 0:
            a = self.A_layer_node_right(a, setting.MAX)
        return a

    def A_layer_persuasion_function2(self, setting, inter_layer, node_i):
        a = inter_layer.two_layer_graph.nodes[node_i]['state']
        if a > 0:
            a = self.A_layer_node_right(a, setting.MAX)
        elif a < 0:
           a = self.A_layer_node_left(a, setting.MIN)
        return a
    def A_layer_compromise_function2(self, setting, inter_layer, node_i):
        a = inter_layer.two_layer_graph.nodes[node_i]['state']
        if a > 0:
           a = self.A_layer_node_left(a, setting.MIN)
        elif a < 0:
            a = self.A_layer_node_right(a, setting.MIN)
        return a

    def A_layer_node_left(self, a, Min):
        if a > Min:
            if a < 0 or a > 1:
                a = a - 1
                self.A_COUNT += 1
            elif a == 1:
                a = -1
                self.A_COUNT += 1
        elif a <= Min:
            a = Min
        return a

    def A_layer_node_right(self, a, Max):
        if a < Max:
            if a > 0 or a < -1:
                a = a + 1
                self.A_COUNT += 1
            elif a == -1:
                a = 1
                self.A_COUNT += 1
        elif a >= Max:
            a = Max
        return a


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
        result = opinion.A_layer_dynamics1(setting, inter_layer, 0.2, node_i_names={'A_0', 'A_1', 'A_2', 'A_3'})
        print(result)
        print(inter_layer.two_layer_graph.nodes[0]['state'],inter_layer.two_layer_graph.nodes[1]['state'], inter_layer.two_layer_graph.nodes[2]['state'], inter_layer.two_layer_graph.nodes[3]['state'] )
    state = 0
    for i in range(setting.A_node):
        state += inter_layer.two_layer_graph.nodes[i]['state']
    print(state)
    end = time.time()
    print(end - start)

