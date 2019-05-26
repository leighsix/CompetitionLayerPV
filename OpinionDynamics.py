import random
import copy
import math
import numpy as np
import networkx as nx
import SettingSimulationValue
import InterconnectedLayerModeling
import time


class OpinionDynamics:
    def __init__(self, setting, inter_layer, p, v, order=0, using_prob=False, unchanged_nodes=None):
        self.A_COUNT = 0
        A_dynamics_result = self.A_layer_dynamics(setting, inter_layer, p, v, order, using_prob, unchanged_nodes)
        self.A_inter_layer = A_dynamics_result[0]
        self.persuasion_prob = A_dynamics_result[1]
        self.compromise_prob = A_dynamics_result[2]
        self.volatility_prob = A_dynamics_result[3]

    def A_layer_dynamics(self, setting, inter_layer, p, v, order, using_prob, unchanged_nodes):
        A_dynamics_result = []
        if order == 0:
            A_dynamics_result = self.A_layer_sequential_dynamics(setting, inter_layer, p, using_prob, unchanged_nodes)
        elif order == 1:
            A_dynamics_result = self.A_layer_simultaneous_dynamics(setting, inter_layer, p, using_prob, unchanged_nodes)
        elif order == 2:
            A_dynamics_result = self.A_layer_random_dynamics(setting, inter_layer, p, unchanged_nodes)
        elif order == 3:
            A_dynamics_result = self.AB_layer_sequential_dynamics(setting, inter_layer, p, v, using_prob, unchanged_nodes)
        elif order == 4:
            A_dynamics_result = self.AB_layer_random_dynamics(setting, inter_layer, p, v, unchanged_nodes)
        return A_dynamics_result

    def A_layer_sequential_dynamics(self, setting, inter_layer, p, using_prob, unchanged_nodes):
        volatility_prob = 0
        sequential_dynamics = []
        if using_prob is False:
            sequential_dynamics = self.A_layer_sequential_dynamics1(setting, inter_layer, p, unchanged_nodes)
        elif using_prob is True:
            sequential_dynamics = self.A_layer_sequential_dynamics2(setting, inter_layer, p, unchanged_nodes)
        return sequential_dynamics[0], sequential_dynamics[1], sequential_dynamics[2], volatility_prob

    def A_layer_simultaneous_dynamics(self, setting, inter_layer, p, using_prob, unchanged_nodes):
        volatility_prob = 0
        simultaneous_dynamics = []
        if using_prob is False:
            simultaneous_dynamics = self.A_layer_simultaneous_dynamics1(setting, inter_layer, p, unchanged_nodes)
        elif using_prob is True:
            simultaneous_dynamics = self.A_layer_simultaneous_dynamics2(setting, inter_layer, p, unchanged_nodes)
        return simultaneous_dynamics[0], simultaneous_dynamics[1], simultaneous_dynamics[2], volatility_prob

    def A_layer_random_dynamics(self, setting, inter_layer, p, unchanged_nodes):  # original_step
        volatility_prob = 0
        persuasion_count = 0
        compromise_count = 0
        if unchanged_nodes is None:
            unchanged_nodes = set()
        edges_list = inter_layer.edges_on_A + inter_layer.edges_on_AB
        random.shuffle(edges_list)
        temp_inter_layer = copy.deepcopy(inter_layer)
        for edges in edges_list:
            if edges[1] < setting.A_node:
                internal_result = self.two_node_in_layer_A(setting, inter_layer, p, unchanged_nodes, edges[0], edges[1])
                inter_layer.two_layer_graph.nodes[edges[0]]['state'] = internal_result[0]
                inter_layer.two_layer_graph.nodes[edges[1]]['state'] = internal_result[1]
            elif edges[1] >= setting.A_node:
                external_result = self.two_node_in_layer_AB(setting, inter_layer, p, unchanged_nodes, edges[0], edges[1])
                inter_layer.two_layer_graph.nodes[edges[0]]['state'] = external_result
        for node_i in inter_layer.A_nodes:
            previous_state = temp_inter_layer.two_layer_graph.nodes[node_i]['state']
            present_state = inter_layer.two_layer_graph.nodes[node_i]['state']
            if previous_state * present_state > 0:
                if abs(previous_state) > abs(present_state):
                    compromise_count += 1
                elif abs(previous_state) < abs(present_state):
                    persuasion_count += 1
                elif abs(previous_state) == abs(present_state) == 2:
                    z = random.random()
                    if z < p:
                        persuasion_count += 1
            else:
                compromise_count += 1
        persuasion_prob = persuasion_count / setting.A_node
        compromise_prob = compromise_count / setting.A_node
        return inter_layer, persuasion_prob, compromise_prob, volatility_prob

    def AB_layer_sequential_dynamics(self, setting, inter_layer, p, v, using_prob, unchanged_nodes):
        sequential_dynamics = 0
        if using_prob is False:
            sequential_dynamics = self.AB_layer_sequential_dynamics1(setting, inter_layer, p, v, unchanged_nodes)
        elif using_prob is True:
            sequential_dynamics = self.AB_layer_sequential_dynamics2(setting, inter_layer, p, v, unchanged_nodes)
        return sequential_dynamics[0], sequential_dynamics[1], sequential_dynamics[2], sequential_dynamics[3]

    def AB_layer_random_dynamics(self, setting, inter_layer, p, v, unchanged_nodes):
        persuasion_count = 0
        compromise_count = 0
        volatility_count = 0
        if unchanged_nodes is None:
            unchanged_nodes = set()
        edges_list = inter_layer.edges_on_A + inter_layer.edges_on_AB + inter_layer.B_nodes
        random.shuffle(edges_list)
        temp_inter_layer = copy.deepcopy(inter_layer)
        for edges in edges_list:
            if type(edges) is tuple:
                if edges[1] < setting.A_node:
                    internal_result = self.two_node_in_layer_A(setting, inter_layer, p, unchanged_nodes, edges[0], edges[1])
                    inter_layer.two_layer_graph.nodes[edges[0]]['state'] = internal_result[0]
                    inter_layer.two_layer_graph.nodes[edges[1]]['state'] = internal_result[1]
                elif edges[1] >= setting.A_node:
                    external_result = self.two_node_in_layer_AB(setting, inter_layer, p, unchanged_nodes, edges[0], edges[1])
                    inter_layer.two_layer_graph.nodes[edges[0]]['state'] = external_result
            elif type(edges) is int:
                if edges not in unchanged_nodes:
                    edges_neighbors = np.array(sorted(nx.neighbors(inter_layer.two_layer_graph, edges)))
                    edges_neighbors_state = []
                    for edges_neighbor in edges_neighbors:
                        edges_neighbors_state.append(inter_layer.two_layer_graph.nodes[edges_neighbor]['state'])
                    edges_neighbors_array = np.array(edges_neighbors_state)
                    edges_same_orientation = int((np.sum(edges_neighbors_array * (inter_layer.two_layer_graph.nodes[edges]['state']) > 0)))
                    edges_opposite_orientation = len(edges_neighbors) - edges_same_orientation
                    if edges_opposite_orientation == 0:
                        prob_v = 0
                    else:
                        if v == 0:
                            prob_v = 0
                        else:
                            prob_v = ((edges_opposite_orientation / len(edges_neighbors)) ** (1 / v)) * (len(edges_neighbors) / edges_opposite_orientation)
                    z = random.random()
                    if z < prob_v:
                        inter_layer.two_layer_graph.nodes[edges]['state'] = -(inter_layer.two_layer_graph.nodes[edges]['state'])
                        self.A_COUNT += 1
                        volatility_count += 1
        for node_i in inter_layer.A_nodes:
            previous_state = temp_inter_layer.two_layer_graph.nodes[node_i]['state']
            present_state = inter_layer.two_layer_graph.nodes[node_i]['state']
            if previous_state * present_state > 0:
                if abs(previous_state) > abs(present_state):
                    compromise_count += 1
                elif abs(previous_state) < abs(present_state):
                    persuasion_count += 1
                elif abs(previous_state) == abs(present_state) == 2:
                    z = random.random()
                    if z < p:
                        persuasion_count += 1
            else:
                compromise_count += 1
        persuasion_prob = persuasion_count / setting.A_node
        compromise_prob = compromise_count / setting.A_node
        volatility_prob = volatility_count / setting.B_node
        return inter_layer, persuasion_prob, compromise_prob, volatility_prob

    def AB_layer_sequential_dynamics1(self, setting, inter_layer, p, v, unchanged_nodes):
        persuasion_count = 0
        compromise_count = 0
        volatility_count = 0
        temp_inter_layer = copy.deepcopy(inter_layer)
        if unchanged_nodes is None:
            unchanged_nodes = set()
        for node_i in inter_layer.A_nodes:
            connected_B_nodes_list = OpinionDynamics.finding_B_node(setting, inter_layer, node_i)
            if len(connected_B_nodes_list) == 1:
                connected_B_node = connected_B_nodes_list[0]
                connected_A_nodes_list = OpinionDynamics.finding_A_node(setting, inter_layer, connected_B_node)
                for connected_A_node in connected_A_nodes_list:
                    neighbor_list = inter_layer.unique_neighbor_dict[connected_A_node]
                    random.shuffle(neighbor_list)
                    for neighbor in neighbor_list:
                        if neighbor < setting.A_node:
                            internal_result = self.two_node_in_layer_A(setting, inter_layer, p, unchanged_nodes, connected_A_node, neighbor)
                            inter_layer.two_layer_graph.nodes[connected_A_node]['state'] = internal_result[0]
                            inter_layer.two_layer_graph.nodes[neighbor]['state'] = internal_result[1]
                        elif neighbor >= setting.A_node:
                            external_result = self.two_node_in_layer_AB(setting, inter_layer, p, unchanged_nodes, connected_A_node, neighbor)
                            inter_layer.two_layer_graph.nodes[connected_A_node]['state'] = external_result
                if connected_B_node not in unchanged_nodes:
                    B_node_neighbors = np.array(sorted(nx.neighbors(inter_layer.two_layer_graph, connected_B_node)))
                    B_node_neighbor_state = []
                    for B_node_neighbor in B_node_neighbors:
                        B_node_neighbor_state.append(inter_layer.two_layer_graph.nodes[B_node_neighbor]['state'])
                    B_node_neighbor_array = np.array(B_node_neighbor_state)
                    B_node_same_orientation = int(np.sum(B_node_neighbor_array * (inter_layer.two_layer_graph.nodes[connected_B_node]['state']) > 0))
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
                        self.A_COUNT += 1
                        volatility_count += 1
            elif len(connected_B_nodes_list) > 1:
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
                for connected_B_node in connected_B_nodes_list:
                    if connected_B_node not in unchanged_nodes:
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
                            self.A_COUNT += 1
                            volatility_count += 1
        for node_i in inter_layer.A_nodes:
            previous_state = temp_inter_layer.two_layer_graph.nodes[node_i]['state']
            present_state = inter_layer.two_layer_graph.nodes[node_i]['state']
            if previous_state * present_state > 0:
                if abs(previous_state) > abs(present_state):
                    compromise_count += 1
                elif abs(previous_state) < abs(present_state):
                    persuasion_count += 1
                elif abs(previous_state) == abs(present_state) == 2:
                    z = random.random()
                    if z < p:
                        persuasion_count += 1
            else:
                compromise_count += 1
        persuasion_prob = persuasion_count / setting.A_node
        compromise_prob = compromise_count / setting.A_node
        volatility_prob = volatility_count / setting.B_node
        return inter_layer, persuasion_prob, compromise_prob, volatility_prob

    def AB_layer_sequential_dynamics2(self, setting, inter_layer, p, v, unchanged_nodes):
        persuasion_count = 0
        compromise_count = 0
        volatility_count = 0
        if unchanged_nodes is None:
            unchanged_nodes = set()
        for node_i in inter_layer.A_nodes:
            connected_B_nodes_list = OpinionDynamics.finding_B_node(setting, inter_layer, node_i)
            if len(connected_B_nodes_list) == 1:
                connected_B_node = connected_B_nodes_list[0]
                connected_A_nodes_list = OpinionDynamics.finding_A_node(setting, inter_layer, connected_B_node)
                for connected_A_node in connected_A_nodes_list:
                    if connected_A_node not in unchanged_nodes:
                        prob = OpinionDynamics.three_probability_of_opinion_dynamics(inter_layer, p, connected_A_node)
                        z = random.random()
                        if z < prob[1]:
                            inter_layer.two_layer_graph.nodes[connected_A_node]['state'] = self.one_node_persuasion_function(setting, inter_layer, connected_A_node)
                            persuasion_count += 1
                        elif z > prob[1] + prob[0]:
                            inter_layer.two_layer_graph.nodes[connected_A_node]['state'] = self.one_node_compromise_function(setting, inter_layer, connected_A_node)
                            compromise_count += 1
                if connected_B_node not in unchanged_nodes:
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
                        self.A_COUNT += 1
                        volatility_count += 1
            elif len(connected_B_nodes_list) > 1:
                if node_i not in unchanged_nodes:
                    prob = OpinionDynamics.three_probability_of_opinion_dynamics(inter_layer, p, node_i)
                    z = random.random()
                    if z < prob[1]:
                        inter_layer.two_layer_graph.nodes[node_i]['state'] = self.one_node_persuasion_function(setting, inter_layer, node_i)
                        persuasion_count += 1
                    elif z > prob[1] + prob[0]:
                        inter_layer.two_layer_graph.nodes[node_i]['state'] = self.one_node_compromise_function(setting, inter_layer, node_i)
                        compromise_count += 1
                for connected_B_node in connected_B_nodes_list:
                    if connected_B_node not in unchanged_nodes:
                        B_node_neighbors = np.array(sorted(nx.neighbors(inter_layer.two_layer_graph, connected_B_node)))
                        B_node_neighbor_state = []
                        for B_node_neighbor in B_node_neighbors:
                            B_node_neighbor_state.append(inter_layer.two_layer_graph.nodes[B_node_neighbor]['state'])
                        B_node_neighbor_array = np.array(B_node_neighbor_state)
                        B_node_same_orientation = int(np.sum(
                            B_node_neighbor_array * inter_layer.two_layer_graph.nodes[connected_B_node]['state'] > 0))
                        B_node_opposite_orientation = len(B_node_neighbors) - B_node_same_orientation
                        if B_node_opposite_orientation == 0:
                            prob_v = 0
                        else:
                            if v == 0:
                                prob_v = 0
                            else:
                                prob_v = (B_node_opposite_orientation / len(B_node_neighbors)) ** (1 / v) * (
                                            len(B_node_neighbors) / B_node_opposite_orientation)
                        z = random.random()
                        if z < prob_v:
                            inter_layer.two_layer_graph.nodes[connected_B_node]['state'] = -(
                            inter_layer.two_layer_graph.nodes[connected_B_node]['state'])
                            self.A_COUNT += 1
                            volatility_count += 1
        persuasion_prob = persuasion_count / setting.A_node
        compromise_prob = compromise_count / setting.A_node
        volatility_prob = volatility_count / setting.B_node
        return inter_layer, persuasion_prob, compromise_prob, volatility_prob

    def A_layer_sequential_dynamics1(self, setting, inter_layer, p, unchanged_nodes):  # original_step
        persuasion_count = 0
        compromise_count = 0
        if unchanged_nodes is None:
            unchanged_nodes = set()
        temp_inter_layer = copy.deepcopy(inter_layer)
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
        for node_i in inter_layer.A_nodes:
            previous_state = temp_inter_layer.two_layer_graph.nodes[node_i]['state']
            present_state = inter_layer.two_layer_graph.nodes[node_i]['state']
            if previous_state * present_state > 0:
                if abs(previous_state) > abs(present_state):
                    compromise_count += 1
                elif abs(previous_state) < abs(present_state):
                    persuasion_count += 1
                elif abs(previous_state) == abs(present_state) == 2:
                    z = random.random()
                    if z < p:
                        persuasion_count += 1
            elif previous_state * present_state < 0:
                compromise_count += 1
        persuasion_prob = persuasion_count / setting.A_node
        compromise_prob = compromise_count / setting.A_node
        return inter_layer, persuasion_prob, compromise_prob

    def A_layer_sequential_dynamics2(self, setting, inter_layer, p, unchanged_nodes):  # probability_step
        persuasion_count = 0
        compromise_count = 0
        if unchanged_nodes is None:
            unchanged_nodes = set()
        for node_i in inter_layer.A_nodes:
            if node_i not in unchanged_nodes:
                prob = OpinionDynamics.three_probability_of_opinion_dynamics(inter_layer, p, node_i)
                z = random.random()
                if z < prob[1]:
                    inter_layer.two_layer_graph.nodes[node_i]['state'] = self.one_node_persuasion_function(setting, inter_layer, node_i)
                    persuasion_count += 1
                elif z > prob[1]+prob[0]:
                    inter_layer.two_layer_graph.nodes[node_i]['state'] = self.one_node_compromise_function(setting, inter_layer, node_i)
                    compromise_count += 1
        persuasion_prob = persuasion_count / setting.A_node
        compromise_prob = compromise_count / setting.A_node
        return inter_layer, persuasion_prob, compromise_prob

    def A_layer_simultaneous_dynamics1(self, setting, inter_layer, p, unchanged_nodes):  # original_same
        temp_inter_layer = copy.deepcopy(inter_layer)
        persuasion_count = 0
        compromise_count = 0
        if unchanged_nodes is None:
            unchanged_nodes = set()
        for node_i in inter_layer.A_nodes:
            neighbor_list = sorted(nx.neighbors(inter_layer.two_layer_graph, node_i))
            random.shuffle(neighbor_list)
            for neighbor in neighbor_list:
                if neighbor < setting.A_node:
                    internal_result = self.one_node_in_layer_A(setting, temp_inter_layer, p, unchanged_nodes, node_i, neighbor)
                    inter_layer.two_layer_graph.nodes[node_i]['state'] = internal_result
                elif neighbor >= setting.A_node:
                    external_result = self.two_node_in_layer_AB(setting, temp_inter_layer, p, unchanged_nodes, node_i, neighbor)
                    inter_layer.two_layer_graph.nodes[node_i]['state'] = external_result
        for node_i in inter_layer.A_nodes:
            previous_state = temp_inter_layer.two_layer_graph.nodes[node_i]['state']
            present_state = inter_layer.two_layer_graph.nodes[node_i]['state']
            if previous_state * present_state > 0:
                if abs(previous_state) > abs(present_state):
                    compromise_count += 1
                elif abs(previous_state) < abs(present_state):
                    persuasion_count += 1
                elif abs(previous_state) == abs(present_state) == 2:
                    z = random.random()
                    if z < p:
                        persuasion_count += 1
            elif previous_state * present_state < 0:
                compromise_count += 1
        persuasion_prob = persuasion_count / setting.A_node
        compromise_prob = compromise_count / setting.A_node
        return inter_layer, persuasion_prob, compromise_prob

    def A_layer_simultaneous_dynamics2(self, setting, inter_layer, p, unchanged_nodes):    # probability same
        temp_inter_layer = copy.deepcopy(inter_layer)
        persuasion_count = 0
        compromise_count = 0
        if unchanged_nodes is None:
            unchanged_nodes = set()
        probability_result = OpinionDynamics.A_state_change_probability_cal(inter_layer, p)
        prob_array = probability_result[0]
        z = np.random.random((setting.A_node, 1))
        prob = np.sum(prob_array < z, axis=1)
        for node_i in inter_layer.A_nodes:
            if node_i not in unchanged_nodes:
                if prob[node_i] == 1:
                    persuasion_result = self.one_node_persuasion_function(setting, temp_inter_layer, node_i)
                    inter_layer.two_layer_graph.nodes[node_i]['state'] = persuasion_result
                    persuasion_count += 1
                elif prob[node_i] == 2:
                    compromise_result = self.one_node_compromise_function(setting, temp_inter_layer, node_i)
                    inter_layer.two_layer_graph.nodes[node_i]['state'] = compromise_result
                    compromise_count += 1
        persuasion_prob = persuasion_count / setting.A_node
        compromise_prob = compromise_count / setting.A_node
        return inter_layer, persuasion_prob, compromise_prob

    @staticmethod
    def A_state_change_probability_cal(inter_layer, p):
        prob_list = []
        prob_per_list = []
        prob_com_list = []
        for node_i in inter_layer.A_nodes:
            prob = OpinionDynamics.three_probability_of_opinion_dynamics(inter_layer, p, node_i)
            prob_list.append((prob[0], prob[0]+prob[1], prob[0]+prob[1]+prob[2]))
            prob_per_list.append(prob[1])
            prob_com_list.append(prob[2])
        prob_array = np.array(prob_list)
        persuasion_prob = sum(prob_per_list) / len(prob_per_list)
        compromise_prob = sum(prob_com_list) / len(prob_com_list)
        return prob_array, persuasion_prob, compromise_prob

    @staticmethod
    def three_probability_of_opinion_dynamics(inter_layer, p, node_i):
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
                n_combi = OpinionDynamics.nCr(same_orientation, n)
                m_combi = OpinionDynamics.nCr(opposite_orientation, m)
                if n == m:
                    node_unchanging_prob += p ** (n + opposite_orientation - m) * ((1 - p) ** (same_orientation - n + m)) * n_combi * m_combi
                elif n > m:
                    node_persuasion_prob += p ** (n + opposite_orientation - m) * ((1 - p) ** (same_orientation - n + m)) * n_combi * m_combi
                elif n < m:
                    node_compromise_prob += p ** (n + opposite_orientation - m) * ((1 - p) ** (same_orientation - n + m)) * n_combi * m_combi
        return node_unchanging_prob, node_persuasion_prob, node_compromise_prob

    @staticmethod
    def finding_B_node(setting, inter_layer, node_i):
        connected_B_nodes_list = []
        neighbors = sorted(nx.neighbors(inter_layer.two_layer_graph, node_i))
        for neighbor in neighbors:
            if neighbor >= setting.A_node:
                connected_B_nodes_list.append(neighbor)
        return connected_B_nodes_list

    @staticmethod
    def finding_A_node(setting, inter_layer, node_i):
        connected_A_nodes_list = []
        neighbors = sorted(nx.neighbors(inter_layer.two_layer_graph, node_i))
        for neighbor in neighbors:
            if neighbor < setting.A_node:
                connected_A_nodes_list.append(neighbor)
        return connected_A_nodes_list

    @staticmethod
    def nCr(n, r):
        f = math.factorial
        return f(n) // f(r) // f(n - r)

    def one_node_in_layer_A(self, setting, temp_inter_layer, p, unchanged_nodes, node_i, neighbor):
        a = temp_inter_layer.two_layer_graph.nodes[node_i]['state']
        b = temp_inter_layer.two_layer_graph.nodes[neighbor]['state']
        result_a = a
        if a * b > 0:
            z = random.random()
            if z < p:
                persuasion_func = self.two_node_persuasion_function(setting, a, b)
                if node_i not in unchanged_nodes:
                    result_a = persuasion_func[0]
        elif a * b < 0:
            z = random.random()
            if z < (1 - p):
                compromise_func = self.two_node_compromise_function(setting, a, b, p, z)
                if node_i not in unchanged_nodes:
                    result_a = compromise_func[0]
        return result_a

    def two_node_in_layer_A(self, setting, temp_inter_layer, p, unchanged_nodes, node_i, neighbor):
        a = temp_inter_layer.two_layer_graph.nodes[node_i]['state']
        b = temp_inter_layer.two_layer_graph.nodes[neighbor]['state']
        result_a = a
        result_b = b
        if a * b > 0:
            z = random.random()
            if z < p:
                persuasion_func = self.two_node_persuasion_function(setting, a, b)
                if node_i not in unchanged_nodes:
                    result_a = persuasion_func[0]
                    if neighbor not in unchanged_nodes:
                        result_b = persuasion_func[1]
                elif (node_i in unchanged_nodes) and (neighbor not in unchanged_nodes):
                    result_b = persuasion_func[1]
        elif a * b < 0:
            z = random.random()
            if z < (1 - p):
                compromise_func = self.two_node_compromise_function(setting, a, b, p, z)
                if node_i not in unchanged_nodes:
                    result_a = compromise_func[0]
                    if neighbor not in unchanged_nodes:
                        result_b = compromise_func[1]
                elif (node_i in unchanged_nodes) and (neighbor not in unchanged_nodes):
                    result_b = compromise_func[1]
        return result_a, result_b

    def two_node_in_layer_AB(self, setting, temp_inter_layer, p, unchanged_nodes, node_i, neighbor):
        a = temp_inter_layer.two_layer_graph.nodes[node_i]['state']
        b = temp_inter_layer.two_layer_graph.nodes[neighbor]['state']
        result_a = a
        if a * b > 0:
            z = random.random()
            if z < p:
                if node_i not in unchanged_nodes:
                    result_a = self.one_node_persuasion_function(setting, temp_inter_layer, node_i)
        elif a * b < 0:
            z = random.random()
            if z < (1 - p):
                if node_i not in unchanged_nodes:
                    result_a = self.one_node_compromise_function(setting, temp_inter_layer, node_i)
        return result_a
    
    def two_node_persuasion_function(self, setting, a, b):  # A layer 중에서 same orientation 에서 일어나는  변동 현상
        if a > 0 and b > 0:
            a = self.A_layer_node_right(a, setting.MAX)
            b = self.A_layer_node_right(b, setting.MAX)
        elif a < 0 and b < 0:
            a = self.A_layer_node_left(a, setting.MIN)
            b = self.A_layer_node_left(b, setting.MIN)
        return a, b

    def two_node_compromise_function(self, setting, a, b, p, z):  # A layer  중에서 opposite orientation 에서 일어나는 변동 현상
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

    def one_node_persuasion_function(self, setting, inter_layer, node_i):
        a = inter_layer.two_layer_graph.nodes[node_i]['state']
        if a > 0:
            a = self.A_layer_node_right(a, setting.MAX)
        elif a < 0:
            a = self.A_layer_node_left(a, setting.MIN)
        return a

    def one_node_compromise_function(self, setting, inter_layer, node_i):
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
    setting = SettingSimulationValue.SettingSimulationValue()
    inter_layer = InterconnectedLayerModeling.InterconnectedLayerModeling(setting)
    state = 0
    for i in inter_layer.A_nodes:
        state += inter_layer.two_layer_graph.nodes[i]['state']
    print(state)
    start = time.time()
    for i in range(10):
        opinion_result = OpinionDynamics(setting, inter_layer, 0.2, 0.5, order=0, using_prob=False,
                                         unchanged_nodes={1, 3, 4})

        print(opinion_result)
        print(inter_layer.two_layer_graph.nodes[1]['state'], inter_layer.two_layer_graph.nodes[3]['state'],
              inter_layer.two_layer_graph.nodes[4]['state'])
        print(opinion_result.A_COUNT, opinion_result.persuasion_prob, opinion_result.compromise_prob, opinion_result.volatility_prob)
    state = 0
    for i in inter_layer.A_nodes:
        state += inter_layer.two_layer_graph.nodes[i]['state']
    print(state)
    end = time.time()
    print(end - start)

