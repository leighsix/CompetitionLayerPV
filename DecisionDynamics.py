import time
import random
import numpy as np
import SettingSimulationValue
import InterconnectedLayerModeling
import networkx as nx


class DecisionDynamics:
    def __init__(self, setting, inter_layer, v, order=0, unchanged_nodes=None):
        self.B_COUNT = 0
        B_dynamics_result = self.B_layer_dynamics(setting, inter_layer, v, order, unchanged_nodes)
        self.B_inter_layer = B_dynamics_result[0]
        self.volatility_prob = B_dynamics_result[1]

    def B_layer_dynamics(self, setting, inter_layer, v, order, unchanged_nodes):
        B_dynamics_result = []
        if order == 0:
            B_dynamics_result = self.B_layer_sequential_dynamics(setting, inter_layer, v, unchanged_nodes)
        elif order == 1:
            B_dynamics_result = self.B_layer_simultaneous_dynamics(setting, inter_layer, v, unchanged_nodes)
        return B_dynamics_result

    def B_layer_sequential_dynamics(self, setting, inter_layer, v, unchanged_nodes):  # B_layer 다이내믹스, 베타 적용 및 언어데스 알고리즘 적용
        volatility_count = 0
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
                        prob_v = 0
                    else:
                        prob_v = (opposite_orientation / len(neighbors)) ** (1 / v) * (len(neighbors) / opposite_orientation)
                z = random.random()
                if z < prob_v:
                    inter_layer.two_layer_graph.nodes[node_i]['state'] = -(inter_layer.two_layer_graph.nodes[node_i]['state'])
                    volatility_count += 1
                    self.B_COUNT += 1
        volatility_prob = volatility_count / setting.B_node
        return inter_layer, volatility_prob

    def B_layer_simultaneous_dynamics(self, setting, inter_layer, v, unchanged_nodes):
        volatility_count = 0
        if unchanged_nodes is None:
            unchanged_nodes = set()
        volatility_result = DecisionDynamics.B_state_change_probability_cal(inter_layer, v)
        prob_v_array = volatility_result[0]
        z = np.random.random(setting.B_node)
        prob = (prob_v_array > z)
        for node_i in inter_layer.B_nodes:
            if node_i not in unchanged_nodes:
                node_j = node_i - setting.A_node
                if prob[node_j] == 1:
                    inter_layer.two_layer_graph.nodes[node_i]['state'] = -(inter_layer.two_layer_graph.nodes[node_i]['state'])
                    volatility_count += 1
                    self.B_COUNT += 1
        volatility_prob = volatility_count / setting.B_node
        return inter_layer, volatility_prob

    @staticmethod
    def B_state_change_probability_cal(inter_layer, v):
        prob_v_list = []
        for node_i in inter_layer.B_nodes:
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
                    prob_v = 0
                else:
                    prob_v = ((opposite_orientation / len(neighbors)) ** (1 / v)) * (len(neighbors) / opposite_orientation)
            prob_v_list.append(prob_v)
        prob_v_array = np.array(prob_v_list)
        prob_v_mean = np.sum(prob_v_array) / len(prob_v_array)
        return prob_v_array, prob_v_mean


if __name__ == "__main__":
    print("DecisionDynamics")
    setting = SettingSimulationValue.SettingSimulationValue()
    inter_layer = InterconnectedLayerModeling.InterconnectedLayerModeling(setting)
    state = 0
    for i in inter_layer.B_nodes:
        state += inter_layer.two_layer_graph.nodes[i]['state']
    print(state)
    start = time.time()
    for i in range(100):
        decision_result = DecisionDynamics(setting, inter_layer, 0.5, order=0, unchanged_nodes={2048, 2049})
        print(decision_result)
        print(inter_layer.two_layer_graph.nodes[2048]['state'], inter_layer.two_layer_graph.nodes[2049]['state'])
        print(decision_result.B_COUNT, decision_result.volatility_prob)
    state = 0
    for i in inter_layer.B_nodes:
        state += inter_layer.two_layer_graph.nodes[i]['state']
    print(state)
    end = time.time()
    print(end-start)



