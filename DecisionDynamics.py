import time
import random
import numpy as np
import Setting_Simulation_Value
import InterconnectedLayerModeling
import networkx as nx


class DecisionDynamics:
    def __init__(self):
        self.B_COUNT = 0

    def B_layer_dynamics(self, setting, inter_layer, v):  # B_layer 다이내믹스, 베타 적용 및 언어데스 알고리즘 적용
        prob_v_list = []
        for node_i in range(setting.A_node, setting.A_node+setting.B_node):
            neighbors = np.array(sorted(nx.neighbors(inter_layer.two_layer_graph, node_i)))
            neighbor_state = []
            for neighbor in neighbors:
                neighbor_state.append(inter_layer.two_layer_graph.nodes[neighbor]['state'])
            neighbor_array = np.array(neighbor_state)
            same_orientation = int(np.sum(neighbor_array * inter_layer.two_layer_graph.nodes[node_i]['state'] > 0))
            opposite_orientation = len(neighbors) - same_orientation
            if opposite_orientation == 0:
                prob_v = 0
            else:
                if v == 0:
                    prob_v = 0
                else:
                    prob_v = (opposite_orientation / len(neighbors)) ** (1 / v) * \
                                (len(neighbors) / opposite_orientation)
            z = random.random()
            if z < prob_v:
                inter_layer.two_layer_graph.nodes[node_i]['state'] = \
                    -(inter_layer.two_layer_graph.nodes[node_i]['state'])
                self.B_COUNT += 1
            prob_v_list.append(prob_v)
        prob_v_array = np.array(prob_v_list)
        return inter_layer, prob_v_array

    def B_layer_simultaneous_dynamics(self, setting, inter_layer, probability):
        temp_inter_layer = inter_layer
        z = np.random.random(setting.B_node)
        prob = (probability > z)
        for node_i in range(setting.A_node, setting.A_node+setting.B_node):
            if prob[node_i-setting.A_node] == 1:
                inter_layer.two_layer_graph.nodes[node_i]['state'] = \
                    -(temp_inter_layer.two_layer_graph.nodes[node_i]['state'])
                self.B_COUNT += 1
        return inter_layer


    def B_state_change_probability_cal(self, setting, inter_layer, v):
        prob_v_list = []
        for node_i in range(setting.A_node, setting.A_node+setting.B_node):
            neighbors = np.array(sorted(nx.neighbors(inter_layer.two_layer_graph, node_i)))
            neighbor_state = []
            for neighbor in neighbors:
                neighbor_state.append(inter_layer.two_layer_graph.nodes[neighbor]['state'])
            neighbor_array = np.array(neighbor_state)
            same_orientation = int(np.sum(neighbor_array * inter_layer.two_layer_graph.nodes[node_i]['state'] > 0))
            opposite_orientation = len(neighbors) - same_orientation
            if opposite_orientation == 0:
                prob_v = 0
            else:
                if v == 0:
                    prob_v = 0
                else:
                    prob_v = (opposite_orientation / len(neighbors)) ** (1 / v) * \
                                (len(neighbors) / opposite_orientation)
            prob_v_list.append(prob_v)
        prob_v_array = np.array(prob_v_list)
        return prob_v_array


if __name__ == "__main__" :
    print("DecisionDynamics")
    setting = Setting_Simulation_Value.Setting_Simulation_Value()
    inter_layer = InterconnectedLayerModeling.InterconnectedLayerModeling(setting)
    state = 0
    for i in range(setting.A_node, setting.A_node+setting.B_node):
        state += inter_layer.two_layer_graph.nodes[i]['state']
    print(state)
    start = time.time()
    decision = DecisionDynamics()
    decision_result = decision.B_layer_dynamics(setting, inter_layer, 0.2)
    inter_layer = decision_result[0]
    prob_v = decision_result[1]
    print(prob_v)
    state = 0
    for i in range(setting.A_node, setting.A_node+setting.B_node):
        state += inter_layer.two_layer_graph.nodes[i]['state']
    print(state)
    end = time.time()
    print(end-start)



