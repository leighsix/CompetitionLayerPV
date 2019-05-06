import time
import random
import numpy as np
import Setting_Simulation_Value
import InterconnectedLayerModeling
import networkx as nx


class DecisionDynamics:
    def __init__(self):
        self.B_COUNT = 0

    def B_layer_dynamics(self, setting, inter_layer, v, node_i_names=None):  # B_layer 다이내믹스, 베타 적용 및 언어데스 알고리즘 적용
        volatility_count = 0
        if node_i_names == None : 
            node_i_names = set()
        for node_i in range(setting.A_node, setting.A_node+setting.B_node):
            if inter_layer.two_layer_graph.nodes[node_i]['name'] not in node_i_names:
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
                        prob_v = (opposite_orientation / len(neighbors)) ** (1 / v) * (len(neighbors) / opposite_orientation)
                z = random.random()
                if z < prob_v:
                    inter_layer.two_layer_graph.nodes[node_i]['state'] = -(inter_layer.two_layer_graph.nodes[node_i]['state'])
                    volatility_count +=1
                    self.B_COUNT += 1
        volatility_prob = volatility_count / setting.B_node
        return inter_layer, volatility_prob

    def B_layer_simultaneous_dynamics(self, setting, inter_layer, v, node_i_names=None):
        volatility_count = 0
        if node_i_names == None : 
            node_i_names = set()
        prob_v_array = self.B_state_change_probability_cal(setting, inter_layer, v)[0]
        temp_inter_layer = inter_layer
        z = np.random.random(setting.B_node)
        prob = (prob_v_array > z)
        for node_i in range(setting.A_node, setting.A_node+setting.B_node):
            if inter_layer.two_layer_graph.nodes[node_i]['name'] not in node_i_names:
                if prob[node_i-setting.A_node] == 1:
                    inter_layer.two_layer_graph.nodes[node_i]['state'] = -(temp_inter_layer.two_layer_graph.nodes[node_i]['state'])
                    volatility_count +=1
                    self.B_COUNT += 1
        volatility_prob = volatility_count / setting.B_node
        return inter_layer, volatility_prob

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
        prob_v_mean = np.sum(prob_v_array) / len(prob_v_array)
        return prob_v_array, prob_v_mean


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
    for i in range(10):
        decision_result = decision.B_layer_dynamics(setting, inter_layer, 0.5, node_i_names={'B_1', 'B_2'})
        inter_layer = decision_result[0]
        print(decision_result[1])
        print(inter_layer.two_layer_graph.nodes[setting.A_node+1]['state'], inter_layer.two_layer_graph.nodes[setting.A_node+2]['state'])
    state = 0
    for i in range(setting.A_node, setting.A_node+setting.B_node):
        state += inter_layer.two_layer_graph.nodes[i]['state']
    print(state)
    end = time.time()
    print(end-start)



