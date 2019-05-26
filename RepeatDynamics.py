import numpy as np
import pandas as pd
import NodeProperty
import SettingSimulationValue
import InterconnectedDynamics
import InterconnectedLayerModeling
import time
import random


class RepeatDynamics:
    def __init__(self, setting, p, v, using_prob=False, select_step=1, select_method=0, select_layer='A_layer',
                 node_number=0, unchanged_state=-1):
        self.repeated_result = RepeatDynamics.repeat_dynamics(setting, p, v, using_prob, select_step, select_method,
                                                              select_layer, node_number, unchanged_state)

    @staticmethod
    def repeat_dynamics(setting, p, v, using_prob=False, select_step=1, select_method=0, select_layer='A_layer',
                        node_number=0, unchanged_state=-1):
        num_data = np.zeros([setting.Limited_step + 1, 16])
        centrality_method = setting.select_method_list[select_method]
        for i in range(setting.Repeating_number):
            inter_layer = InterconnectedLayerModeling.InterconnectedLayerModeling(setting)
            key_nodes = RepeatDynamics.select_keynode(setting, inter_layer, centrality_method, select_layer, node_number, unchanged_state)
            dynamics_result = InterconnectedDynamics.InterconnectedDynamics(setting, key_nodes[2], p, v, using_prob,
                                                                            select_step, key_nodes[0], key_nodes[1])
            print("unchanged_nodelist: %s" % key_nodes[0])
            num_data = num_data + dynamics_result.dynamics_result_array
        Num_Data = num_data / setting.Repeating_number
        panda_db = RepeatDynamics.making_dataframe_per_step(setting, Num_Data)
        panda_db['using_prob'] = using_prob
        if using_prob is False:
            panda_db['Orders'] = setting.step_list1[select_step]
        elif using_prob is True:
            panda_db['Orders'] = setting.step_list2[select_step]
        panda_db['keynode_method'] = centrality_method
        if select_method == '0':
            panda_db['select_layer'] = select_layer
            panda_db['keynode_number'] = 0
            panda_db['unchanged_state'] = 0
        elif select_method != '0':
            panda_db['select_layer'] = select_layer
            panda_db['keynode_number'] = node_number
            panda_db['unchanged_state'] = unchanged_state
        return panda_db

    @staticmethod
    def select_keynode(setting, inter_layer, select_method, select_layer, node_number, unchanged_state):
        if select_method == '0':
            unchanged_nodes = None
            sum_properties = 0
        elif select_method == 'random':
            node_list = []
            if select_layer == 'A_layer':
                node_list = setting.A_nodes
            elif select_layer == 'B_layer':
                node_list = setting.B_nodes
            elif select_layer == 'mixed':
                node_list = sorted(inter_layer.two_layer_graph.nodes)
            select_nodes_list = random.sample(node_list, k=node_number)
            unchanged_nodes = set(select_nodes_list)
            sum_properties = 0
        else:
            select_nodes_list = []
            nodes_properties = []
            select_layer_number = 0
            if select_layer == 'A_layer':
                select_layer_number = 0
            elif select_layer == 'B_layer':
                select_layer_number = 1
            elif select_layer == 'mixed':
                select_layer_number = 2
            nodes_calculation = NodeProperty.NodeProperty(setting, inter_layer, select_method, select_layer_number=select_layer_number)
            ordering = nodes_calculation.nodes_order[0:node_number]
            for i, j in ordering:
                inter_layer.two_layer_graph.nodes[i]['state'] = unchanged_state
                select_nodes_list.append(i)
                nodes_properties.append(j)
            unchanged_nodes = set(select_nodes_list)
            sum_properties = sum(nodes_properties)
        return unchanged_nodes, sum_properties, inter_layer

    @staticmethod
    def making_dataframe_per_step(setting, value_array):
        columns = ['p', 'v', 'prob_v', 'persuasion', 'compromise',
                   'A_plus', 'A_minus', 'B_plus', 'B_minus',
                   'Layer_A_Mean', 'Layer_B_Mean', 'AS',
                   'A_total_edges', 'B_total_edges', 'change_count', 'key_nodes_property']
        df = pd.DataFrame(value_array, columns=columns)
        step = [i for i in range(0, setting.Limited_step+1)]
        df['Model'] = setting.Model
        df['Steps'] = step
        df['Structure'] = setting.Structure
        df['A_node_number'] = setting.A_node
        df['B_node_number'] = setting.B_node
        return df



if __name__ == "__main__":
    print("RepeatDynamics")
    start = time.time()
    setting = SettingSimulationValue.SettingSimulationValue()
    setting.Repeating_number = 10
    inter_layer = InterconnectedLayerModeling.InterconnectedLayerModeling(setting)
    p = 0.2
    v = 0.5
    repeat = RepeatDynamics(setting, p, v, using_prob=False, select_step=14, select_method=1, select_layer='mixed',
                            node_number=8, unchanged_state=-1)
    print(repeat.repeated_result)
    end = time.time()
    print(end - start)
