import numpy as np
import pandas as pd
import NodeProperty
import EdgeProperty
import SettingSimulationValue
import InterconnectedDynamics
import InterconnectedLayerModeling
import time
import random

step_list1 = [r'$O(s, o) \leftrightarrow D(s)$', r'$O(o, o) \to D(o)$', r'$O(o, o) \leftarrow D(o)$',
              r'$O(s, o) \to D(o)$', r'$O(s, o) \leftarrow D(o)$', r'$O(o, o) \to D(s)$',
              r'$O(o, o) \leftarrow D(s)$', r'$O(s, o) \to D(s)$',
              r'$O(s, o) \leftarrow D(s)$', r'$O(o, o) \Leftrightarrow D(o)$',
              r'$O(r, r) \to D(o)$', r'$O(r, r) \leftarrow D(o)$',
              r'$O(r, r) \to D(s)$', r'$O(r, r) \leftarrow D(s)$',
              r'$O(r, r) \Leftrightarrow D(r)$']

step_list2 = [r'$O(s, s) \leftrightarrow D(s)$', r'$O(o, s) \to D(o)$', r'$O(o, s) \leftarrow D(o)$',
              r'$O(s, s) \to D(o)$', r'$O(s, s) \leftarrow D(o)$', r'$O(o, s) \to D(s)$',
              r'$O(o, s) \leftarrow D(s)$', r'$O(s, s) \to D(s)$',
              r'$O(s, s) \leftarrow D(s)$', r'$O(o, s) \Leftrightarrow D(o)$']


class RepeatDynamics:
    def __init__(self, setting, p, v, using_prob=False, select_step=1,
                 select_node_layer='A_layer', select_node_method='0',  node_number=0, unchanged_state='None',
                 select_edge_layer='A_internal', select_edge_method='0',  edge_number=0):
        self.repeated_result = RepeatDynamics.repeat_dynamics(setting, p, v, using_prob, select_step,
                                                              select_node_layer, select_node_method, node_number, unchanged_state,
                                                              select_edge_layer, select_edge_method, edge_number)

    @staticmethod
    def repeat_dynamics(setting, p, v, using_prob, select_step,
                        select_node_layer, select_node_method, node_number, unchanged_state,
                        select_edge_layer, select_edge_method, edge_number):
        num_data = np.zeros([setting.Limited_step+1, 17])
        for repeat in range(setting.Repeating_number):
            inter_layer = InterconnectedLayerModeling.InterconnectedLayerModeling(setting)
            key_nodes = RepeatDynamics.select_keynode(setting, inter_layer, select_node_layer, select_node_method, node_number)
            key_edges = RepeatDynamics.select_keyedge(setting, key_nodes[2], select_edge_layer, select_edge_method, edge_number)
            dynamics_result = InterconnectedDynamics.InterconnectedDynamics(setting, key_edges[2], p, v, using_prob,
                                                                            select_step, key_nodes[0], key_nodes[1], key_edges[1])
            # print("unchanged_nodelist: %s " % len(key_nodes[0]) + "  removed_edgelist: %s" % key_edges[0])
            num_data = num_data + dynamics_result.dynamics_result_array
        Num_Data = num_data / setting.Repeating_number
        panda_db = RepeatDynamics.making_dataframe_per_step(setting, Num_Data)
        panda_db['using_prob'] = using_prob
        if using_prob is False:
            panda_db['Orders'] = step_list1[select_step]
        elif using_prob is True:
            panda_db['Orders'] = step_list2[select_step]
        panda_db['keynode_method'] = select_node_method
        if select_node_method == '0':
            panda_db['select_node_layer'] = select_node_layer
            panda_db['keynode_number'] = 0
            panda_db['unchanged_state'] = 0
        elif select_node_method != '0':
            panda_db['select_node_layer'] = select_node_layer
            panda_db['keynode_number'] = node_number
            panda_db['unchanged_state'] = unchanged_state
        panda_db['keyedge_method'] = select_edge_method
        if select_edge_method == '0':
            panda_db['select_edge_layer'] = select_edge_layer
            panda_db['keyedge_number'] = 0
        elif select_edge_method != '0':
            panda_db['select_edge_layer'] = select_edge_layer
            panda_db['keyedge_number'] = edge_number
        return panda_db

    @staticmethod
    def select_keynode(setting, inter_layer, select_node_layer, select_node_method, node_number):
        if select_node_method == '0':
            unchanged_nodes = None
            sum_properties = 0
        elif select_node_method == 'random':
            node_list = []
            if select_node_layer == 'A_layer':
                node_list = inter_layer.A_nodes
            elif select_node_layer == 'B_layer':
                node_list = inter_layer.B_nodes
            elif select_node_layer == 'mixed':
                node_list = sorted(inter_layer.two_layer_graph.nodes)
            select_nodes_list = random.sample(node_list, k=node_number)
            unchanged_nodes = set(select_nodes_list)
            sum_properties = 0
        else:
            select_nodes_list = []
            nodes_properties = []
            select_layer_number = 0
            if select_node_layer == 'A_layer':
                select_layer_number = 0
            elif select_node_layer == 'B_layer':
                select_layer_number = 1
            elif select_node_layer == 'mixed':
                select_layer_number = 2
            nodes_calculation = NodeProperty.NodeProperty(setting, inter_layer, select_layer_number, select_node_method)
            ordering = nodes_calculation.nodes_order[0:node_number]
            for i, j in ordering:
                select_nodes_list.append(i)
                nodes_properties.append(j)
            unchanged_nodes = set(select_nodes_list)
            sum_properties = sum(nodes_properties)
        return unchanged_nodes, sum_properties, inter_layer

    @staticmethod
    def select_keyedge(setting, inter_layer, select_edge_layer, select_edge_method, edge_number):
        if select_edge_method == '0':
            select_edges_list = None
            edges_properties = 0
        elif select_edge_method == 'random':
            edge_list = []
            if select_edge_layer == 'A_internal':
                edge_list = inter_layer.edges_on_A
            elif select_edge_layer == 'A_mixed':
                edge_list = inter_layer.edges_on_A + inter_layer.edges_on_AB
            elif select_edge_layer == 'B_internal':
                edge_list = inter_layer.edges_on_B
            elif select_edge_layer == 'B_mixed':
                edge_list = inter_layer.edges_on_B + inter_layer.edges_on_AB
            elif select_edge_layer == 'external':
                edge_list = inter_layer.edges_on_AB
            elif select_edge_layer == 'mixed':
                edge_list = inter_layer.edges_on_A + inter_layer.edges_on_B + inter_layer.edges_on_AB
            select_edges_list = random.sample(edge_list, k=edge_number)
            inter_layer.two_layer_graph.remove_edges_from(select_edges_list)
            for edge in select_edges_list:
                if edge[0] < setting.A_node and edge[1] < setting.A_node:
                    inter_layer.edges_on_A.remove(edge)
                    inter_layer.A_layer_graph.remove_edges_from(select_edges_list)
                elif edge[0] >= setting.A_node and edge[1] >= setting.A_node:
                    inter_layer.edges_on_B.remove(edge)
                    inter_layer.B_layer_graph.remove_edges_from(select_edges_list)
                else:
                    inter_layer.edges_on_AB.remove(edge)
                inter_layer.unique_neighbor_dict[edge[0]].remove(edge[1])
            edges_properties = 0
        else:
            select_edges_list = []
            edges_properties_list = []
            select_edges_number = 0
            edges_properties = 0
            if select_edge_layer == 'A_internal':
                select_edges_number = 0
            elif select_edge_layer == 'A_mixed':
                select_edges_number = 1
            elif select_edge_layer == 'B_internal':
                select_edges_number = 2
            elif select_edge_layer == 'B_mixed':
                select_edges_number = 3
            elif select_edge_layer == 'external':
                select_edges_number = 4
            elif select_edge_layer == 'mixed':
                select_edges_number = 5
            if select_edge_method.split('_')[-1] != 'sequential':
                edges_calculation = EdgeProperty.EdgeProperty(setting, inter_layer, select_edges_number, select_edge_method)
                ordering = edges_calculation.edges_order[0:edge_number]
                for i, j in ordering:
                    select_edges_list.append(i)
                    edges_properties_list.append(j)
                inter_layer.two_layer_graph.remove_edges_from(select_edges_list)
                for edge in select_edges_list:
                    if edge[0] < setting.A_node and edge[1] < setting.A_node:
                        inter_layer.edges_on_A.remove(edge)
                        inter_layer.A_layer_graph.remove_edges_from(select_edges_list)
                    elif edge[0] >= setting.A_node and edge[1] >= setting.A_node:
                        inter_layer.edges_on_B.remove(edge)
                        inter_layer.B_layer_graph.remove_edges_from(select_edges_list)
                    else:
                        inter_layer.edges_on_AB.remove(edge)
                    inter_layer.unique_neighbor_dict[edge[0]].remove(edge[1])
                edges_properties = sum(edges_properties_list)
            elif select_edge_method.split('_')[-1] == 'sequential':
                for removed_number in range(edge_number):
                    edges_calculation = EdgeProperty.EdgeProperty(setting, inter_layer, select_edges_number, select_edge_method)
                    ordering = edges_calculation.edges_order[0]
                    edge = ordering[0]
                    values = ordering[1]
                    select_edges_list.append(edge)
                    edges_properties_list.append(values)
                    inter_layer.two_layer_graph.remove_edge(*edge)
                    if edge[0] < setting.A_node and edge[1] < setting.A_node:
                        inter_layer.edges_on_A.remove(edge)
                        inter_layer.A_layer_graph.remove_edge(*edge)
                    elif edge[0] >= setting.A_node and edge[1] >= setting.A_node:
                        inter_layer.edges_on_B.remove(edge)
                        inter_layer.B_layer_graph.remove_edge(*edge)
                    else:
                        inter_layer.edges_on_AB.remove(edge)
                    inter_layer.unique_neighbor_dict[edge[0]].remove(edge[1])
                edges_properties = sum(edges_properties_list)
        return select_edges_list, edges_properties, inter_layer

    @staticmethod
    def making_dataframe_per_step(setting, value_array):
        columns = ['p', 'v', 'prob_v', 'persuasion', 'compromise',
                   'A_plus', 'A_minus', 'B_plus', 'B_minus',
                   'Layer_A_Mean', 'Layer_B_Mean', 'AS',
                   'A_total_edges', 'B_total_edges', 'change_count',
                   'key_nodes_property', 'key_edges_property']
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
    settings = SettingSimulationValue.SettingSimulationValue()
    settings.Repeating_number = 10
    P = 0.1
    V = 0.1
    res = RepeatDynamics(settings, P, V, using_prob=False, select_step=1,
                         select_node_layer='A_layer', select_node_method='pagerank', node_number=5, unchanged_state='pos',
                         select_edge_layer='A_internal', select_edge_method='0', edge_number=0)
    print(res.repeated_result)
    end = time.time()
    print(end - start)
