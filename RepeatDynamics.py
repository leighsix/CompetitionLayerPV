import numpy as np
import pandas as pd
import NodeProperty
import EdgeProperty
import SettingSimulationValue
import InterconnectedDynamics
import InterconnectedLayerModeling
import time
import random
from concurrent import futures
from tqdm import tqdm

updating_rule_list1 = [r'$O(s, o) \leftrightarrow D(s)$', r'$O(o, o) \to D(o)$', r'$O(o, o) \leftarrow D(o)$',
                       r'$O(s, o) \to D(o)$', r'$O(s, o) \leftarrow D(o)$', r'$O(o, o) \to D(s)$',
                       r'$O(o, o) \leftarrow D(s)$', r'$O(s, o) \to D(s)$',
                       r'$O(s, o) \leftarrow D(s)$', r'$O(o, o) \Leftrightarrow D(o)$',
                       r'$O(r, r) \to D(o)$', r'$O(r, r) \leftarrow D(o)$',
                       r'$O(r, r) \to D(s)$', r'$O(r, r) \leftarrow D(s)$',
                       r'$O(r, r) \Leftrightarrow D(r)$']

updating_rule_list2 = [r'$O(s, s) \leftrightarrow D(s)$', r'$O(o, s) \to D(o)$', r'$O(o, s) \leftarrow D(o)$',
                       r'$O(s, s) \to D(o)$', r'$O(s, s) \leftarrow D(o)$', r'$O(o, s) \to D(s)$',
                       r'$O(o, s) \leftarrow D(s)$', r'$O(s, s) \to D(s)$',
                       r'$O(s, s) \leftarrow D(s)$', r'$O(o, s) \Leftrightarrow D(o)$']


class RepeatDynamics:
    def __init__(self, setting, using_prob=False, updating_rule=1,
                 node_layer_list=None, node_method_list=None,
                 edge_layer_list=None, edge_method_list=None,  edge_numbers=0):
        self.repeated_result = RepeatDynamics.many_execute_for_repeating(setting, using_prob, updating_rule,
                                                                         node_layer_list, node_method_list,
                                                                         edge_layer_list, edge_method_list, edge_numbers)

    @staticmethod
    def many_execute_for_repeating(setting, using_prob, updating_rule,
                                   node_layer_list, node_method_list,
                                   edge_layer_list, edge_method_list, edge_numbers):
        num_data = np.zeros(25)
        with futures.ProcessPoolExecutor(max_workers=setting.workers) as executor:
            to_do_map = {}
            for repeat in range(setting.Repeating_number):
                future = executor.submit(RepeatDynamics.combined_dynamics, setting, using_prob, updating_rule,
                                         node_layer_list, node_method_list,
                                         edge_layer_list, edge_method_list, edge_numbers)
                to_do_map[future] = repeat
            done_iter = futures.as_completed(to_do_map)
            done_iter = tqdm(done_iter, total=setting.Repeating_number)
            for future in done_iter:
                result_array = future.result()
                num_data = num_data + result_array
        Num_Data = num_data / setting.Repeating_number
        panda_db = RepeatDynamics.making_dataframe_per_step(setting, Num_Data)
        panda_db['using_prob'] = using_prob
        if using_prob is False:
            panda_db['Orders'] = updating_rule_list1[updating_rule]
        elif using_prob is True:
            panda_db['Orders'] = updating_rule_list2[updating_rule]
        return panda_db

    @staticmethod
    def combined_dynamics(setting, using_prob, updating_rule,
                          node_layer_list, node_method_list,
                          edge_layer_list, edge_method_list, edge_numbers):
        if node_layer_list is None:
            node_layer_list = ['0']
        if edge_layer_list is None:
            edge_layer_list = ['0']
        if node_method_list is None:
            node_method_list = ['0']
        if edge_method_list is None:
            edge_method_list = ['0']
        inter_layer = InterconnectedLayerModeling.InterconnectedLayerModeling(setting)
        result_array = np.zeros(25)
        for edge_layer in edge_layer_list:
            edge_layer_number = RepeatDynamics.naming_edge_layer(edge_layer)
            dic_key_edges = RepeatDynamics.dictionary_edges(setting, inter_layer, edge_layer_number, edge_method_list, edge_numbers)
            for edge_method in edge_method_list:
                key_edges = dic_key_edges[edge_method]
                for edge_number in range(edge_numbers + 1):
                    inter_layer = RepeatDynamics.remove_edges(setting, inter_layer, key_edges[0])
                    for node_layer in node_layer_list:
                        node_layer_number = RepeatDynamics.naming_node_layer(node_layer)
                        if node_layer_number == 0:
                            p = [0.2]
                            v = [0.4]
                            gap = 1
                            node_numbers = 64
                        elif node_layer_number == 1:
                            p = [0.3]
                            v = [0.5]
                            gap = 1
                            node_numbers = 256
                        elif node_layer_number == 2:
                            p = [0.2]
                            v = [0.4]
                            gap = 1
                            node_numbers = 50
                        else:
                            p = [0, 1]
                            v = [0, 1]
                            gap = 30
                            node_numbers = 0
                        dic_key_nodes = RepeatDynamics.dictionary_centralities(setting, inter_layer, node_layer_number,
                                                                               node_method_list, node_numbers)
                        for node_method in node_method_list:
                            if node_layer == 'A_layer' and node_method != '0':
                                unchanged_state = 1
                            elif node_layer == 'B_layer' and node_method != '0':
                                unchanged_state = 2
                            else:
                                unchanged_state = 0
                            key_nodes = dic_key_nodes[node_method]
                            keynode_method = RepeatDynamics.naming_keynode_method(node_method)
                            keyedge_method = RepeatDynamics.naming_keyedge_method(edge_method)
                            p_list = np.linspace(p[0], p[-1], gap)
                            v_list = np.linspace(v[0], v[-1], gap)
                            for p_value in p_list:
                                for v_value in v_list:
                                    dynamics_result = InterconnectedDynamics.InterconnectedDynamics(setting, inter_layer, p_value, v_value,
                                                                                                    using_prob,
                                                                                                    updating_rule, key_nodes[0],
                                                                                                    key_nodes[1],
                                                                                                    key_edges[1][edge_number],
                                                                                                    edge_number, keynode_method,
                                                                                                    keyedge_method, unchanged_state,
                                                                                                    node_layer_number, edge_layer_number)
                                    result_array = np.vstack([result_array, dynamics_result.dynamics_result_array])
        result_array = result_array[1:]
        return result_array

    @staticmethod
    def dictionary_centralities(setting, inter_layer, node_layer_number, node_method_list, node_numbers):
        dic_centralities = {}
        for node_method in node_method_list:
            keynode = RepeatDynamics.select_keynode(setting, inter_layer, node_layer_number, node_method, node_numbers)
            dic_centralities[node_method] = [keynode[0], keynode[1]]
        return dic_centralities

    @staticmethod
    def dictionary_edges(setting, inter_layer, edge_layer_number, edge_method_list, edge_numbers):
        dic_edges = {}
        for edge_method in edge_method_list:
            keyedge = RepeatDynamics.select_keyedge(setting, inter_layer, edge_layer_number, edge_method, edge_numbers)
            dic_edges[edge_method] = [keyedge[0], keyedge[1]]
        return dic_edges

    @staticmethod
    def select_keynode(setting, inter_layer, node_layer_number, node_method, node_numbers):
        if node_method == '0':
            unchanged_nodes_list = None
            nodes_properties_list = [0]
        elif node_method == 'random':
            unchanged_nodes_list = []
            nodes_properties_list = []
            node_list = []
            if node_layer_number == 0:
                node_list = inter_layer.A_nodes
            elif node_layer_number == 1:
                node_list = inter_layer.B_nodes
            elif node_layer_number == 2:
                node_list = sorted(inter_layer.two_layer_graph.nodes)
            for node_number in range(1, node_numbers+1):
                select_nodes_list = random.sample(node_list, k=node_number)
                unchanged_nodes = set(select_nodes_list)
                unchanged_nodes_list.append(unchanged_nodes)
                nodes_properties_list.append(0)
        else:
            unchanged_nodes_list = []
            nodes_properties_list = []
            select_nodes_list = []
            nodes_properties = []
            nodes_calculation = NodeProperty.NodeProperty(setting, inter_layer, node_layer_number, node_method)
            for node_number in range(1, node_numbers+1):
                ordering = nodes_calculation.nodes_order[0:node_number]
                for i, j in ordering:
                    select_nodes_list.append(i)
                    nodes_properties.append(j)
                unchanged_nodes = set(select_nodes_list)
                sum_properties = sum(nodes_properties)
                unchanged_nodes_list.append(unchanged_nodes)
                nodes_properties_list.append(sum_properties)
        return unchanged_nodes_list, nodes_properties_list

    @staticmethod
    def select_keyedge(setting, inter_layer, edge_layer_number, edge_method, edge_numbers):
        if edge_method == '0':
            select_edges_list = []
            edges_properties_list = [0]
        elif edge_method == 'random':
            edge_list = []
            select_edges_list = []
            edges_properties_list = []
            if edge_layer_number == 0:
                edge_list = inter_layer.edges_on_A
            elif edge_layer_number == 1:
                edge_list = inter_layer.edges_on_A + inter_layer.edges_on_AB
            elif edge_layer_number == 2:
                edge_list = inter_layer.edges_on_B
            elif edge_layer_number == 3:
                edge_list = inter_layer.edges_on_B + inter_layer.edges_on_AB
            elif edge_layer_number == 4:
                edge_list = inter_layer.edges_on_AB
            elif edge_layer_number == 5:
                edge_list = inter_layer.edges_on_A + inter_layer.edges_on_B + inter_layer.edges_on_AB
            for edge_number in range(1, edge_numbers+1):
                select_edges = random.sample(edge_list, k=edge_number)
                select_edges_list.append(select_edges)
                edges_properties_list.append(0)
        else:
            select_edges_list = []
            edges_properties_list = []
            edges_calculation = EdgeProperty.EdgeProperty(setting, inter_layer, edge_layer_number, edge_method)
            for edge_number in range(1, edge_numbers + 1):
                ordering = edges_calculation.edges_order[0:edge_number]
                edges_proerties=[]
                for i, j in ordering:
                    select_edges_list.append(i)
                    edges_proerties.append(j)
                select_edges_list = list(set(select_edges_list))
                edges_properties_list.append(sum(edges_proerties))
        return select_edges_list, edges_properties_list

    @staticmethod
    def remove_edges(setting, inter_layer, select_edges_list):
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
        return inter_layer

    @staticmethod
    def making_dataframe_per_step(setting, value_array):
        columns = ['p', 'v', 'prob_v', 'persuasion', 'compromise',
                   'A_plus', 'A_minus', 'B_plus', 'B_minus',
                   'Layer_A_Mean', 'Layer_B_Mean', 'AS',
                   'A_total_edges', 'B_total_edges', 'change_count',
                   'key_nodes_property', 'key_edges_property', 'keynode_number',
                   'keyedge_number', 'Steps', 'keynode_method', 'keyedge_method', 'unchanged_state',
                   'select_node_layer', 'select_edge_layer']
        df = pd.DataFrame(value_array, columns=columns)
        df['Model'] = setting.Model
        df['Structure'] = setting.Structure
        df['A_node_number'] = setting.A_node
        df['B_node_number'] = setting.B_node
        return df

    @staticmethod
    def naming_node_layer(node_layer):
        node_layer_number = 0
        if node_layer == 'A_layer': node_layer_number = 0
        elif node_layer == 'B_layer': node_layer_number = 1
        elif node_layer == 'mixed': node_layer_number = 2
        elif node_layer == '0': node_layer_number = 3
        return node_layer_number

    @staticmethod
    def naming_edge_layer(edge_layer):
        edge_layer_number = 0
        if edge_layer == 'A_internal': edge_layer_number = 0
        elif edge_layer == 'A_mixed': edge_layer_number = 1
        elif edge_layer == 'B_internal': edge_layer_number = 2
        elif edge_layer == 'B_mixed': edge_layer_number = 3
        elif edge_layer == 'external': edge_layer_number = 4
        elif edge_layer == 'mixed': edge_layer_number = 5
        elif edge_layer == '0': edge_layer_number = 6
        return edge_layer_number

    @staticmethod
    def naming_keynode_method(node_method):
        keynode_method = 0
        if node_method == '0': keynode_method = 1
        elif node_method == 'degree': keynode_method = 2
        elif node_method == 'pagerank': keynode_method = 3
        elif node_method == 'random': keynode_method = 4
        elif node_method == 'eigenvector': keynode_method = 5
        elif node_method == 'closeness': keynode_method = 6
        elif node_method == 'betweenness': keynode_method = 7
        elif node_method == 'PR+DE': keynode_method = 8
        elif node_method == 'PR+DE+BE': keynode_method = 9
        elif node_method == 'PR+BE': keynode_method = 10
        elif node_method == 'DE+BE': keynode_method = 11
        elif node_method == 'load': keynode_method = 12
        elif node_method == 'pagerank_individual': keynode_method = 13
        elif node_method == 'AB_pagerank': keynode_method = 14
        elif node_method == 'AB_eigenvector': keynode_method = 15
        elif node_method == 'AB_degree': keynode_method = 16
        elif node_method == 'AB_betweenness': keynode_method = 17
        elif node_method == 'AB_closeness': keynode_method = 18
        elif node_method == 'AB_load': keynode_method = 19
        elif node_method == 'PR+EI': keynode_method = 20
        elif node_method == 'DE+EI': keynode_method = 21
        elif node_method == 'PR+DE+EI': keynode_method = 22
        return keynode_method

    @staticmethod
    def naming_keyedge_method(edge_method):
        keyedge_method = 0
        if edge_method == '0': keyedge_method = 0
        elif edge_method == 'edge_pagerank': keyedge_method = 1
        elif edge_method == 'edge_betweenness': keyedge_method = 2
        elif edge_method == 'edge_degree': keyedge_method = 3
        elif edge_method == 'edge_eigenvector': keyedge_method = 4
        elif edge_method == 'edge_closeness': keyedge_method = 5
        elif edge_method == 'edge_load': keyedge_method = 6
        elif edge_method == 'edge_jaccard': keyedge_method = 7
        elif edge_method == 'random': keyedge_method = 8
        return keyedge_method



if __name__ == "__main__":
    print("RepeatDynamics")
    start = time.time()
    settings = SettingSimulationValue.SettingSimulationValue()
    settings.A_node = 512
    settings.B_node = 512
    settings.Repeating_number = 1
    settings.workers = 1
    inter_layer = InterconnectedLayerModeling.InterconnectedLayerModeling(settings)
    res = RepeatDynamics(settings, using_prob=False, updating_rule=1,
                         node_layer_list=['A_layer', 'B_layer'],
                         node_method_list=['degree', 'pagerank', 'eigenvector', 'random',
                                           'betweenness', 'closeness', 'PR+DE', 'PR+BE', 'DE+BE', 'PR+DE+BE'],
                         edge_layer_list=None, edge_method_list=None, edge_numbers=0)
    print(res.repeated_result[0:5])
    end = time.time()
    print(end - start)
