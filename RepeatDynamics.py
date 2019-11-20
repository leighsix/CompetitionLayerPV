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
    def __init__(self, setting, p, v, using_prob=False, updating_rule=1,
                 node_layer='A_layer', node_method_list=None,  node_numbers=0,
                 edge_layer='A_internal', edge_method_list=None,  edge_numbers=0):
        self.repeated_result = RepeatDynamics.many_execute_for_repeating(setting, p, v, using_prob, updating_rule,
                                                                         node_layer, node_method_list, node_numbers,
                                                                         edge_layer, edge_method_list, edge_numbers)

    @staticmethod
    def many_execute_for_repeating(setting, p, v, using_prob, updating_rule,
                                   node_layer, node_method_list, node_numbers,
                                   edge_layer, edge_method_list, edge_numbers):
        num_data = np.zeros(23)
        with futures.ProcessPoolExecutor(max_workers=setting.workers) as executor:
            to_do_map = {}
            for repeat in range(setting.Repeating_number):
                future = executor.submit(RepeatDynamics.combined_dynamics, setting, p, v, using_prob, updating_rule,
                                         node_layer, node_method_list, node_numbers,
                                         edge_layer, edge_method_list, edge_numbers)
                to_do_map[future] = repeat
            done_iter = futures.as_completed(to_do_map)
            done_iter = tqdm(done_iter, total=setting.Repeating_number)
            for future in done_iter:
                result_array = future.result()
                num_data = num_data + result_array
        #         print("result1: %s" %num_data)
        # print("result2: %s" %num_data)
        Num_Data = num_data / setting.Repeating_number
        panda_db = RepeatDynamics.making_dataframe_per_step(setting, Num_Data)
        panda_db['select_node_layer'] = node_layer
        panda_db['select_edge_layer'] = edge_layer
        panda_db['using_prob'] = using_prob
        if using_prob is False:
            panda_db['Orders'] = updating_rule_list1[updating_rule]
        elif using_prob is True:
            panda_db['Orders'] = updating_rule_list2[updating_rule]
        return panda_db

    @staticmethod
    def combined_dynamics(setting, p, v, using_prob, updating_rule,
                          node_layer, node_method_list, node_numbers,
                          edge_layer, edge_method_list, edge_numbers):
        if node_method_list is None:
            node_method_list = ['0']
        if edge_method_list is None:
            edge_method_list = ['0']
        inter_layer = InterconnectedLayerModeling.InterconnectedLayerModeling(setting)
        dic_key_edges = RepeatDynamics.dictionary_edges(setting, inter_layer, edge_layer, edge_method_list, edge_numbers)
        result_array = np.zeros(23)
        for edge_method in edge_method_list:
            key_edges = dic_key_edges[edge_method]
            for edge_number in range(edge_numbers + 1):
                inter_layer = RepeatDynamics.remove_edges(setting, inter_layer, key_edges[0])
                dic_key_nodes = RepeatDynamics.dictionary_centralities(setting, inter_layer, node_layer,
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
                    dynamics_result = InterconnectedDynamics.InterconnectedDynamics(setting, inter_layer, p, v,
                                                                                    using_prob,
                                                                                    updating_rule, key_nodes[0],
                                                                                    key_nodes[1],
                                                                                    key_edges[1][edge_number],
                                                                                    edge_number, keynode_method,
                                                                                    keyedge_method, unchanged_state)
                    result_array = np.vstack([result_array, dynamics_result.dynamics_result_array])
        result_array = result_array[1:]
        return result_array

    @staticmethod
    def dictionary_centralities(setting, inter_layer, select_node_layer, node_method_list, node_numbers):
        dic_centralities = {}
        for node_method in node_method_list:
            keynode = RepeatDynamics.select_keynode(setting, inter_layer, select_node_layer, node_method, node_numbers)
            dic_centralities[node_method] = [keynode[0], keynode[1]]
        return dic_centralities

    @staticmethod
    def dictionary_edges(setting, inter_layer, select_edge_layer, edge_method_list, edge_numbers):
        dic_edges = {}
        for edge_method in edge_method_list:
            keyedge = RepeatDynamics.select_keyedge(setting, inter_layer, select_edge_layer, edge_method, edge_numbers)
            dic_edges[edge_method] = [keyedge[0], keyedge[1]]
        return dic_edges

    @staticmethod
    def select_keynode(setting, inter_layer, select_node_layer, node_method, node_numbers):
        if node_method == '0':
            unchanged_nodes_list = None
            nodes_properties_list = [0]
        elif node_method == 'random':
            unchanged_nodes_list = []
            nodes_properties_list = []
            node_list = []
            if select_node_layer == 'A_layer':
                node_list = inter_layer.A_nodes
            elif select_node_layer == 'B_layer':
                node_list = inter_layer.B_nodes
            elif select_node_layer == 'mixed':
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
            select_layer_number = 0
            if select_node_layer == 'A_layer':
                select_layer_number = 0
            elif select_node_layer == 'B_layer':
                select_layer_number = 1
            elif select_node_layer == 'mixed':
                select_layer_number = 2
            nodes_calculation = NodeProperty.NodeProperty(setting, inter_layer, select_layer_number, node_method)
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
    def select_keyedge(setting, inter_layer, select_edge_layer, edge_method, edge_numbers):
        if edge_method == '0':
            select_edges_list = []
            edges_properties_list = [0]
        elif edge_method == 'random':
            edge_list = []
            select_edges_list = []
            edges_properties_list = []
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
            for edge_number in range(1, edge_numbers+1):
                select_edges = random.sample(edge_list, k=edge_number)
                select_edges_list.append(select_edges)
                edges_properties_list.append(0)
        else:
            select_edges_list = []
            edges_properties_list = []
            select_edges_number = 0
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
            edges_calculation = EdgeProperty.EdgeProperty(setting, inter_layer, select_edges_number, edge_method)
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
                   'keyedge_number', 'Steps', 'keynode_method', 'keyedge_method', 'unchanged_state']
        df = pd.DataFrame(value_array, columns=columns)
        df = RepeatDynamics.naming_method_in_df(df)
        df['Model'] = setting.Model
        df['Structure'] = setting.Structure
        df['A_node_number'] = setting.A_node
        df['B_node_number'] = setting.B_node
        return df

    @staticmethod
    def naming_method_in_df(df):
        for i in range(len(df)):
            df.iloc[i, 20] = RepeatDynamics.renaming_keynode_method(df['keynode_method'][i])
            df.iloc[i, 21] = RepeatDynamics.renaming_keyedge_method(df['keyedge_method'][i])
            df.iloc[i, 22] = RepeatDynamics.renaming_unchanged_state(df['unchanged_state'][i])
        return df

    @staticmethod
    def renaming_unchanged_state(unchanged_state_number):
        unchanged_state = '0'
        if unchanged_state_number == 0: unchanged_state = '0'
        elif unchanged_state_number == 1: unchanged_state = 'pos'
        elif unchanged_state_number == 2: unchanged_state = 'neg'
        return unchanged_state

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
        elif node_method == 'load': keynode_method = 10
        elif node_method == 'pagerank_individual': keynode_method = 11
        elif node_method == 'AB_pagerank': keynode_method = 12
        elif node_method == 'AB_eigenvector': keynode_method = 13
        elif node_method == 'AB_degree': keynode_method = 14
        elif node_method == 'AB_betweenness': keynode_method = 15
        elif node_method == 'AB_closeness': keynode_method = 16
        elif node_method == 'AB_load': keynode_method = 17
        return keynode_method

    @staticmethod
    def renaming_keynode_method(keynode_method):
        node_method = '0'
        if keynode_method == 1: node_method = '0'
        elif  keynode_method == 2: node_method = 'degree'
        elif  keynode_method == 3: node_method = 'pagerank'
        elif  keynode_method == 4: node_method = 'random'
        elif  keynode_method == 5: node_method = 'eigenvector'
        elif  keynode_method == 6: node_method = 'closeness'
        elif  keynode_method == 7: node_method = 'betweenness'
        elif  keynode_method == 8: node_method = 'PR+DE'
        elif  keynode_method == 9: node_method = 'PR+DE+BE'
        elif  keynode_method == 10: node_method = 'load'
        elif  keynode_method == 11: node_method = 'pagerank_individual'
        elif  keynode_method == 12: node_method = 'AB_pagerank'
        elif  keynode_method == 13: node_method = 'AB_eigenvector'
        elif  keynode_method == 14: node_method = 'AB_degree'
        elif  keynode_method == 15: node_method = 'AB_betweenness'
        elif  keynode_method == 16: node_method = 'AB_closeness'
        elif  keynode_method == 17: node_method = 'AB_load'
        return node_method

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

    @staticmethod
    def renaming_keyedge_method(keyedge_method):
        edge_method = '0'
        if keyedge_method == 0: edge_method = '0'
        elif  keyedge_method == 1: edge_method = 'edge_pagerank'
        elif  keyedge_method == 2: edge_method = 'edge_betweenness'
        elif keyedge_method == 3: edge_method = 'edge_degree'
        elif keyedge_method == 4: edge_method = 'edge_eigenvector'
        elif keyedge_method == 5: edge_method = 'edge_closeness'
        elif keyedge_method == 6: edge_method = 'edge_load'
        elif keyedge_method == 7: edge_method = 'edge_jaccard'
        elif keyedge_method == 8: edge_method = 'random'
        return edge_method

if __name__ == "__main__":
    print("RepeatDynamics")
    start = time.time()
    settings = SettingSimulationValue.SettingSimulationValue()
    settings.Repeating_number = 2
    settings.workers = 1
    inter_layer = InterconnectedLayerModeling.InterconnectedLayerModeling(settings)
    p = 0.1
    v = 0.1
    res = RepeatDynamics(settings, p, v, using_prob=False, updating_rule=1,
                         node_layer='A_layer', node_method_list=['0', 'pagerank', 'degree'], node_numbers=5,
                         edge_layer='A_internal', edge_method_list=None, edge_numbers=0)
    print(res.repeated_result)
    end = time.time()
    print(end - start)
