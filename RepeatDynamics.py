import numpy as np
import MakingPandas
import NodeProperty
import Setting_Simulation_Value
import InterconnectedDynamics
import InterconnectedLayerModeling
import time


class RepeatDynamics:
    def __init__(self):
        self.inter_dynamics = InterconnectedDynamics.InterconnectedDynamics()
        self.mp = MakingPandas.MakingPandas()
        self.node_property = NodeProperty.NodeProperty()

    def repeat_dynamics(self, setting, p, v, select_step=1, select_method=0, node_number=0):
        num_data = np.zeros([setting.Limited_step + 1, 16])
        centrality_method = setting.select_method_list[select_method]
        for i in range(setting.Repeating_number):
            inter_layer = InterconnectedLayerModeling.InterconnectedLayerModeling(setting)
            key_nodes = self.select_key_A_node(inter_layer, centrality_method, node_number)
            node_i_names = key_nodes[0]
            sum_properties = key_nodes[1]
            if select_step == 0:    # 'O(s)<->D(s)'
                total_array = self.inter_dynamics.interconnected_dynamics0(setting, inter_layer, p, v, node_i_names, sum_properties)
                num_data = num_data + total_array
            elif select_step == 1:  # 'O(o)->D(o)'
                total_array = self.inter_dynamics.interconnected_dynamics1(setting, inter_layer, p, v, node_i_names, sum_properties)
                num_data = num_data + total_array
            elif select_step == 2:  # 'O(o)<-D(o)'
                total_array = self.inter_dynamics.interconnected_dynamics2(setting, inter_layer, p, v, node_i_names, sum_properties)
                num_data = num_data + total_array
            elif select_step == 3:  # 'O(s)->D(o)'
                total_array = self.inter_dynamics.interconnected_dynamics3(setting, inter_layer, p, v, node_i_names, sum_properties)
                num_data = num_data + total_array
            elif select_step == 4:  # 'O(s)<-D(o)'
                total_array = self.inter_dynamics.interconnected_dynamics4(setting, inter_layer, p, v, node_i_names, sum_properties)
                num_data = num_data + total_array
            elif select_step == 5:  # 'O(o)->D(s)'
                total_array = self.inter_dynamics.interconnected_dynamics5(setting, inter_layer, p, v, node_i_names, sum_properties)
                num_data = num_data + total_array
            elif select_step == 6:  # 'O(o)<-D(s)'
                total_array = self.inter_dynamics.interconnected_dynamics6(setting, inter_layer, p, v, node_i_names, sum_properties)
                num_data = num_data + total_array
            elif select_step == 7:  # 'O(s)->D(s)'
                total_array = self.inter_dynamics.interconnected_dynamics7(setting, inter_layer, p, v, node_i_names, sum_properties)
                num_data = num_data + total_array
            elif select_step == 8:  # 'O(s)<-D(s)'
                total_array = self.inter_dynamics.interconnected_dynamics8(setting, inter_layer, p, v, node_i_names, sum_properties)
                num_data = num_data + total_array
            elif select_step == 9:  # 'O(o)<=>D(o)'
                total_array = self.inter_dynamics.interconnected_dynamics9(setting, inter_layer, p, v, node_i_names, sum_properties)
                num_data = num_data + total_array
        Num_Data = num_data / setting.Repeating_number
        panda_db = self.mp.making_dataframe_per_step(setting, Num_Data)
        panda_db['Order'] = setting.step_list[select_step]
        panda_db['keynode_method'] = centrality_method
        if select_method == '0':
            panda_db['keynode_number'] = 0
        elif select_method != '0':
            panda_db['keynode_number'] = node_number
        return panda_db

    def select_key_A_node(self, inter_layer, select_method, node_number):
        node_i_names = {}
        sum_properties = 0
        if select_method == '0':
            node_i_names = None
            sum_properties = 0
        elif select_method != '0':
            select_nodes_list = []
            nodes_properties = []
            select_nodes = self.node_property.ordering_A_node(inter_layer, select_method)[0:node_number]
            for i, j in select_nodes:
                select_nodes_list.append('A_%s' % i)
                nodes_properties.append(j)
            node_i_names = set(select_nodes_list)
            sum_properties = sum(nodes_properties)
        return node_i_names, sum_properties

    def select_key_B_node(self, inter_layer, select_method, node_number):
        node_i_names = {}
        sum_properties = 0
        if select_method == '0':
            node_i_names = None
            sum_properties = 0
        elif select_method != '0':
            select_nodes_list = []
            nodes_properties = []
            select_nodes = self.node_property.ordering_A_node(inter_layer, select_method)[0:node_number]
            for i, j in select_nodes:
                select_nodes_list.append('B_%s' % i-len(sorted(inter_layer.A_edges())))
                nodes_properties.append(j)
            node_i_names = set(select_nodes_list)
            sum_properties = sum(nodes_properties)
        return node_i_names, sum_properties


if __name__ == "__main__":
    print("RepeatDynamics")
    start = time.time()
    setting = Setting_Simulation_Value.Setting_Simulation_Value()
    inter_layer = InterconnectedLayerModeling.InterconnectedLayerModeling(setting)
    p = 0.2
    v = 0.5
    repeat = RepeatDynamics()
    result = repeat.repeat_dynamics(setting, p, v, select_step=1, select_method=1, node_number=1)
    print(result)
    end = time.time()
    print(end - start)
