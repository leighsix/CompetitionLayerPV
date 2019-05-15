import numpy as np
import random

class Setting_Simulation_Value:
    def __init__(self):
        self.database = 'pv_variable'  # 'competition  renew_competition'
        self.table = 'comparison_order_table2'
        self.MODEL = 'BA-BA'
        self.Structure = 'BA-BA'

        self.Limited_step = 100
        self.Repeating_number = 100

        self.A_state = [1, 2]
        self.A_node = 2048
        self.A_edge = 4
        self.A_inter_edges = 1
        self.A = self.static_making_A_array()
        self.MAX = 2
        self.MIN = -2

        self.B_state = [-1]
        self.B_node = 2048
        self.B_edge = 4
        self.B_inter_edges = int(self.A_node / self.B_node)
        self.B = self.static_making_B_array()

        self.DB = 'MySQL'
        self.gap = 30
        simulation_condition = self.simulation_condition(self.gap)
        self.P = simulation_condition[0]
        self.V = simulation_condition[1]
        self.variable_list = self.p_and_v_list(self.P, self.V)
        self.workers = 5

        self.select_method_list = ['0', 'pagerank', 'betweenness', 'number_degree', 'degree', 'eigenvector', 'closeness',
                                   'hub', 'authority', 'load', 'AB_hub', 'AB_authority', 'AB_pagerank', 'AB_eigenvector',
                                   'AB_degree', 'AB_betweenness', 'AB_closeness', 'AB_load', 'AB_number_degree', 'random']

        self.step_list1 = [r'$O(s, o) \leftrightarrow D(s)$', r'$O(o, o) \to D(o)$', r'$O(o, o) \leftarrow D(o)$', r'$O(s, o) \to D(o)$',
                           r'$O(s, o) \leftarrow D(o)$', r'$O(o, o) \to D(s)$', r'$O(o, o) \leftarrow D(s)$', r'$O(s, o) \to D(s)$',
                           r'$O(s, o) \leftarrow D(s)$', r'$O(o, o) \Leftrightarrow D(o)$']

        self.step_list2 = [r'$O(s, s) \leftrightarrow D(s)$', r'$O(o, s) \to D(o)$', r'$O(o, s) \leftarrow D(o)$', r'$O(s, s) \to D(o)$',
                           r'$O(s, s) \leftarrow D(o)$', r'$O(o, s) \to D(s)$', r'$O(o, s) \leftarrow D(s)$', r'$O(s, s) \to D(s)$',
                           r'$O(s, s) \leftarrow D(s)$', r'$O(o, s) \Leftrightarrow D(o)$']

        self.x_list = ['Steps', 'keynode_number']

        self.y_list = ['AS', 'prob_v', 'persuasion', 'compromise', 'change_count']

    def simulation_condition(self, gap):
        self.P = np.linspace(0, 1, gap)
        self.V = np.linspace(0, 1, gap)
        return self.P, self.V

    def p_and_v_list(self, p_list, v_list):
        self.variable_list = []
        for p in p_list:
            for v in v_list:
                self.variable_list.append((p, v))
        return self.variable_list

    def static_making_A_array(self):
        values = self.A_state * int(self.A_node / len(self.A_state))
        self.A = np.array(values)
        random.shuffle(self.A)
        return self.A

    def static_making_B_array(self):
        values = self.B_state * int(self.B_node / len(self.B_state))
        self.B = np.array(values)
        random.shuffle(self.B)
        return self.B

if __name__ == "__main__":
    SS = Setting_Simulation_Value()
    #layer_A1 = Layer_A_Modeling.Layer_A_Modeling(SS)
    print(SS.A_node)
    #print(len(layer_A1.A))
    #layer_A2 = Layer_A_Modeling.Layer_A_Modeling(SS)
    print(SS.B_node)
    print(SS.A)
    print(SS.variable_list)
    #print(len(layer_A2.A))
