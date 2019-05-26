import numpy as np


class SettingSimulationValue:
    def __init__(self):
        self.database = 'pv_variable'  # 'competition  renew_competition'
        self.table = 'keynode_table'
        self.Model = 'BA-BA'
        self.Structure = 'BA-BA'

        self.Limited_step = 100
        self.Repeating_number = 100

        self.A_node = 2048
        self.A_edge = 4
        self.A = SettingSimulationValue.static_making_A_array(self.A_node, A_state=[1, 2])
        self.MAX = 2
        self.MIN = -2

        self.B_node = 2048
        self.B_edge = 4
        self.B = SettingSimulationValue.static_making_B_array(self.B_node, B_state=[-1])
        self.variable_list = SettingSimulationValue.p_and_v_list(gap=30)
        self.workers = 5

        self.select_method_list = ['0', 'pagerank', 'betweenness', 'degree', 'eigenvector', 'closeness',
                                   'load', 'AB_pagerank', 'AB_eigenvector', 'AB_degree', 'AB_betweenness',
                                   'AB_closeness', 'AB_load', 'random']

        self.step_list1 = [r'$O(s, o) \leftrightarrow D(s)$', r'$O(o, o) \to D(o)$', r'$O(o, o) \leftarrow D(o)$',
                           r'$O(s, o) \to D(o)$', r'$O(s, o) \leftarrow D(o)$', r'$O(o, o) \to D(s)$',
                           r'$O(o, o) \leftarrow D(s)$', r'$O(s, o) \to D(s)$',
                           r'$O(s, o) \leftarrow D(s)$', r'$O(o, o) \Leftrightarrow D(o)$',
                           r'$O(r, r) \to D(o)$', r'$O(r, r) \leftarrow D(o)$',
                           r'$O(r, r) \to D(s)$', r'$O(r, r) \leftarrow D(s)$',
                           r'$O(r, r) \Leftrightarrow D(r)$']

        self.step_list2 = [r'$O(s, s) \leftrightarrow D(s)$', r'$O(o, s) \to D(o)$', r'$O(o, s) \leftarrow D(o)$',
                           r'$O(s, s) \to D(o)$', r'$O(s, s) \leftarrow D(o)$', r'$O(o, s) \to D(s)$',
                           r'$O(o, s) \leftarrow D(s)$', r'$O(s, s) \to D(s)$',
                           r'$O(s, s) \leftarrow D(s)$', r'$O(o, s) \Leftrightarrow D(o)$']

    @staticmethod
    def p_and_v_list(gap):
        p_list = np.linspace(0, 1, gap)
        v_list = np.linspace(0, 1, gap)
        variable_list = []
        for p in p_list:
            for v in v_list:
                variable_list.append((p, v))
        return variable_list

    @staticmethod
    def static_making_A_array(A_node, A_state):
        values = A_state * int(A_node / len(A_state))
        A = np.array(values)
        return A

    @staticmethod
    def static_making_B_array(B_node, B_state):
        values = B_state * int(B_node / len(B_state))
        B = np.array(values)
        return B


if __name__ == "__main__":
    SS = SettingSimulationValue()
    SS.A_node =128
    #layer_A1 = Layer_A_Modeling.Layer_A_Modeling(SS)
    # print(SS.A_node)
    #print(len(layer_A1.A))
    #layer_A2 = Layer_A_Modeling.Layer_A_Modeling(SS)
    # print(SS.B_node)
    # print(SS.variable_list)
    #print(len(layer_A2.A))
