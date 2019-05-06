import numpy as np
import math
import random


class Setting_Simulation_Value:
    def __init__(self):
        self.database = 'pv_variable'  # 'competition  renew_competition'
        self.table = 'test'
        self.MODEL = 'test'
        self.Structure = 'BA-BA'

        self.Limited_step = 100
        self.Repeating_number = 10

        self.A_state = [1, 2]
        self.A_node = 2048
        self.A_edge = 5
        self.A_inter_edges = 1
        self.A = self.static_making_A_array()
        self.MAX = 2
        self.MIN = -2

        self.B_state = [-1]
        self.B_node = 2048
        self.B_edge = 5
        self.B_inter_edges = int(self.A_node / self.B_node)
        self.B = self.static_making_B_array()

        self.DB = 'MySQL'
        self.gap = 30
        simulation_condition = self.simulation_condition(self.gap)
        self.P = simulation_condition[0]
        self.V = simulation_condition[1]
        self.variable_list = self.p_and_v_list(self.P, self.V)
        self.workers = 1

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
