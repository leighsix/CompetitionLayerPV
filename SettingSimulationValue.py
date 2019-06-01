import numpy as np


class SettingSimulationValue:
    def __init__(self):
        self.database = 'pv_variable2'  # 'competition  renew_competition'
        self.table = 'total_variable'
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
        self.workers = 5

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
