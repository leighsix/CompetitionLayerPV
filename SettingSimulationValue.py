import numpy as np


class SettingSimulationValue:
    def __init__(self):
        self.database = 'pv_variable'  # 'competition  renew_competition'
        self.table = 'pv_variable3'
        self.Model = 'HM(2)'
        self.Structure = 'RR-RR'

        self.Limited_step = 100
        self.Repeating_number = 100
        self.A_node = 2048

        self.A_edge = 5
        self.A = SettingSimulationValue.static_making_A_array(self.A_node, A_state=[1, 2])
        self.MAX = 2
        self.MIN = -2

        self.B_node = 1024
        self.B_edge = 5
        self.B = SettingSimulationValue.static_making_B_array(self.B_node, B_state=[-1])
        self.workers = 5
        self.NodeColorDict = {1: 'orangered', 2: 'red', -1: 'royalblue', -2: 'blue'}
        self.EdgeColorDict = {1: 'yellowgreen', 2: 'hotpink', 4: 'red', -1: 'royalblue', -2: 'blue', -4: 'darkblue'}

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
    print(SS.A)
    #layer_A1 = Layer_A_Modeling.Layer_A_Modeling(SS)
    # print(SS.A_node)
    #print(len(layer_A1.A))
    #layer_A2 = Layer_A_Modeling.Layer_A_Modeling(SS)
    # print(SS.B_node)
    # print(SS.variable_list)
    #print(len(layer_A2.A))
