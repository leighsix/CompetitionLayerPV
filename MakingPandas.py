import pandas as pd
import numpy as np
import InterconnectedLayerModeling
import Setting_Simulation_Value


class MakingPandas:
    def making_dataframe_per_step(self, setting, value_array):
        columns = ['p', 'v', 'prob_v',
                   'A_plus', 'A_minus', 'B_plus', 'B_minus',
                   'Layer_A_Mean', 'Layer_B_Mean', 'AS',
                   'A_total_edges', 'B_total_edges', 'change_count']
        df = pd.DataFrame(value_array, columns=columns)
        step = [i for i in range(0, setting.Limited_step+1)]
        df['MODEL'] = setting.MODEL
        df['Steps'] = step
        df['Structure'] = setting.Structure
        df['A_node_number'] = setting.A_node
        df['B_node_number'] = setting.B_node
        df['A_external_edges'] = setting.A_inter_edges
        df['B_external_edges'] = setting.B_inter_edges
        return df


    def interacting_property(self, setting, inter_layer):
        property_A = []
        property_B = []
        for i in range(setting.A_node):
            property_A.append(inter_layer.two_layer_graph.nodes[i]['state'])
        for i in range(setting.A_node, setting.A_node + setting.B_node):
            property_B.append(inter_layer.two_layer_graph.nodes[i]['state'])
        judge_A = np.array(property_A)
        judge_B = np.array(property_B)

        A_plus = int(np.sum(judge_A > 0))
        A_minus = int(np.sum(judge_A < 0))
        B_plus = int(np.sum(judge_B > 0))
        B_minus = int(np.sum(judge_B < 0))
        layer_A_mean = int(np.sum(judge_A)) / setting.A_node
        layer_B_mean = int(np.sum(judge_B)) / setting.B_node
        average_state = ((layer_A_mean / setting.MAX) + layer_B_mean) / 2

        return A_plus, A_minus, B_plus, B_minus, layer_A_mean, layer_B_mean, average_state

if __name__ == "__main__":
    print("MakingPandas")
    setting = Setting_Simulation_Value.Setting_Simulation_Value()
    inter_layer = InterconnectedLayerModeling.InterconnectedLayerModeling(setting)
    mp = MakingPandas()
