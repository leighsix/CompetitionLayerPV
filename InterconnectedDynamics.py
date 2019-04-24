import numpy as np
import Setting_Simulation_Value
import OpinionDynamics
import DecisionDynamics
import MakingPandas
import InterconnectedLayerModeling
import matplotlib
import time
matplotlib.use("Agg")

class InterconnectedDynamics:
    def __init__(self):
        self.opinion = OpinionDynamics.OpinionDynamics()
        self.decision = DecisionDynamics.DecisionDynamics()
        self.mp = MakingPandas.MakingPandas()

    def interconnected_dynamics0(self, setting, inter_layer, p, v):  #same:same:same
        total_value = np.zeros(13)
        for step_number in range(setting.Limited_step+1):
            temp_inter_layer = inter_layer
            if step_number == 0:
                probability = self.decision.B_state_change_probability_cal(setting, inter_layer, v)
                initial_value = self.making_properties_array(setting, inter_layer, p, v, probability)
                total_value = total_value + initial_value
            elif step_number >= 1:
                opinion_result = self.opinion.A_layer_simultaneous_dynamics1(setting, temp_inter_layer, p)
                probability = self.decision.B_state_change_probability_cal(setting, temp_inter_layer, v)
                decision_result = self.decision.B_layer_simultaneous_dynamics(setting, temp_inter_layer, probability)
                for node_A in range(setting.A_node):
                    inter_layer.two_layer_graph.nodes[node_A]['state'] = opinion_result.two_layer_graph.nodes[node_A]['state']
                for node_B in range(setting.A_node, setting.A_node+setting.B_node):
                    inter_layer.two_layer_graph.nodes[node_B]['state'] = decision_result.two_layer_graph.nodes[node_B]['state']
                array_value = self.making_properties_array(setting, inter_layer, p, v, probability)
                total_value = np.vstack([total_value, array_value])
        self.opinion.A_COUNT = 0
        self.decision.B_COUNT = 0
        return total_value

    def interconnected_dynamics1(self, setting, inter_layer, p, v):  # step:step:opinion
        total_value = np.zeros(13)
        for step_number in range(setting.Limited_step+1):
            if step_number == 0:
                probability = self.decision.B_state_change_probability_cal(setting, inter_layer, v)
                initial_value = self.making_properties_array(setting, inter_layer, p, v, probability)
                total_value = total_value + initial_value
            elif step_number >= 1:
                inter_layer = self.opinion.A_layer_dynamics1(setting, inter_layer, p)
                decision_result = self.decision.B_layer_dynamics(setting, inter_layer, v)
                inter_layer = decision_result[0]
                probability = decision_result[1]
                array_value = self.making_properties_array(setting, inter_layer, p, v, probability)
                total_value = np.vstack([total_value, array_value])
        self.opinion.A_COUNT = 0
        self.decision.B_COUNT = 0
        return total_value

    def interconnected_dynamics2(self, setting, inter_layer, p, v):  # step:step:decision
        total_value = np.zeros(13)
        for step_number in range(setting.Limited_step+1):
            if step_number == 0:
                probability = self.decision.B_state_change_probability_cal(setting, inter_layer, v)
                initial_value = self.making_properties_array(setting, inter_layer, p, v, probability)
                total_value = total_value + initial_value
            elif step_number >= 1:
                decision_result = self.decision.B_layer_dynamics(setting, inter_layer, v)
                inter_layer = decision_result[0]
                probability = decision_result[1]
                inter_layer = self.opinion.A_layer_dynamics1(setting, inter_layer, p)
                array_value = self.making_properties_array(setting, inter_layer, p, v, probability)
                total_value = np.vstack([total_value, array_value])
        self.opinion.A_COUNT = 0
        self.decision.B_COUNT = 0
        return total_value

    def interconnected_dynamics3(self, setting, inter_layer, p, v):  # same:step:opinion
        total_value = np.zeros(13)
        for step_number in range(setting.Limited_step+1):
            if step_number == 0:
                probability = self.decision.B_state_change_probability_cal(setting, inter_layer, v)
                initial_value = self.making_properties_array(setting, inter_layer, p, v, probability)
                total_value = total_value + initial_value
            elif step_number >= 1:
                inter_layer = self.opinion.A_layer_simultaneous_dynamics1(setting, inter_layer, p)
                decision_result = self.decision.B_layer_dynamics(setting, inter_layer, v)
                inter_layer = decision_result[0]
                probability = decision_result[1]
                array_value = self.making_properties_array(setting, inter_layer, p, v, probability)
                total_value = np.vstack([total_value, array_value])
        self.opinion.A_COUNT = 0
        self.decision.B_COUNT = 0
        return total_value

    def interconnected_dynamics4(self, setting, inter_layer, p, v):  # same:step:decision
        total_value = np.zeros(13)
        for step_number in range(setting.Limited_step+1):
            if step_number == 0:
                probability = self.decision.B_state_change_probability_cal(setting, inter_layer, v)
                initial_value = self.making_properties_array(setting, inter_layer, p, v, probability)
                total_value = total_value + initial_value
            elif step_number >= 1:
                decision_result = self.decision.B_layer_dynamics(setting, inter_layer, v)
                inter_layer = decision_result[0]
                probability = decision_result[1]
                inter_layer = self.opinion.A_layer_simultaneous_dynamics1(setting, inter_layer, p)
                array_value = self.making_properties_array(setting, inter_layer, p, v, probability)
                total_value = np.vstack([total_value, array_value])
        self.opinion.A_COUNT = 0
        self.decision.B_COUNT = 0
        return total_value

    def interconnected_dynamics5(self, setting, inter_layer, p, v):  # step:same:opinion
        total_value = np.zeros(13)
        for step_number in range(setting.Limited_step+1):
            if step_number == 0:
                probability = self.decision.B_state_change_probability_cal(setting, inter_layer, v)
                initial_value = self.making_properties_array(setting, inter_layer, p, v, probability)
                total_value = total_value + initial_value
            elif step_number >= 1:
                inter_layer = self.opinion.A_layer_dynamics1(setting, inter_layer, p)
                probability = self.decision.B_state_change_probability_cal(setting, inter_layer, v)
                inter_layer = self.decision.B_layer_simultaneous_dynamics(setting, inter_layer, probability)
                array_value = self.making_properties_array(setting, inter_layer, p, v, probability)
                total_value = np.vstack([total_value, array_value])
        self.opinion.A_COUNT = 0
        self.decision.B_COUNT = 0
        return total_value

    def interconnected_dynamics6(self, setting, inter_layer, p, v):  # step:same:decision
        total_value = np.zeros(13)
        for step_number in range(setting.Limited_step+1):
            if step_number == 0:
                probability = self.decision.B_state_change_probability_cal(setting, inter_layer, v)
                initial_value = self.making_properties_array(setting, inter_layer, p, v, probability)
                total_value = total_value + initial_value
            elif step_number >= 1:
                probability = self.decision.B_state_change_probability_cal(setting, inter_layer, v)
                inter_layer = self.decision.B_layer_simultaneous_dynamics(setting, inter_layer, probability)
                inter_layer = self.opinion.A_layer_dynamics1(setting, inter_layer, p)
                array_value = self.making_properties_array(setting, inter_layer, p, v, probability)
                total_value = np.vstack([total_value, array_value])
        self.opinion.A_COUNT = 0
        self.decision.B_COUNT = 0
        return total_value

    def interconnected_dynamics7(self, setting, inter_layer, p, v):  # same:same:opinion
        total_value = np.zeros(13)
        for step_number in range(setting.Limited_step+1):
            if step_number == 0:
                probability = self.decision.B_state_change_probability_cal(setting, inter_layer, v)
                initial_value = self.making_properties_array(setting, inter_layer, p, v, probability)
                total_value = total_value + initial_value
            elif step_number >= 1:
                inter_layer = self.opinion.A_layer_simultaneous_dynamics1(setting, inter_layer, p)
                probability = self.decision.B_state_change_probability_cal(setting, inter_layer, v)
                inter_layer = self.decision.B_layer_simultaneous_dynamics(setting, inter_layer, probability)
                array_value = self.making_properties_array(setting, inter_layer, p, v, probability)
                total_value = np.vstack([total_value, array_value])
        self.opinion.A_COUNT = 0
        self.decision.B_COUNT = 0
        return total_value

    def interconnected_dynamics8(self, setting, inter_layer, p, v):  # same:same:decision
        total_value = np.zeros(13)
        for step_number in range(setting.Limited_step+1):
            if step_number == 0:
                probability = self.decision.B_state_change_probability_cal(setting, inter_layer, v)
                initial_value = self.making_properties_array(setting, inter_layer, p, v, probability)
                total_value = total_value + initial_value
            elif step_number >= 1:
                probability = self.decision.B_state_change_probability_cal(setting, inter_layer, v)
                inter_layer = self.decision.B_layer_simultaneous_dynamics(setting, inter_layer, probability)
                inter_layer = self.opinion.A_layer_simultaneous_dynamics1(setting, inter_layer, p)
                array_value = self.making_properties_array(setting, inter_layer, p, v, probability)
                total_value = np.vstack([total_value, array_value])
        self.opinion.A_COUNT = 0
        self.decision.B_COUNT = 0
        return total_value

    def interconnected_dynamics9(self, setting, inter_layer, p, v):  # step:step:opinion-decision
        total_value = np.zeros(13)
        for step_number in range(setting.Limited_step+1):
            if step_number == 0:
                probability = self.decision.B_state_change_probability_cal(setting, inter_layer, v)
                initial_value = self.making_properties_array(setting, inter_layer, p, v, probability)
                total_value = total_value + initial_value
            elif step_number >= 1:
                for i, j in sorted(inter_layer.A_edges.edges()):

                for node_i in range(setting.A_node, setting.A_node + setting.B_node):

        #             probability = self.decision.B_state_change_probability_cal(setting, inter_layer, v)
    #             inter_layer = self.decision.B_layer_dynamics(setting, inter_layer, probability)
    #             inter_layer = self.opinion.A_layer_dynamics(setting, inter_layer, p)
    #             array_value = self.making_properties_array(setting, inter_layer, p, v, probability)
    #             total_value = np.vstack([total_value, array_value])
        self.opinion.A_COUNT = 0
        self.decision.B_COUNT = 0
        return total_value


    def making_properties_array(self, setting, inter_layer, p, v, probability):
        prob_v_mean = np.sum(probability) / len(probability)
        interacting_properties = self.mp.interacting_property(setting, inter_layer)
        change_count = self.opinion.A_COUNT + self.decision.B_COUNT
        array_value = np.array([p, v, prob_v_mean,
                                interacting_properties[0], interacting_properties[1],
                                interacting_properties[2], interacting_properties[3],
                                interacting_properties[4], interacting_properties[5],
                                interacting_properties[6],
                                len(sorted(inter_layer.A_edges.edges)), len(inter_layer.B_edges),
                                change_count])
        return array_value

if __name__ == "__main__":
    print("InterconnectedDynamics")
    start = time.time()
    setting = Setting_Simulation_Value.Setting_Simulation_Value()
    inter_layer = InterconnectedLayerModeling.InterconnectedLayerModeling(setting)
    p = 0.5
    v = 0.5
    state = 0
    for i in range(setting.A_node):
        state += inter_layer.two_layer_graph.nodes[i]['state']
    print(state)
    inter_dynamics = InterconnectedDynamics()
    array = inter_dynamics.interconnected_dynamics1(setting, inter_layer, p, v)
    print(array[0:3])
    state = 0
    for i in range(setting.A_node):
        state += inter_layer.two_layer_graph.nodes[i]['state']
    print(state)
    end = time.time()
    print(end-start)











