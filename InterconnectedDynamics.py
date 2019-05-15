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

    def interconnected_dynamics0(self, setting, inter_layer, p, v, using_prob=False, node_i_names=None, sum_properties=0):  #same:same:same
        total_value = np.zeros(16)
        for step_number in range(setting.Limited_step+1):
            temp_inter_layer = inter_layer
            if step_number == 0:
                decision_prob = self.decision.B_state_change_probability_cal(setting, inter_layer, v)
                opinion_prob = self.opinion.A_state_change_probability_cal(inter_layer, p)
                initial_value = self.making_properties_array(setting, inter_layer, p, v, opinion_prob[1],
                                                             opinion_prob[2],  decision_prob[1], sum_properties)
                total_value = total_value + initial_value
            elif step_number >= 1:
                opinion_result = self.opinion.A_layer_simultaneous_dynamics(setting, temp_inter_layer, p, using_prob, node_i_names)
                decision_result = self.decision.B_layer_simultaneous_dynamics(setting, temp_inter_layer, v, node_i_names)
                for node_A in range(setting.A_node):
                    inter_layer.two_layer_graph.nodes[node_A]['state'] = opinion_result[0].two_layer_graph.nodes[node_A]['state']
                for node_B in range(setting.A_node, setting.A_node+setting.B_node):
                    inter_layer.two_layer_graph.nodes[node_B]['state'] = decision_result[0].two_layer_graph.nodes[node_B]['state']
                array_value = self.making_properties_array(setting, inter_layer, p, v, opinion_result[1],
                                                           opinion_result[2], decision_result[1], sum_properties)
                total_value = np.vstack([total_value, array_value])
        self.opinion.A_COUNT = 0
        self.decision.B_COUNT = 0
        return total_value

    def interconnected_dynamics1(self, setting, inter_layer, p, v, using_prob=False, node_i_names=None, sum_properties=0):  # step:step:opinion
        total_value = np.zeros(16)
        for step_number in range(setting.Limited_step+1):
            if step_number == 0:
                decision_prob = self.decision.B_state_change_probability_cal(setting, inter_layer, v)
                opinion_prob = self.opinion.A_state_change_probability_cal(inter_layer, p)
                initial_value = self.making_properties_array(setting, inter_layer, p, v, opinion_prob[1],
                                                             opinion_prob[2], decision_prob[1], sum_properties)
                total_value = total_value + initial_value
            elif step_number >= 1:
                opinion_result = self.opinion.A_layer_sequential_dynamics(setting, inter_layer, p, using_prob, node_i_names)
                decision_result = self.decision.B_layer_sequential_dynamics(setting, opinion_result[0], v, node_i_names)
                array_value = self.making_properties_array(setting, decision_result[0], p, v, opinion_result[1],
                                                           opinion_result[2], decision_result[1], sum_properties)
                total_value = np.vstack([total_value, array_value])
        self.opinion.A_COUNT = 0
        self.decision.B_COUNT = 0
        return total_value

    def interconnected_dynamics2(self, setting, inter_layer, p, v, using_prob=False, node_i_names=None, sum_properties=0):  # step:step:decision
        total_value = np.zeros(16)
        for step_number in range(setting.Limited_step+1):
            if step_number == 0:
                decision_prob = self.decision.B_state_change_probability_cal(setting, inter_layer, v)
                opinion_prob = self.opinion.A_state_change_probability_cal(inter_layer, p)
                initial_value = self.making_properties_array(setting, inter_layer, p, v, opinion_prob[1],
                                                             opinion_prob[2], decision_prob[1], sum_properties)
                total_value = total_value + initial_value
            elif step_number >= 1:
                decision_result = self.decision.B_layer_sequential_dynamics(setting, inter_layer, v, node_i_names)
                opinion_result = self.opinion.A_layer_sequential_dynamics(setting, decision_result[0], p, using_prob, node_i_names)
                array_value = self.making_properties_array(setting, opinion_result[0], p, v, opinion_result[1],
                                                           opinion_result[2], decision_result[1], sum_properties)
                total_value = np.vstack([total_value, array_value])
        self.opinion.A_COUNT = 0
        self.decision.B_COUNT = 0
        return total_value

    def interconnected_dynamics3(self, setting, inter_layer, p, v, using_prob=False, node_i_names=None, sum_properties=0):  # same:step:opinion
        total_value = np.zeros(16)
        for step_number in range(setting.Limited_step+1):
            if step_number == 0:
                decision_prob = self.decision.B_state_change_probability_cal(setting, inter_layer, v)
                opinion_prob = self.opinion.A_state_change_probability_cal(inter_layer, p)
                initial_value = self.making_properties_array(setting, inter_layer, p, v, opinion_prob[1],
                                                             opinion_prob[2], decision_prob[1], sum_properties)
                total_value = total_value + initial_value
            elif step_number >= 1:
                opinion_result = self.opinion.A_layer_simultaneous_dynamics(setting, inter_layer, p, using_prob, node_i_names)
                decision_result = self.decision.B_layer_sequential_dynamics(setting, opinion_result[0], v, node_i_names)
                array_value = self.making_properties_array(setting, decision_result[0], p, v, opinion_result[1],
                                                           opinion_result[2], decision_result[1], sum_properties)
                total_value = np.vstack([total_value, array_value])
        self.opinion.A_COUNT = 0
        self.decision.B_COUNT = 0
        return total_value

    def interconnected_dynamics4(self, setting, inter_layer, p, v, using_prob=False, node_i_names=None, sum_properties=0):  # same:step:decision
        total_value = np.zeros(16)
        for step_number in range(setting.Limited_step+1):
            if step_number == 0:
                decision_prob = self.decision.B_state_change_probability_cal(setting, inter_layer, v)
                opinion_prob = self.opinion.A_state_change_probability_cal(inter_layer, p)
                initial_value = self.making_properties_array(setting, inter_layer, p, v, opinion_prob[1],
                                                             opinion_prob[2], decision_prob[1], sum_properties)
                total_value = total_value + initial_value
            elif step_number >= 1:
                decision_result = self.decision.B_layer_sequential_dynamics(setting, inter_layer, v, node_i_names)
                opinion_result = self.opinion.A_layer_simultaneous_dynamics(setting, decision_result[0], p, using_prob, node_i_names)
                array_value = self.making_properties_array(setting, opinion_result[0], p, v, opinion_result[1],
                                                           opinion_result[2], decision_result[1], sum_properties)
                total_value = np.vstack([total_value, array_value])
        self.opinion.A_COUNT = 0
        self.decision.B_COUNT = 0
        return total_value

    def interconnected_dynamics5(self, setting, inter_layer, p, v, using_prob=False, node_i_names=None, sum_properties=0):  # step:same:opinion
        total_value = np.zeros(16)
        for step_number in range(setting.Limited_step+1):
            if step_number == 0:
                decision_prob = self.decision.B_state_change_probability_cal(setting, inter_layer, v)
                opinion_prob = self.opinion.A_state_change_probability_cal(inter_layer, p)
                initial_value = self.making_properties_array(setting, inter_layer, p, v, opinion_prob[1],
                                                             opinion_prob[2], decision_prob[1], sum_properties)
                total_value = total_value + initial_value
            elif step_number >= 1:
                opinion_result = self.opinion.A_layer_sequential_dynamics(setting, inter_layer, p, using_prob, node_i_names)
                decision_result = self.decision.B_layer_simultaneous_dynamics(setting, opinion_result[0], v, node_i_names)
                array_value = self.making_properties_array(setting, decision_result[0], p, v, opinion_result[1],
                                                           opinion_result[2], decision_result[1], sum_properties)
                total_value = np.vstack([total_value, array_value])
        self.opinion.A_COUNT = 0
        self.decision.B_COUNT = 0
        return total_value

    def interconnected_dynamics6(self, setting, inter_layer, p, v, using_prob=False, node_i_names=None, sum_properties=0):  # step:same:decision
        total_value = np.zeros(16)
        for step_number in range(setting.Limited_step+1):
            if step_number == 0:
                decision_prob = self.decision.B_state_change_probability_cal(setting, inter_layer, v)
                opinion_prob = self.opinion.A_state_change_probability_cal(inter_layer, p)
                initial_value = self.making_properties_array(setting, inter_layer, p, v, opinion_prob[1],
                                                             opinion_prob[2], decision_prob[1], sum_properties)
                total_value = total_value + initial_value
            elif step_number >= 1:
                decision_result = self.decision.B_layer_simultaneous_dynamics(setting, inter_layer, v, node_i_names)
                opinion_result = self.opinion.A_layer_sequential_dynamics(setting, decision_result[0], p, using_prob, node_i_names)
                array_value = self.making_properties_array(setting, decision_result[0], p, v, opinion_result[1],
                                                           opinion_result[2], decision_result[1], sum_properties)
                total_value = np.vstack([total_value, array_value])
        self.opinion.A_COUNT = 0
        self.decision.B_COUNT = 0
        return total_value

    def interconnected_dynamics7(self, setting, inter_layer, p, v, using_prob=False, node_i_names=None, sum_properties=0):  # same:same:opinion
        total_value = np.zeros(16)
        for step_number in range(setting.Limited_step+1):
            if step_number == 0:
                decision_prob = self.decision.B_state_change_probability_cal(setting, inter_layer, v)
                opinion_prob = self.opinion.A_state_change_probability_cal(inter_layer, p)
                initial_value = self.making_properties_array(setting, inter_layer, p, v, opinion_prob[1],
                                                             opinion_prob[2], decision_prob[1], sum_properties)
                total_value = total_value + initial_value
            elif step_number >= 1:
                opinion_result = self.opinion.A_layer_simultaneous_dynamics(setting, inter_layer, p, using_prob, node_i_names)
                decision_result = self.decision.B_layer_simultaneous_dynamics(setting, opinion_result[0], v, node_i_names)
                array_value = self.making_properties_array(setting, decision_result[0], p, v, opinion_result[1],
                                                           opinion_result[2], decision_result[1], sum_properties)
                total_value = np.vstack([total_value, array_value])
        self.opinion.A_COUNT = 0
        self.decision.B_COUNT = 0
        return total_value

    def interconnected_dynamics8(self, setting, inter_layer, p, v, using_prob=False, node_i_names=None, sum_properties=0):  # same:same:decision
        total_value = np.zeros(16)
        for step_number in range(setting.Limited_step+1):
            if step_number == 0:
                decision_prob = self.decision.B_state_change_probability_cal(setting, inter_layer, v)
                opinion_prob = self.opinion.A_state_change_probability_cal(inter_layer, p)
                initial_value = self.making_properties_array(setting, inter_layer, p, v, opinion_prob[1],
                                                             opinion_prob[2], decision_prob[1], sum_properties)
                total_value = total_value + initial_value
            elif step_number >= 1:
                decision_result = self.decision.B_layer_simultaneous_dynamics(setting, inter_layer, v, node_i_names)
                opinion_result = self.opinion.A_layer_simultaneous_dynamics(setting, decision_result[0], p, using_prob, node_i_names)
                array_value = self.making_properties_array(setting, decision_result[0], p, v, opinion_result[1],
                                                           opinion_result[2], decision_result[1], sum_properties)
                total_value = np.vstack([total_value, array_value])
        self.opinion.A_COUNT = 0
        self.decision.B_COUNT = 0
        return total_value

    def interconnected_dynamics9(self, setting, inter_layer, p, v, using_prob=False, node_i_names=None, sum_properties=0):  # step:step:opinion-decision
        total_value = np.zeros(16)
        for step_number in range(setting.Limited_step+1):
            if step_number == 0:
                decision_prob = self.decision.B_state_change_probability_cal(setting, inter_layer, v)
                opinion_prob = self.opinion.A_state_change_probability_cal(inter_layer, p)
                initial_value = self.making_properties_array(setting, inter_layer, p, v, opinion_prob[1],
                                                             opinion_prob[2], decision_prob[1], sum_properties)
                total_value = total_value + initial_value
            elif step_number >= 1:
                AB_dynamics_result = self.opinion.AB_layer_sequential_dynamics(setting, inter_layer, p, v, using_prob, node_i_names)
                array_value = self.making_properties_array(setting, AB_dynamics_result[0], p, v, AB_dynamics_result[1],
                                                           AB_dynamics_result[2], AB_dynamics_result[3], sum_properties)
                total_value = np.vstack([total_value, array_value])
        self.opinion.A_COUNT = 0
        self.decision.B_COUNT = 0
        return total_value


    def making_properties_array(self, setting, inter_layer, p, v, persuasion_prob, compromise_prob, volatility_prob, sum_properties):
        interacting_properties = self.mp.interacting_property(setting, inter_layer)
        change_count = self.opinion.A_COUNT + self.decision.B_COUNT
        array_value = np.array([p, v, volatility_prob, persuasion_prob, compromise_prob,
                                interacting_properties[0], interacting_properties[1],
                                interacting_properties[2], interacting_properties[3],
                                interacting_properties[4], interacting_properties[5],
                                interacting_properties[6],
                                len(sorted(inter_layer.A_edges.edges)), len(inter_layer.B_edges),
                                change_count, sum_properties])
        return array_value

if __name__ == "__main__":
    print("InterconnectedDynamics")
    start = time.time()
    setting = Setting_Simulation_Value.Setting_Simulation_Value()
    inter_layer = InterconnectedLayerModeling.InterconnectedLayerModeling(setting)
    p = 0.2
    v = 0.3
    state = 0
    for i in range(setting.A_node):
        state += inter_layer.two_layer_graph.nodes[i]['state']
    print(state)
    inter_dynamics = InterconnectedDynamics()
    array = inter_dynamics.interconnected_dynamics1(setting, inter_layer, p, v, node_i_names={'A_0', 'A_1'})
    print(array)
    state = 0
    for i in range(setting.A_node):
        state += inter_layer.two_layer_graph.nodes[i]['state']
    print(state)
    end = time.time()
    print(end-start)











