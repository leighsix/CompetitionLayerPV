import numpy as np
import SettingSimulationValue
import OpinionDynamics
import DecisionDynamics
import InterconnectedLayerModeling
import time
import math
import networkx as nx
import copy

class InterconnectedDynamics:
    def __init__(self, setting, inter_layer, p, v, using_prob=False, select_step=1, unchanged_nodes=None,
                 sum_properties=0, edges_properties=0):
        self.dynamics_result_array = InterconnectedDynamics.interconnected_dynamics(setting, inter_layer, p, v, using_prob,
                                                                                    select_step, unchanged_nodes,
                                                                                    sum_properties, edges_properties)

    @staticmethod
    def interconnected_dynamics(setting, inter_layer, p, v, using_prob, select_step, unchanged_nodes, sum_properties, edges_properties):
        total_array = np.zeros([setting.Limited_step + 1, 17])
        if select_step == 0:  # 'O(s)<->D(s)'
            total_array = InterconnectedDynamics.interconnected_dynamics0(setting, inter_layer, p, v, using_prob, unchanged_nodes, sum_properties, edges_properties)
        elif select_step == 1:  # 'O(o)->D(o)'
            total_array = InterconnectedDynamics.interconnected_dynamics1(setting, inter_layer, p, v, using_prob, unchanged_nodes, sum_properties, edges_properties)
        elif select_step == 2:  # 'O(o)<-D(o)'
            total_array = InterconnectedDynamics.interconnected_dynamics2(setting, inter_layer, p, v, using_prob, unchanged_nodes, sum_properties, edges_properties)
        elif select_step == 3:  # 'O(s)->D(o)'
            total_array = InterconnectedDynamics.interconnected_dynamics3(setting, inter_layer, p, v, using_prob, unchanged_nodes, sum_properties, edges_properties)
        elif select_step == 4:  # 'O(s)<-D(o)'
            total_array = InterconnectedDynamics.interconnected_dynamics4(setting, inter_layer, p, v, using_prob, unchanged_nodes, sum_properties, edges_properties)
        elif select_step == 5:  # 'O(o)->D(s)'
            total_array = InterconnectedDynamics.interconnected_dynamics5(setting, inter_layer, p, v, using_prob, unchanged_nodes, sum_properties, edges_properties)
        elif select_step == 6:  # 'O(o)<-D(s)'
            total_array = InterconnectedDynamics.interconnected_dynamics6(setting, inter_layer, p, v, using_prob, unchanged_nodes, sum_properties, edges_properties)
        elif select_step == 7:  # 'O(s)->D(s)'
            total_array = InterconnectedDynamics.interconnected_dynamics7(setting, inter_layer, p, v, using_prob, unchanged_nodes, sum_properties, edges_properties)
        elif select_step == 8:  # 'O(s)<-D(s)'
            total_array = InterconnectedDynamics.interconnected_dynamics8(setting, inter_layer, p, v, using_prob, unchanged_nodes, sum_properties, edges_properties)
        elif select_step == 9:  # 'O(o)<=>D(o)'
            total_array = InterconnectedDynamics.interconnected_dynamics9(setting, inter_layer, p, v, using_prob, unchanged_nodes, sum_properties, edges_properties)
        elif select_step == 10:  # 'O(r)->D(o)'
            total_array = InterconnectedDynamics.interconnected_dynamics10(setting, inter_layer, p, v, using_prob, unchanged_nodes, sum_properties, edges_properties)
        elif select_step == 11:  # 'O(r)<-D(o)'
            total_array = InterconnectedDynamics.interconnected_dynamics11(setting, inter_layer, p, v, using_prob, unchanged_nodes, sum_properties, edges_properties)
        elif select_step == 12:  # 'O(r)->D(s)'
            total_array = InterconnectedDynamics.interconnected_dynamics12(setting, inter_layer, p, v, using_prob, unchanged_nodes, sum_properties, edges_properties)
        elif select_step == 13:  # 'O(r)<-D(s)'
            total_array = InterconnectedDynamics.interconnected_dynamics13(setting, inter_layer, p, v, using_prob, unchanged_nodes, sum_properties, edges_properties)
        elif select_step == 14:  # 'O(r)<=>D(r)'
            total_array = InterconnectedDynamics.interconnected_dynamics14(setting, inter_layer, p, v, using_prob, unchanged_nodes, sum_properties, edges_properties)
        return total_array

    @staticmethod
    def interconnected_dynamics0(setting, inter_layer, p, v, using_prob, unchanged_nodes, sum_properties, edges_properties):
        #same:same:same
        total_value = np.zeros(17)
        change_count = 0
        for step_number in range(setting.Limited_step+1):
            temp_inter_layer1 = copy.deepcopy(inter_layer)
            temp_inter_layer2 = copy.deepcopy(inter_layer)
            if step_number == 0:
                opinion_prob = InterconnectedDynamics.A_state_change_probability_cal(inter_layer, p)
                decision_prob = InterconnectedDynamics.B_state_change_probability_cal(inter_layer, v)
                initial_value = InterconnectedDynamics.making_properties_array(setting, inter_layer, p, v, opinion_prob[1],
                                                                               opinion_prob[2],  decision_prob[1],
                                                                               change_count, sum_properties, edges_properties)
                total_value = total_value + initial_value
            elif step_number >= 1:
                opinion_result = OpinionDynamics.OpinionDynamics(setting, temp_inter_layer1, p, v, 1, using_prob, unchanged_nodes)
                decision_result = DecisionDynamics.DecisionDynamics(setting, temp_inter_layer2, v, 1, unchanged_nodes)
                change_count += opinion_result.A_COUNT + decision_result.B_COUNT
                for node_A in inter_layer.A_nodes:
                    inter_layer.two_layer_graph.nodes[node_A]['state'] = opinion_result.A_inter_layer.two_layer_graph.nodes[node_A]['state']
                for node_B in inter_layer.B_nodes:
                    inter_layer.two_layer_graph.nodes[node_B]['state'] = decision_result.B_inter_layer.two_layer_graph.nodes[node_B]['state']
                array_value = InterconnectedDynamics.making_properties_array(setting, inter_layer, p, v,
                                                                             opinion_result.persuasion_prob,
                                                                             opinion_result.compromise_prob,
                                                                             decision_result.volatility_prob,
                                                                             change_count, sum_properties, edges_properties)
                total_value = np.vstack([total_value, array_value])
        return total_value

    @staticmethod
    def interconnected_dynamics1(setting, inter_layer, p, v, using_prob, unchanged_nodes, sum_properties, edges_properties):
        # step:step:opinion
        total_value = np.zeros(17)
        change_count = 0
        for step_number in range(setting.Limited_step+1):
            if step_number == 0:
                opinion_prob = InterconnectedDynamics.A_state_change_probability_cal(inter_layer, p)
                decision_prob = InterconnectedDynamics.B_state_change_probability_cal(inter_layer, v)
                initial_value = InterconnectedDynamics.making_properties_array(setting, inter_layer, p, v, opinion_prob[1],
                                                                               opinion_prob[2], decision_prob[1],
                                                                               change_count, sum_properties, edges_properties)
                total_value = total_value + initial_value
            elif step_number >= 1:
                opinion_result = OpinionDynamics.OpinionDynamics(setting, inter_layer, p, v, 0, using_prob, unchanged_nodes)
                decision_result = DecisionDynamics.DecisionDynamics(setting, opinion_result.A_inter_layer, v, 0, unchanged_nodes)
                change_count += opinion_result.A_COUNT + decision_result.B_COUNT
                array_value = InterconnectedDynamics.making_properties_array(setting, decision_result.B_inter_layer, p, v,
                                                                             opinion_result.persuasion_prob,
                                                                             opinion_result.compromise_prob,
                                                                             decision_result.volatility_prob,
                                                                             change_count, sum_properties, edges_properties)
                total_value = np.vstack([total_value, array_value])
        return total_value

    @staticmethod
    def interconnected_dynamics2(setting, inter_layer, p, v, using_prob, unchanged_nodes, sum_properties, edges_properties):
        # step:step:decision
        total_value = np.zeros(17)
        change_count = 0
        for step_number in range(setting.Limited_step+1):
            if step_number == 0:
                opinion_prob = InterconnectedDynamics.A_state_change_probability_cal(inter_layer, p)
                decision_prob = InterconnectedDynamics.B_state_change_probability_cal(inter_layer, v)
                initial_value = InterconnectedDynamics.making_properties_array(setting, inter_layer, p, v, opinion_prob[1],
                                                                               opinion_prob[2], decision_prob[1],
                                                                               change_count, sum_properties, edges_properties)
                total_value = total_value + initial_value
            elif step_number >= 1:
                decision_result = DecisionDynamics.DecisionDynamics(setting, inter_layer, v, 0, unchanged_nodes)
                opinion_result = OpinionDynamics.OpinionDynamics(setting, decision_result.B_inter_layer, p, v, 0,
                                                                 using_prob, unchanged_nodes)
                change_count += opinion_result.A_COUNT + decision_result.B_COUNT
                array_value = InterconnectedDynamics.making_properties_array(setting, opinion_result.A_inter_layer, p, v,
                                                                             opinion_result.persuasion_prob,
                                                                             opinion_result.compromise_prob,
                                                                             decision_result.volatility_prob,
                                                                             change_count, sum_properties, edges_properties)
                total_value = np.vstack([total_value, array_value])
        return total_value

    @staticmethod
    def interconnected_dynamics3(setting, inter_layer, p, v, using_prob, unchanged_nodes, sum_properties, edges_properties):
        # same:step:opinion
        total_value = np.zeros(17)
        change_count = 0
        for step_number in range(setting.Limited_step+1):
            if step_number == 0:
                opinion_prob = InterconnectedDynamics.A_state_change_probability_cal(inter_layer, p)
                decision_prob = InterconnectedDynamics.B_state_change_probability_cal(inter_layer, v)
                initial_value = InterconnectedDynamics.making_properties_array(setting, inter_layer, p, v, opinion_prob[1],
                                                                               opinion_prob[2], decision_prob[1],
                                                                               change_count, sum_properties, edges_properties)
                total_value = total_value + initial_value
            elif step_number >= 1:
                opinion_result = OpinionDynamics.OpinionDynamics(setting, inter_layer, p, v, 1, using_prob, unchanged_nodes)
                decision_result = DecisionDynamics.DecisionDynamics(setting, opinion_result.A_inter_layer, v, 0, unchanged_nodes)
                change_count += opinion_result.A_COUNT + decision_result.B_COUNT
                array_value = InterconnectedDynamics.making_properties_array(setting, decision_result.B_inter_layer, p, v,
                                                                             opinion_result.persuasion_prob,
                                                                             opinion_result.compromise_prob,
                                                                             decision_result.volatility_prob,
                                                                             change_count, sum_properties, edges_properties)
                total_value = np.vstack([total_value, array_value])
        return total_value

    @staticmethod
    def interconnected_dynamics4(setting, inter_layer, p, v, using_prob, unchanged_nodes, sum_properties, edges_properties):
        # same:step:decision
        total_value = np.zeros(17)
        change_count = 0
        for step_number in range(setting.Limited_step+1):
            if step_number == 0:
                opinion_prob = InterconnectedDynamics.A_state_change_probability_cal(inter_layer, p)
                decision_prob = InterconnectedDynamics.B_state_change_probability_cal(inter_layer, v)
                initial_value = InterconnectedDynamics.making_properties_array(setting, inter_layer, p, v, opinion_prob[1],
                                                                               opinion_prob[2], decision_prob[1],
                                                                               change_count, sum_properties, edges_properties)
                total_value = total_value + initial_value
            elif step_number >= 1:
                decision_result = DecisionDynamics.DecisionDynamics(setting, inter_layer, v, 0, unchanged_nodes)
                opinion_result = OpinionDynamics.OpinionDynamics(setting, decision_result.B_inter_layer, p, v, 1,
                                                                 using_prob, unchanged_nodes)
                change_count += opinion_result.A_COUNT + decision_result.B_COUNT
                array_value = InterconnectedDynamics.making_properties_array(setting, opinion_result.A_inter_layer, p, v,
                                                                             opinion_result.persuasion_prob,
                                                                             opinion_result.compromise_prob,
                                                                             decision_result.volatility_prob,
                                                                             change_count, sum_properties, edges_properties)
                total_value = np.vstack([total_value, array_value])
        return total_value

    @staticmethod
    def interconnected_dynamics5(setting, inter_layer, p, v, using_prob, unchanged_nodes, sum_properties, edges_properties):
        # step:same:opinion
        total_value = np.zeros(17)
        change_count = 0
        for step_number in range(setting.Limited_step+1):
            if step_number == 0:
                opinion_prob = InterconnectedDynamics.A_state_change_probability_cal(inter_layer, p)
                decision_prob = InterconnectedDynamics.B_state_change_probability_cal(inter_layer, v)
                initial_value = InterconnectedDynamics.making_properties_array(setting, inter_layer, p, v, opinion_prob[1],
                                                                               opinion_prob[2], decision_prob[1],
                                                                               change_count, sum_properties, edges_properties)
                total_value = total_value + initial_value
            elif step_number >= 1:
                opinion_result = OpinionDynamics.OpinionDynamics(setting, inter_layer, p, v, 0, using_prob, unchanged_nodes)
                decision_result = DecisionDynamics.DecisionDynamics(setting, opinion_result.A_inter_layer, v, 1, unchanged_nodes)
                change_count += opinion_result.A_COUNT + decision_result.B_COUNT
                array_value = InterconnectedDynamics.making_properties_array(setting, decision_result.B_inter_layer, p, v,
                                                                             opinion_result.persuasion_prob,
                                                                             opinion_result.compromise_prob,
                                                                             decision_result.volatility_prob,
                                                                             change_count, sum_properties, edges_properties)
                total_value = np.vstack([total_value, array_value])
        return total_value

    @staticmethod
    def interconnected_dynamics6(setting, inter_layer, p, v, using_prob, unchanged_nodes, sum_properties, edges_properties):
        # step:same:decision
        total_value = np.zeros(17)
        change_count = 0
        for step_number in range(setting.Limited_step+1):
            if step_number == 0:
                opinion_prob = InterconnectedDynamics.A_state_change_probability_cal(inter_layer, p)
                decision_prob = InterconnectedDynamics.B_state_change_probability_cal(inter_layer, v)
                initial_value = InterconnectedDynamics.making_properties_array(setting, inter_layer, p, v, opinion_prob[1],
                                                                               opinion_prob[2], decision_prob[1],
                                                                               change_count, sum_properties, edges_properties)
                total_value = total_value + initial_value
            elif step_number >= 1:
                decision_result = DecisionDynamics.DecisionDynamics(setting, inter_layer, v, 1, unchanged_nodes)
                opinion_result = OpinionDynamics.OpinionDynamics(setting, decision_result.B_inter_layer, p, v, 0, using_prob, unchanged_nodes)
                change_count += opinion_result.A_COUNT + decision_result.B_COUNT
                array_value = InterconnectedDynamics.making_properties_array(setting, opinion_result.A_inter_layer, p, v,
                                                                             opinion_result.persuasion_prob,
                                                                             opinion_result.compromise_prob,
                                                                             decision_result.volatility_prob,
                                                                             change_count, sum_properties, edges_properties)
                total_value = np.vstack([total_value, array_value])
        return total_value

    @staticmethod
    def interconnected_dynamics7(setting, inter_layer, p, v, using_prob, unchanged_nodes, sum_properties, edges_properties):
        # same:same:opinion
        total_value = np.zeros(17)
        change_count = 0
        for step_number in range(setting.Limited_step+1):
            if step_number == 0:
                opinion_prob = InterconnectedDynamics.A_state_change_probability_cal(inter_layer, p)
                decision_prob = InterconnectedDynamics.B_state_change_probability_cal(inter_layer, v)
                initial_value = InterconnectedDynamics.making_properties_array(setting, inter_layer, p, v, opinion_prob[1],
                                                                               opinion_prob[2], decision_prob[1],
                                                                               change_count, sum_properties, edges_properties)
                total_value = total_value + initial_value
            elif step_number >= 1:
                opinion_result = OpinionDynamics.OpinionDynamics(setting, inter_layer, p, v, 1, using_prob, unchanged_nodes)
                decision_result = DecisionDynamics.DecisionDynamics(setting, opinion_result.A_inter_layer, v, 1, unchanged_nodes)
                change_count += opinion_result.A_COUNT + decision_result.B_COUNT
                array_value = InterconnectedDynamics.making_properties_array(setting, decision_result.B_inter_layer, p, v,
                                                                             opinion_result.persuasion_prob,
                                                                             opinion_result.compromise_prob,
                                                                             decision_result.volatility_prob,
                                                                             change_count, sum_properties, edges_properties)
                total_value = np.vstack([total_value, array_value])
        return total_value

    @staticmethod
    def interconnected_dynamics8(setting, inter_layer, p, v, using_prob, unchanged_nodes, sum_properties, edges_properties):
        # same:same:decision
        total_value = np.zeros(17)
        change_count = 0

        for step_number in range(setting.Limited_step+1):
            if step_number == 0:
                opinion_prob = InterconnectedDynamics.A_state_change_probability_cal(inter_layer, p)
                decision_prob = InterconnectedDynamics.B_state_change_probability_cal(inter_layer, v)
                initial_value = InterconnectedDynamics.making_properties_array(setting, inter_layer, p, v, opinion_prob[1],
                                                                               opinion_prob[2], decision_prob[1],
                                                                               change_count, sum_properties, edges_properties)
                total_value = total_value + initial_value
            elif step_number >= 1:
                decision_result = DecisionDynamics.DecisionDynamics(setting, inter_layer, v, 1, unchanged_nodes)
                opinion_result = OpinionDynamics.OpinionDynamics(setting, decision_result.B_inter_layer, p, v, 1, using_prob, unchanged_nodes)
                change_count += opinion_result.A_COUNT + decision_result.B_COUNT
                array_value = InterconnectedDynamics.making_properties_array(setting, opinion_result.A_inter_layer, p, v,
                                                                             opinion_result.persuasion_prob,
                                                                             opinion_result.compromise_prob,
                                                                             decision_result.volatility_prob,
                                                                             change_count, sum_properties, edges_properties)
                total_value = np.vstack([total_value, array_value])
        return total_value

    @staticmethod
    def interconnected_dynamics9(setting, inter_layer, p, v, using_prob, unchanged_nodes, sum_properties, edges_properties):
        # step:step:opinion-decision
        total_value = np.zeros(17)
        change_count = 0
        for step_number in range(setting.Limited_step+1):
            if step_number == 0:
                opinion_prob = InterconnectedDynamics.A_state_change_probability_cal(inter_layer, p)
                decision_prob = InterconnectedDynamics.B_state_change_probability_cal(inter_layer, v)
                initial_value = InterconnectedDynamics.making_properties_array(setting, inter_layer, p, v, opinion_prob[1],
                                                                               opinion_prob[2], decision_prob[1],
                                                                               change_count, sum_properties, edges_properties)
                total_value = total_value + initial_value
            elif step_number >= 1:
                AB_dynamics_result = OpinionDynamics.OpinionDynamics(setting, inter_layer, p, v, 3, using_prob, unchanged_nodes)
                change_count += AB_dynamics_result.A_COUNT
                array_value = InterconnectedDynamics.making_properties_array(setting, AB_dynamics_result.A_inter_layer, p, v,
                                                                             AB_dynamics_result.persuasion_prob,
                                                                             AB_dynamics_result.compromise_prob,
                                                                             AB_dynamics_result.volatility_prob,
                                                                             change_count, sum_properties, edges_properties)
                total_value = np.vstack([total_value, array_value])
        return total_value

    @staticmethod
    def interconnected_dynamics10(setting, inter_layer, p, v, using_prob, unchanged_nodes, sum_properties, edges_properties):
        # random:order:opinion
        total_value = np.zeros(17)
        change_count = 0
        for step_number in range(setting.Limited_step+1):
            if step_number == 0:
                opinion_prob = InterconnectedDynamics.A_state_change_probability_cal(inter_layer, p)
                decision_prob = InterconnectedDynamics.B_state_change_probability_cal(inter_layer, v)
                initial_value = InterconnectedDynamics.making_properties_array(setting, inter_layer, p, v, opinion_prob[1],
                                                                               opinion_prob[2], decision_prob[1],
                                                                               change_count, sum_properties, edges_properties)
                total_value = total_value + initial_value
            elif step_number >= 1:
                opinion_result = OpinionDynamics.OpinionDynamics(setting, inter_layer, p, v, 2, using_prob, unchanged_nodes)
                decision_result = DecisionDynamics.DecisionDynamics(setting, opinion_result.A_inter_layer, v, 0, unchanged_nodes)
                change_count += opinion_result.A_COUNT + decision_result.B_COUNT
                array_value = InterconnectedDynamics.making_properties_array(setting, decision_result.B_inter_layer, p, v,
                                                                             opinion_result.persuasion_prob,
                                                                             opinion_result.compromise_prob,
                                                                             decision_result.volatility_prob,
                                                                             change_count, sum_properties, edges_properties)
                total_value = np.vstack([total_value, array_value])
        return total_value

    @staticmethod
    def interconnected_dynamics11(setting, inter_layer, p, v, using_prob, unchanged_nodes, sum_properties, edges_properties):
        # random:order:decision
        total_value = np.zeros(17)
        change_count = 0
        for step_number in range(setting.Limited_step+1):
            if step_number == 0:
                opinion_prob = InterconnectedDynamics.A_state_change_probability_cal(inter_layer, p)
                decision_prob = InterconnectedDynamics.B_state_change_probability_cal(inter_layer, v)
                initial_value = InterconnectedDynamics.making_properties_array(setting, inter_layer, p, v, opinion_prob[1],
                                                                               opinion_prob[2], decision_prob[1],
                                                                               change_count, sum_properties, edges_properties)
                total_value = total_value + initial_value
            elif step_number >= 1:
                decision_result = DecisionDynamics.DecisionDynamics(setting, inter_layer, v, 0, unchanged_nodes)
                opinion_result = OpinionDynamics.OpinionDynamics(setting, decision_result.B_inter_layer, p, v, 2, using_prob, unchanged_nodes)
                change_count += opinion_result.A_COUNT + decision_result.B_COUNT
                array_value = InterconnectedDynamics.making_properties_array(setting, opinion_result.A_inter_layer, p, v,
                                                                             opinion_result.persuasion_prob,
                                                                             opinion_result.compromise_prob,
                                                                             decision_result.volatility_prob,
                                                                             change_count, sum_properties, edges_properties)
                total_value = np.vstack([total_value, array_value])

        return total_value

    @staticmethod
    def interconnected_dynamics12(setting, inter_layer, p, v, using_prob, unchanged_nodes, sum_properties, edges_properties):
        # random:same:opinion
        total_value = np.zeros(17)
        change_count = 0
        for step_number in range(setting.Limited_step+1):
            if step_number == 0:
                opinion_prob = InterconnectedDynamics.A_state_change_probability_cal(inter_layer, p)
                decision_prob = InterconnectedDynamics.B_state_change_probability_cal(inter_layer, v)
                initial_value = InterconnectedDynamics.making_properties_array(setting, inter_layer, p, v, opinion_prob[1],
                                                                               opinion_prob[2], decision_prob[1],
                                                                               change_count, sum_properties, edges_properties)
                total_value = total_value + initial_value
            elif step_number >= 1:
                opinion_result = OpinionDynamics.OpinionDynamics(setting, inter_layer, p, v, 2, using_prob, unchanged_nodes)
                decision_result = DecisionDynamics.DecisionDynamics(setting, opinion_result.A_inter_layer, v, 1, unchanged_nodes)
                change_count += opinion_result.A_COUNT + decision_result.B_COUNT
                array_value = InterconnectedDynamics.making_properties_array(setting, decision_result.B_inter_layer, p, v,
                                                                             opinion_result.persuasion_prob,
                                                                             opinion_result.compromise_prob,
                                                                             decision_result.volatility_prob,
                                                                             change_count, sum_properties, edges_properties)
                total_value = np.vstack([total_value, array_value])

        return total_value

    @staticmethod
    def interconnected_dynamics13(setting, inter_layer, p, v, using_prob, unchanged_nodes, sum_properties, edges_properties):
        # random:same:decision
        total_value = np.zeros(17)
        change_count = 0
        for step_number in range(setting.Limited_step+1):
            if step_number == 0:
                opinion_prob = InterconnectedDynamics.A_state_change_probability_cal(inter_layer, p)
                decision_prob = InterconnectedDynamics.B_state_change_probability_cal(inter_layer, v)
                initial_value = InterconnectedDynamics.making_properties_array(setting, inter_layer, p, v, opinion_prob[1],
                                                                               opinion_prob[2], decision_prob[1],
                                                                               change_count, sum_properties, edges_properties)
                total_value = total_value + initial_value
            elif step_number >= 1:
                decision_result = DecisionDynamics.DecisionDynamics(setting, inter_layer, v, 1, unchanged_nodes)
                opinion_result = OpinionDynamics.OpinionDynamics(setting, decision_result.B_inter_layer, p, v, 2, using_prob, unchanged_nodes)
                change_count += opinion_result.A_COUNT + decision_result.B_COUNT
                array_value = InterconnectedDynamics.making_properties_array(setting, opinion_result.A_inter_layer, p, v,
                                                                             opinion_result.persuasion_prob,
                                                                             opinion_result.compromise_prob,
                                                                             decision_result.volatility_prob,
                                                                             change_count, sum_properties, edges_properties)
                total_value = np.vstack([total_value, array_value])
        return total_value

    @staticmethod
    def interconnected_dynamics14(setting, inter_layer, p, v, using_prob, unchanged_nodes, sum_properties, edges_properties):
        # random:random:random
        total_value = np.zeros(17)
        change_count = 0
        for step_number in range(setting.Limited_step+1):
            if step_number == 0:
                opinion_prob = InterconnectedDynamics.A_state_change_probability_cal(inter_layer, p)
                decision_prob = InterconnectedDynamics.B_state_change_probability_cal(inter_layer, v)
                initial_value = InterconnectedDynamics.making_properties_array(setting, inter_layer, p, v, opinion_prob[1],
                                                                               opinion_prob[2], decision_prob[1],
                                                                               change_count, sum_properties, edges_properties)
                total_value = total_value + initial_value
            elif step_number >= 1:
                AB_dynamics_result = OpinionDynamics.OpinionDynamics(setting, inter_layer, p, v, 4, using_prob, unchanged_nodes)
                change_count += AB_dynamics_result.A_COUNT
                array_value = InterconnectedDynamics.making_properties_array(setting, AB_dynamics_result.A_inter_layer, p, v,
                                                                             AB_dynamics_result.persuasion_prob,
                                                                             AB_dynamics_result.compromise_prob,
                                                                             AB_dynamics_result.volatility_prob,
                                                                             change_count, sum_properties, edges_properties)
                total_value = np.vstack([total_value, array_value])
        return total_value

    @staticmethod
    def making_properties_array(setting, inter_layer, p, v, persuasion_prob, compromise_prob, volatility_prob,
                                change_count, sum_properties, edges_properties):
        interacting_properties = InterconnectedDynamics.interacting_property(setting, inter_layer)
        array_value = np.array([p, v, volatility_prob, persuasion_prob, compromise_prob,
                                interacting_properties[0], interacting_properties[1],
                                interacting_properties[2], interacting_properties[3],
                                interacting_properties[4], interacting_properties[5],
                                interacting_properties[6],
                                len(inter_layer.edges_on_A), len(inter_layer.edges_on_B),
                                change_count, sum_properties, edges_properties])
        return array_value

    @staticmethod
    def interacting_property(setting, inter_layer):
        property_A = []
        property_B = []
        for i in inter_layer.A_nodes:
            property_A.append(inter_layer.two_layer_graph.nodes[i]['state'])
        for j in inter_layer.B_nodes:
            property_B.append(inter_layer.two_layer_graph.nodes[j]['state'])
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

    @staticmethod
    def B_state_change_probability_cal(inter_layer, v):
        prob_v_list = []
        for node_i in inter_layer.B_nodes:
            neighbors = np.array(sorted(nx.neighbors(inter_layer.two_layer_graph, node_i)))
            neighbor_state = []
            for neighbor in neighbors:
                neighbor_state.append(inter_layer.two_layer_graph.nodes[neighbor]['state'])
            neighbor_array = np.array(neighbor_state)
            same_orientation = int(np.sum(neighbor_array * (inter_layer.two_layer_graph.nodes[node_i]['state']) > 0))
            opposite_orientation = len(neighbors) - same_orientation
            if opposite_orientation == 0:
                prob_v = 0
            else:
                if v == 0:
                    prob_v = 0
                else:
                    prob_v = ((opposite_orientation / len(neighbors)) ** (1 / v)) * (len(neighbors) / opposite_orientation)
            prob_v_list.append(prob_v)
        prob_v_array = np.array(prob_v_list)
        prob_v_mean = np.sum(prob_v_array) / len(prob_v_array)
        return prob_v_array, prob_v_mean

    @staticmethod
    def A_state_change_probability_cal(inter_layer, p):
        prob_list = []
        prob_per_list = []
        prob_com_list = []
        for node_i in inter_layer.A_nodes:
            prob = InterconnectedDynamics.three_probability_of_opinion_dynamics(inter_layer, p, node_i)
            prob_list.append((prob[0], prob[0]+prob[1], prob[0]+prob[1]+prob[2]))
            prob_per_list.append(prob[1])
            prob_com_list.append(prob[2])
        prob_array = np.array(prob_list)
        persuasion_prob = sum(prob_per_list) / len(prob_per_list)
        compromise_prob = sum(prob_com_list) / len(prob_com_list)
        return prob_array, persuasion_prob, compromise_prob

    @staticmethod
    def three_probability_of_opinion_dynamics(inter_layer, p, node_i):
        neighbors = np.array(sorted(nx.neighbors(inter_layer.two_layer_graph, node_i)))
        neighbor_state = []
        for neighbor in neighbors:
            neighbor_state.append(inter_layer.two_layer_graph.nodes[neighbor]['state'])
        neighbor_array = np.array(neighbor_state)
        same_orientation = int(np.sum(neighbor_array * inter_layer.two_layer_graph.nodes[node_i]['state'] > 0))
        opposite_orientation = len(neighbors) - same_orientation
        node_unchanging_prob = 0
        node_persuasion_prob = 0
        node_compromise_prob = 0
        for n in range(0, same_orientation + 1):
            for m in range(0, opposite_orientation + 1):
                n_combi = InterconnectedDynamics.nCr(same_orientation, n)
                m_combi = InterconnectedDynamics.nCr(opposite_orientation, m)
                if n == m:
                    node_unchanging_prob += p ** (n + opposite_orientation - m) * ((1 - p) ** (same_orientation - n + m)) * n_combi * m_combi
                elif n > m:
                    node_persuasion_prob += p ** (n + opposite_orientation - m) * ((1 - p) ** (same_orientation - n + m)) * n_combi * m_combi
                elif n < m:
                    node_compromise_prob += p ** (n + opposite_orientation - m) * ((1 - p) ** (same_orientation - n + m)) * n_combi * m_combi
        return node_unchanging_prob, node_persuasion_prob, node_compromise_prob

    @staticmethod
    def nCr(n, r):
        f = math.factorial
        return f(n) // f(r) // f(n - r)


if __name__ == "__main__":
    print("InterconnectedDynamics")
    start = time.time()
    setting = SettingSimulationValue.SettingSimulationValue()
    inter_layer = InterconnectedLayerModeling.InterconnectedLayerModeling(setting)
    p = 0.1
    v = 0.3
    state = 0
    for i in inter_layer.A_nodes:
        state += inter_layer.two_layer_graph.nodes[i]['state']
    print(state)
    print(inter_layer.two_layer_graph.nodes[0]['state'], inter_layer.two_layer_graph.nodes[1]['state'],
          inter_layer.two_layer_graph.nodes[2]['state'], inter_layer.two_layer_graph.nodes[3]['state'])
    inter_dynamics = InterconnectedDynamics(setting, inter_layer, p, v,
                                            using_prob=False, select_step=1, unchanged_nodes={0, 1, 2, 3},
                                            sum_properties=0, edges_properties=0)
    print(inter_layer.two_layer_graph.nodes[0]['state'], inter_layer.two_layer_graph.nodes[1]['state'],
          inter_layer.two_layer_graph.nodes[2]['state'], inter_layer.two_layer_graph.nodes[3]['state'])
    print(inter_dynamics.dynamics_result_array)
    state = 0
    for i in inter_layer.A_nodes:
        state += inter_layer.two_layer_graph.nodes[i]['state']
    print(state)
    end = time.time()
    print(end-start)











