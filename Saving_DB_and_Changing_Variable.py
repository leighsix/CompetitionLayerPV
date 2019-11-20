import SettingSimulationValue
import RepeatDynamics
import sqlalchemy
import numpy as np
from concurrent import futures
from tqdm import tqdm

class Saving_DB_and_Changing_Variable:
    def __init__(self, setting, p=(0, 1), v=(0, 1), gap=30, select_using_prob=None, updating_rules_list=None,
                 select_node_layers=None, select_node_methods=None, node_numbers=0,
                 select_edge_layers=None, select_edge_methods=None, edge_numbers=0):
        if select_using_prob is None: select_using_prob = [False]
        if updating_rules_list is None: updating_rules_list = [1]
        if select_node_layers is None:
            select_node_layers = [0]
            select_node_methods = ['0']
            node_numbers = 0
            unchanged_state = 'None'
        if select_edge_layers is None:
            select_edge_layers = [0]
            select_edge_methods = ['0']
            edge_numbers = 0
        Saving_DB_and_Changing_Variable.saving_DB(setting, p, v, gap, select_using_prob, updating_rules_list,
                                                  select_node_layers, select_node_methods, node_numbers,
                                                  select_edge_layers, select_edge_methods, edge_numbers)

    @staticmethod
    def saving_DB(setting, p, v, gap, select_using_prob, updating_rules_list,
                  select_node_layers_list, node_method_list, node_numbers,
                  select_edge_layers_list, edge_method_list, edge_numbers):
        engine = sqlalchemy.create_engine('mysql+pymysql://root:2853@localhost:3306/%s' % setting.database)
        p_list = np.linspace(p[0], p[-1], gap)
        v_list = np.linspace(v[0], v[-1], gap)
        for p_value in p_list:
            for v_value in v_list:
                for using_prob in select_using_prob:
                    updating_rules = []
                    if using_prob is False:
                        if updating_rules_list == ['all']:
                            updating_rules = [i for i in range(15)]
                        elif updating_rules_list != ['all']:
                            updating_rules = updating_rules_list
                    elif using_prob is True:
                        if updating_rules_list == ['all']:
                            updating_rules = [i for i in range(10)]
                        elif updating_rules_list != ['all']:
                            updating_rules = updating_rules_list
                    for updating_rule in updating_rules:
                        for node_layer in select_node_layers_list:
                            for edge_layer in select_edge_layers_list:
                                res = Saving_DB_and_Changing_Variable.calculate_for_simulation\
                                    (setting, p_value, v_value, using_prob, updating_rule,
                                     node_layer, node_method_list, node_numbers,
                                     edge_layer, edge_method_list, edge_numbers)
                                res.to_sql(name='%s' % setting.table, con=engine, index=False, if_exists='append')

    @staticmethod
    def calculate_for_simulation(setting, p_value, v_value, using_prob, updating_rule, node_layer, node_method_list,
                                 node_numbers, edge_layer, edge_method_list, edge_numbers):
        repeat_result = RepeatDynamics.RepeatDynamics(setting, p_value, v_value, using_prob, updating_rule,
                                                      node_layer, node_method_list, node_numbers,
                                                      edge_layer, edge_method_list, edge_numbers)
        result_panda = repeat_result.repeated_result
        return result_panda

if __name__ == "__main__":
    print("Saving_DB")
    settings = SettingSimulationValue.SettingSimulationValue()
    settings.workers = 1
    settings.database = 'test'
    settings.table = 'test'
    Saving_DB_and_Changing_Variable(settings, p=[0.2], v=[0.4], gap=1, select_using_prob=None, updating_rules_list=None,
                                    select_node_layers=['A_layer'],
                                    select_node_methods=['degree', 'pagerank', 'eigenvector', 'random', 'betweenness', 'closeness','PR+DE', 'PR+DE+BE'],
                                    node_numbers=200,
                                    select_edge_layers=None, select_edge_methods=None, edge_numbers=0)
    print("Operating end")

# select_node_layers_list = ['A_layer', 'B_layer', 'mixed']
# select_edge_layers_list = ['A_internal', 'A_mixed', 'B_internal', 'B_mixed', 'external', 'mixed']
# node_methods = ['pagerank', 'degree', 'eigenvector', 'betweenness', 'closeness', 'random', 'PR+DE', 'PR+DE+BE]
# edge_methods = ['random', 'edge_pagerank', 'edge_degree', 'edge_eigenvector', 'edge_betweenness', 'edge_closeness',
#                 'edge_load', 'edge_jaccard']