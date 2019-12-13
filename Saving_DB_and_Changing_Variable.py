import SettingSimulationValue
import RepeatDynamics
import sqlalchemy
import numpy as np

class Saving_DB_and_Changing_Variable:
    def __init__(self, setting, select_using_prob=None, updating_rules_list=None,
                 node_layer_list=None, node_method_list=None,
                 edge_layer_list=None, edge_method_list=None, edge_numbers=0):
        if select_using_prob is None: select_using_prob = [False]
        if updating_rules_list is None: updating_rules_list = [1]
        if node_layer_list is None:
            node_method_list = ['0']
        if edge_layer_list is None:
            edge_method_list = ['0']
            edge_numbers = 0
        Saving_DB_and_Changing_Variable.saving_DB(setting, select_using_prob, updating_rules_list,
                                                  node_layer_list, node_method_list,
                                                  node_layer_list, edge_method_list, edge_numbers)

    @staticmethod
    def saving_DB(setting, select_using_prob, updating_rules_list,
                  node_layer_list, node_method_list,
                  edge_layer_list, edge_method_list, edge_numbers):
        engine = sqlalchemy.create_engine('mysql+pymysql://root:2853@localhost:3306/%s' % setting.database)
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
                res = Saving_DB_and_Changing_Variable.calculate_for_simulation(setting, using_prob, updating_rule,
                                                                               node_layer_list, node_method_list,
                                                                               edge_layer_list, edge_method_list, edge_numbers)
                res.to_sql(name='%s' % setting.table, con=engine, index=False, if_exists='append')

    @staticmethod
    def calculate_for_simulation(setting, using_prob, updating_rule, node_layer_list, node_method_list,
                                 edge_layer_list, edge_method_list, edge_numbers):
        repeat_result = RepeatDynamics.RepeatDynamics(setting, using_prob, updating_rule,
                                                      node_layer_list, node_method_list,
                                                      edge_layer_list, edge_method_list, edge_numbers)
        result_panda = repeat_result.repeated_result
        return result_panda

if __name__ == "__main__":
    print("Saving_DB")
    settings = SettingSimulationValue.SettingSimulationValue()
    Saving_DB_and_Changing_Variable(settings, select_using_prob=None, updating_rules_list=None,
                                    node_layer_list=['A_layer', 'B_layer'],
                                    # node_method_list=['degree', 'pagerank', 'eigenvector', 'random',
                                    #                   'PR+DE', 'DE+EI', 'PR+EI', 'PR+DE+EI'],
                                    node_method_list=['degree', 'pagerank', 'eigenvector', 'random',
                                                      'betweenness', 'closeness', 'PR+DE', 'PR+BE', 'DE+BE', 'PR+DE+BE'],
                                    edge_layer_list=None, edge_method_list=None, edge_numbers=0)
    print("Operating end")

# select_node_layers_list = ['A_layer', 'B_layer', 'mixed']
# select_edge_layers_list = ['A_internal', 'A_mixed', 'B_internal', 'B_mixed', 'external', 'mixed']
# node_methods = ['pagerank', 'degree', 'eigenvector', 'betweenness', 'closeness', 'random', 'PR+DE', 'PR+BE','DE+BE', 'PR+DE+BE]
# edge_methods = ['random', 'edge_pagerank', 'edge_degree', 'edge_eigenvector', 'edge_betweenness', 'edge_closeness',
#                 'edge_load', 'edge_jaccard']