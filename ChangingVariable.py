import SettingSimulationValue
import RepeatDynamics
import sqlalchemy
import numpy as np
from concurrent import futures
from tqdm import tqdm

select_node_layers_list = ['A_layer', 'B_layer', 'mixed']
select_edge_layers_list = ['A_internal', 'A_mixed', 'B_internal', 'B_mixed', 'external', 'mixed']
select_node_method_list = ['0', 'degree', 'pagerank', 'random', 'eigenvector', 'closeness', 'betweenness',
                           'pagerank_individual', 'load', 'AB_pagerank', 'AB_eigenvector', 'AB_degree', 'AB_betweenness',
                           'AB_closeness', 'AB_load']
select_edge_method_list = ['0', 'edge_pagerank', 'edge_betweenness', 'edge_degree', 'edge_eigenvector', 'edge_closeness',
                           'edge_load', 'random', 'edge_pagerank_sequential', 'edge_betweenness_sequential',
                           'edge_degree_sequential', 'edge_eigenvector_sequential',
                           'edge_closeness_sequential', 'edge_load_sequential']


class ChangingVariable:
    def __init__(self, setting, p=(0, 1), v=(0, 1), gap=30, select_using_prob=(True, False), steps=(1, 2),
                 select_node_layers=(0, 0), select_node_methods=(0, 0), node_numbers=(0, 0), unchanged_state='None',
                 select_edge_layers=(0, 0), select_edge_methods=(0, 0), edge_numbers=(0, 0)):
        ChangingVariable.many_execute_for_simulation(setting, p, v, gap, select_using_prob, steps,
                                                     select_node_layers, select_node_methods, node_numbers, unchanged_state,
                                                     select_edge_layers, select_edge_methods, edge_numbers)

    @staticmethod
    def many_execute_for_simulation(setting, p, v, gap, select_using_prob, steps,
                                    select_node_layers, select_node_methods, node_numbers, unchanged_state,
                                    select_edge_layers, select_edge_methods, edge_numbers):
        setting_variable_list = ChangingVariable.making_variable_tuples_list(setting, p, v, gap, select_using_prob, steps,
                                                                             select_node_layers, select_node_methods, node_numbers, unchanged_state,
                                                                             select_edge_layers, select_edge_methods, edge_numbers)
        engine = sqlalchemy.create_engine('mysql+pymysql://root:2853@localhost:3306/%s' % setting.database)
        with futures.ProcessPoolExecutor(max_workers=setting.workers) as executor:
            to_do_map = {}
            for setting_variable_tuple in sorted(setting_variable_list):
                future = executor.submit(ChangingVariable.calculate_for_simulation, setting_variable_tuple)
                to_do_map[future] = setting_variable_tuple
            done_iter = futures.as_completed(to_do_map)
            done_iter = tqdm(done_iter, total=len(setting_variable_list))
            for future in done_iter:
                res = future.result()
                res.to_sql(name='%s' % setting.table, con=engine, index=False, if_exists='append')

    @staticmethod
    def calculate_for_simulation(setting_variable_tuple):
        repeat_result = RepeatDynamics.RepeatDynamics(setting_variable_tuple[0],
                                                      setting_variable_tuple[1],
                                                      setting_variable_tuple[2],
                                                      using_prob=setting_variable_tuple[3],
                                                      select_step=setting_variable_tuple[4],
                                                      select_node_layer=setting_variable_tuple[5],
                                                      select_node_method=setting_variable_tuple[6],
                                                      node_number=setting_variable_tuple[7],
                                                      unchanged_state=setting_variable_tuple[8],
                                                      select_edge_layer=setting_variable_tuple[9],
                                                      select_edge_method=setting_variable_tuple[10],
                                                      edge_number=setting_variable_tuple[11])
        result_panda = repeat_result.repeated_result
        return result_panda

    @staticmethod
    def making_variable_tuples_list(setting, p, v, gap, select_using_prob, steps,
                                    select_node_layers, select_node_methods, node_numbers, unchanged_state,
                                    select_edge_layers, select_edge_methods,  edge_numbers):
        p_list = np.linspace(p[0], p[-1], gap)
        v_list = np.linspace(v[0], v[-1], gap)
        setting_variable_list = []
        for p_value in p_list:
            for v_value in v_list:
                for using_prob in select_using_prob:
                    select_steps = []
                    if using_prob is False:
                        select_steps_list = [i for i in range(15)]
                        if steps == ['all']:
                            select_steps = select_steps_list
                        elif steps != ['all']:
                            select_steps = [select_steps_list[step] for step in steps]
                    elif using_prob is True:
                        select_steps_list = [i for i in range(10)]
                        if steps == ['all']:
                            select_steps = select_steps_list
                        elif steps != ['all']:
                            select_steps = [select_steps_list[step] for step in steps]
                    for select_step in select_steps:
                        for node_layer in select_node_layers_list[select_node_layers[0]:select_node_layers[-1]+1]:
                            for select_node_method in select_node_method_list[select_node_methods[0]:select_node_methods[-1]+1]:
                                for node_number in range(node_numbers[0], node_numbers[-1] + 1):
                                    for edge_layer in select_edge_layers_list[select_edge_layers[0]:select_edge_layers[-1]+1]:
                                        for select_edge_method in select_edge_method_list[select_edge_methods[0]:select_edge_methods[-1]+1]:
                                            for edge_number in range(edge_numbers[0], edge_numbers[-1] + 1):
                                                setting_variable_list.append(
                                                    (setting, p_value, v_value, using_prob, select_step,
                                                     node_layer, select_node_method, node_number, unchanged_state,
                                                     edge_layer, select_edge_method, edge_number))
        return setting_variable_list


if __name__ == "__main__":
    print("Changing_Variable")
    settings = SettingSimulationValue.SettingSimulationValue()
    ChangingVariable(settings,  p=[0, 1], v=[0, 1], gap=20, select_using_prob=[False], steps=[1],
                     select_node_layers=[0], select_node_methods=[0, 0], node_numbers=[0, 0], unchanged_state='None',
                     select_edge_layers=(0, 0), select_edge_methods=(0, 0), edge_numbers=(0, 0))
    print("Operating end")
