import SettingSimulationValue
import RepeatDynamics
import sqlalchemy
from concurrent import futures
from tqdm import tqdm


class ChangingVariable:
    def __init__(self, setting, p=0.4, v=0.4, unchanged_state=-1, select_layer='A_layer',
                 select_using_prob=(True, False), steps=(1, 2), select_methods=(1, 2), node_numbers=50):
        ChangingVariable.many_execute_for_simulation(setting, p, v, unchanged_state, select_layer, select_using_prob,
                                                     steps, select_methods, node_numbers)

    @staticmethod
    def many_execute_for_simulation(setting, p, v, unchanged_state, select_layer, select_using_prob,
                                    steps, select_methods, node_numbers):
        setting_variable_list = ChangingVariable.making_variable_tuples_list(setting, p, v, unchanged_state,
                                                                             select_layer, select_using_prob,
                                                                             steps, select_methods, node_numbers)
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
    def making_variable_tuples_list(setting, p, v, unchanged_state, select_layer, select_using_prob,
                                    steps, select_methods, node_numbers):
        setting_variable_list = []
        for using_prob in list(select_using_prob):
            select_steps_list = []
            if using_prob is False:
                select_steps_list = [i for i in range(15)]
            elif using_prob is True:
                select_steps_list = [i for i in range(10)]
            select_steps = [select_steps_list[step] for step in list(steps)]
            for select_step in select_steps:
                for select_method in list(select_methods):
                    for node_number in range(1, node_numbers+1):
                        setting_variable_list.append((setting, p, v, using_prob, select_step, select_method,
                                                      select_layer, node_number, unchanged_state))
        return setting_variable_list

    @staticmethod
    def calculate_for_simulation(setting_variable_tuple):
        repeat_result = RepeatDynamics.RepeatDynamics(setting_variable_tuple[0], setting_variable_tuple[1], setting_variable_tuple[2],
                                                      using_prob=setting_variable_tuple[3], select_step=setting_variable_tuple[4],
                                                      select_method=setting_variable_tuple[5], select_layer=setting_variable_tuple[6],
                                                      node_number=setting_variable_tuple[7], unchanged_state=setting_variable_tuple[8])
        result_panda = repeat_result.repeated_result
        return result_panda


if __name__ == "__main__":
    print("Changing_Variable")
    setting = SettingSimulationValue.SettingSimulationValue()
    ChangingVariable(setting, p=0.4, v=0.4, unchanged_state=-1, select_layer='A_layer',
                     select_using_prob=[False], steps=[1], select_methods=[4], node_numbers=50)
    print("Operating end")
