import Setting_Simulation_Value
import RepeatDynamics
import sqlalchemy
from concurrent import futures
from tqdm import tqdm

class Changing_Variable:
    def __init__(self):
        self.repeat_dynamics = RepeatDynamics.RepeatDynamics()

    def many_execute_for_simulation(self, setting):
        setting_variable_list = self.making_variable_tuples_list(setting)
        engine = sqlalchemy.create_engine('mysql+pymysql://root:2853@localhost:3306/%s' % setting.database)
        with futures.ProcessPoolExecutor(max_workers=setting.workers) as executor:
            to_do_map = {}
            for setting_variable_tuple in sorted(setting_variable_list):
                future = executor.submit(self.calculate_for_simulation, setting_variable_tuple)
                to_do_map[future] = setting_variable_tuple
            done_iter = futures.as_completed(to_do_map)
            done_iter = tqdm(done_iter, total=len(setting_variable_list))
            for future in done_iter:
                res = future.result()
                res.to_sql(name='%s' % setting.table, con=engine, index=False, if_exists='append')

    def making_variable_tuples_list(self, setting):
        p = 0.4
        v = 0.4
        setting_variable_list = []
        for select_step in range(1, 2):
            for select_method in range(0, 5):
                for node_number in [1, 2, 3, 4]:
                    setting_variable_list.append((setting, p, v, select_step, select_method, node_number))
        return setting_variable_list

    def calculate_for_simulation(self, setting_variable_tuple):
        panda_db = self.repeat_dynamics.repeat_dynamics(setting_variable_tuple[0],
                                                        setting_variable_tuple[1],
                                                        setting_variable_tuple[2],
                                                        select_step=setting_variable_tuple[3],
                                                        select_method=setting_variable_tuple[4],
                                                        node_number=setting_variable_tuple[5])
        return panda_db


if __name__ == "__main__":
    print("Changing_Variable")
    setting = Setting_Simulation_Value.Setting_Simulation_Value()
    changing_variable = Changing_Variable()
    # lis = changing_variable.making_variable_tuples_list(setting)
    # for tu in lis:
    #     print(tu)
    changing_variable.many_execute_for_simulation(setting)
    print("Operating end")