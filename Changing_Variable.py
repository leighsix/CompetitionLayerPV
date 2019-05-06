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
            for setting_variable_tuple in setting_variable_list:
                future = executor.submit(self.calculate_for_simulation, setting_variable_tuple)
                to_do_map[future] = setting_variable_tuple
            done_iter = futures.as_completed(to_do_map)
            done_iter = tqdm(done_iter, total=len(setting_variable_list))
            for future in done_iter:
                res = future.result()
                res.to_sql(name='%s' % setting.table, con=engine, index=False, if_exists='append')

    def calculate_for_simulation(self, setting_variable_tuple):
        p = 0.3
        v = 0.3
        panda_db = self.repeat_dynamics.repeat_dynamics(setting_variable_tuple[0], p, v, select_step=setting_variable_tuple[1])
        return panda_db

    def making_variable_tuples_list(self, setting):
        setting_variable_list = []
        for i in range(9):
            setting_variable_list.append((setting, i))
        return setting_variable_list


    # def calculate_and_input_database(self, setting_variable_tuple):
    #     p = 0.3
    #     v = 0.3
    #     panda_db = self.repeat_dynamics.repeat_dynamics(setting_variable_tuple[0], p, v, select_step=setting_variable_tuple[1])
    #     print(panda_db.loc[0])
    #     engine = sqlalchemy.create_engine('mysql+pymysql://root:2853@localhost:3306/%s' % setting_variable_tuple[0].database)
    #     panda_db.to_sql(name='%s' % setting_variable_tuple[0].table, con=engine, index=False, if_exists='append')
    #
    # def paralleled_work(self, setting):
    #     workers = setting.workers
    #     setting_variable_list = self.making_variable_tuples_list(setting)
    #     with Pool(workers) as p:
    #         p.map(self.calculate_and_input_database, setting_variable_list)




if __name__ == "__main__":
    print("Changing_Variable")
    setting = Setting_Simulation_Value.Setting_Simulation_Value()
    changing_variable = Changing_Variable()
    changing_variable.many_execute_for_simulation(setting)
    # changing_variable.paralleled_work(setting)
    print("Operating end")



