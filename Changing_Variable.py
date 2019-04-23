import numpy as np
import Setting_Simulation_Value
import MakingPandas
import RepeatDynamics
import sqlalchemy
from multiprocessing import Pool


class Changing_Variable:
    def __init__(self):
        self.mp = MakingPandas.MakingPandas()
        self.repeat_dynamics = RepeatDynamics.RepeatDynamics()

    def simultaneous_calculate_and_input_database(self, setting_variable_tuple):
        p = setting_variable_tuple[1][0]
        v = setting_variable_tuple[1][1]
        Num_Data = self.repeat_dynamics.repeat_simultaneous_dynamics(setting_variable_tuple[0], p, v)
        panda_db = self.mp.making_dataframe_per_step(setting_variable_tuple[0], Num_Data)
        print(panda_db.loc[0])
        engine = sqlalchemy.create_engine('mysql+pymysql://root:2853@localhost:3306/%s' % setting_variable_tuple[0].database)
        panda_db.to_sql(name='%s' % setting_variable_tuple[0].table, con=engine, index=False, if_exists='append')

    def calculate_and_input_database(self, setting_variable_tuple):
        p = setting_variable_tuple[1][0]
        v = setting_variable_tuple[1][1]
        Num_Data = self.repeat_dynamics.repeat_dynamics(setting_variable_tuple[0], p, v)
        panda_db = self.mp.making_dataframe_per_step(setting_variable_tuple[0], Num_Data)
        print(panda_db.loc[0])
        engine = sqlalchemy.create_engine('mysql+pymysql://root:2853@localhost:3306/%s' % setting_variable_tuple[0].database)
        panda_db.to_sql(name='%s' % setting_variable_tuple[0].table, con=engine, index=False, if_exists='append')

    def simultaneous_paralleled_work(self, setting):
        workers = setting.workers
        setting_variable_list = []
        for i in setting.variable_list:
            setting_variable_list.append((setting, i))
        with Pool(workers) as p:
            p.map(self.simultaneous_calculate_and_input_database, setting_variable_list)

    def paralleled_work(self, setting):
        workers = setting.workers
        setting_variable_list = []
        for i in setting.variable_list:
            setting_variable_list.append((setting, i))
        with Pool(workers) as p:
            p.map(self.calculate_and_input_database, setting_variable_list)

if __name__ == "__main__":
    print("Changing_Variable")
    setting = Setting_Simulation_Value.Setting_Simulation_Value()
    changing_variable = Changing_Variable()
    changing_variable.paralleled_work(setting)
    # changing_variable.simultaneous_paralleled_work(setting)
    print("Operating end")



