import Setting_Simulation_Value
import RepeatDynamics
import sqlalchemy
from multiprocessing import Pool


class Changing_Variable:
    def __init__(self):
        self.repeat_dynamics = RepeatDynamics.RepeatDynamics()

    def calculate_and_input_database(self, setting_variable_tuple):
        p = setting_variable_tuple[1][0]
        v = setting_variable_tuple[1][1]
        panda_db = self.repeat_dynamics.select_repeat_dynamics(setting_variable_tuple[0], p, v, select_step=0)
        print(panda_db.loc[0])
        engine = sqlalchemy.create_engine('mysql+pymysql://root:2853@localhost:3306/%s' % setting_variable_tuple[0].database)
        panda_db.to_sql(name='%s' % setting_variable_tuple[0].table, con=engine, index=False, if_exists='append')

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
    print("Operating end")



