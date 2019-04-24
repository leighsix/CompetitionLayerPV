import numpy as np
import MakingPandas
import Setting_Simulation_Value
import InterconnectedDynamics
import InterconnectedLayerModeling
import time


class RepeatDynamics:
    def __init__(self):
        self.inter_dynamics = InterconnectedDynamics.InterconnectedDynamics()
        self.mp = MakingPandas.MakingPandas()

    def select_repeat_dynamics(self, setting, p, v, select_step):
        num_data = np.zeros([setting.Limited_step + 1, 13])
        for i in range(setting.Repeating_number):
            inter_layer = InterconnectedLayerModeling.InterconnectedLayerModeling(setting)
            if select_step == 0:
                total_array = self.inter_dynamics.interconnected_dynamics0(setting, inter_layer, p, v)
                num_data = num_data + total_array
            elif select_step == 1:
                total_array = self.inter_dynamics.interconnected_dynamics1(setting, inter_layer, p, v)
                num_data = num_data + total_array
            elif select_step == 2:
                total_array = self.inter_dynamics.interconnected_dynamics2(setting, inter_layer, p, v)
                num_data = num_data + total_array
            elif select_step == 3:
                total_array = self.inter_dynamics.interconnected_dynamics3(setting, inter_layer, p, v)
                num_data = num_data + total_array
            elif select_step == 4:
                total_array = self.inter_dynamics.interconnected_dynamics4(setting, inter_layer, p, v)
                num_data = num_data + total_array
            elif select_step == 5:
                total_array = self.inter_dynamics.interconnected_dynamics5(setting, inter_layer, p, v)
                num_data = num_data + total_array
            elif select_step == 6:
                total_array = self.inter_dynamics.interconnected_dynamics6(setting, inter_layer, p, v)
                num_data = num_data + total_array
            elif select_step == 7:
                total_array = self.inter_dynamics.interconnected_dynamics7(setting, inter_layer, p, v)
                num_data = num_data + total_array
            elif select_step == 8:
                total_array = self.inter_dynamics.interconnected_dynamics8(setting, inter_layer, p, v)
                num_data = num_data + total_array
            elif select_step == 9:
                total_array = self.inter_dynamics.interconnected_dynamics9(setting, inter_layer, p, v)
                num_data = num_data + total_array
        Num_Data = num_data / setting.Repeating_number
        panda_db = self.mp.making_dataframe_per_step(setting, Num_Data)
        return panda_db


if __name__ == "__main__":
    print("RepeatDynamics")
    start = time.time()
    setting = Setting_Simulation_Value.Setting_Simulation_Value()
    p = 0.2
    v = 0.5
    repeat = RepeatDynamics()
    result = repeat.select_repeat_dynamics(setting, p, v, 0)
    print(result)
    end = time.time()
    print(end - start)
