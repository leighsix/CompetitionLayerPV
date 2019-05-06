import SelectDB
import numpy as np
import Setting_Simulation_Value
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
import pandas as pd
from sympy import *
from matplotlib import cycler
from mpl_toolkits.mplot3d.axes3d import *
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
matplotlib.use("TkAgg")


class Visualization:
    def plot_2D_p_for_average_state(self, df, v_values):  # v_values =[]
        marker = ['-o', '-x', '-v', '-^', '-s', '-d']
        plt.style.use('seaborn-whitegrid')
        ax = fig.add_subplot(111)
        ax.tick_params(axis='both', labelsize=14)
        v_list = Visualization.making_select_list(df, 'v')
        for i, v_value in enumerate(v_values):
            temp_value = Visualization.covert_to_select_list_value(v_list, v_value)
            df = df[df.v == temp_value]
            plt.plot(df['p'], df['AS'], marker[i], label=r'$v$=%.2f' % temp_value,
                     markersize=6, linewidth=1.5, markeredgewidth=1)
        plt.legend(framealpha=1, frameon=True, prop={'size': 12})
        plt.ylim(-1.5, 1.5)
        plt.xlabel(r'$p$', fontsize=18, labelpad=4)
        plt.ylabel('AS', fontsize=18, labelpad=4)


    def plot_2D_v_for_average_state(self, df, p_values):
        marker = ['-o', '-x', '-v', '-^', '-s', '-d']
        plt.style.use('seaborn-whitegrid')
        ax = fig.add_subplot(111)
        ax.tick_params(axis='both', labelsize=14)
        p_list = Visualization.making_select_list(df, 'p')
        for i, p_value in enumerate(p_values):
            temp_value = Visualization.covert_to_select_list_value(p_list, p_value)
            df = df[df.p == temp_value]
            plt.plot(df['v'], df['AS'], marker[i], label=r'$p$=%.2f' % temp_value,
                     markersize=6, linewidth=1.5, markeredgewidth=1)
        plt.legend(framealpha=1, frameon=True, prop={'size': 12})
        plt.ylim(-1.5, 1.5)
        plt.xlabel(r'$v$', fontsize=18, labelpad=4)
        plt.ylabel('AS', fontsize=18, labelpad=4)



    def plot_3D_scatter_for_average_state(self, df):
        plt.style.use('seaborn-whitegrid')
        ax = plt.axes(projection='3d')
        ax.scatter(df['v'], df['p'], df['AS'], c=df['AS'], cmap='RdBu', linewidth=0.2)
        ax.set_xlabel(r'$v$', fontsize=18, labelpad=8)
        ax.set_ylabel(r'$prob.p$', fontsize=18, labelpad=8)
        ax.set_zlabel('AS', fontsize=18, labelpad=8)
        ax.set_title(r'$v$-$prob.p$-AS', fontsize=18)
        ax.tick_params(axis='both', labelsize=14)
        ax.view_init(45, 45)


    def plot_3D_trisurf_for_average_state(self, df):
        plt.style.use('seaborn-whitegrid')
        ax = plt.axes(projection='3d')
        ax.plot_trisurf(df['v'], df['p'], df['AS'], cmap='RdBu', edgecolor='none')
        ax.set_xlabel(r'$v$', fontsize=18, labelpad=8)
        ax.set_ylabel(r'$prob.p$', fontsize=18, labelpad=8)
        ax.set_zlabel('AS', fontsize=18, labelpad=8)
        ax.set_title(r'$v$-$prob.p$-AS', fontsize=18)
        ax.tick_params(axis='both', labelsize=14)
        ax.view_init(45, 45)


    def plot_3D_contour_for_average_state(self, df):
        plt.style.use('seaborn-whitegrid')
        v_list = Visualization.making_select_list(df, 'v')  # list이지만 실제로는 array
        p_list = Visualization.making_select_list(df, 'p')
        X, Y = np.meshgrid(v_list, p_list)
        Z = Visualization.state_list_function(df, p_list, v_list)
        ax = plt.axes(projection='3d')
        ax.contour3D(X, Y, Z, 50, cmap='RdBu')
        ax.set_xlabel(r'$v$', fontsize=18, labelpad=6)
        ax.set_ylabel(r'$prob.p$', fontsize=18, labelpad=6)
        ax.set_zlabel('AS', fontsize=18, labelpad=6)
        ax.set_title(r'$v$-$prob.p$-AS', fontsize=18)
        ax.view_init(45, 45)

    def plot_3D_to_2D_contour_for_average_state(self, df):
        plt.style.use('seaborn-whitegrid')
        ax = fig.add_subplot(111)
        ax.tick_params(axis='both', labelsize=14)
        v_list = Visualization.making_select_list(df, 'v')  # list이지만 실제로는 array
        p_list = Visualization.making_select_list(df, 'p')
        X, Y = np.meshgrid(v_list, p_list)
        Z = Visualization.state_list_function(df, p_list, v_list)
        plt.contourf(X, Y, Z, 50, cmap='RdBu')
        cb = plt.colorbar()
        cb.set_label(label='AS', size=15, labelpad=10)
        cb.ax.tick_params(labelsize=12)
        plt.clim(-1, 1)
        plt.xlabel(r'$v$', fontsize=18, labelpad=6)
        plt.ylabel(r'$p$', fontsize=18, labelpad=6)
        #plt.clabel(contours, inline=True, fontsize=8)

    def average_state_for_steps(self, df, v_value, p_value):
        v_list = Visualization.making_select_list(df, 'v')  # list이지만 실제로는 array
        p_list = Visualization.making_select_list(df, 'p')
        temp_v_value = Visualization.covert_to_select_list_value(v_list, v_value)
        temp_p_value = Visualization.covert_to_select_list_value(p_list, p_value)
        df1 = df[df.p == temp_p_value]
        df2 = df1[df1.v == temp_v_value]
        df3 = df2.sort_values(by='Steps', ascending=True)
        plt.plot(df3['Steps'], df3['AS'], linestyle=':', marker='o', markersize=2, linewidth=0.3)
        plt.ylabel('A', fontsize=18, labelpad=4)
        plt.xlabel('time(step)', fontsize=18, labelpad=4)


    def flow_prob_v_chart(self, df, v_values, p_values):
        v_list = Visualization.making_select_list(df, 'v')  # list이지만 실제로는 array
        p_list = Visualization.making_select_list(df, 'p')
        v_min = Visualization.covert_to_select_list_value(v_list, v_values[0])
        v_max = Visualization.covert_to_select_list_value(v_list, v_values[1])
        p_min = Visualization.covert_to_select_list_value(p_list, p_values[0])
        p_max = Visualization.covert_to_select_list_value(p_list, p_values[1])
        df = df[df.p >= p_min]
        df = df[df.p <= p_max]
        df = df[df.v >= v_min]
        df = df[df.v <= v_max]
        p_df = df['p'].drop_duplicates()
        v_df = df['v'].drop_duplicates()
        p_array = np.array(p_df)
        v_array = np.array(v_df)
        for i in sorted(p_array):
            for j in sorted(v_array):
                df1 = df[df.p == i]
                df2 = df1[df1.v == j]
                plt.plot(df2['Steps'], df2['prob_v'], '-', markersize=2, linewidth=0.3)
        plt.ylabel('probability for layer B', fontsize=18, labelpad=4)
        plt.xlabel('time(step)', fontsize=18, labelpad=4)

    def average_state_for_steps_scale(self, df, v_values, p_values):
        v_list = Visualization.making_select_list(df, 'v')  # list이지만 실제로는 array
        p_list = Visualization.making_select_list(df, 'p')
        v_min = Visualization.covert_to_select_list_value(v_list, v_values[0])
        v_max = Visualization.covert_to_select_list_value(v_list, v_values[1])
        p_min = Visualization.covert_to_select_list_value(p_list, p_values[0])
        p_max = Visualization.covert_to_select_list_value(p_list, p_values[1])
        df = df[df.p >= p_min]
        df = df[df.p <= p_max]
        df = df[df.v >= v_min]
        df = df[df.v <= v_max]
        p_df = df['p'].drop_duplicates()
        v_df = df['v'].drop_duplicates()
        p_array = np.array(p_df)
        v_array = np.array(v_df)
        for i in sorted(p_array):
            for j in sorted(v_array):
                df1 = df[df.p == i]
                df2 = df1[df1.v == j]
                plt.plot(df2['Steps'], df2['AS'], linewidth=0.5)
        plt.ylabel('AS', fontsize=18, labelpad=6)
        plt.xlabel('time(step)', fontsize=18, labelpad=6)

    def average_state_for_steps_regarding_order(self, df, p_value, v_value):
        v_list = Visualization.making_select_list(df, 'v')  # list이지만 실제로는 array
        p_list = Visualization.making_select_list(df, 'p')
        v = Visualization.covert_to_select_list_value(v_list, v_value)
        p = Visualization.covert_to_select_list_value(p_list, p_value)
        df = df[df.v == v]
        df = df[df.p == p]
        orders = df['Order'].unique()
        for order in orders:
            df1 = df[df.Order == order]
            print(len(df1))
            plt.plot(df1['Steps'], df1['AS'], label=r'%s' % order, linewidth=1.5)
        plt.legend(framealpha=1, frameon=True, prop={'size': 12})
        plt.ylabel('AS', fontsize=18, labelpad=6)
        plt.xlabel('time(step)', fontsize=18, labelpad=6)


    @staticmethod
    def state_list_function(df, p_list, v_list):
        Z = np.zeros([len(p_list), len(v_list)])
        for i, p in enumerate(p_list):
            for j, v in enumerate(v_list):
                df1 = df[df.p == p]
                df2 = df1[df1.v == v]
                if len(df2) == 0:
                    Z[i][j] = 0
                else:
                    Z[i][j] = df2['AS'].iloc[0]
        return Z

    @staticmethod
    def covert_to_select_list_value(select_list, input_value):  # list가 만들어져 있는 곳에 사용
        loc = np.sum(select_list <= input_value)  # select_list는 making_select_list를 사용, array로 만들어져 있음
        temp_value = select_list[loc - 1]
        return temp_value

    @staticmethod
    def making_select_list(df, list_name):
        list = []
        df = df[list_name]
        select_list = np.array(df.drop_duplicates())
        for i in range(len(select_list)):
            list.append(select_list[i])
        return np.array(sorted(list))

if __name__ == "__main__":
    print("Visualization")
    setting = Setting_Simulation_Value.Setting_Simulation_Value()
    # print(df)
    # array1 = Visualization.making_select_list(df, 'p')
    # array2 = Visualization.making_select_list(df, 'v')
    # temp1 = Visualization.covert_to_select_list_value(array1, 0.1)
    # temp2 = Visualization.covert_to_select_list_value(array2, 0.6)
    # df1 = df[df.p == temp1]
    # df2 = df1[df1.v == temp2]
    # print(df2)
    # setting.database = 'pv_variable'
    # setting.table = 'step_same_table'
    setting.database = 'paper_revised_data'
    setting.table = 'simulation_table3'
    select_db = SelectDB.SelectDB()
    df = select_db.select_data_from_DB(setting)
    df = df[df.Steps == 100]

    visualization = Visualization()
    fig = plt.figure()
    sns.set()
    visualization.plot_3D_trisurf_for_average_state(df)

    # visualization.average_state_for_steps_regarding_order(df, 0.5, 0.5)
    # visualization.average_state_for_steps_scale(df, [0, 1], [0, 1])
    plt.show()
    plt.close()



    #previous_research

    # visualization.average_state_for_steps(setting, df, [0, 0.5], [0, 0.5])
    # fig = plt.figure()
    #
    # plt.show()
    # plt.close()
    #visualization.plot_3D_to_2D_contour_for_average_state(setting, df)
    #visualization.plot_3D_to_2D_contour_for_average_state()
    #visualization.plot_3D_contour_for_average_state('previous_research')
    #visualization.plot_3D_scatter_for_average_state('average_layer_state')    #previous_research
    # fig = visualization.flow_prob_beta_chart([0, 3], [0, 2])
    # fig = visualization.different_state_ratio_chart([0, 3], [0, 2], 'B')
    # visualization.plot_2D_beta_for_average_state(0.2)
    # visualization.plot_2D_beta_for_average_state(0.4)

    #visualization.plot_2D_beta_for_average_state('previous_research', 0.4)
    #visualization.plot_2D_beta_for_average_state('previous_research', 0.6)
    print("paint finished")