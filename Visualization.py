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
    def plot_2D_for_average_state(self, df, p_values=None, v_values=None):  # v_values =[]
        marker = ['-o', '-x', '-v', '-^', '-s', '-d']
        plt.style.use('seaborn-whitegrid')
        ax = fig.add_subplot(111)
        ax.tick_params(axis='both', labelsize=14)
        if p_values is not None:
            p_list = Visualization.making_select_list(df, 'p')
            temp_values = Visualization.covert_to_select_list_value(p_list, p_values)
            for i, p_value in enumerate(temp_values):
                df1 = df[df.p == p_value]
                df1 = df1.sort_values(by='v', ascending=True)
                plt.plot(df1['v'], df1['AS'], marker[i], label=r'$p$=%.2f' % p_value,
                         markersize=6, linewidth=1.5, markeredgewidth=1)
                plt.xlabel(r'$v$', fontsize=18, labelpad=4)
                plt.legend(framealpha=1, frameon=True, prop={'size': 12})
                plt.ylim(-1.5, 1.5)
                plt.ylabel('AS', fontsize=18, labelpad=4)
        elif v_values is not None:
            v_list = Visualization.making_select_list(df, 'v')
            temp_values = Visualization.covert_to_select_list_value(v_list, v_values)
            for i, v_value in enumerate(temp_values):
                df1 = df[df.v == v_value]
                df1 = df1.sort_values(by='p', ascending=True)
                plt.plot(df1['p'], df1['AS'], marker[i], label=r'$v$=%.2f' % v_value,
                         markersize=6, linewidth=1.5, markeredgewidth=1)
                plt.xlabel(r'$p$', fontsize=18, labelpad=4)
                plt.legend(framealpha=1, frameon=True, prop={'size': 12})
                plt.ylim(-1.5, 1.5)
                plt.ylabel('AS', fontsize=18, labelpad=4)
        elif p_values is None and v_values is None:
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
            # plt.clabel(contours, inline=True, fontsize=8)

    def plot_3D_for_average_state(self, df, chart_type):
        plt.style.use('seaborn-whitegrid')
        ax = plt.axes(projection='3d')
        if chart_type == 'scatter':
            ax.scatter(df['v'], df['p'], df['AS'], c=df['AS'], cmap='RdBu', linewidth=0.2)
        elif chart_type == 'trisurf':
            ax.plot_trisurf(df['v'], df['p'], df['AS'], cmap='RdBu', edgecolor='none')
        elif chart_type == 'contour':
            v_list = Visualization.making_select_list(df, 'v')  # list이지만 실제로는 array
            p_list = Visualization.making_select_list(df, 'p')
            X, Y = np.meshgrid(v_list, p_list)
            Z = Visualization.state_list_function(df, p_list, v_list)
            ax.contour3D(X, Y, Z, 50, cmap='RdBu')
        ax.set_xlabel(r'$v$', fontsize=18, labelpad=8)
        ax.set_ylabel(r'$prob.p$', fontsize=18, labelpad=8)
        ax.set_zlabel('AS', fontsize=18, labelpad=8)
        ax.set_title(r'$v$-$prob.p$-AS', fontsize=18)
        ax.tick_params(axis='both', labelsize=14)
        ax.view_init(45, 45)

    def timeflow_chart(self, setting, df, x_list=0, y_list=0, p_values=(0, 1), v_values=(0, 1), order=(False, 1),
                       keynode_method=(False, 0), keynode_number=(False, 1)):
        p_list = Visualization.making_select_list(df, 'p')
        v_list = Visualization.making_select_list(df, 'v')
        temp_p_values = Visualization.covert_to_select_list_value(p_list, p_values)
        temp_v_values = Visualization.covert_to_select_list_value(v_list, v_values)
        if p_values == (0, 1) and v_values == (0, 1):
            df1 = df[df.p >= p_list[0]]
            df2 = df1[df1.p <= p_list[-1]]
            df3 = df2[df2.v >= v_list[0]]
            pv_df = df3[df3.v <= v_list[-1]]
            p_df = sorted(pv_df['p'].unique())
            v_df = sorted(pv_df['v'].unique())
            print(p_df, v_df)
            for i in p_df:
                for j in v_df:
                    pv_df1 = pv_df[pv_df.p == i]
                    pv_df2 = pv_df1[pv_df1.v == j]
                    orders = pv_df2['Order'].unique()
                    for ordering in orders:
                        pv_df3 = pv_df2[pv_df2.Order == ordering]
                        pv_df4 = pv_df3[pv_df3.keynode_method == setting.select_method_list[keynode_method[1]]]
                        pv_df4 = pv_df4.sort_values(by='Steps', ascending=True)
                        plt.plot(pv_df4['Steps'], pv_df4['AS'], linewidth=1.5)
        else:
            for i in range(len(temp_p_values)):
                df1 = df[df.p == temp_p_values[i]]
                pv_df = df1[df1.v == temp_v_values[i]]
                if order[0] is True:
                    orders = pv_df['Order'].unique()
                    for ordering in orders:
                        pv_df2 = pv_df[pv_df.Order == ordering]
                        pv_df3 = pv_df2[pv_df2.keynode_method == setting.select_method_list[keynode_method[1]]]
                        pv_df3 = pv_df3.sort_values(by='Steps', ascending=True)
                        plt.plot(pv_df3['Steps'], pv_df3['AS'], label=r'%s' % ordering, linewidth=1.5)
                        plt.legend(framealpha=1, frameon=True, prop={'size': 10})
                elif order[0] is False:
                    pv_df2 = pv_df[pv_df.Order == setting.step_list[order[1]]]
                    if keynode_method[0] is True:
                        key_methods = pv_df['keynode_method'].unique()
                        for key_method in key_methods:
                            pv_df3 = pv_df2[pv_df2.keynode_method == key_method]
                            if keynode_number[0] is False:
                                pv_df4 = pv_df3[pv_df3.keynode_number == keynode_method[1]]
                                pv_df4 = pv_df4.sort_values(by='Steps', ascending=True)
                                plt.plot(pv_df4['Steps'], pv_df4['AS'], label=r'%s' % key_method,
                                         linewidth=1.5)
                                plt.legend(framealpha=1, frameon=True, prop={'size': 10})
                            elif keynode_number[0] is True:
                                pv_df4 = pv_df3[pv_df3.Steps == setting.Limited_step]
                                pv_df4 = pv_df4.sort_values(by='keynode_number', ascending=True)
                                plt.plot(pv_df4['keynode_number'] / setting.A_node, pv_df4['AS'],
                                         marker='o', label=r'%s' % key_method, linewidth=1.5)
                                plt.ylim(-1.2, 1.2)
                                plt.legend(framealpha=1, frameon=True, prop={'size': 10})
        # plt.xlabel('%s' % setting.x_list[x_list], fontsize=18, labelpad=6)
        # plt.ylabel('%s' % setting.y_list[y_list], fontsize=18, labelpad=6)


    def average_state_for_steps_regarding_order(self, df, p_value, v_value):
        v_list = Visualization.making_select_list(df, 'v')  # list이지만 실제로는 array
        p_list = Visualization.making_select_list(df, 'p')
        p = Visualization.covert_to_select_list_value(p_list, p_value)
        v = Visualization.covert_to_select_list_value(v_list, v_value)
        df1 = df[df.p == p[0]]
        df2 = df1[df1.v == v[0]]
        orders = df['Order'].unique()
        for order in orders:
            df3 = df2[df2.Order == order]
            plt.plot(df3['Steps'], df3['AS'], label=r'%s' % order, linewidth=1.5)
        plt.legend(framealpha=1, frameon=True, prop={'size': 10})
        plt.ylabel('AS', fontsize=18, labelpad=6)
        plt.xlabel('time(step)', fontsize=18, labelpad=6)
        plt.title('AS comparison according to dynamics order')
        plt.text(20, -0.13, r'$p=%.2f, v=%.2f$' % (p[0], v[0]))


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
    def covert_to_select_list_value(select_list, input_values):  # list가 만들어져 있는 곳에 사용
        temp_value = 0
        if len(input_values) == 1:
            for input_value in input_values:
                loc = np.sum(select_list <= input_value)  # select_list는 making_select_list를 사용, array로 만들어져 있음
                temp_v = select_list[loc-1]
                temp_value = [temp_v]
        elif len(input_values) > 1:
            temp_value = []
            for input_value in input_values:
                loc = np.sum(select_list <= input_value)
                temp_v = select_list[loc-1]
                temp_value.append(temp_v)
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
    setting.database = 'pv_variable'
    setting.table = 'keynode_table'   #'step_same_table'  #'comparison_os_table'
    select_db = SelectDB.SelectDB()
    df = select_db.select_data_from_DB(setting)
    df = df.fillna(0)
    visualization = Visualization()
    fig = plt.figure()
    sns.set()
    # visualization.plot_3D_to_2D_contour_for_average_state(df)
    # visualization.plot_2D_for_average_state(df)
    # select_list = Visualization.making_select_list(df, 'v')
    # temp = Visualization.covert_to_select_list_value(select_list, [0.1, 0.2])
    visualization.timeflow_chart(setting, df, x_list=0, y_list=0, p_values=(0, 1), v_values=(0, 1),
                                 order=(False, 1), keynode_method=(False, 0), keynode_number=(False, 1))
    # visualization.average_state_for_steps_regarding_order(df, [0.4], [0.4])
    # visualization.average_state_for_steps_regarding_order(df, [0.5], [0.5])

    # visualization.average_state_for_steps_scale(df, [0, 1], [0, 1])
    plt.show()
    plt.close()

    print("paint finished")