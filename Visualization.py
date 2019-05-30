import SettingSimulationValue
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
import random
import pandas as pd
import sqlalchemy
import sqlite3
from sympy import *
from matplotlib import cycler
from mpl_toolkits.mplot3d.axes3d import *
matplotlib.use("TkAgg")

marker = ['-o', '-x', '-v', '-^', '-s', '-d', '']
linestyle = ['-r', '--r', '-.r', ':r', '-g', '--g', '-.g', ':g', '-b', '--b', '-.b', ':b', '-c', '--c', '-.c',
             ':c', '-m', '--m', '-.m', ':m', '-y', '--y', '-.y', ':y', '-k', '--k', '-.k', ':k']
random.shuffle(linestyle)

x_list = ['Steps', 'keynode_number', 'keyedge_number']
y_list = ['AS', 'prob_v', 'persuasion', 'compromise', 'change_count', 'consensus_index']

class Visualization:
    def __init__(self, setting):
        self.df = Visualization.select_data_from_DB(setting)

    def run(self, setting, plot_type='2D', p_value_list=None, v_value_list=None, steps_2d=100,
            chart_type='scatter', steps_3d=100,
            x_index=0, y_index=0, p_values=(0, 1), v_values=(0, 1), order=(False, 1),
            keynode_method=False, select_layer='A_layer', keynode_number=(False, 1),
            keyedge_method=False, select_edge_layer='A_internal', keyedge_number=(False, 1), steps_timeflow=100,
            steps_hist=100):
        Visualization.making_chart(self.df, setting, plot_type, p_value_list, v_value_list, steps_2d,
                                   chart_type, steps_3d, x_index, y_index, p_values, v_values, order,
                                   keynode_method, select_layer, keynode_number,
                                   keyedge_method, select_edge_layer, keyedge_number, steps_timeflow,
                                   steps_hist)
        plt.show()
        plt.close()

    @staticmethod
    def making_chart(df, setting, plot_type, p_value_list, v_value_list, steps_2d,
                     chart_type, steps_3d, x_index, y_index, p_values, v_values, order,
                     keynode_method, select_layer, keynode_number,
                     keyedge_method, select_edge_layer, keyedge_number, steps_timeflow,
                     steps_hist):
        if plot_type == '2D':
            Visualization.plot_2D_for_average_state(df, p_value_list, v_value_list, steps_2d)
        elif plot_type == '3D':
            Visualization.plot_3D_for_average_state(df, chart_type, steps_3d)
        elif plot_type == 'timeflow':
            Visualization.timeflow_chart(setting, df, x_index, y_index, p_values, v_values, order,
                                         keynode_method, select_layer, keynode_number,
                                         keyedge_method, select_edge_layer, keyedge_number, steps_timeflow)
        elif plot_type == 'hist':
            Visualization.making_mixed_hist(df, steps_hist)

    @staticmethod
    def plot_2D_for_average_state(df, p_value_list, v_value_list, steps_2d):  # v_values =[]
        fig = plt.figure()
        sns.set()
        plt.style.use('seaborn-whitegrid')
        ax = fig.add_subplot(111)
        ax.tick_params(axis='both', labelsize=14)
        df = df[df.Steps == steps_2d]
        if p_value_list is not None:
            p_list = Visualization.making_select_list(df, 'p')
            temp_values = Visualization.covert_to_select_list_value(p_list, p_value_list)
            for i, p_value in enumerate(temp_values):
                df1 = df[df.p == p_value]
                df1 = df1.sort_values(by='v', ascending=True)
                plt.plot(df1['v'], df1['AS'], marker[i], label=r'$p$=%.2f' % p_value,
                         markersize=6, linewidth=1.5, markeredgewidth=1)
                plt.xlabel(r'$v$', fontsize=18, labelpad=4)
                plt.legend(framealpha=1, frameon=True, prop={'size': 12})
                plt.ylim(-1.5, 1.5)
                plt.ylabel('AS', fontsize=18, labelpad=4)
        elif v_value_list is not None:
            v_list = Visualization.making_select_list(df, 'v')
            temp_values = Visualization.covert_to_select_list_value(v_list, v_value_list)
            for i, v_value in enumerate(temp_values):
                df1 = df[df.v == v_value]
                df1 = df1.sort_values(by='p', ascending=True)
                plt.plot(df1['p'], df1['AS'], marker[i], label=r'$v$=%.2f' % v_value,
                         markersize=6, linewidth=1.5, markeredgewidth=1)
                plt.xlabel(r'$p$', fontsize=18, labelpad=4)
                plt.legend(framealpha=1, frameon=True, prop={'size': 12})
                plt.ylim(-1.5, 1.5)
                plt.ylabel('AS', fontsize=18, labelpad=4)
        elif p_value_list is None and v_value_list is None:
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

    @staticmethod
    def plot_3D_for_average_state(df, chart_type, steps_3d):
        plt.figure()
        sns.set()
        plt.style.use('seaborn-whitegrid')
        ax = plt.axes(projection='3d')
        df = df[df.Steps == steps_3d]
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

    @staticmethod
    def select_data_from_DB(setting):
        select_query = ('''SELECT * FROM %s;''' % str(setting.table))
        engine = sqlalchemy.create_engine('mysql+pymysql://root:2853@localhost:3306/%s' % setting.database)
        query = select_query
        df = pd.read_sql_query(query, engine)
        df = df.fillna(0)
        df['consensus_index'] = ((df['A_plus'] * df['B_minus']) + (df['A_minus'] * df['B_plus'])) / (setting.A_node * setting.B_node)
        return df

    @staticmethod
    def timeflow_chart(setting, df, x_index, y_index, p_values, v_values, order,
                       keynode_method, select_node_layer, keynode_number,
                       keyedge_method, select_edge_layer, keyedge_number, steps_timeflow):
        plt.figure()
        sns.set()
        plt.style.use('seaborn-whitegrid')
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
            for i in p_df:
                for j in v_df:
                    pv_df1 = pv_df[pv_df.p == i]
                    pv_df2 = pv_df1[pv_df1.v == j]
                    orders = pv_df2['Orders'].unique()
                    for ordering in orders:
                        pv_df3 = pv_df2[pv_df2.Orders == ordering]
                        pv_df4 = pv_df3[pv_df3.keynode_method == '0']
                        pv_df4 = pv_df4.sort_values(by='Steps', ascending=True)
                        plt.plot(pv_df4[x_list[x_index]], pv_df4[y_list[y_index]], linewidth=1.5)
        else:
            for i in range(len(temp_p_values)):
                df1 = df[df.p == temp_p_values[i]]
                pv_df = df1[df1.v == temp_v_values[i]]
                if order is True:
                    orders = pv_df['Orders'].unique()
                    # orders = [r'$O(o, o) \to D(o)$', r'$O(r, r) \to D(o)$', r'$O(s, o) \to D(o)$', r'$O(o, s) \to D(o)$',
                    #           r'$O(s, s) \to D(o)$']
                    # # orders = [r'$O(o, o) \to D(o)$', r'$O(o, o) \leftarrow D(o)$', r'$O(s, o) \to D(s)$', r'$O(s, o) \leftarrow D(s)$']
                    # orders = [r'$O(o, o) \to D(o)$', r'$O(s, s) \leftrightarrow D(s)$', r'$O(r, r) \Leftrightarrow D(r)$']
                    # # orders = [r'$O(o, o) \to D(o)$', r'$O(o, o) \to D(s)$', r'$O(s, o) \to D(o)$', r'$O(o, s) \to D(o)$',
                    # #           r'$O(s, o) \to D(s)$', r'$O(o, s) \to D(s)$', r'$O(s, s) \to D(o)$',
                    # #           r'$O(s, s) \to D(s)$']
                    for style, ordering in enumerate(orders):
                        pv_df2 = pv_df[pv_df.Orders == ordering]
                        pv_df3 = pv_df2[pv_df2.keynode_method == '0']
                        pv_df3 = pv_df3.sort_values(by='Steps', ascending=True)
                        plt.plot(pv_df3[x_list[x_index]], pv_df3[y_list[y_index]], linestyle[style],
                                 label=r'%s' % ordering, linewidth=1.5)
                        plt.legend(framealpha=1, frameon=True, prop={'size': 10})
                elif order is False:
                    pv_df['Orders'] = r'$O(o, o) \to D(o)$'
                    pv_df2 = pv_df[pv_df.Orders == r'$O(o, o) \to D(o)$']
                    if keynode_method is True:
                        pv_df3 = pv_df2[pv_df.select_node_layer == select_node_layer]
                        key_methods = pv_df3['keynode_method'].unique()
                        for key_method in key_methods:
                            pv_df4 = pv_df3[pv_df3.keynode_method == key_method]
                            if keynode_number[0] is False:
                                pv_df5 = pv_df4[pv_df4.keynode_number == keynode_number[1]]
                                pv_df5 = pv_df5.sort_values(by='Steps', ascending=True)
                                plt.plot(pv_df5[x_list[x_index]], pv_df5[y_list[y_index]],
                                         label=r'%s(%s)' % (key_method, select_node_layer.split('_')[0]), linewidth=1.5)
                                plt.legend(framealpha=1, frameon=True, prop={'size': 10})
                            elif keynode_number[0] is True:
                                pv_df5 = pv_df4[pv_df4.Steps == steps_timeflow]
                                pv_df5 = pv_df5.sort_values(by='keynode_number', ascending=True)
                                plt.plot(pv_df5[x_list[x_index]] / setting.A_node, pv_df5[y_list[y_index]],
                                         marker='o', label=r'%s(%s)' % (key_method, select_node_layer.split('_')[0]), linewidth=1.5)
                                plt.legend(framealpha=1, frameon=True, prop={'size': 10})
                    if keyedge_method is True:
                        pv_df3 = pv_df2[pv_df.select_edge_layer == select_edge_layer]
                        key_methods = pv_df3['keyedge_method'].unique()
                        for key_method in key_methods:
                            pv_df4 = pv_df3[pv_df3.keyedge_method == key_method]
                            if keyedge_number[0] is False:
                                pv_df5 = pv_df4[pv_df4.keyedge_number == keyedge_number[1]]
                                pv_df5 = pv_df5.sort_values(by='Steps', ascending=True)
                                plt.plot(pv_df5[x_list[x_index]], pv_df5[y_list[y_index]],
                                         label=r'%s(%s)' % (key_method, select_node_layer.split('_')[0]), linewidth=1.5)
                                plt.legend(framealpha=1, frameon=True, prop={'size': 10})
                            elif keyedge_number[0] is True:
                                pv_df5 = pv_df4[pv_df4.Steps == steps_timeflow]
                                pv_df5 = pv_df5.sort_values(by='keyedge_number', ascending=True)
                                plt.plot(pv_df5[x_list[x_index]] / pv_df5['A_total_edges'], pv_df5[y_list[y_index]],
                                         marker='o', label=r'%s(%s)' % (key_method, select_edge_layer), linewidth=1.5)
                                plt.legend(framealpha=1, frameon=True, prop={'size': 10})
        # plt.xlabel('%s' % x_list[x_index], fontsize=16, labelpad=6)
        # plt.xlabel('ratio of unchanged nodes', fontsize=16, labelpad=6)
        plt.xlabel('ratio of removed edges', fontsize=16, labelpad=6)
        plt.ylabel('%s' % y_list[y_index], fontsize=16, labelpad=6)
        plt.title('AS comparison according to dynamics order')
        # plt.text(20, -0.13, r'$p=%.2f, v=%.2f$' % (p[0], v[0]))

    @staticmethod
    def making_mixed_hist(df, steps_hist):   # Model, ASs, PCRs, NCRs, CRs
        df_hist = Visualization.making_property_array_for_hist(df, steps_hist)
        fig = plt.figure()  # 그래프 창생성
        ax = fig.add_subplot(111)
        N = len(df_hist)
        tuples = Visualization.making_tuple_data_for_hist(df_hist)
        ASs = tuples[1]
        PCRs = tuples[2]  # 남학생 수
        NCRs = tuples[3]  # 여학생 수
        ind = np.arange(N)  # x축
        width = 0.2  # 너비
        p1 = ax.bar(ind - (width / 2), PCRs, width, color='SkyBlue')
        p2 = ax.bar(ind - (width / 2), NCRs, width, color='IndianRed', bottom=PCRs)
        p3 = ax.bar(ind + (width / 2), ASs, width, color='palegreen', label='AS')
        ax.set_xlabel('model', fontsize=16)  # x축 라벨
        ax.set_title('Competition results', fontsize=18)  # subplot의 제목
        ax.set_yticks(np.arange(0, 1.2, 0.2))  # 0 ~ 81까지 10간격식으로 y축 틱설정
        ax.set_xticks(ind)  # x축 틱설정
        ax.set_xticklabels(tuples[0])  # x축 틱 라벨설정
        ax.tick_params(labelsize=13)
        plt.legend((p1[0], p2[0], p3[0]), ("PCR", "NCR", "AS total"), loc=0, fontsize=14)

    @staticmethod
    def making_property_array_for_hist(df, steps_hist):
        property_array = np.zeros(5)
        df_step = df[df.Steps == steps_hist]
        model = df_step['Model']
        model = sorted(model.unique())
        for m in model:
            df_model = df_step[df_step.Model == m]
            AS_total = sum(df_model['AS']) / len(df_model)
            pcr, ncr = Visualization.calculate_consensus_number_for_hist(df_model)
            initial_value = np.array([m, AS_total, pcr, ncr, pcr + ncr])
            property_array = np.vstack([property_array, initial_value])
        property_data = property_array[1:]
        columns = ['Model', 'AS_total', 'PCR', 'NCR', 'CR']
        hist_df = pd.DataFrame(property_data, columns=columns)
        return hist_df

    @staticmethod
    def calculate_consensus_number_for_hist(df):
        pos_con = 0
        neg_con = 0
        AS_series= df['AS']
        for i in range(len(df)):
            if AS_series.iloc[i] > 0.95:
                pos_con += 1
            elif AS_series.iloc[i] < -0.95:
                neg_con += 1
        return pos_con / len(df), neg_con / len(df)

    @staticmethod
    def making_tuple_data_for_hist(df):
        Model = tuple([i for i in df['Model']])
        ASs = tuple([i for i in df['AS_total']])
        PCRs = tuple([i for i in df['PCR']])
        NCRs = tuple([i for i in df['NCR']])
        CRs = tuple([i for i in df['CR']])
        return Model, ASs, PCRs, NCRs, CRs

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
    setting = SettingSimulationValue.SettingSimulationValue()
    setting.database = 'pv_variable'
    # setting.table = 'comparison_order_table3'   #'step_same_table'  #'comparison_order_table3'
    setting.table = 'keyedge_table'
    visualization = Visualization(setting)
    visualization.run(setting, plot_type='timeflow', p_value_list=None, v_value_list=None, steps_2d=100,
                      chart_type='scatter', steps_3d=100,
                      x_index=2, y_index=0, p_values=[0.4], v_values=[0.4], order=False,
                      keynode_method=False, select_layer='A_layer', keynode_number=(False, 1),
                      keyedge_method=True, select_edge_layer='A_mixed', keyedge_number=(True, 1), steps_timeflow=100,
                      steps_hist=100)

    print("paint finished")
