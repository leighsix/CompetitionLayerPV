import pandas as pd
import numpy as np
import SettingSimulationValue
import sqlalchemy
import sqlite3


class SelectDB:
    def select_data_from_DB(self, setting):
        select_query = ('''SELECT * FROM %s;''' % str(setting.table))
        engine = sqlalchemy.create_engine('mysql+pymysql://root:2853@localhost:3306/%s' % setting.database)
        query = select_query
        df = pd.read_sql_query(query, engine)
        return df

    def select_data_from_setting(self, setting):
        select_query = ('''
            SELECT * FROM %s 
            WHERE Structure = '%s' AND A_internal_edges = %s AND B_internal_edges = %s 
            AND A_external_edges = %s AND B_external_edges = %s AND A_node_number = %s 
            AND B_node_number = %s;''' % (str(setting.table), str(setting.Structure), int(setting.A_edge),
                                          int(setting.B_edge), int(setting.A_inter_edges),
                                          int(setting.B_inter_edges), int(setting.A_node), int(setting.B_node)))
        engine = sqlalchemy.create_engine('mysql+pymysql://root:2853@localhost:3306/%s' % setting.database)
        query = select_query
        df = pd.read_sql_query(query, engine)
        return df


    def select_duplicates_from_DB(self, setting):
        duplicate_query = ('''
            SELECT * FROM %s
            GROUP BY 
                Structure, A_node_number, B_node_number, A_internal_edges, B_internal_edges,
                A_external_edges, B_external_edges, beta, gamma, Steps 
            HAVING  COUNT(Structure) > 1 AND COUNT(A_node_number) > 1 AND COUNT(B_node_number) > 1
                AND COUNT(A_internal_edges) >1 AND COUNT(A_external_edges) >1
                AND COUNT(B_internal_edges) >1 AND COUNT(B_external_edges) >1
                AND COUNT(beta) >1 AND COUNT(gamma) >1 AND COUNT(Steps) > 1; ''' % setting.table)
        engine = sqlalchemy.create_engine('mysql+pymysql://root:2853@localhost:3306/%s' % setting.database)
        query = duplicate_query
        df = pd.read_sql_query(query, engine)
        return df


class SelectSQlite:
    def select_data_from_sqlite(self, setting):
        con = sqlite3.connect("C:/Users/Purple/CompetingLayer/CompetitionLayer.db")
        select_query = ('''
            SELECT * FROM %s ;''' % str(setting.table))
        df = pd.read_sql(select_query, con, index_col=None)
        return df



if __name__ == "__main__":
    print("Select DB")
    setting = SettingSimulationValue.SettingSimulationValue()
    select = SelectSQlite()
    db = select.select_data_from_sqlite(setting)
    print(db)
