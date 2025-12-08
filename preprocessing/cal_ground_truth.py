import pandas as pd
import numpy as np
from scipy.stats import norm

def cal_ground_truth(df_dat, data_name):

    def decide_long(row, data_name, w_qt_h,w_qt_l):
        if data_name == 'KuaiComt':
            if row['comment_stay_time'] > w_qt_h:
                return 1
            else:
                return 0
    w_qt_h = df_dat['play_time_truncate'].quantile(0.70) #23
    w_qt_l = df_dat['play_time_truncate'].quantile(0.50) #5
    print(w_qt_h, w_qt_l)
    if data_name == 'KuaiComt':
        w_qt_h = df_dat['comment_stay_time'].quantile(0.70) #23
        w_qt_l = df_dat['comment_stay_time'].quantile(0.50) #5
    print(w_qt_h, w_qt_l)
    df_dat['long_view2'] = df_dat.apply(lambda row: decide_long(row,data_name,w_qt_h,w_qt_l), axis=1)
    return df_dat

if __name__=="__main__":
    pass