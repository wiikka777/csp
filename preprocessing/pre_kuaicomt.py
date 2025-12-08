import pandas as pd
import numpy as np
import datetime

def contain_ls(ls):
    result_ls = []
    for x in ls:
        result_ls.extend(x)
    return result_ls

def compare_max(cat_ls, frac_dict):
    frac_ls = np.array([frac_dict[c] for c in cat_ls])
    cat_ls = np.array(cat_ls)
    frac_sort_cat_ls = cat_ls[np.argsort(frac_ls)][::-1]
    return frac_sort_cat_ls[0]

def get_range_label(days):
    if 8 <= days <= 14:
        return '8-14'
    elif 15 <= days <= 30:
        return '15-30'
    elif 31 <= days <= 60:
        return '31-60'
    elif 61 <= days <= 90:
        return '61-90'
    elif 91 <= days <= 180:
        return '91-180'
    elif 181 <= days <= 365:
        return '181-365'
    elif 366 <= days <= 730:
        return '366-730'
    else:  # days > 730
        return '730+'

def pre_kuaicomt():

    # KuaiComt
    df_kuaiComt_interaction_ls = []
    file_names = ['../rec_datasets/KuaiComt/user_photo_final_filtered_table.csv']
    for file_name in file_names:
        df_kuaiComt_interaction_ls.append(pd.read_csv(file_name, sep='\t'))
    df_kuaiComt_usr_fe = pd.read_csv('../rec_datasets/KuaiComt/user_table_final.csv',sep='\t')
    df_kuaiComt_video_fe_basic = pd.read_csv('../rec_datasets/KuaiComt/photo_table_final.csv',sep='\t', lineterminator='\n')


    df_kuaiComt_interaction_standard = pd.concat(df_kuaiComt_interaction_ls,axis=0)

    df_kuaiComt_interaction_standard['play_time_truncate'] = df_kuaiComt_interaction_standard.apply(lambda row:row['play_time_ms'] if row['play_time_ms']<row['duration_ms'] else row['duration_ms'],axis=1)
    df_kuaiComt_interaction_standard['duration_ms'] = np.round(df_kuaiComt_interaction_standard['duration_ms'].values/1e3)
    df_kuaiComt_interaction_standard['play_time_ms'] = np.round(df_kuaiComt_interaction_standard['play_time_ms'].values/1e3)
    df_kuaiComt_interaction_standard['play_time_truncate'] = np.round(df_kuaiComt_interaction_standard['play_time_truncate'].values/1e3)

    df_kuaiComt_interaction_standard['comment_stay_time'] = df_kuaiComt_interaction_standard['comment_stay_time'].values/1e3
    df_kuaiComt_interaction_standard['open_comment'] =df_kuaiComt_interaction_standard['comment_stay_time'].apply(lambda x: 1 if x>0 else 0)

    df_kuaiComt_interaction_standard['date'] = df_kuaiComt_interaction_standard['time_ms'].apply(lambda x:datetime.datetime.fromtimestamp(x/1000).strftime('%Y%m%d%H'))


    # preprocess the user feature
    dic_user_active_degree = {'full_active':4,'high_active':3, 'middle_active':2,'2_14_day_new':0,'low_active':1,'single_low_active':1,'30day_retention':0,'day_new':0, 'UNKNOWN':0}
    df_kuaiComt_usr_fe['user_active_degree'] = df_kuaiComt_usr_fe['user_active_degree'].apply(lambda x: dic_user_active_degree[x])

    dic_follow_user_num_range = {'0':0, '(0,10]':1, '(10,50]':2, '(50,100]':3, '(100,150]':4, '(150,250]':5, '(250,500]':6, '500+':7}
    df_kuaiComt_usr_fe['follow_user_num_range'] = df_kuaiComt_usr_fe['follow_user_num_range'].apply(lambda x: dic_follow_user_num_range[x])

    dic_fans_user_num_range = {'0':0, '[1,10)':1, '[10,100)':2, '[100,1k)':3, '[1k,5k)':4, '[5k,1w)':5, '[1w,10w)':6,'[10w,100w)':6,'[100w,1000w)':6}
    df_kuaiComt_usr_fe['fans_user_num_range'] = df_kuaiComt_usr_fe['fans_user_num_range'].apply(lambda x: dic_fans_user_num_range[x])

    dic_friend_user_num_range = {'0':0, '[1,5)':1, '[5,30)':2, '[30,60)':3, '[60,120)':4, '[120,250)':5, '250+':6}
    df_kuaiComt_usr_fe['friend_user_num_range'] = df_kuaiComt_usr_fe['friend_user_num_range'].apply(lambda x: dic_friend_user_num_range[x])

    dic_register_days_range = {'8-14':0,'15-30':0, '31-60':1, '61-90':2, '91-180':3, '181-365':4, '366-730':5, '730+':6}
    df_kuaiComt_usr_fe['register_days_range'] = df_kuaiComt_usr_fe['register_days'].apply(lambda x: dic_register_days_range[get_range_label(x)])

    # preprocess the video feature
    dic_video_type = {'NORMAL':1,'AD':0,'UNKNOWN':0}
    df_kuaiComt_video_fe_basic['video_type'] = df_kuaiComt_video_fe_basic['photo_type'].apply(lambda x: dic_video_type[x])
    df_kuaiComt_video_fe_basic['category'] = df_kuaiComt_video_fe_basic['category'].apply(lambda x: str(x).split(','))

    total_ls = contain_ls(df_kuaiComt_video_fe_basic['category'].values)
    stat_series = pd.Series(total_ls).value_counts()
    count_info = dict(zip(stat_series.index,stat_series.values))

    df_kuaiComt_video_fe_basic['category'] = df_kuaiComt_video_fe_basic['category'].apply(lambda x: compare_max(x, count_info))

    # merge the dataframe
    df_kuaiComt_interaction_standard = pd.merge(df_kuaiComt_interaction_standard, df_kuaiComt_usr_fe, on=['user_id'], how='left')
    df_kuaiComt_interaction_standard = pd.merge(df_kuaiComt_interaction_standard, df_kuaiComt_video_fe_basic, on=['photo_id'], how='left')
    df_kuaiComt_interaction_standard['video_id'] = df_kuaiComt_interaction_standard['photo_id']

    # select duration range and featrues
    df_sel_dat = df_kuaiComt_interaction_standard[(df_kuaiComt_interaction_standard['duration_ms']>=5) & (df_kuaiComt_interaction_standard['duration_ms']<=400)]
    df_sel_dat = df_sel_dat[['date', 'time_ms', 'user_id','video_id','author_id','category','video_type',
                            'is_like','is_follow','is_comment','is_forward','is_hate',
                            'profile_stay_time','comment_stay_time','follow_user_num_range','register_days_range',
                            'fans_user_num_range','friend_user_num_range','user_active_degree','duration_ms','play_time_truncate','play_time_ms','open_comment',
                             'sampled_comments_reindexed','user_clicked','comments_score','comment0_id','comment1_id','comment2_id','comment3_id','comment4_id','comment5_id']]
    df_sel_dat['category'] =  df_sel_dat['category'].apply(lambda x: 999 if pd.isna(x) else x)

    user_id_map = dict(zip(np.sort(df_sel_dat['user_id'].unique()),range(len(df_sel_dat['user_id'].unique()))))
    video_id_map = dict(zip(np.sort(df_sel_dat['video_id'].unique()),range(len(df_sel_dat['video_id'].unique()))))
    author_id_map = dict(zip(np.sort(df_sel_dat['author_id'].unique()),range(len(df_sel_dat['author_id'].unique()))))
    category_map = dict(zip(np.sort(df_sel_dat['category'].unique()),range(len(df_sel_dat['category'].unique()))))

    df_sel_dat['user_id'] = df_sel_dat['user_id'].apply(lambda x: user_id_map[x])
    df_sel_dat['video_id'] = df_sel_dat['video_id'].apply(lambda x: video_id_map[x])
    df_sel_dat['author_id'] = df_sel_dat['author_id'].apply(lambda x: author_id_map[x])
    df_sel_dat['category'] = df_sel_dat['category'].apply(lambda x: category_map[x])

    return df_sel_dat

if __name__=="__main__":
    pass