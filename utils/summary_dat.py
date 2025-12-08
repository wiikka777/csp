import numpy as np
import pandas as pd

def make_feature(df, data_name):
    if data_name == 'KuaiComt':
        fe_names = ['user_id', 'follow_user_num_range','register_days_range', 'fans_user_num_range', 'friend_user_num_range','user_active_degree',
                    'video_id', 'author_id', ]
    return df[fe_names].values

def make_feature_with_comments(df, data_name):
    if data_name == 'KuaiComt':
        fe_names = ['user_id', 'follow_user_num_range','register_days_range', 'fans_user_num_range', 'friend_user_num_range','user_active_degree',
                    'video_id', 'author_id', 
                    'comment0_id', 'comment1_id', 'comment2_id', 'comment3_id', 'comment4_id', 'comment5_id',]
    return df[fe_names].values

def cal_field_dims(df, data_name):
    if data_name == 'KuaiComt':
        fe_names = ['user_id', 'follow_user_num_range','register_days_range', 'fans_user_num_range', 'friend_user_num_range','user_active_degree',
                    'video_id', 'author_id', ]
    field_dims = [int(df[fe].max()) + 1 for fe in fe_names]
    print(fe_names)
    print(field_dims)
    print([df[fe].max() for fe in fe_names])
    return field_dims

def cal_comments_dims(df, data_name):
    if data_name == 'KuaiComt':
        fe_names = ['comment0_id', 'comment1_id', 'comment2_id', 'comment3_id', 'comment4_id', 'comment5_id']
    
    # 计算每一列的最大值
    max_values = [df[fe].max() for fe in fe_names]
    field_dim = max(max_values) + 1  # 加 1 是为了将最大值作为合法索引
    
    print(f"Comments feature names: {fe_names}")
    print(f"Total unique field dimensions: {field_dim}")
    
    return field_dim