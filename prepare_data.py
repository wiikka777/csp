import argparse
import numpy as np
import pandas as pd
from argparse import ArgumentTypeError
from preprocessing.pre_kuaicomt import pre_kuaicomt
from preprocessing.cal_baseline_label import cal_baseline_label
from preprocessing.cal_gmm_label import cal_gmm_label
from preprocessing.cal_ground_truth import cal_ground_truth

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise ArgumentTypeError('Boolean value expected.')

def main():
    parser = argparse.ArgumentParser(description="prepare datasets")
    parser.add_argument('-g', '--group_num', type=int, default=30, help="Groups of percentile_label")
    parser.add_argument('-t', '--windows_size', type=int, default=10, help='Windows size of moving average')
    parser.add_argument('-e', '--eps', type=float, default=0.3, help='smooth scale of GMM')
    parser.add_argument('--bias_point', type=float, default=0.15, help='smooth scale of GMM')
    parser.add_argument('--noise_point', type=float, default=10, help='smooth scale of GMM')
    parser.add_argument('--dat_name', type=str, default='KuaiRand', choices=['KuaiRand', 'WeChat','KuaiShou2018','KuaiComt'])
    parser.add_argument('--is_load', type=str2bool, nargs='?', default=False)
    args = parser.parse_args()

    group_num = args.group_num
    dat_name = args.dat_name
    windows_size = args.windows_size
    eps = args.eps
    is_load = args.is_load
    bias_point = args.bias_point
    noise_point = args.noise_point

    if dat_name == 'KuaiComt':
        if is_load == True:
            print('Load Processed Data...')
            kuaicomt_dat = pd.read_csv('../rec_datasets/WM_KuaiComt/KuaiComt_subset.csv')

            print('Cal Ground Truth Labels...')
            kuaicomt_dat = cal_ground_truth(kuaicomt_dat, dat_name)

            kuaicomt_dat.to_csv('../rec_datasets/WM_KuaiComt/KuaiComt_subset.csv')
            print(kuaicomt_dat.head(10))
        else:
            print('Load Raw Data...')
            kuaicomt_dat = pre_kuaicomt()

            print('Cal Baseline Labels...')
            kuaicomt_dat = cal_baseline_label(kuaicomt_dat, group_num, dat_name, windows_size)

            print('Cal Ground Truth Labels...')
            kuaicomt_dat = cal_ground_truth(kuaicomt_dat, dat_name)

            kuaicomt_dat.to_csv('../rec_datasets/WM_KuaiComt/KuaiComt_subset.csv')
            print(kuaicomt_dat.head(10))

if __name__ == "__main__":
    main()