#!/bin/bash
set -e
set -x

gpu_id=0
lambda1=0.001
lambda2=0.1

randseed=61
dataname="KuaiComt"
windows_size=3
eps=0.5

# nohup python -u prepare_data.py --group_num 60 --windows_size ${windows_size} --eps ${eps} --dat_name ${dataname} --is_load 0 &> kuaicomt_dataset.log &
#python prepare_data.py --group_num 60 --windows_size ${windows_size} --eps ${eps} --dat_name ${dataname} --is_load 0

randseed=61
c_inv=40
sigma=2
epo_nm=6
groupnum=30

modelname="DCN"
labelname="WLR"
label1name="user_clicked"
label2name="comments_score"

# CUDA_VISIBLE_DEVICES=${gpu_id} nohup python -u main.py --fout ../rec_datasets/WM_KuaiComt/${modelname}_${labelname}_${lambda1}_${lambda2}_test_${c_inv}_${sigma}_${randseed} --dat_name ${dataname} --model_name ${modelname} --label_name ${labelname} --randseed ${randseed} --load_to_eval 0 --epoch_num ${epo_nm} --label1_name ${label1name} --label2_name ${label2name} --lambda1 ${lambda1} --lambda2 ${lambda2} > log/output_${modelname}_${labelname}.log 2>&1 &
CUDA_VISIBLE_DEVICES=${gpu_id} python main.py --fout ../rec_datasets/WM_KuaiComt/${modelname}_${labelname}_${lambda1}_${lambda2}_test_${c_inv}_${sigma}_${randseed} --dat_name ${dataname} --model_name ${modelname} --label_name ${labelname} --randseed ${randseed} --load_to_eval 0 --epoch_num ${epo_nm} --label1_name ${label1name} --label2_name ${label2name} --lambda1 ${lambda1} --lambda2 ${lambda2}