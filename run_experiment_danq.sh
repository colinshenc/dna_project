#!/bin/bash

cd /ubc/cs/research/shield/projects/cshen001/dna_project/human_mouse_dna_classification/

CUDA_VISIBLE_DEVICES=1 /ubc/cs/research/shield/projects/cshen001/anaconda3/envs/fal_gan_py3.7/bin/python train.py \
--model 'danq' --num_epochs 60 --batch_size 100 --lr 1e-3 --feature_multiplier -1  \
--exp_name '' --data_file 'dataset_chr_left_out_1_bigger_test_no_mega.h5' --init_func 'no_init'
#--mpool_kernel_size 2 --mpool_stride 2 --kernel_size 19 #--resume #--init_func ortho