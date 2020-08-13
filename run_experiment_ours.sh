#!/bin/bash

cd /ubc/cs/research/shield/projects/cshen001/dna_project/human_mouse_dna_classification/

CUDA_VISIBLE_DEVICES=1 /ubc/cs/research/shield/projects/cshen001/anaconda3/envs/fal_gan_py3.7/bin/python train.py \
--model 'ours' --num_epochs 10 --batch_size 100 --lr 3e-3 \
--exp_name '' --kernel_size 19 --data_file 'dataset_chr_left_out_1_bigger_test_no_mega.h5' --mpool_kernel_size 2 \
--mpool_stride 2 --init_func 'no_init' #--resume #--init_func ortho