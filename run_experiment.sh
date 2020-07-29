#!/bin/bash

cd /ubc/cs/research/shield/projects/cshen001/dna_project/human_mouse_dna_classification/

CUDA_VISIBLE_DEVICES=3 /ubc/cs/research/shield/projects/cshen001/anaconda3/envs/fal_gan_py3.7/bin/python train.py \
--num_epochs 15 --batch_size 2048 --lr 1e-5 \
--exp_name '07-28-2020-20:20:26_bs_2048_lr_5e-05' --resume #--init_func ortho