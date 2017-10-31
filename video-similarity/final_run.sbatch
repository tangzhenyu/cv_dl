#!/bin/bash
#SBATCH --partition=cosmos
#SBATCH --nodelist=blazar
#SBATCH --time=17:59:00
#SBATCH --gres=gpu:1,gpu_mem:5000
#SBATCH --mem=6000
#SBATCH --qos=intermediate

python train.py --return_outputs 1 --name=15a --lr 0.00001 --train_file_positive ./annotation_files/positive_annotations_train_val_generic.txt --train_file_negative  ./annotation_files/negative_annotations_train_val_generic.txt > result_15a
#python eval.py --model  --filename --label 

