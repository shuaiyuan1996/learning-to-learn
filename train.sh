#!/bin/bash
#SBATCH --job-name=nV8+4
#SBATCH --time=100:00:00
#SBATCH --mem=192G
#SBATCH --gres=gpu:1
#SBATCH --partition=compsci-gpu
#SBATCH --output=job/job%j.out

python3 train_mlp_optimizer.py \
--nsample 2000 --mode val --val_split 0.5 --noise 0 \
--logdir /usr/xtmp/shuai/lol_work/mlp_optimizer_mnist/final/TbV1000+1000/
