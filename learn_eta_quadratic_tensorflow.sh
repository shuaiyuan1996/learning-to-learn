#!/bin/bash
#SBATCH --job-name=leqtf
#SBATCH --time=100:00:00
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --partition=compsci-gpu
#SBATCH --output=job/job%j.out

python3 learn_eta_quadratic_tensorflow.py \
--ndim 20 --t 80 --eta0 0.1 --num_tasks 1000 \
--meta_train_opt SGD --meta_train_opt_lr 0.001 \
--train_dir /usr/xtmp/shuai/lol_work/learn_eta_quadratic_0609/

python3 learn_eta_quadratic.py \
--ndim 20 --t 80 --eta0 0.1 --num_tasks 1000 \
--train_dir /usr/xtmp/shuai/lol_work/learn_eta_quadratic_0609/
