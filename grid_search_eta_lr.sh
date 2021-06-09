#!/bin/bash
#SBATCH --job-name=lgs2
#SBATCH --time=100:00:00
#SBATCH --mem=16G
#SBATCH --partition=compsci
#SBATCH --output=job/job%j.out

python3 grid_search_eta_lr.py \
--ndim 1000 --t 40 --nsample 500 --sigma 1 --num_tasks 300 --num_tasks_test 10 \
--train_dir /usr/xtmp/shuai/lol_work/grid_search_meta_lr/
