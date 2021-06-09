#!/bin/bash
#SBATCH --job-name=10T10
#SBATCH --time=2:00:00
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --partition=compsci-gpu
#SBATCH --output=job/job%j.out


python3 test_mlp_optimizer.py \
--nsample 1000 --noise 0 --ntest 5 --niter 4000 \
--logdir /usr/xtmp/shuai/lol_work/mlp_optimizer_mnist/final/TbV1000+1000/
