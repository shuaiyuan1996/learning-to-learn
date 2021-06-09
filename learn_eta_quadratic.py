from __future__ import print_function

import os, time
import argparse
import numpy as np
import pandas as pd
from collections import defaultdict

# config
parser = argparse.ArgumentParser()
parser.add_argument('--ndim', type=int, default=20, help="Dimension of the quadratic problem")
parser.add_argument('--t', type=int, default=80, help="Inner training length")
parser.add_argument('--eta0', type=float, default=0.1, help="The initial eta")
parser.add_argument('--num_tasks', type=int, default=1000, help="The number of tasks, i.e. the number of meta steps")
parser.add_argument('--train_dir', default="/usr/xtmp/shuai/lol_work/", help="Directory to store parameters and results")

args = parser.parse_args()
ndim = args.ndim
t = args.t
eta0 = args.eta0
num_tasks = args.num_tasks

def inner_objective(w, H):
    return w.T.dot(H).dot(w) / 2

def inner_gradient(w, H):
    return H.dot(w)

def inner_train(w0, H, eta):
    w = w0
    inner_objective_list = []
    for i in range(t):
        w_grad = inner_gradient(w, H)
        w = w - eta * w_grad
        inner_objective_list.append(inner_objective(w, H))
        
    return inner_objective(w, H)

def meta_objective(eta, w0, H):
    return np.log(inner_train(w0, H, eta)) / t
      
def meta_gradient(eta, w0, H):
    lam, _ = np.linalg.eig(H)
    L = lam[0]
    alpha = lam[-1]
    
    if eta <= 2 / (alpha + L):
        scale = 1 - eta * alpha
    else:
        scale = eta * L - 1
    
    it = (np.eye(ndim) - eta * H) / scale
    it_2t_1 = np.eye(ndim)
    for i in range(2 * t - 1):
        it_2t_1 = it_2t_1.dot(it)
    it_2t = it_2t_1.dot(it)

    numerator = w0.T.dot(it_2t_1).dot(H).dot(H).dot(w0) / scale
    denominator = w0.T.dot(it_2t).dot(H).dot(w0)
    return -2 * numerator / denominator
  
if __name__ == "__main__":
    
    # The fixed starting point of each inner trajectory
    w0 = np.random.normal(size=(ndim)) 
    X = np.random.normal(size=(ndim, ndim))
    H = X.T.dot(X) # The quadratic problem
    
    log_path = os.path.join(args.train_dir, "dim{}_t{}_init_eta{}.csv".format(ndim, t, eta0))
    
    eta = eta0
    meta_grad_cur = meta_gradient(eta0, w0, H)
    res = defaultdict(list)
    res["iter"].append(0)
    res["eta"].append(eta0)
    res["metaobj"].append(meta_grad_cur)
    for i in range(1, num_tasks + 1):
        eta  = eta - 1 / np.sqrt(i) / 100 * meta_grad_cur
        if eta < 0:
            eta = 0
        res["iter"].append(i)
        res["eta"].append(eta)
        meta_grad_cur = meta_gradient(eta, w0, H)
        res["metaobj"].append(meta_grad_cur)

    pd.DataFrame(res).to_csv(log_path, index=False)

