from __future__ import print_function

import os, time
import argparse
import numpy as np
import pandas as pd
from problems import datasets
from collections import defaultdict

# config
parser = argparse.ArgumentParser()
parser.add_argument('--ndim', type=int, default=1000, help="Dimension of the quadratic problem")
parser.add_argument('--t', type=int, default=40, help="Inner training length")
parser.add_argument('--nsample', type=int, default=500, help="The total number of samples")
parser.add_argument('--sigma', type=float, default=1, help="The noise standard deviation")
parser.add_argument('--num_tasks', type=int, default=300, help="The number of tasks for training")
parser.add_argument('--num_tasks_test', type=int, default=10, help="The number of tasks for testing; We repeat this number of independent tests")
parser.add_argument('--train_dir', default="/usr/xtmp/shuai/lol_work/", help="Directory to store parameters and results")

args = parser.parse_args()
ndim = args.ndim
unroll_length = args.t
nsample = args.nsample
sigma = args.sigma
num_tasks = args.num_tasks
num_tasks_test = args.num_tasks_test

logdir = os.path.join(args.train_dir, "dim{}_sigma{}_sample{}/".format(ndim, sigma, nsample))

if not os.path.exists(logdir):
    os.mkdir(logdir)
log_path = os.path.join(logdir, "metaobj.csv")
eta_path = os.path.join(logdir, "eta_learned.csv")

def rmse(w, dataset):
    return np.sqrt(np.mean((dataset.data.dot(w) - dataset.labels.reshape(-1)) ** 2))

def inner_objective(w, dataset):
    return np.mean((dataset.data.dot(w) - dataset.labels.reshape(-1)) ** 2) / 2

def inner_gradient(w, dataset):
    return (dataset.data.T.dot(dataset.data.dot(w) - dataset.labels.reshape(-1))) / dataset.size

# For train-by-train, we assign the same dataset to dataset_train and dataset_val
def inner_train(dataset_train, dataset_val, eta):
    w = np.zeros(ndim, dtype=np.float32)
    for i in range(unroll_length):
        w_grad = inner_gradient(w, dataset_train)
        w = w - eta * w_grad
        if np.linalg.norm(w) >= 40 * sigma:
            u = np.random.normal(size=(ndim))
            u = u / np.linalg.norm(u)
            w = 40 * sigma * u
            break
    return inner_objective(w, dataset_val)
    
def meta_objective(eta):
    TbT_objs = []
    TbV_objs = []
    for i in range(num_tasks):
        w = np.random.normal(size=(ndim))
        w = w / np.linalg.norm(w)
        dataset = datasets.lr_dataset(w, nsample, sigma)
        
        # Train by train
        TbT_objs.append(inner_train(dataset, dataset, eta))
        
        # Train by val
        dataset_train, dataset_val = dataset.train_test_split(test_size=0.5)
        TbV_objs.append(inner_train(dataset_train, dataset_val, eta))
    
    return np.mean(TbT_objs), np.mean(TbV_objs)
    
def test(dataset_train, dataset_test, eta):
    w = np.zeros(ndim, dtype=np.float32)
    test_record = {}
    test_record["iter"] = [0]
    test_record["objective"] = [inner_objective(w, dataset_train)]
    test_record["rmse_train"] = [rmse(w, dataset_train)]
    test_record["rmse_test"] = [rmse(w, dataset_test)]
    
    for i in range(unroll_length):
        w_grad = inner_gradient(w, dataset_train)
        w = w - eta * w_grad
        test_record["iter"].append(i + 1)
        test_record["objective"].append(inner_objective(w, dataset_train))
        test_record["rmse_train"].append(rmse(w, dataset_train))
        test_record["rmse_test"].append(rmse(w, dataset_test))
        
    return test_record
    
if __name__ == "__main__":
    eta_list = list(np.logspace(-6, 0, 25))
    res = defaultdict(list)
    
    for eta in eta_list:
        metaobj_TbT, metaobj_TbV = meta_objective(eta)
        res["eta"].append(eta)
        res["metaobj_TbT"].append(metaobj_TbT)
        res["metaobj_TbV"].append(metaobj_TbV)
    
    eta_TbT = res["eta"][np.argmin(res["metaobj_TbT"])]
    eta_TbV = res["eta"][np.argmin(res["metaobj_TbV"])]
    
    pd.DataFrame(res).to_csv(log_path, index=False)
    pd.DataFrame({"TbT": [eta_TbT], "TbV": [eta_TbV]}).to_csv(eta_path, index=False)
    
    # Test the best eta
    for i in range(num_tasks_test):
        w = np.random.normal(size=(ndim))
        w = w / np.linalg.norm(w)
        dataset_train = datasets.lr_dataset(w, nsample, sigma)
        dataset_test = datasets.lr_dataset(w, nsample, sigma)
    
        test_record_TbT = test(dataset_train, dataset_test, eta_TbT)
        test_record_TbV = test(dataset_train, dataset_test, eta_TbV)
    
        pd.DataFrame(test_record_TbT).to_csv(os.path.join(logdir, "TbT_test{}.csv".format(i)), index=False)
        pd.DataFrame(test_record_TbV).to_csv(os.path.join(logdir, "TbV_test{}.csv".format(i)), index=False)    
