# Copyright 2017 Google, Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#         http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Scripts for meta-optimization."""

from __future__ import print_function

import os, time

import tensorflow as tf
import numpy as np
import pandas as pd
import argparse

import metaopt
from optimizer import mlp
from optimizer import global_learning_rate
from problems import problem_sets as ps
from problems import problem_spec as pspec
from problems import datasets
from problems import problem_generator as pg

# config
parser = argparse.ArgumentParser()
parser.add_argument('--nsample', type=int, default=1000, help="The total number of samples")
parser.add_argument('--noise', type=float, default=0, help="The noise level (probability to flip the label)")
parser.add_argument('--logdir', default="/usr/xtmp/shuai/lol_work/", help="The directory that has the model to test")
parser.add_argument('--niter', type=int, default=4000, help="The number of iterations for each trajectory")
parser.add_argument('--ntest', type=int, default=5, help="The number of independent tests")

args = parser.parse_args()
subset_n_sample = args.nsample
noise = args.noise
num_iter = args.niter
logdir = args.logdir

tf.app.flags.DEFINE_integer("task", 0, """Task id of the replica running the training.""")
tf.app.flags.DEFINE_integer("worker_tasks", 1, """Number of tasks in the worker job.""")

FLAGS = tf.app.flags.FLAGS

def main(unused_argv):

    problem_spec = pspec.Spec(pg.FullyConnected, (28 * 28, 10), {"hidden_sizes": (100, 20), "activation": tf.nn.relu})
    dataset = datasets.mnist_dataset_noised(noise_ratio=noise)
    dataset_test = datasets.mnist_dataset_noised(split="test", noise_ratio=noise)
    
    subset_n_class = -1 # we use all the classes
    batch_size = 32
    
    # get the optimizer class and arguments, learned optimizer
    optimizer_cls = mlp.MLP
    optimizer_args = (32, 1)
    optimizer_kwargs = {
        "use_attention": False,
        "use_log_objective": False,
        "use_second_derivatives": False,
        "use_numerator_epsilon": False,
        "decay": [0.5, 0.9, 0.99, 0.999, 0.9999],
        "step_multiplier": 0.001, 
        "magnitude_rate": 0.001
    }
    optimizer_spec = pspec.Spec(optimizer_cls, optimizer_args, optimizer_kwargs)   
    
    #######################################################################
    # test our optimizers
    optimizer = optimizer_spec.build()
    for i in range(args.ntest):
        problem = problem_spec.build()

        objective_values, parameters, records = metaopt.test_optimizer(
            optimizer=optimizer,
            problem=problem,
            num_iter=num_iter,
            subset_n_class=subset_n_class,
            subset_n_sample=subset_n_sample,
            dataset=dataset,
            dataset_test=dataset_test,
            batch_size=batch_size,
            logdir=logdir,
            record_every=1)

        pd.DataFrame(records["start_record"]).to_csv(os.path.join(logdir, "result{}_start_record_{}.csv".format(i, subset_n_sample)), index=False)
        pd.DataFrame(records["regular_record"]).to_csv(os.path.join(logdir, "result{}_regular_record_{}.csv".format(i, subset_n_sample)), index=False)
    
    #######################################################################
    # whether to test the baseline SGD; turn on or off by hand
    if True:
        # We use a default global learning rate optimizer as the SGD baseline
        # We fix the initial learning rate and don't train it, making it equivalent to a normal SGD
        optimizer_spec = pspec.Spec(global_learning_rate.GlobalLearningRate, (), {
                "initial_rate": 0.1,
                "use_attention": False,
                "use_log_objective": False,
                "obj_train_max_multiplier": -1,
                "use_second_derivatives": False,
                "use_numerator_epsilon": False,
        })
        optimizer = optimizer_spec.build()
        problem = problem_spec.build()

        objective_values, parameters, records = metaopt.test_optimizer(
            optimizer=optimizer,
            problem=problem,
            num_iter=num_iter,
            subset_n_class=subset_n_class,
            subset_n_sample=subset_n_sample,
            dataset=dataset,
            dataset_test=dataset_test,
            batch_size=batch_size,
            logdir=None, # No model needed
            record_every=1)
            
        pd.DataFrame(records["start_record"]).to_csv(os.path.join(logdir, "result_SGD_start_record_{}.csv".format(subset_n_sample)), index=False)
        pd.DataFrame(records["regular_record"]).to_csv(os.path.join(logdir, "result_SGD_regular_record_{}.csv".format(subset_n_sample)), index=False)
    
    return 0
    

if __name__ == "__main__":
    tf.compat.v1.app.run()
    
