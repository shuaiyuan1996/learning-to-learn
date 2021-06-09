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
import numpy as np

import tensorflow as tf
import argparse

import metaopt
from optimizer import mlp
from problems import problem_sets as ps
from problems import problem_spec, datasets
from problems import problem_generator as pg

# config
parser = argparse.ArgumentParser()
parser.add_argument('--nsample', type=int, default=500, help="The total number of samples")
parser.add_argument('--mode', default="train", help="train: train-by-train; val: train-by-val")
parser.add_argument('--val_split', type=float, default=0.5, help="If train-by-val, the percentage of data to be split to the validation set")
parser.add_argument('--noise', type=float, default=0, help="The noise level (probability to flip the label)")
parser.add_argument('--logdir', default="/usr/xtmp/shuai/lol_work/", help="Directory to store parameters and results")

args = parser.parse_args()
nsample = args.nsample
noise = args.noise
logdir = args.logdir
val_split = args.val_split

t_start_str = time.strftime("%Y-%m-%d_%H%M%S", time.localtime(time.time()))

tf.app.flags.DEFINE_string("train_dir", "/usr/xtmp/shuai/lol_work", """Directory to store parameters and results.""")

tf.app.flags.DEFINE_integer("task", 0, """Task id of the replica running the training.""")
tf.app.flags.DEFINE_integer("worker_tasks", 1, """Number of tasks in the worker job.""")

tf.app.flags.DEFINE_integer("num_problems", 100, """Number of sub-problems to run.""")
tf.app.flags.DEFINE_integer("num_meta_iterations", 3, """Number of meta-iterations to optimize.""")
tf.app.flags.DEFINE_integer("num_unroll_scale", 30, """The scale parameter of the exponential
                                                    distribution from which the number of partial
                                                    unrolls is drawn""")
tf.app.flags.DEFINE_integer("min_num_unrolls", 10, """The minimum number of unrolls per problem.""")
tf.app.flags.DEFINE_integer("num_partial_unroll_itr_scale", 100,
                                                        """The scale parameter of the exponential
                                                            distribution from which the number of iterations
                                                            per unroll is drawn.""")
tf.app.flags.DEFINE_integer("min_num_itr_partial_unroll", 50,
                                                        """The minimum number of iterations for one unroll.""")

# Metaoptimization parameters
tf.app.flags.DEFINE_float("meta_learning_rate", 1e-2, """The learning rate for the meta-optimizer.""")
tf.app.flags.DEFINE_float("gradient_clip_level", 1e4, """The level to clip gradients to.""")

# Optimizer parameters: major features
tf.app.flags.DEFINE_boolean("use_log_objective", False, """Whether to use the log of the scaled objective
                                                             rather than just the scaled obj for training.""")
tf.app.flags.DEFINE_boolean("use_attention", False, """Whether to learn where to attend.""")
tf.app.flags.DEFINE_boolean("use_second_derivatives", False, """Whether to use second derivatives.""")
tf.app.flags.DEFINE_boolean("use_numerator_epsilon", False, """Whether to use epsilon in the numerator of the log objective.""")

FLAGS = tf.app.flags.FLAGS

def print_flags(out):
    for name, value in sorted(FLAGS.__flags.items()):
        value = value.value
        out.write(name + ': ' + str(value) + '\n')
    out.flush()


def main(unused_argv):
    # get the optimizer class and arguments
    optimizer_cls = mlp.MLP
    optimizer_args = (32, 1)
    optimizer_kwargs = {
        "use_attention": FLAGS.use_attention,
        "use_log_objective": FLAGS.use_log_objective,
        "use_second_derivatives": FLAGS.use_second_derivatives,
        "use_numerator_epsilon": FLAGS.use_numerator_epsilon,
        "decay": [0.5, 0.9, 0.99, 0.999, 0.9999],
        "step_multiplier": 0.001, 
        "magnitude_rate": 0.001
    }
    optimizer_spec = problem_spec.Spec(
            optimizer_cls, optimizer_args, optimizer_kwargs)
    
    # mnist 10 class mlp relu noised 
    problems_and_data = [(problem_spec.Spec(pg.FullyConnected, (28 * 28, 10), {"hidden_sizes": (100, 20), "activation": tf.nn.relu}), datasets.mnist_dataset_noised(noise_ratio=noise), 32)]
    
    ####################################################################
    # make log directory
    if os.path.exists(logdir) and logdir[-5:] != "/try/":
        raise NameError("The log directory {} already exists!".format(logdir))
    tf.io.gfile.makedirs(logdir)

    is_chief = FLAGS.task == 0
    # if this is a distributed run, make the chief run through problems in order
    select_random_problems = FLAGS.worker_tasks == 1 or not is_chief

    def num_unrolls():
        return metaopt.sample_numiter(FLAGS.num_unroll_scale, FLAGS.min_num_unrolls)

    def num_partial_unroll_itrs(itr=None, total=None):
        return metaopt.sample_numiter(FLAGS.num_partial_unroll_itr_scale, FLAGS.min_num_itr_partial_unroll)

    with open(os.path.join(logdir, 'flags.txt'), 'a+') as flag_out:
        print_flags(flag_out)
        flag_out.write("optimizer_cls: {}\n".format(optimizer_cls.__name__))
        flag_out.write("optimizer_args: {}\noptimizer_kwargs: \n".format(optimizer_args))
        for key, value in optimizer_kwargs.items():
            flag_out.write("    {}: {}\n".format(key, value))
        flag_out.write("problems: {}\n".format(problems_and_data))

    # train
    with open(os.path.join(logdir, 'log.txt'), 'w') as out:
        #optimizer_kwargs_ = optimizer_kwargs.copy()
        #optimizer_kwargs_['initial_log_rate'] = optimizer_kwargs['initial_log_rate']+ w_noise[i]
        #optimizer_spec = problem_spec.Spec(optimizer_cls, optimizer_args, optimizer_kwargs)
        t_start = time.time()
        t_start_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(t_start))
        out.write("Program starts at {}.\n".format(t_start_str))
        
        meta_grads_and_params = metaopt.train_optimizer(
            logdir,
            optimizer_spec,
            problems_and_data,
            FLAGS.num_problems,
            FLAGS.num_meta_iterations,
            num_unrolls,
            num_partial_unroll_itrs,
            subset_n_class=-1, # using all classes
            subset_n_sample=nsample,
            meta_loss_obj=args.mode,
            learning_rate=FLAGS.meta_learning_rate,
            gradient_clip=FLAGS.gradient_clip_level,
            is_chief=is_chief,
            select_random_problems=select_random_problems,
            callbacks=[],
            out=out,
            val_split = val_split)
        
        t_end = time.time()
        t_end_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(t_end))
        out.write("\nProgram completed at {}. Total running time: {}s.".format(t_end_str, int(t_end - t_start)))

    return 0


if __name__ == "__main__":
    tf.compat.v1.app.run()
    