from __future__ import print_function

import os, time
import numpy as np
import tensorflow as tf
import argparse

import metaopt
from optimizer import global_learning_rate
from problems import problem_sets as ps
from problems import problem_spec, datasets
from problems import problem_generator as pg

# config
parser = argparse.ArgumentParser()
parser.add_argument('--ndim', type=int, default=20, help="Dimension of the quadratic problem")
parser.add_argument('--t', type=int, default=80, help="Inner training length")
parser.add_argument('--eta0', type=float, default=0.1, help="The initial eta")
parser.add_argument('--num_tasks', type=int, default=1000, help="The number of tasks, i.e. the number of meta steps")
parser.add_argument('--meta_train_opt', default="SGD", help="The optimizer used to train the trainable optimizer")
parser.add_argument('--meta_train_opt_lr', default=0.001, help="The learning rate of the meta training process")
parser.add_argument('--train_dir', default="/usr/xtmp/shuai/lol_work/", help="Directory to store parameters and results")

args = parser.parse_args()
ndim = args.ndim
t = args.t
eta0 = args.eta0
num_tasks = args.num_tasks

t_start_str = time.strftime("%Y-%m-%d_%H%M%S", time.localtime(time.time()))

tf.app.flags.DEFINE_string("train_dir", args.train_dir, """Directory to store parameters and results.""")
tf.app.flags.DEFINE_integer("task", 0, """Task id of the replica running the training.""")
tf.app.flags.DEFINE_integer("worker_tasks", 1, """Number of tasks in the worker job.""")
tf.app.flags.DEFINE_integer("num_problems", num_tasks, """Number of sub-problems to run.""")
tf.app.flags.DEFINE_integer("num_meta_iterations", 1, """Number of meta-iterations to optimize.""")

# Metaoptimization parameters
tf.app.flags.DEFINE_float("meta_learning_rate", args.meta_train_opt_lr, """The learning rate for the meta-optimizer.""")
tf.app.flags.DEFINE_float("gradient_clip_level", 1e4, """The level to clip gradients to.""")

# Optimizer parameters: major features
tf.app.flags.DEFINE_boolean("use_log_objective", True, """Whether to use the log of the scaled objective rather than just the scaled obj for training.""")
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
    
    # The fixed starting point of each inner trajectory
    w0 = np.random.normal(size=(ndim)) 
    X = np.random.normal(size=(ndim, ndim))
    H = X.T.dot(X) # The quadratic problem
    
    # get the optimizer class and arguments
    optimizer_cls = global_learning_rate.GlobalLearningRate
    optimizer_args = (eta0, )
    optimizer_kwargs = {
        "use_attention": FLAGS.use_attention,
        "use_log_objective": FLAGS.use_log_objective,
        "use_second_derivatives": FLAGS.use_second_derivatives,
        "use_numerator_epsilon": FLAGS.use_numerator_epsilon
    }
    optimizer_spec = problem_spec.Spec(
            optimizer_cls, optimizer_args, optimizer_kwargs)
    
    problems_and_data = [(problem_spec.Spec(pg.MyQuadratic, (H, w0), {}), None, None)]
    logdir = os.path.join(FLAGS.train_dir, "dim{}_t{}_init_eta{}_tf_{}/".format(ndim, t, eta0, args.meta_train_opt))
    #logdir = os.path.join(FLAGS.train_dir, "learn_eta_quadratic/try/")
    
    ####################################################################
    # make log directory
    if os.path.exists(logdir) and logdir[-5:] != "/try/":
        raise NameError("The log directory {} already exists!".format(logdir))
    tf.io.gfile.makedirs(logdir)

    is_chief = FLAGS.task == 0
    # if this is a distributed run, make the chief run through problems in order
    select_random_problems = FLAGS.worker_tasks == 1 or not is_chief

    def num_unrolls():
        # In this experiment, we only have one unroll for each trajectory
        return 1

    def num_partial_unroll_itrs(itr=None, total=None):
        # In this experiment, we have fixed the unroll length to t
        return t

    with open(os.path.join(logdir, 'flags.txt'), 'a+') as flag_out:
        print_flags(flag_out)
        flag_out.write("optimizer_cls: {}\n".format(optimizer_cls.__name__))
        flag_out.write("optimizer_args: {}\noptimizer_kwargs: \n".format(optimizer_args))
        for key, value in optimizer_kwargs.items():
            flag_out.write("    {}: {}\n".format(key, value))
        flag_out.write("problems: {}\n".format(problems_and_data))

    # train
    with open(os.path.join(logdir, 'log.txt'), 'w') as out:
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
            learning_rate=FLAGS.meta_learning_rate,
            gradient_clip=FLAGS.gradient_clip_level,
            is_chief=is_chief,
            select_random_problems=select_random_problems,
            callbacks=[],
            out=out,
            meta_method="last",  # Whether the meta objective is the average of all inner objectives or just the last inner objective
            meta_train_opt=args.meta_train_opt
        )
        
        t_end = time.time()
        t_end_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(t_end))
        out.write("\nProgram completed at {}. Total running time: {}s.".format(t_end_str, int(t_end - t_start)))

    return 0


if __name__ == "__main__":
    tf.compat.v1.app.run()
    