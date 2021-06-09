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

"""Helper utilities for training and testing optimizers."""

from collections import defaultdict
import random
import sys, os
import time

import numpy as np
from six.moves import xrange
import tensorflow as tf

from optimizer import trainable_optimizer
from optimizer import utils
from problems import datasets
from problems import problem_generator

tf.app.flags.DEFINE_integer("ps_tasks", 0, """Number of tasks in the ps job. If 0 no ps job is used.""")
tf.app.flags.DEFINE_float("l2_reg", 0., """Lambda value for parameter regularization.""")
tf.app.flags.DEFINE_float("rms_decay", 0.9, """Decay value for the RMSProp metaoptimizer.""")
tf.app.flags.DEFINE_float("rms_epsilon", 1e-10, """Epsilon value for the RMSProp metaoptimizer.""")
tf.app.flags.DEFINE_boolean("set_profiling", False, """Enable memory usage and computation time tracing for tensorflow nodes (available in TensorBoard).""")
tf.app.flags.DEFINE_boolean("reset_rnn_params", False, """Reset the parameters of the optimizer from one meta-iteration to the next./ whether to reset the states only at the start of each problem; otherwise, at the start of each trajectory""")

FLAGS = tf.app.flags.FLAGS

OPTIMIZER_SCOPE = "LOL"
OPT_SUM_COLLECTION = "LOL_summaries"

CONFIG = tf.compat.v1.ConfigProto()
CONFIG.gpu_options.allow_growth = True
CONFIG.allow_soft_placement = True


def sample_numiter(scale, min_steps=50):
    """Samples a number of iterations from an exponential distribution.

    Args:
        scale: parameter for the exponential distribution
        min_steps: minimum number of steps to run (additive)

    Returns:
      num_steps: An integer equal to a rounded sample from the exponential
                 distribution + the value of min_steps.
    """
    if scale == 0:
        return min_steps
    else:
        return int(np.round(np.random.exponential(scale=scale)) + min_steps)


def train_optimizer(logdir,
                    optimizer_spec,
                    problems_and_data,
                    num_problems,
                    num_meta_iterations,
                    num_unroll_func,
                    num_partial_unroll_itrs_func,
                    subset_n_class=-1,    # sample subsets with how many label classes
                    learning_rate=1e-4,
                    gradient_clip=5.,
                    is_chief=False,
                    select_random_problems=True,
                    callbacks=None,
                    out=sys.stdout,
                    meta_loss_obj="train",
                    meta_method="average",
                    subset_n_sample=None,
                    sample_meta_loss=None,
                    val_split=0.5,
                    meta_train_opt="RMSProp"
                    ):
    """Trains the meta-parameters of this optimizer.

    Args:
        logdir: a directory filepath for storing model checkpoints (must exist)
        optimizer_spec: specification for an Optimizer (see utils.Spec)
        problems_and_data: a list of tuples containing three elements: a problem
            specification (see utils.Spec), a dataset (see datasets.Dataset), and
            a batch_size (int) for generating a problem and corresponding dataset. If
            the problem doesn't have data, set dataset to None.
        num_problems: the number of problems to sample during meta-training
        num_meta_iterations: the number of iterations (steps) to run the
            meta-optimizer for on each subproblem.
        num_unroll_func: called once per meta iteration and returns the number of
            unrolls to do for that meta iteration.
        num_partial_unroll_itrs_func: called once per unroll and returns the number
            of iterations to do for that unroll.
        learning_rate: learning rate of the RMSProp meta-optimizer (Default: 1e-4)
        gradient_clip: value to clip gradients at (Default: 5.0)
        is_chief: whether this is the chief task (Default: False)
        select_random_problems: whether to select training problems randomly
                (Default: True)
        callbacks: a list of callback functions that is run after every random
                problem draw
        obj_train_max_multiplier: the maximum increase in the objective value over
                a single training run. Ignored if < 0.
        out: where to write output to, e.g. a file handle (Default: sys.stdout)
        meta_loss_obj: when we compute the meta objective, do we use the inner train obj or inner val obj?
        sample_meta_loss: when we compute the meta objective, we need to first compute the inner obj over the whole training or validation set,
                which can be very hard due to memory issues. Therefore, do we sample a small batch to evaluate inner obj? if so, this is the
                number of samples; otherwise, None.
        val_split: if train-by-validation, the percentage of data that is split to the val set
        meta_train_opt: The optimizer used to perform the meta-training, i.e., the optimizer used to train the trainable optimizer

    Raises:
        ValueError: If one of the subproblems has a negative objective value.
    """

    if select_random_problems:
        # iterate over random draws of problem / dataset pairs
        sampler = (random.choice(problems_and_data) for _ in range(num_problems))
    else:
        # iterate over a random shuffle of problems, looping if necessary
        num_repeats = (num_problems / len(problems_and_data)) + 1
        random.shuffle(problems_and_data)
        sampler = (problems_and_data * num_repeats)[:num_problems]

    train_logger = tf.compat.v1.summary.FileWriter(os.path.join(logdir, 'train_summary'))
    if not sample_meta_loss is None:
        out.write("When computing the meta-objective, {} data will be sampled.\n".format(sample_meta_loss))

    for problem_itr, (problem_spec, raw_dataset, batch_size) in enumerate(sampler):
        
        # timer used to time how long it takes to initialize a problem
        out.write("--------- Problem #{} ---------\n".format(problem_itr))
        problem_start_time = time.time()
        
        # if dataset is None, use the EMPTY_DATASET
        if raw_dataset is None:
            dataset = datasets.EMPTY_DATASET
            dataset_val = datasets.EMPTY_DATASET
            batch_size = dataset.size
        else: 
            if hasattr(problem_spec.callable, "accuracy"): # for classification
                if meta_loss_obj == "train":
                    dataset, labels_chosen = raw_dataset.generate_subset_by_label(n_label=subset_n_class, n_sample=subset_n_sample)
                    dataset_val = dataset
                    out.write("Dataset class: {}\n".format(labels_chosen))
                    out.write("Training set number of samples: {}\n".format(dataset.size))

                elif meta_loss_obj == "val":
                    dataset, labels_chosen = raw_dataset.generate_subset_by_label(n_label=subset_n_class, n_sample=subset_n_sample)
                    out.write("Dataset class: {}\n".format(labels_chosen))
                    dataset, dataset_val = dataset.train_test_split(test_size=val_split)
                    out.write("Training set number of samples: {}\n".format(dataset.size))
                    out.write("Validation set number of samples: {}\n".format(dataset_val.size))
                else:
                    raise ValueError("Unknown meta loss objective type: {}".format(meta_loss_obj))
            else:  # for regression
                dataset = raw_dataset
                if meta_loss_obj == "train":
                    dataset_val = dataset
                    out.write("Training set number of samples: {}\n".format(dataset.size))
                elif meta_loss_obj == "val":
                    dataset, dataset_val = dataset.train_test_split(test_size=0.5)
                    out.write("Training set number of samples: {}\n".format(dataset.size))
                    out.write("Validation set number of samples: {}\n".format(dataset_val.size))
                else:
                    raise ValueError("Unknown meta loss objective type: {}".format(meta_loss_obj))

            # default batch size is the entire dataset
            batch_size = dataset.size if batch_size is None else batch_size


        ###################################################
        ##    start building a new graph for this problem
        ###################################################
        graph = tf.Graph()
        real_device_setter = tf.compat.v1.train.replica_device_setter(FLAGS.ps_tasks) # ?

        def custom_device_setter(op):
            # Places the local variables onto the workers.
            if trainable_optimizer.is_local_state_variable(op):
                return "/job:worker"
            else:
                return real_device_setter(op)

        if real_device_setter:
            device_setter = custom_device_setter
        else:
            device_setter = None

        with graph.as_default(), graph.device(device_setter):

            # initialize a problem
            problem = problem_spec.build()

            # build the optimizer
            opt = optimizer_spec.build()

            # get the meta-objective for training the optimizer
            train_output = opt.train(problem, dataset, dataset_val, sample_meta_loss=sample_meta_loss, meta_method=meta_method)

            state_keys = opt.state_keys
            for key, val in zip(state_keys, train_output.output_state[0]):
                if not val.dtype == tf.int32:        # skip finite check for int32 type (for learning_rate_schedule)
                    finite_val = utils.make_finite(val, replacement=tf.zeros_like(val))
                    tf.compat.v1.summary.histogram("State/{}".format(key), finite_val,
                                                             collections=[OPT_SUM_COLLECTION])

            tf.compat.v1.summary.scalar("MetaObjective", train_output.metaobj,
                                                collections=[OPT_SUM_COLLECTION])

            # Per-problem meta-objective
            tf.compat.v1.summary.scalar(problem_spec.callable.__name__ + "_MetaObjective",
                                                train_output.metaobj,
                                                collections=[OPT_SUM_COLLECTION])

            # create the meta-train_op
            global_step = tf.Variable(0, name="global_step", trainable=False)
            meta_parameters = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.GLOBAL_VARIABLES,
                                                                                    scope=OPTIMIZER_SCOPE)
            # parameter regularization
            reg_l2 = FLAGS.l2_reg * sum([tf.reduce_sum(param ** 2) for param in meta_parameters])

            # compute the meta-gradients
            if meta_train_opt == "RMSProp":
                meta_opt = tf.compat.v1.train.RMSPropOptimizer(learning_rate, decay=FLAGS.rms_decay, use_locking=True, epsilon=FLAGS.rms_epsilon)
            elif meta_train_opt == "SGD":
                meta_opt = tf.compat.v1.train.GradientDescentOptimizer(learning_rate, use_locking=True)
            else:
                raise ValueError("Unrecognized meta optimizer {}. Please choose from RMSProp and SGD".format(meta_train_opt))
            
            grads_and_vars = meta_opt.compute_gradients(train_output.metaobj + reg_l2, meta_parameters)

            # clip the gradients
            clipped_grads_and_vars = []
            for grad, var in grads_and_vars:
                clipped_grad = tf.clip_by_value(
                        utils.make_finite(grad, replacement=tf.zeros_like(var)),
                        -gradient_clip, gradient_clip)
                clipped_grads_and_vars.append((clipped_grad, var))

            # histogram summary of grads and vars
            for grad, var in grads_and_vars:
                var_name = '_'.join(var.name.split(":"))
                tf.compat.v1.summary.histogram(
                        var_name + "_rawgrad",
                        utils.make_finite(
                                grad, replacement=tf.zeros_like(grad)),
                        collections=[OPT_SUM_COLLECTION])
            for grad, var in clipped_grads_and_vars:
                var_name = '_'.join(var.name.split(":"))
                tf.compat.v1.summary.histogram(var_name + "_var", var,
                                                         collections=[OPT_SUM_COLLECTION])
                tf.compat.v1.summary.histogram(var_name + "_grad", grad,
                                                         collections=[OPT_SUM_COLLECTION])

            # builds the train and summary operations
            train_op = meta_opt.apply_gradients(clipped_grads_and_vars, global_step=global_step)

            # only grab summaries defined for LOL, not inside the problem
            summary_op = tf.compat.v1.summary.merge_all(key=OPT_SUM_COLLECTION)

            # make sure the state gets propagated after the gradients and summaries
            # were computed.
            with tf.control_dependencies([train_op, summary_op]):
                propagate_loop_state_ops = []
                for dest, src in zip(
                        train_output.init_loop_vars, train_output.output_loop_vars):
                    propagate_loop_state_ops.append(dest.assign(src))
                propagate_loop_state_op = tf.group(*propagate_loop_state_ops)

            ###################################################
            ## graph set up; start the training session
            ###################################################
            # create the supervisor
            sv = tf.compat.v1.train.Supervisor(
                graph=graph,
                is_chief=is_chief,
                logdir=logdir,
                save_summaries_secs=0,        # only create the summary folder; we write summaries manually
                summary_writer=None,
                save_model_secs=0,            # we save checkpoints manually
                global_step=global_step
            )

            with sv.managed_session(config=CONFIG) as sess:
                
                timing = defaultdict(list)
                init_time = time.time() - problem_start_time
                if problem_spec.callable.__name__ != "MyQuadratic":
                    out.write("Model: {callable.__name__}{args}{kwargs}\n".format(
                        **problem_spec.get_dict()))
                out.write("Took {} seconds to initialize.\n".format(init_time))
                out.flush()

                # For profiling summaries
                if FLAGS.set_profiling:
                    summary_writer = tf.summary.FileWriter(os.path.join(logdir, 'problem{}_summary'.format(problem_itr)), graph=sess.graph)

                # used to store information during training
                metadata = defaultdict(list)

                for k in range(num_meta_iterations):

                    if sv.should_stop():
                        break

                    problem.init_fn(sess)

                    # set run options (for profiling)
                    full_trace_opt = tf.compat.v1.RunOptions(trace_level=tf.compat.v1.RunOptions.FULL_TRACE)
                    run_options = full_trace_opt if FLAGS.set_profiling else None
                    run_metadata = tf.RunMetadata() if FLAGS.set_profiling else None

                    num_unrolls = num_unroll_func()
                    partial_unroll_iters = [
                        num_partial_unroll_itrs_func(it, num_unrolls) for it in xrange(num_unrolls)
                    ]
                    total_num_iter = sum(partial_unroll_iters)

                    objective_weights = [np.ones(num) / float(num) for num in partial_unroll_iters]
                    db = dataset.batch_indices(total_num_iter, batch_size)
                    dataset_batches = []
                    last_index = 0
                    for num in partial_unroll_iters:
                        dataset_batches.append(db[last_index:last_index + num])
                        last_index += num

                    train_start_time = time.time()
                    additional_log_info = ""

                    for unroll_itr in range(num_unrolls):
                        first_unroll = unroll_itr == 0
                        if FLAGS.reset_rnn_params:
                            reset_state = first_unroll and k == 0
                        else:
                            reset_state = first_unroll
                        
                        if sample_meta_loss is not None:
                            dataset_batches_val = []
                            for _ in range(partial_unroll_iters[unroll_itr]): 
                                dataset_batches_val.append(np.random.choice(len(dataset_val.labels), size=sample_meta_loss, replace=False))
                        else:
                            dataset_batches_val = [] # not used if sample_meta_loss == None

                        feed = {
                            train_output.obj_weights: objective_weights[unroll_itr],
                            train_output.batches: dataset_batches[unroll_itr],
                            train_output.batches_val: dataset_batches_val,
                            train_output.first_unroll: first_unroll,
                            train_output.reset_state: reset_state,
                        }

                        # run the train and summary ops
                        # when a "save_diagnostics" flag is turned on

                        fetches_list = [
                            train_output.metaobj,
                            train_output.problem_objectives,
                            train_output.initial_obj,
                            summary_op,
                            clipped_grads_and_vars,
                            train_op
                        ]
                                
                        if unroll_itr + 1 < num_unrolls:
                            fetches_list += [propagate_loop_state_op]

                        ######################
                        # should not do this sess.run multiple times in one unroll. the first_unroll, reset_state only works once!
                        fetched = sess.run(fetches_list, feed_dict=feed, options=run_options, run_metadata=run_metadata)
                        
                        meta_obj = fetched[0]
                        sub_obj = fetched[1]
                        init_obj = fetched[2]
                        summ = fetched[3]
                        meta_grads_and_params = fetched[4]
                
                        # only the chief task is allowed to write the summary
                        if is_chief:
                            global_step_eval = sess.run(global_step)
                            train_logger.add_summary(summ, global_step_eval)

                        metadata["subproblem_objs"].append(sub_obj)
                        # store training metadata to pass to the callback
                        metadata["meta_objs"].append(meta_obj)
                        metadata["meta_grads_and_params"].append(meta_grads_and_params)
                        
                        # (end of for loop: unroll_itr)

                    optimization_time = time.time() - train_start_time

                    if FLAGS.set_profiling:
                        summary_name = "%02d_iter%04d_%02d" % (FLAGS.task, problem_itr, k)
                        summary_writer.add_run_metadata(run_metadata, summary_name)

                    metadata["global_step"].append(sess.run(global_step))
                    metadata["runtimes"].append(optimization_time)

                    # write a diagnostic message to the output
                    args = (k, meta_obj, optimization_time,
                                    sum(partial_unroll_iters[:unroll_itr+1]))
                    out.write("    [{0:02}] meta_obj: {1}, (time: {2:8.4f}s), {3:} iters ".format(*args))
                    out.write("(unrolled {} steps)".format(
                            ", ".join([str(s) for s in partial_unroll_iters[:unroll_itr+1]])))
                    #out.write("\n    grad: {}, param: {}".format(*meta_grads_and_params[0]))
                    out.write("{}\n".format(additional_log_info))
                    out.flush()

                    # (end of for loop: k)

                if FLAGS.set_profiling:
                    summary_writer.close()

                # force a checkpoint save before we load a new problem
                # only the chief task has the save_path and can write the checkpoint
                if is_chief:
                    sv.saver.save(sess, sv.save_path, global_step=global_step)

                # (end of sess)
            # (end of graph)

        # run the callbacks on the chief
        if is_chief and callbacks is not None:
            for callback in callbacks:
                if hasattr(callback, "__call__"):
                    problem_name = problem_spec.callable.__name__
                    callback(problem_name, problem_itr, logdir, metadata)

        # return meta_grads_and_params

        # (end of for loop: problem_iter)


def test_optimizer(optimizer,
                   problem,
                   num_iter,
                   subset_n_class=-1,
                   subset_n_sample=None,
                   dataset=None,
                   dataset_test=None,
                   batch_size=None,
                   seed=None,
                   graph=None,
                   logdir=None,
                   record_every=None):
    """Tests an optimization algorithm on a given problem.

    Args:
        optimizer: Either a tf.train.Optimizer instance, or an Optimizer instance
            inheriting from trainable_optimizer.py
        problem: A Problem instance that defines an optimization problem to solve
        num_iter: The number of iterations of the optimizer to run
        dataset: The dataset to train the problem against
        batch_size: The number of samples per batch. If None (default), the
            batch size is set to the full batch (dataset.size)
        seed: A random seed used for drawing the initial parameters, or a list of
            numpy arrays used to explicitly initialize the parameters.
        graph: The tensorflow graph to execute (if None, uses the default graph)
        logdir: A directory containing model checkpoints. If given, then the
            parameters of the optimizer are loaded from the latest checkpoint
            in this folder.
        record_every: if an integer, stores the parameters, objective, and gradient
            every recored_every iterations. If None, nothing is stored

    Returns:
        objective_values: A list of the objective values during optimization
        parameters: The parameters obtained after training
        records: A dictionary containing lists of the parameters and gradients
            during optimization saved every record_every iterations (empty if
            record_every is set to None)
    """

    ### //// temp
    num_param = np.sum([np.prod(item) for item in problem.param_shapes])
    print("Number of parameters: {}".format(num_param))
    grad_eval_flat_old = np.zeros(num_param) # number of parameters for mlp for mnist 01 (80562), 10class(98402)
    params_eval_flat_old = np.zeros(num_param)
    full_grad_eval_flat_old = np.zeros(num_param)
    ### ////

    if dataset is None:
        dataset = datasets.EMPTY_DATASET
        dataset_test = datasets.EMPTY_DATASET
        batch_size = dataset.size
    else:
        if dataset_test is None:
            # default batch size is the entire dataset
            dataset, labels_chosen = dataset.generate_subset_by_label(n_label=subset_n_class, n_sample=subset_n_sample)
            dataset, dataset_test = dataset.train_test_split(test_size=0.5)
            print("Dataset class: {}".format(labels_chosen))
            batch_size = dataset.size if batch_size is None else batch_size
        else:
            # default batch size is the entire dataset
            if hasattr(problem, "accuracy"): # for classification
                dataset, labels_chosen = dataset.generate_subset_by_label(n_label=subset_n_class, n_sample=subset_n_sample)
                dataset_test, _ = dataset_test.generate_subset_by_label(labels=labels_chosen)
                print("Dataset class: {}".format(labels_chosen))
                batch_size = dataset.size if batch_size is None else batch_size
            else:       
                # for regression
                batch_size = dataset.size if batch_size is None else batch_size

    graph = tf.compat.v1.get_default_graph() if graph is None else graph
    
    with graph.as_default():
        # define the parameters of the optimization problem
        if isinstance(seed, (list, tuple)):
            # seed is a list of arrays
            params = problem_generator.init_fixed_variables(seed)
        else:
            # seed is an int or None
            params = problem.init_variables(seed)

        data_placeholder = tf.compat.v1.placeholder(tf.float32)
        if hasattr(problem, "accuracy"): # for classification
            labels_placeholder = tf.compat.v1.placeholder(tf.int32)
        else: # for regression
            labels_placeholder = tf.compat.v1.placeholder(tf.float32)

        # get the problem objective and gradient(s)
        obj = problem.objective(params, data_placeholder, labels_placeholder)
        if hasattr(problem, "accuracy"):
            acc = problem.accuracy(params, data_placeholder, labels_placeholder)
        if hasattr(problem, "mse"):
            mse = problem.mse(params, data_placeholder, labels_placeholder)
        gradients= problem.gradients(obj, params)

        vars_to_preinitialize = params

    with tf.compat.v1.Session(graph=graph, config=CONFIG) as sess:
        # create the train operation and training variables
        with tf.control_dependencies([obj] + gradients + params):
            train_op = optimizer.apply_gradients(zip(gradients, params))
            if hasattr(optimizer, "get_grad_input_scale"):
                with tf.control_dependencies([train_op]):
                    step_size, inp, grad_step_size_to_inp = optimizer.get_grad_input_scale()  # get the computed mean gradient and their corresponding root mean squared input (the scale)
        
        # vars_to_preinitialize: parameters of the problem, the starting point of optimization
        # vars_to_restore: parameters of the optimizer
        # vars_to_initialize: variables in the optimizer like the states, steps, etc.
        sess.run(tf.compat.v1.variables_initializer(vars_to_preinitialize))

        vars_to_restore = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.GLOBAL_VARIABLES, scope=OPTIMIZER_SCOPE)
        vars_to_initialize = list(set(tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.GLOBAL_VARIABLES)) - set(vars_to_restore) - set(vars_to_preinitialize))

        # load or initialize optimizer variables
        if logdir is not None:
            restorer = tf.compat.v1.train.Saver(var_list=vars_to_restore)
            ckpt = tf.train.latest_checkpoint(logdir)
            restorer.restore(sess, ckpt)
        else:
            sess.run(tf.variables_initializer(vars_to_restore))
        
        # initialize all the other variables
        
        sess.run(tf.compat.v1.variables_initializer(vars_to_initialize))
        problem.init_fn(sess)

        # generate the minibatch indices
        batch_inds = dataset.batch_indices(num_iter + 1, batch_size)  # + 1 so that the last iteration can also be recorded. For example, if num_iter=4000, running the following for loop will only record for itr up to 3999.

        # run the train operation for n iterations and save the objectives
        records = {}
        records["start_record"] = defaultdict(list)
        records["regular_record"] = defaultdict(list)
        objective_values = []

        full_feed = {data_placeholder: dataset.data,
                    labels_placeholder: dataset.labels}
        test_feed = {data_placeholder: dataset_test.data,
                    labels_placeholder: dataset_test.labels}
        
        for itr, batch in enumerate(batch_inds):
            # data to feed in
            feed = {data_placeholder: dataset.data[batch],
                    labels_placeholder: dataset.labels[batch]}

            # record stuff            
            if record_every is not None and (itr % record_every) == 0:
                # access values
                obj_eval, params_eval, gradients_eval = sess.run((obj, params, gradients), feed_dict=feed) # should access gradients and train_op together; otherwise, the noise will be resampled!

                grad_eval_flat = np.hstack([g.reshape((-1)) for g in gradients_eval])
                params_eval_flat = np.hstack([g.reshape((-1)) for g in params_eval])
                
                if (itr <= 500 and itr % 10 == 0):
                    records["start_record"]["iter"].append(itr)
                    records["start_record"]["obj"].append(obj_eval)
                    records["start_record"]["step_size"].append(((params_eval_flat - params_eval_flat_old) ** 2).sum() ** 0.5)
                    grad_eval_flat_normed = grad_eval_flat_old / ((grad_eval_flat_old ** 2).sum() ** 0.5)
                    displacement_eval_flat_normed = (params_eval_flat - params_eval_flat_old) / (((params_eval_flat - params_eval_flat_old) ** 2).sum() ** 0.5)
                    records["start_record"]["cos_angle"].append((grad_eval_flat_normed * displacement_eval_flat_normed).sum())
                
                if (num_iter <= 2000 and itr % 10 == 0) or (num_iter > 2000 and itr % 100 == 0):
                    records["regular_record"]["iter"].append(itr)
                    records["regular_record"]["obj"].append(obj_eval)
                    records["regular_record"]["step_size"].append(((params_eval_flat - params_eval_flat_old) ** 2).sum() ** 0.5)
                    grad_eval_flat_normed = grad_eval_flat_old / ((grad_eval_flat_old ** 2).sum() ** 0.5)
                    displacement_eval_flat_normed = (params_eval_flat - params_eval_flat_old) / (((params_eval_flat - params_eval_flat_old) ** 2).sum() ** 0.5)
                    records["regular_record"]["cos_angle"].append((grad_eval_flat_normed * displacement_eval_flat_normed).sum())
                
                objective_values.append(obj_eval)

                grad_eval_flat_old = grad_eval_flat.copy()
                params_eval_flat_old = params_eval_flat.copy()  

                if hasattr(problem, "accuracy"):
                    if (itr <= 2000 and itr % 10 == 0) or (itr % 100 == 0):
                        train_acc_eval, full_grad_eval = sess.run((acc, gradients), feed_dict=full_feed)
                        test_acc_eval = sess.run((acc), feed_dict=test_feed)
                        
                        if itr <= 500 and itr % 10 == 0:
                            records["start_record"]["train_acc"].append(train_acc_eval)
                            records["start_record"]["test_acc"].append(test_acc_eval)
                        
                        if (num_iter <= 2000 and itr % 10 == 0) or (num_iter > 2000 and itr % 100 == 0):
                            records["regular_record"]["train_acc"].append(train_acc_eval)
                            records["regular_record"]["test_acc"].append(test_acc_eval)
                            
                if hasattr(problem, "mse"):
                    if (itr <= 2000 and itr % 10 == 0) or (itr % 100 == 0):
                        train_mse_eval, full_grad_eval = sess.run((mse, gradients), feed_dict=full_feed)
                        test_mse_eval = sess.run((mse), feed_dict=test_feed)
                        
                        if itr <= 500 and itr % 10 == 0:
                            records["start_record"]["train_mse"].append(train_mse_eval)
                            records["start_record"]["test_mse"].append(test_mse_eval)
                        
                        if (num_iter <= 2000 and itr % 10 == 0) or (num_iter > 2000 and itr % 100 == 0):
                            records["regular_record"]["train_mse"].append(train_mse_eval)
                            records["regular_record"]["test_mse"].append(test_mse_eval)

                # update
                _ = sess.run(train_op, feed_dict=feed)

            else:
                # run the optimization train operation
                objective_values.append(sess.run([train_op, obj], feed_dict=feed)[1])


        # final parameters
        parameters = [sess.run(p) for p in params]

    return objective_values, parameters, records

