# Guarantees for Tuning the Step Size using a Learning-to-Learn Approach

## Experiment environment

The code is tested in the following environment:

* Python 3.6.9
* Tensorflow 1.14.0
* CUDA 10.1
* numpy 1.18.1
* pandas 0.25.1

(Note that there could be harmless future warnings if you use numpy>=1.17. Though unnecessary, downgrading to 1.14 can solve this.)

Other required packages are:

* argparse
* collections
* itertools

## Pretrained models

All the pretrained models, as well as log files, training summaries and test results, are available as a zip file [here](https://drive.google.com/file/d/1LCdEITWMWdKO6SInAawSArnzQCKj4xfK/view?usp=sharing) (size: ~2.2GB). 

## Code credits

The overall structure of this code is borrowed from the [github repo](https://github.com/tensorflow/models/tree/master/research/learned_optimizer) of Olga Wichrowska et al., appeared in their publication [_Learned Optimizers that Scale and Generalize_](https://arxiv.org/abs/1703.04813).

The MLP optimizer code is adapted from the [github repo](https://github.com/google-research/google-research/tree/master/task_specific_learned_opt) of Luke Metz et al. from their paper [_Understanding and correcting pathologies in the training of learned optimizers_](https://arxiv.org/abs/1810.10180)


## Code structure

	.
	├── grid_search_eta_lr.py
	├── grid_search_eta_lr.sh
	├── learn_eta_quadratic.py
	├── learn_eta_quadratic_tensorflow.py
	├── learn_eta_quadratic_tensorflow.sh
	├── metaopt.py
	├── optimizer
	│   ├── global_learning_rate.py
	│   ├── __init__.py
	│   ├── mlp.py
	│   ├── trainable_optimizer.py
	│   └── utils.py
	├── problems
	│   ├── datasets.py
	│   ├── __init__.py
	│   ├── model_adapter.py
	│   ├── problem_generator.py
	│   ├── problem_sets.py
	│   └── problem_spec.py
	├── README.md
	├── test_mlp_optimizer.py
	├── test.sh
	├── train_mlp_optimizer.py
	└── train.sh

A brief overview of the code folder:

* The `learn_eta_quadratic.py`, `learn_eta_quadratic.py`, `learn_eta_quadratic_tensorflow.py`, `train.py`, `test.py` files are the scripts we run.
* The `metaopt.py` file implements train and test code for the meta-learning process, which is called by all the scripts we run.
* The `optimizer` folder contains the class definitions for the trainable optimizers that we use. Note that `trainable_optimizer.py` defines the most basic class of the trainable optimizer, whereas `global_learning_rate.py` and `mlp.py` define the specific optimizers that we use.
* The `problems` folder defines the classes for the inner-problems to be used. That includes generating the datasets (`datasets.py`) and generating the inner model (`problem_generator.py`). The `problem_sets.py` file contains shortcuts to define a set of problems, which is not used in our final experiments.

## Running the code

### Experiment 1: Optimizing step size for quadratic objective

For our **hand-derived** version, please run 

	python3 learn_eta_quadratic.py \
	--ndim 20 --t 80 --eta0 0.1  --num_tasks 1000 \
	--train_dir <TRAIN_DIR>

The parameters:

* `ndim` is the dimension;
* `t` is the inner training length;
* `eta0` is the initial value for the learning rate eta;
* `num_tasks` is the number of tasks used for training, i.e. the number of meta steps;
* `train_dir` is the your own directory that will save the results.

After running this command, you will find a `dim20_t80_init_eta0.1.csv` file under your `<TRAIN_DIR>`, which is a talbe that collects the eta value and its gradient at each step, for which you can visualize the convergence. Note that we don't reflect the `num_tasks` in the file name because we always use the default value. You can change the file name at Line 69 in `learn_eta_quadratic.py`.

For the **Tensorflow** version, please run

	python3 learn_eta_quadratic_tensorflow.py \
	--ndim 20 --t 80 --eta0 0.1 --num_tasks 1000 \
	--meta_train_opt RMSProp --meta_train_opt_lr 0.001 \
	--train_dir <TRAIN_DIR>

The parameters:

* `ndim`, `t`, `eta0`, `num_tasks`, `train_dir` are the same as before;
* `meta_train_opt` specifies the meta optimizer to use; please choose from "RMSProp" and "SGD" (it is actually GD in our case);
* `meta_train_opt_lr ` is the meta learning rate.

After running this command, you will see a `dim20_t80_init_eta0.1_tf_RMSProp` folder under your `<TRAIN_DIR>` that contains the model and results. Under that folder, `log.txt`  shows the meta-objectives at each meta-step. `train_summary` is the summary file that you can use tensorboard to open and read, where you can see the meta-objective, meta-gradient and value of `eta` throughout training. From there, you can see whether the model converges or not.

You can also find the two commands mentioned above in `learn_eta_quadratic_tensorflow.sh`.

### Experiment 2: Train-by-train vs. train-by-validation, synthetic data

To run grid search and test for both train-by-trian and train-by-val settings, please run

	python3 grid_search_eta_lr.py \
	--ndim 1000 --t 40 --nsample 500 --sigma 1 --num_tasks 300 --num_tasks_test 10 \
	--train_dir <TRAIN_DIR>

The parameters:

* `ndim` is the dimension;
* `t` is the inner training length;
* `nsample` is the total number of samples used; we split train/val set by half by default, so `nsample=500` means that we will test TbT500 and TbV250+250;
* `sigma` is the noise standard deviation;
* `num_tasks` is the number of tasks used for training, i.e. the number of meta steps;
* `num_tasks_test` is the number of independent tests that will be performed.
* `train_dir` is the your own directory that will save the results.

After running this command, you will see a result folder `dim1000_sigma1.0_sample500` under your `<TRAIN_DIR>`. Note that the `t`, `num_tasks`, `num_tasks_test` are not reflected in the file name because we always default values in this experiment. Feel free to change the file name at Line 28 in `grid_search_eta_lr.py`.

In the result folder, there will be a `eta_learned.csv` file that records the optimal `eta` found by the train-by-train and train-by-val settings, respectively. There will also be a `metaobj.csv` file that records all the candidate `eta` values, as well as the meta-objectives, evaluated in the grid search process. In addition, there will be multilple `TbT_test#.csv` and `TbV_test#.csv` files, where "#" stands for the test number and should be within [0, `num_tasks_test`-1]. These files show the objective and the training and test RMSEs at each step for all the independent runs.

### Experiment 3: Train-by-train vs. train-by-validation, MLP optimizer on MNIST

#### Training:

To train MLP optimizer on the MNIST classification problem, please run

	python3 train_mlp_optimizer.py \
	--nsample 2000 --mode val --val_split 0.5 --noise 0 \
	--logdir <LOG_DIR>

The parameters:

* `nsample` is the **total** number of samples used; for train-by-val, that includes the number of samples for both the training and validation set combined;
* `mode` is whether we are doing train-by-train ("train") or train-by-val ("val");
* `val_split` is the portion of the data that we split to the validation set; for example, for TbV1000+1000, do `nsample=2000, val_split=0.5`; for TbV50000+10000, do `nsample=60000, val_split=0.166666`;
* `noise` is the noise level (the probability that a label will be randomly changed); do `noise=0.2` for 20% noise.
* `logdir` is your own directory that will save the results.

After running this command, your `<LOG_DIR>` will be created. In that directory, you will find the following files:

* `log.txt` is the log file as you train the model, which shows problem number, classes, iteration splits, meta objective of the last step, as well as the running time;
* `flags.txt` saves all the hyper-parameters used;
* `checkpoint` saves the path of the checkpoints; **IMPORTANT: The paths are absolute. Please edit the paths in this `checkpoint` file to the actual paths on your system before testing our pretrained models. If you train it yourself, that's fine.**
* `model.ckpt-xxxx' saves the checkpoint of the model at different steps;
* `train_summary` is a summary folder that you can use tensorboard to visualize. After installing tensorboard, do `tensorboard --logdir train_summary`.

#### Testing:

To test the trained MLP optimizer, please run

	python3 test_mlp_optimizer.py \
	--nsample 1000 --noise 0 --ntest 5 --niter 4000 \
	--logdir <LOG_DIR>

The parameters:

* `nsample` is the number of training samples used in the problem. Note that there is no distinction of train-by-train or train-by-val at this stage because those are just two different ways to find the optimal trainable optimizer. When we apply the trained optimizer, we don't need to split the training set. We use all test samples by default.
* `noise` is the noise level (the probability that a label will be randomly changed); do `noise=0.2` for 20% noise;
* `ntest` is the number of independent tests;
* `niter` is the number of train iterations;
* `logdir` is the directory that you saved your training model before.

After running this, the test results will be saved under `<LOG_DIR>`. You will find `result#_regular_record_1000.csv`, which saves the objective, training and test accuracies along the trajecotry, as well as `result#_start_record_1000.csv`, which only records the starting 500 iterations but for higher frequency. `result_SGD_xxx_record_1000.csv` shows the result for our baseline SGD method.










