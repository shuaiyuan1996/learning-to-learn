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

"""A trainable optimizer that learns a mlp to ."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from . import trainable_optimizer
from . import utils


class MLP_LR_GROUP(trainable_optimizer.TrainableOptimizer):
    """The MLP Optimizer that outputs step sizes for each parameter group separtely"""

    def __init__(self, hidden_size=32, hidden_layer=1, decay=[0.5, 0.9, 0.99, 0.999, 0.9999], step_multiplier=0.001, magnitude_rate=0.001, **kwargs):
        self._reuse_vars = False
        self.hidden_layer_size = [31] + [hidden_size] * hidden_layer + [2]
        self.decay = decay
        self.W = []
        self.b = []
        self.step_multiplier = step_multiplier
        self.magnitude_rate = magnitude_rate
        with tf.compat.v1.variable_scope(trainable_optimizer.OPTIMIZER_SCOPE):
            for i in range(1, len(self.hidden_layer_size)):
                self.W.append(tf.compat.v1.get_variable("mlp_weight_{}".format(i), shape=(self.hidden_layer_size[i - 1], self.hidden_layer_size[i]), initializer=tf.random_normal_initializer()))
                self.b.append(tf.compat.v1.get_variable("mlp_bias_{}".format(i), shape=(self.hidden_layer_size[i]), initializer=tf.constant_initializer(0.1)))

        state_keys = []
        for i, decay in enumerate(decay):
            state_keys.append("ms_{}".format(i))
            state_keys.append("rms_{}".format(i))
        state_keys.append("step")
        super(MLP, self).__init__("MLP", state_keys, **kwargs)

        self.sum_of_grad = []
        self.sum_of_squared_input = []
        self.sum_of_grad_times_abs_input = []
        self.num_param = tf.Variable(0)
        self.sum_of_change_ratio = []

    def _initialize_state(self, var):
        """Returns a dictionary mapping names of state variables to their values."""
        vectorized_shape = var.get_shape().num_elements(), 1
        res = {}
        for key in self.state_keys:
            if key != "step":
                res[key] = tf.zeros(vectorized_shape)
        res["step"] = tf.zeros((1))
        return res

        #return {key: tf.zeros(vectorized_shape) for key in self.state_keys}

    def mod(self, inp):    # the mlp model
        for i in range(len(self.W)):
            inp = tf.matmul(inp, self.W[i]) + self.b[i]
            if i < len(self.W) - 1:
                inp = tf.nn.relu(inp)
        return inp

    def _compute_update(self, param, grad, state):
        with tf.compat.v1.variable_scope(trainable_optimizer.OPTIMIZER_SCOPE) as scope:
            if self._reuse_vars:
                scope.reuse_variables()
            else:
                self._reuse_vars = True

            (grad_values, first_moment, second_moment, timestep, grad_indices) = self._extract_gradients_and_internal_state(grad, state, tf.shape(param))
            
            if grad_indices != None:
                raise NotImplementedError("Sparse gradient with non-none grad_indices not implemented")

            grad_values = tf.reshape(grad_values, [-1, 1])
            old_param_shape = tf.shape(param)
            param = tf.reshape(param, [-1, 1])

            new_timestep = timestep + 1
            new_first_moment = []
            new_second_moment = []
            for i in range(len(self.decay)):
                new_first_moment.append(self._update_decay_estimate(first_moment[i], grad_values, self.decay[i]))    # This only works for grad_indices = None
                new_second_moment.append(self._update_decay_estimate(second_moment[i], tf.square(grad_values), self.decay[i]))

            '''
            if grad_indices is None:
                grad_used = utils.slice_tensor(grad_values, grad_indices, tf.shape(param))
            else:
                grad_used = grad_values
            '''
            inp = self._expand_features(grad_values, param, new_first_moment, new_second_moment, new_timestep)
            
            output = self.mod(inp)

            direction = output[:, 0:1]
            magnitude = output[:, 1:2]
            step = direction * tf.exp(magnitude * self.magnitude_rate) * self.step_multiplier
            #step = grad_values * tf.exp(magnitude * self.magnitude_rate) * self.step_multiplier
            
            grad_step_to_input = tf.compat.v1.gradients(step, inp)[0]
            
            self.sum_of_grad.append(tf.reduce_sum(grad_step_to_input, axis=0))
            self.sum_of_squared_input.append(tf.reduce_sum(tf.square(inp), axis=0))
            self.sum_of_grad_times_abs_input.append(tf.reduce_sum(grad_step_to_input * tf.math.abs(inp), axis=0))
            self.num_param = self.num_param + tf.shape(param)[0]

            new_param = param - step
            new_param = tf.reshape(new_param, old_param_shape)

            new_state = {}
            for i in range(len(self.decay)):
                new_state["ms_{}".format(i)] = new_first_moment[i]
                new_state["rms_{}".format(i)] = new_second_moment[i]
            new_state["step"] = new_timestep

        return new_param, new_state

    
    def get_grad_input_scale(self):
        grad = self.sum_of_grad[0]
        inp = self.sum_of_squared_input[0]
        grad_times_abs_input = self.sum_of_grad_times_abs_input[0]
        for i in range(1, len(self.sum_of_grad)):
            grad += self.sum_of_grad[i]
            inp += self.sum_of_squared_input[i]
            grad_times_abs_input += self.sum_of_grad_times_abs_input[i]
        num_param = tf.cast(self.num_param, tf.float32)
        return grad / num_param, tf.sqrt(inp / num_param), grad_times_abs_input / num_param

    def _update_decay_estimate(self, estimate, value, beta):
        """Returns a beta-weighted average of estimate and value."""
        return (beta * estimate) + ((1 - beta) * value)

    def _expand_features(self, flat_g, flat_v, first_moment, second_moment, timestep):
        m = tf.concat(first_moment, 1)
        rms = tf.concat(second_moment, 1)
        rsqrt = tf.math.rsqrt(rms + 1e-6)
        norm_g = m * rsqrt

        inp = tf.concat([flat_g, norm_g, flat_v, m, rms, rsqrt], 1)
        inp = self.second_moment_normalize(inp, is_training=True)

        step = self.tanh_embedding(timestep)
        stack_step = tf.tile(tf.reshape(step, [1, -1]), tf.stack([tf.shape(flat_g)[0], 1]))
        inp = tf.concat([inp, stack_step], axis=1)

        return inp

    def second_moment_normalize(self, x, is_training=True):
        normed = x * tf.math.rsqrt(1e-5 + tf.reduce_mean(tf.square(x), axis=0, keepdims=True))
        return normed

    def tanh_embedding(self, x):
        """Embed time in a format usable by a neural network.

        This embedding involves dividing x by different timescales and running through
        a squashing function.
        Args:
            x: tf.Tensor
        Returns:
            tf.Tensor
        """
        mix_proj = []
        for i in [3, 10, 30, 100, 300, 1000, 3000, 10000, 300000]:
            mix_proj.append(tf.tanh(tf.cast(x, tf.float32) / float(i) - 1.))
        return tf.stack(mix_proj)

    def _extract_gradients_and_internal_state(self, grad, state, param_shape):
        """Extracts the gradients and relevant internal state.

        If the gradient is sparse, extracts the appropriate slices from the state.

        Args:
            grad: The current gradient.
            state: The current state.
            param_shape: The shape of the parameter (used if gradient is sparse).

        Returns:
            grad_values: The gradient value tensor.
            first_moment: The first moment tensor (internal state).
            second_moment: The second moment tensor (internal state).
            timestep: The current timestep (internal state).
            grad_indices: The indices for the gradient tensor, if sparse.
                    None otherwise.
        """
        grad_values = grad
        grad_indices = None
        first_moment = [state["ms_{}".format(i)] for i in    range(len(self.decay))]
        second_moment = [state["rms_{}".format(i)] for i in range(len(self.decay))]
        timestep = state["step"]

        if isinstance(grad, tf.IndexedSlices):
            grad_indices, grad_values = utils.accumulate_sparse_gradients(grad)
            for i in range(len(self.decay)):
                first_moment[i] = utils.slice_tensor(first_moment[i], grad_indices, param_shape)
                second_moment[i] = utils.slice_tensor(second_moment[i], grad_indices, param_shape)
            timestep = utils.slice_tensor(timestep, grad_indices, param_shape)

        return grad_values, first_moment, second_moment, timestep, grad_indices