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
    """The MLP Optimizer that outputs learning rates for each parameter group separtely"""

    def __init__(self, hidden_size=32, hidden_layer=1, decay=[0.5, 0.9, 0.99, 0.999, 0.9999], step_multiplier=0.001, magnitude_rate=0.001, **kwargs):
        self._reuse_vars = False
        self.hidden_layer_size = [21] + [hidden_size] * hidden_layer + [1]
        self.decay = decay
        self.step_embedding_scale = [3, 10, 30, 100, 300, 1000, 3000, 10000, 300000]
        self.feature_names = ["log grad norm"] + ["log grad first moment {}".format(d) for d in decay] + ["inner product {}".format(d) for d in decay] + ["param"] + ["step embed {}".format(s) for s in self.step_embedding_scale]
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
        state_keys.append("step")
        super(MLP_LR_GROUP, self).__init__("MLP_LR_GROUP", state_keys, **kwargs)

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

            (grad_values, first_moment, timestep, grad_indices) = self._extract_gradients_and_internal_state(grad, state, tf.shape(param))
            
            if grad_indices != None:
                raise NotImplementedError("Sparse gradient with non-none grad_indices not implemented")

            grad_values_flat = tf.reshape(grad_values, [-1, 1])
            param_flat = tf.reshape(param, [-1, 1])
            
            # compute updated parameters
            inp = self._expand_features(grad_values_flat, param_flat, first_moment, timestep)
            output = tf.reshape(self.mod(inp), ()) # reshape to scalar
            learned_lr = tf.exp(output * self.magnitude_rate) * self.step_multiplier
            new_param_flat = param_flat - learned_lr * grad_values_flat
            new_param = tf.reshape(new_param_flat, tf.shape(param))
            
            # update the states
            new_state = {}
            for i in range(len(self.decay)):
                new_first_moment = self._update_decay_estimate(first_moment[i], grad_values_flat, self.decay[i])  # This only works for grad_indices = None
                new_state["ms_{}".format(i)] = new_first_moment
            new_state["step"] = timestep + 1

        return new_param, new_state

    '''
    def get_grad_input_scale(self):
        return self.step_size, self.inp, self.grad_step_size_to_inp
    '''

    def _update_decay_estimate(self, estimate, value, beta):
        """Returns a beta-weighted average of estimate and value."""
        return (beta * estimate) + ((1 - beta) * value)

    def _expand_features(self, flat_g, flat_v, first_moment, timestep):
        n_param = tf.cast(tf.shape(flat_g)[0], tf.float32)
        log_grad_scale = tf.math.log(tf.norm(flat_g, axis=0) / tf.sqrt(n_param) + 1e-8)
        
        first_moment = tf.concat(first_moment, axis=1)
        log_grad_first_moment_scale = tf.math.log(tf.norm(first_moment, axis=0) / tf.sqrt(n_param) + 1e-8)
        
        grad_inner_product = tf.reduce_sum(first_moment * flat_g, 0)
        param_scale = tf.norm(flat_v, axis=0) / tf.sqrt(n_param)
        step = tf.reshape(self.tanh_embedding(timestep), (-1, ))
        
        inp = tf.concat([log_grad_scale, log_grad_first_moment_scale, grad_inner_product, param_scale, step], 0)
        inp = tf.reshape(inp, (1, -1))
        return inp

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
        for i in self.step_embedding_scale:
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
        timestep = state["step"]

        if isinstance(grad, tf.IndexedSlices):
            grad_indices, grad_values = utils.accumulate_sparse_gradients(grad)
            for i in range(len(self.decay)):
                first_moment[i] = utils.slice_tensor(first_moment[i], grad_indices, param_shape)

        return grad_values, first_moment, timestep, grad_indices