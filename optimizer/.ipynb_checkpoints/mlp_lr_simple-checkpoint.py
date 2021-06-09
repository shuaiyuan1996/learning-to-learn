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


class MLP_LR_SIMPLE(trainable_optimizer.TrainableOptimizer):
    """The MLP Optimizer applied for the whole parameter set, which outputs a step size"""

    def __init__(self, hidden_size=32, hidden_layer=1, decay=[0.5, 0.9, 0.99, 0.999, 0.9999], step_multiplier=0.001, magnitude_rate=0.001, **kwargs):
        self._reuse_vars = False
        self.step_embedding_scale = [3, 10, 30, 100, 300, 1000, 3000, 10000, 300000]
        self.feature_names = ["step embed {}".format(s) for s in self.step_embedding_scale]
        self.hidden_layer_size = [len(self.feature_names)] + [hidden_size] * hidden_layer + [1]
        self.W = []
        self.b = []
        self.step_multiplier = step_multiplier
        self.magnitude_rate = magnitude_rate
        with tf.compat.v1.variable_scope(trainable_optimizer.OPTIMIZER_SCOPE):
            for i in range(1, len(self.hidden_layer_size)):
                self.W.append(tf.compat.v1.get_variable("mlp_weight_{}".format(i), shape=(self.hidden_layer_size[i - 1], self.hidden_layer_size[i]), initializer=tf.random_normal_initializer()))
                self.b.append(tf.compat.v1.get_variable("mlp_bias_{}".format(i), shape=(self.hidden_layer_size[i]), initializer=tf.constant_initializer(0.1)))

        state_keys = []
        state_keys.append("step")
        super(MLP_LR_SIMPLE, self).__init__("MLP_LR", state_keys, **kwargs)
        #self.sum_of_grad = []
        #self.sum_of_squared_input = []
        #self.sum_of_grad_times_abs_input = []
        #self.num_param = tf.Variable(0)
        #self.sum_of_change_ratio = []

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

    
    # override this to apply updates to all parameters at the same time
    # adapted from the source tensorflow package
    def apply_gradients(self, grads_and_vars, global_step=None, name=None):
        """Apply gradients to variables.
        This is the second part of `minimize()`. It returns an `Operation` that
        applies gradients.
        Args:
          grads_and_vars: List of (gradient, variable) pairs as returned by
            `compute_gradients()`.
          global_step: Optional `Variable` to increment by one after the
            variables have been updated.
          name: Optional name for the returned operation.  Default to the
            name passed to the `Optimizer` constructor.
        Returns:
          An `Operation` that applies the specified gradients. If `global_step`
          was not None, that operation also increments `global_step`.
        Raises:
          TypeError: If `grads_and_vars` is malformed.
          ValueError: If none of the variables have gradients.
          RuntimeError: If you should use `_distributed_apply()` instead.
        """
        # This is a default implementation of apply_gradients() that can be shared
        # by most optimizers.  It relies on the subclass implementing the following
        # methods: _create_slots(), _prepare(), _apply_dense(), and _apply_sparse().

        # TODO(isaprykin): Get rid of `has_strategy()` check by
        # always calling _distributed_apply(), using the default distribution
        # as needed.
        
        #import IPython; IPython.embed()
        from tensorflow.python.framework import ops
        grads_and_vars = tuple(grads_and_vars)  # Make sure repeat iteration works.
        
        if not grads_and_vars:
            raise ValueError("No variables provided.")
            
        var_list = [v for g, v in grads_and_vars]
        grad_list = [g for g, v in grads_and_vars]
        if not var_list:
            raise ValueError("No gradients provided for any variable: %s." %
                              ([str(v) for _, v in grads_and_vars],))
        
        with ops.init_scope():
            self._create_slots(var_list)      
        update_ops = []
        
        with ops.name_scope(name, self._name) as name:
            self._prepare()
            states = []
            for grad, var in grads_and_vars:
                keys = self.get_slot_names()
                state = {}
                for key in keys:
                    state[key] = self.get_slot(var, key)
                states.append(state)
            
            new_params, new_states, _, _ = self._compute_updates(var_list, grad_list, states, None) # the global state is ignored
            
            for state, new_state, var, new_var in zip(states, new_states, var_list, new_params):
                state_assign_ops = []
                for key, state_var in state.items():
                    state_assign_ops.append(tf.compat.v1.assign(state_var, new_state[key]))
                
                with tf.control_dependencies(state_assign_ops):
                    update_op = var.assign(new_var)
                if var.constraint is not None:
                    raise NotImplementedError("var.constraint")
                update_ops.append(update_op)
         
            if global_step is None:
                apply_updates = self._finish(update_ops, name)
            else:
                raise NotImplementedError("global_step is not None")
            
            train_op = ops.get_collection_ref(ops.GraphKeys.TRAIN_OP)
            if apply_updates not in train_op:
                train_op.append(apply_updates)

            return apply_updates
        
    # to update for all parameters at the same time
    def _compute_updates(self, params, grads, states, global_state):
        # Zip up the arguments to _compute_update.
        
        with tf.compat.v1.variable_scope(trainable_optimizer.OPTIMIZER_SCOPE) as scope:
            if self._reuse_vars:
                scope.reuse_variables()
            else:
                self._reuse_vars = True
            
            inp_list = []
            new_states = []
            grad_values_flat_list = []
            param_flat_list = []
            
            for param, grad, state in zip(params, grads, states):
                timestep = state["step"]

                # compute new states
                new_state = {}           
                new_state["step"] = timestep + 1
                new_states.append(new_state)
            # end for
            
            inp = tf.reshape(self.tanh_embedding(states[0]["step"]), (-1, ))
            inp = tf.reshape(inp, (1, -1))
            #inp = self._expand_features(grad_values_flat_all, param_flat_all, states)
            
            output = tf.reshape(self.mod(inp), ()) # reshape to scalar
            learned_lr = tf.exp(output * self.magnitude_rate) * self.step_multiplier
            self.learned_lr = learned_lr
            self.inp = tf.reshape(inp, (-1, ))
            self.grad_step_size_to_inp = tf.reshape(tf.compat.v1.gradients(learned_lr, inp), (-1, ))
            
            # apply the updates
            new_params = []
            for param, grad in zip(params, grads):
                new_param = param - learned_lr * grad
                new_params.append(new_param)

        # Global state is unused in the basic case, just pass it through.

        return list(new_params), list(new_states), global_state, list(new_params)
    
    def get_grad_input_scale(self):
        return self.learned_lr, self.inp, self.grad_step_size_to_inp
    
    
    def _update_decay_estimate(self, estimate, value, beta):
        """Returns a beta-weighted average of estimate and value."""
        return (beta * estimate) + ((1 - beta) * value)

    '''
    def _expand_features(self, flat_g, flat_v, states):
        n_param = tf.cast(tf.shape(flat_g)[0], tf.float32)
        log_grad_scale = tf.math.log(tf.norm(flat_g, axis=0) / tf.sqrt(n_param) + 1e-8)
        
        grad_first_moment = []
        for i in range(len(self.decay)):
            moments = []
            for state in states:
                moments.append(state["ms_{}".format(i)]) 
            grad_first_moment.append(tf.concat(moments, 0))
        grad_first_moment = tf.concat(grad_first_moment, 1)
        log_grad_first_moment_scale = tf.math.log(tf.norm(grad_first_moment, axis=0) / tf.sqrt(n_param) + 1e-8)
        
        grad_inner_product = tf.reduce_sum(grad_first_moment * flat_g, 0)
        param_scale = tf.norm(flat_v, axis=0) / tf.sqrt(n_param)
        step = tf.reshape(self.tanh_embedding(states[0]["step"]), (-1, ))
        
        inp = tf.concat([log_grad_scale, log_grad_first_moment_scale, grad_inner_product, param_scale, step], 0)
        inp = tf.reshape(inp, (1, -1))
        return inp
    '''

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

    '''
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
        timestep = state["step"]

        if isinstance(grad, tf.IndexedSlices):
            grad_indices, grad_values = utils.accumulate_sparse_gradients(grad)

        return grad_values, None, timestep, grad_indices
    '''