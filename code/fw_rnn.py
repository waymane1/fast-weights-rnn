from tensorflow.python.ops import array_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops
from tensorflow.contrib.rnn import RNNCell
from tensorflow.python.ops import variable_scope as vs
from tensorflow.python.platform import tf_logging as logging
from tensorflow.contrib.layers.python.layers import layer_norm
from tensorflow.python.util import nest
import tensorflow as tf
import numpy as np

class FWCell(RNNCell):

  def __init__(self, hidden_units, bias=1.0, reuse=False,
               activation=nn_ops.relu, layer_norm=True, norm_gain=1.0, norm_shift=0.0,steps=1, rate_dec=0.9, rate_learn=0.5, dpo_prob=1.0):
    self._g = norm_gain
    self._b = norm_shift
    self._hidden_units = hidden_units
    self._activation = activation
    self._S = steps
    self._eta = rate_learn
    self._lambda = rate_dec
    self._bias = bias
    self._norm_re = reuse
    self._keep_prob = dpo_prob
    self._layer_norm = layer_norm

  def _norm(self, inp, scope=None):
    reuse = tf.get_variable_scope().reuse
    with vs.variable_scope(scope or "norm") as scope:
      normalized = layer_norm(inp, reuse=reuse, scope=scope)
      return normalized

  def fw_calc(self, args, output_size, scope=None):
    assert len(args) == 2
    assert args[0].get_shape().as_list()[1] == output_size

    typ = [a.dtype for a in args][0]

    with vs.variable_scope(scope or "linear"):
      matrixW = vs.get_variable(
        "matrix_w", dtype=typ, initializer=tf.convert_to_tensor(np.eye(output_size, dtype=np.float32) * .05))

      matrixC = vs.get_variable(
        "matrix_c", [args[1].get_shape().as_list()[1], output_size], dtype=typ)

      res = tf.matmul(args[0], matrixW) + tf.matmul(args[1], matrixC)
      return res

  def fw_zero(self, batch_size, dtype):

    state_size = self.state_size

    zeros = array_ops.zeros(
        tf.stack([batch_size, state_size, state_size]), dtype=dtype)
    zeros.set_shape([None, state_size, state_size])

    return zeros

  def _vec2mat(self, vector):
    memory_size = vector.get_shape().as_list()[1]
    return tf.reshape(vector, [-1, memory_size, 1])

  def _mat2vec(self, matrix):
    return tf.squeeze(matrix, [2])

  def __call__(self, inputs, state, scope=None):
    state, fw = state
    with vs.variable_scope(scope or type(self).__name__) as scope:
      """Wh(t) + Cx(t)"""
      linear = self.fw_calc([state, inputs], self._hidden_units, False)
      """h_0(t+1) = f(Wh(t) + Cx(t))"""
      if not self._norm_re:
        h = self._activation(self._norm(linear, scope="Norm0"))
      else:
        h = self._activation(self._norm(linear))
      h = self._vec2mat(h)
      linear = self._vec2mat(linear)
      for i in range(self._S):
        """
        h_{s+1}(t+1) = f([Wh(t) + Cx(t)] + A(t) h_s(t+1)), S times.
        From Eqn (2).
        """
        if not self._norm_re:
          h = self._activation(self._norm(linear +
                                          tf.matmul(fw, h), scope="Norm%d" % (i + 1)))
        else:
          h = self._activation(self._norm(linear +
                                          math_ops.batch_matmul(fw, h)))

      """
      Compute A(t+1)  according to Eqn (4)
      """
      state = self._vec2mat(state)
      new_fw = self._lambda * fw + self._eta * tf.matmul(state, state, adjoint_b=True)

      h = self._mat2vec(h)

      return h, (h, new_fw)
