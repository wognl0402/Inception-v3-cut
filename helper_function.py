from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.eager import context
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import confusion_matrix
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import weights_broadcast_ops
from tensorflow.python.ops.losses import util
from tensorflow.python.util.deprecation import deprecated_args
from tensorflow.python.util.deprecation import deprecated_argument_lookup
from tensorflow.python.util.tf_export import tf_export

import tensorflow as tf

@tf_export("losses.Reduction")
class Reduction(object):
  """Types of loss reduction.
  Contains the following values:
  `NONE`: Un-reduced weighted losses with the same shape as input.
  `SUM`: Scalar sum of weighted losses.
  `MEAN`: Scalar `SUM` divided by sum of weights.
  `SUM_OVER_BATCH_SIZE`: Scalar `SUM` divided by number of elements in losses.
  `SUM_OVER_NONZERO_WEIGHTS`: Scalar `SUM` divided by number of non-zero
     weights.
  `SUM_BY_NONZERO_WEIGHTS`: Same as `SUM_OVER_NONZERO_WEIGHTS`.
  """

  NONE = "none"

  SUM = "weighted_sum"

  MEAN = "weighted_mean"

  SUM_OVER_BATCH_SIZE = "weighted_sum_over_batch_size"

  SUM_BY_NONZERO_WEIGHTS = "weighted_sum_by_nonzero_weights"
  SUM_OVER_NONZERO_WEIGHTS = SUM_BY_NONZERO_WEIGHTS

  @classmethod
  def all(cls):
    return (
        cls.NONE,
        cls.SUM,
        cls.MEAN,
        cls.SUM_OVER_BATCH_SIZE,
        cls.SUM_OVER_NONZERO_WEIGHTS,
        cls.SUM_BY_NONZERO_WEIGHTS)

  @classmethod
  def validate(cls, key):
    if key not in cls.all():
      raise ValueError("Invalid ReductionKey %s." % key)

def _safe_div(numerator, denominator, name="value"):
  """Computes a safe divide which returns 0 if the denominator is zero.
  Note that the function contains an additional conditional check that is
  necessary for avoiding situations where the loss is zero causing NaNs to
  creep into the gradient computation.
  Args:
    numerator: An arbitrary `Tensor`.
    denominator: `Tensor` whose shape matches `numerator` and whose values are
      assumed to be non-negative.
    name: An optional name for the returned op.
  Returns:
    The element-wise value of the numerator divided by the denominator.
  """
  return array_ops.where(
      math_ops.greater(denominator, 0),
      math_ops.div(numerator, array_ops.where(
          math_ops.equal(denominator, 0),
          array_ops.ones_like(denominator), denominator)),
      array_ops.zeros_like(numerator),
name=name)


def _safe_mean(losses, num_present):
  """Computes a safe mean of the losses.
  Args:
    losses: `Tensor` whose elements contain individual loss measurements.
    num_present: The number of measurable elements in `losses`.
  Returns:
    A scalar representing the mean of `losses`. If `num_present` is zero,
      then zero is returned.
  """
  total_loss = math_ops.reduce_sum(losses)
  return _safe_div(total_loss, num_present)

@tf_export("losses.compute_weighted_loss")
def compute_weighted_loss(
  losses, weights=1.0, scope=None, loss_collection=ops.GraphKeys.LOSSES,
  reduction=Reduction.SUM_BY_NONZERO_WEIGHTS):
  """Computes the weighted loss.
  Args:
    losses: `Tensor` of shape `[batch_size, d1, ... dN]`.
    weights: Optional `Tensor` whose rank is either 0, or the same rank as
      `losses`, and must be broadcastable to `losses` (i.e., all dimensions must
      be either `1`, or the same as the corresponding `losses` dimension).
    scope: the scope for the operations performed in computing the loss.
    loss_collection: the loss will be added to these collections.
    reduction: Type of reduction to apply to loss.
  Returns:
    Weighted loss `Tensor` of the same type as `losses`. If `reduction` is
    `NONE`, this has the same shape as `losses`; otherwise, it is scalar.
  Raises:
    ValueError: If `weights` is `None` or the shape is not compatible with
      `losses`, or if the number of dimensions (rank) of either `losses` or
      `weights` is missing.
  Note:
    When calculating the gradient of a weighted loss contributions from
    both `losses` and `weights` are considered. If your `weights` depend
    on some model parameters but you do not want this to affect the loss
    gradient, you need to apply `tf.stop_gradient` to `weights` before
    passing them to `compute_weighted_loss`.
  @compatibility(eager)
  The `loss_collection` argument is ignored when executing eagerly. Consider
  holding on to the return value or collecting losses via a `tf.keras.Model`.
  @end_compatibility
  """
  Reduction.validate(reduction)
  with ops.name_scope(scope, "weighted_loss", (losses, weights)):
    # Save the `reduction` argument for loss normalization when distributing
    # to multiple towers.
    # TODO(josh11b): Associate it with the returned op for more precision.
    ops.get_default_graph()._last_loss_reduction = reduction  # pylint: disable=protected-access

    with ops.control_dependencies((
        weights_broadcast_ops.assert_broadcastable(weights, losses),)):
      losses = ops.convert_to_tensor(losses)
      input_dtype = losses.dtype
      losses = math_ops.to_float(losses)
      weights = math_ops.to_float(weights)
      weighted_losses = math_ops.multiply(losses, weights)
      if reduction == Reduction.NONE:
        loss = weighted_losses
      else:
        loss = math_ops.reduce_sum(weighted_losses)
        if reduction == Reduction.MEAN:
          loss = _safe_mean(
              loss,
              math_ops.reduce_sum(array_ops.ones_like(losses) * weights))
        elif (reduction == Reduction.SUM_BY_NONZERO_WEIGHTS or
              reduction == Reduction.SUM_OVER_NONZERO_WEIGHTS):
          loss = _safe_mean(loss, _num_present(losses, weights))
        elif reduction == Reduction.SUM_OVER_BATCH_SIZE:
          loss = _safe_mean(loss, _num_elements(losses))

      # Convert the result back to the input type.
      loss = math_ops.cast(loss, input_dtype)
      # util.add_loss(loss, loss_collection)
  return loss

@tf_export("losses.softmax_cross_entropy")
def softmax_cross_entropy(
    onehot_labels, logits, weights=1.0, label_smoothing=0, scope=None,
    loss_collection=ops.GraphKeys.LOSSES,
    reduction=Reduction.SUM_BY_NONZERO_WEIGHTS):
  """Creates a cross-entropy loss using tf.nn.softmax_cross_entropy_with_logits_v2.
  `weights` acts as a coefficient for the loss. If a scalar is provided,
  then the loss is simply scaled by the given value. If `weights` is a
  tensor of shape `[batch_size]`, then the loss weights apply to each
  corresponding sample.
  If `label_smoothing` is nonzero, smooth the labels towards 1/num_classes:
      new_onehot_labels = onehot_labels * (1 - label_smoothing)
                          + label_smoothing / num_classes
  Note that `onehot_labels` and `logits` must have the same shape,
  e.g. `[batch_size, num_classes]`. The shape of `weights` must be
  broadcastable to loss, whose shape is decided by the shape of `logits`.
  In case the shape of `logits` is `[batch_size, num_classes]`, loss is
  a `Tensor` of shape `[batch_size]`.
  Args:
    onehot_labels: One-hot-encoded labels.
    logits: Logits outputs of the network.
    weights: Optional `Tensor` that is broadcastable to loss.
    label_smoothing: If greater than 0 then smooth the labels.
    scope: the scope for the operations performed in computing the loss.
    loss_collection: collection to which the loss will be added.
    reduction: Type of reduction to apply to loss.
  Returns:
    Weighted loss `Tensor` of the same type as `logits`. If `reduction` is
    `NONE`, this has shape `[batch_size]`; otherwise, it is scalar.
  Raises:
    ValueError: If the shape of `logits` doesn't match that of `onehot_labels`
      or if the shape of `weights` is invalid or if `weights` is None.  Also if
      `onehot_labels` or `logits` is None.
  @compatibility(eager)
  The `loss_collection` argument is ignored when executing eagerly. Consider
  holding on to the return value or collecting losses via a `tf.keras.Model`.
  @end_compatibility
  """
  if onehot_labels is None:
    raise ValueError("onehot_labels must not be None.")
  if logits is None:
    raise ValueError("logits must not be None.")
  with ops.name_scope(scope, "softmax_cross_entropy_loss",
                      (logits, onehot_labels, weights)) as scope:
    logits = ops.convert_to_tensor(logits)
    onehot_labels = math_ops.cast(onehot_labels, logits.dtype)
    logits.get_shape().assert_is_compatible_with(onehot_labels.get_shape())

    if label_smoothing > 0:
      num_classes = math_ops.cast(
          array_ops.shape(onehot_labels)[1], logits.dtype)
      smooth_positives = 1.0 - label_smoothing
      smooth_negatives = label_smoothing / num_classes
      onehot_labels = onehot_labels * smooth_positives + smooth_negatives

    onehot_labels = array_ops.stop_gradient(
        onehot_labels, name="labels_stop_gradient")
    losses = nn.softmax_cross_entropy_with_logits_v2(
        labels=onehot_labels, logits=logits, name="xentropy")

    return compute_weighted_loss(
losses, weights, scope, loss_collection, reduction=reduction)

def cross_entropy_loss(logits, one_hot_labels, label_smoothing=0,
                       weight=1.0, scope=None):
  """Define a Cross Entropy loss using softmax_cross_entropy_with_logits.

  It can scale the loss by weight factor, and smooth the labels.

  Args:
    logits: [batch_size, num_classes] logits outputs of the network .
    one_hot_labels: [batch_size, num_classes] target one_hot_encoded labels.
    label_smoothing: if greater than 0 then smooth the labels.
    weight: scale the loss by this factor.
    scope: Optional scope for name_scope.

  Returns:
    A tensor with the softmax_cross_entropy loss.
  """
  logits.get_shape().assert_is_compatible_with(one_hot_labels.get_shape())
  with tf.name_scope(scope, 'CrossEntropyLoss', [logits, one_hot_labels]):
    num_classes = one_hot_labels.get_shape()[-1].value
    one_hot_labels = tf.cast(one_hot_labels, logits.dtype)
    if label_smoothing > 0:
      smooth_positives = 1.0 - label_smoothing
      smooth_negatives = label_smoothing / num_classes
      one_hot_labels = one_hot_labels * smooth_positives + smooth_negatives
    cross_entropy = tf.contrib.nn.deprecated_flipped_softmax_cross_entropy_with_logits(
        logits, one_hot_labels, name='xentropy')

    weight = tf.convert_to_tensor(weight,
                                  dtype=logits.dtype.base_dtype,
                                  name='loss_weight')
    loss = tf.multiply(weight, tf.reduce_mean(cross_entropy), name='value')
    #tf.add_to_collection(LOSSES_COLLECTION, loss)
    return loss

def inception_loss(logits, aux_logits, labels, batch_size=None):
  """Adds all losses for the model.

  Note the final loss is not returned. Instead, the list of losses are collected
  by slim.losses. The losses are accumulated in tower_loss() and summed to
  calculate the total loss.

  Args:
    logits: List of logits from inference(). Each entry is a 2-D float Tensor.
    labels: Labels from distorted_inputs or inputs(). 1-D tensor
            of shape [batch_size]
    batch_size: integer
  """
  if not batch_size:
    batch_size = 1

  #print(batch_size)
  # Reshape the labels into a dense Tensor of
  # shape [FLAGS.batch_size, num_classes].
  sparse_labels = tf.reshape(labels, [batch_size, 1])
  indices = tf.reshape(tf.range(batch_size), [batch_size, 1])
  concated = tf.concat(axis=1, values=[indices, sparse_labels])
  num_classes = logits.get_shape()[-1].value
  dense_labels = tf.sparse_to_dense(concated,
                                    [batch_size, num_classes],
                                    1.0, 0.0)

  print(dense_labels)
  print(logits[0])
  # Cross entropy loss for the main softmax prediction.
  loss = cross_entropy_loss(logits,
                                 dense_labels,
                                 label_smoothing=0.1,
                                 weight=1.0)

  # Cross entropy loss for the auxiliary softmax head.
  loss += cross_entropy_loss(aux_logits,
                                 dense_labels,
                                 label_smoothing=0.1,
                                 weight=0.4,
                                 scope='aux_loss')
  # loss = softmax_cross_entropy(dense_labels,
  #                               logits[0],
  #                                label_smoothing=0.1,
  #                                weights=1.0)

  # # Cross entropy loss for the auxiliary softmax head.
  # loss += softmax_cross_entropy(dense_labels, logits[1],
  #                                label_smoothing=0.1,
  #                                weights=0.4,
  #                                scope='aux_loss')
  return loss
