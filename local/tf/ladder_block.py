#!/usr/bin/env python

"""
# Created by feit on 10/2/19
email: feit@uber.com
blocks for ladder network (python2)
"""

from __future__ import print_function
import tensorflow as tf
import numpy
import math
import os
import sys
from models import Model
from tf_block import batch_norm_wrapper
from ze_utils import set_cuda_visible_devices
import collections
from tensorflow.python.compat import compat
from tensorflow.python.eager import context
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors_impl
from tensorflow.python.framework import graph_util
from tensorflow.python.framework import ops
from tensorflow.python.framework import random_seed
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import check_ops
from tensorflow.python.ops import gen_nn_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import random_ops
VAR2STD_EPSILON = 0.00001


def _get_sequence(value, n, channel_index, name):
  """Formats a value input for gen_nn_ops."""
  if value is None:
    value = [1]
  elif not isinstance(value, collections.Sized):
    value = [value]

  current_n = len(value)
  if current_n == n + 2:
    return value
  elif current_n == 1:
    value = list((value[0],) * n)
  elif current_n == n:
    value = list(value)
  else:
    raise ValueError("{} should be of length 1, {} or {} but was {}".format(
        name, n, n + 2, current_n))

  if channel_index == 1:
    return [1, 1] + value
  else:
    return [1] + value + [1]


def conv1d_transpose(
    input,  # pylint: disable=redefined-builtin
    filters,
    output_shape,
    strides,
    padding="SAME",
    data_format="NWC",
    dilations=None,
    name=None):
  """The transpose of `conv1d`.
  This operation is sometimes called "deconvolution" after [Deconvolutional
  Networks](https://www.matthewzeiler.com/mattzeiler/deconvolutionalnetworks.pdf),
  but is really the transpose (gradient) of `conv1d` rather than an actual
  deconvolution.
  Args:
    input: A 3-D `Tensor` of type `float` and shape
      `[batch, in_width, in_channels]` for `NWC` data format or
      `[batch, in_channels, in_width]` for `NCW` data format.
    filters: A 3-D `Tensor` with the same type as `value` and shape
      `[filter_width, output_channels, in_channels]`.  `filter`'s
      `in_channels` dimension must match that of `value`.
    output_shape: A 1-D `Tensor`, containing three elements, representing the
      output shape of the deconvolution op.
    strides: An int or list of `ints` that has length `1` or `3`.  The number of
      entries by which the filter is moved right at each step.
    padding: A string, either `'VALID'` or `'SAME'`. The padding algorithm.
      See the "returns" section of `tf.nn.convolution` for details.
    data_format: A string. `'NWC'` and `'NCW'` are supported.
    dilations: An int or list of `ints` that has length `1` or `3` which
      defaults to 1. The dilation factor for each dimension of input. If set to
      k > 1, there will be k-1 skipped cells between each filter element on that
      dimension. Dilations in the batch and depth dimensions must be 1.
    name: Optional name for the returned tensor.
  Returns:
    A `Tensor` with the same type as `value`.
  Raises:
    ValueError: If input/output depth does not match `filter`'s shape, if
      `output_shape` is not at 3-element vector, if `padding` is other than
      `'VALID'` or `'SAME'`, or if `data_format` is invalid.
  """
  with ops.name_scope(name, "conv1d_transpose",
                      [input, filters, output_shape]) as name:
    # The format could be either NWC or NCW, map to NHWC or NCHW
    if data_format is None or data_format == "NWC":
      data_format = "NHWC"
      spatial_start_dim = 1
      channel_index = 2
    elif data_format == "NCW":
      data_format = "NCHW"
      spatial_start_dim = 2
      channel_index = 1
    else:
      raise ValueError("data_format must be \"NWC\" or \"NCW\".")

    # Reshape the input tensor to [batch, 1, in_width, in_channels]
    strides = [1] + _get_sequence(strides, 1, channel_index, "stride")
    dilations = [1] + _get_sequence(dilations, 1, channel_index, "dilations")

    input = array_ops.expand_dims(input, spatial_start_dim)
    filters = array_ops.expand_dims(filters, 0)
    output_shape = list(output_shape) if not isinstance(
        output_shape, ops.Tensor) else output_shape
    output_shape = array_ops.concat([output_shape[: spatial_start_dim], [1],
                                     output_shape[spatial_start_dim:]], 0)

    result = gen_nn_ops.conv2d_backprop_input(
        input_sizes=output_shape,
        filter=filters,
        out_backprop=input,
        strides=strides,
        padding=padding,
        data_format=data_format,
        dilations=dilations,
        name=name)
    return array_ops.squeeze(result, spatial_start_dim)




# noinspection PyAttributeOutsideInit
class ModelWithoutDropoutLadderTdnn(Model):

    def __init__(self):
        super(ModelWithoutDropoutLadderTdnn, self).__init__()

    def build_model(self, num_classes, input_feature_dim, output_dir, logger=None):

        # logger_dir = '/media/feit/Work/Work/SpeakerID/Kaldi_Voxceleb/outputlog'
        layer_sizes = [input_feature_dim, 512, 512, 512, 512, 3 * 512]
        kernel_sizes = [1, 5, 3, 3, 1, 1]
        embedding_sizes = [512, 512]
        dilation_rates = [1, 1, 2, 3, 1, 1]
        noise_std = 0.3  # scaling factor for noise used in corrupted encoder
        # hyperparameters that denote the importance of each layer
        denoising_cost = [10.0, 5.0, 0.1, 0.1, 0.1, 0.1]

        if logger is not None:
            logger.info("Start building the model ...")

        tf.reset_default_graph()
        self.graph = tf.Graph()

        with self.graph.as_default():
            self.num_classes = num_classes

            # placeholder for parameter
            self.learning_rate = tf.placeholder(tf.float32, name="learning_rate")
            self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")
            self.phase = tf.placeholder(tf.bool, name="phase")

            # Placeholders for regular data
            self.input_x = tf.placeholder(tf.float32, [None, None, input_feature_dim], name="input_x")
            self.input_y = tf.placeholder(tf.float32, [None, num_classes], name="input_y")

            set_false = tf.constant(False, dtype=tf.bool)
            L = len(layer_sizes)

            inputs = self.input_x

            # batchnorm scalor
            def bi(inits, size, name):
                return tf.Variable(inits * tf.ones([size]), name=name)

            # layer bias
            def wbi(size, name):
                return tf.Variable(tf.constant(0.1, shape=[size]), name=name)

            # layer weights
            def wi(shape, name):
                return tf.Variable(tf.truncated_normal(shape, stddev=0.1), name=name)

            # shared parameters
            shapes = zip(kernel_sizes[1:], layer_sizes[:-1], layer_sizes[1:])  # shapes of linear layers
            weights = {'W': [wi(s, "W") for s in shapes],
                       # Encoder weights
                       'V': [wi(s, "V") for s in shapes],  # Decoder weights
                       'BW': [wbi(layer_sizes[l], "B") for l in range(1, L)],  # Encoder bias
                       'BV': [wbi(layer_sizes[l], "B") for l in range(0, L-1)]}  # Decoder bias

            training = self.phase

            ewma = tf.train.ExponentialMovingAverage(
                decay=0.95)  # to calculate the moving averages of mean and variance
            bn_assigns = []  # this list stores the updates to be made to average mean and variance

            def batch_normalization(batch, mean=None, var=None, keep_dims=True):
                if mean is None or var is None:
                    mean, var = tf.nn.moments(batch, axes=[0], keep_dims=keep_dims)
                return (batch - mean) / tf.sqrt(var + tf.constant(1e-10))

            # average mean and variance of all layers
            running_mean = [tf.Variable(tf.constant(0.0, shape=[1, kernel_sizes[l], layer_sizes[l]]), trainable=False)
                            for l in range(1,L)]
            running_var = [tf.Variable(tf.constant(1.0, shape=[1, kernel_sizes[l], layer_sizes[l]]), trainable=False)
                           for l in range(1,L)]

            def update_batch_normalization(batch, l, keep_dims=True):
                "batch normalize + update average mean and variance of layer l"
                mean, var = tf.nn.moments(batch, axes=[0], keep_dims=keep_dims)
                assign_mean = running_mean[l - 1].assign(mean)
                assign_var = running_var[l - 1].assign(var)
                bn_assigns.append(ewma.apply([running_mean[l - 1], running_var[l - 1]]))
                with tf.control_dependencies([assign_mean, assign_var]):
                    return (batch - mean) / tf.sqrt(var + 1e-10)

            def encoder(inputs, noise_std):
                h_l = inputs + tf.random_normal(tf.shape(inputs)) * noise_std  # add noise to input for labeled
                h_u = inputs + tf.random_normal(tf.shape(inputs)) * noise_std  # add noise to input for unlabeled
                d = {}  # to store the pre-activation, activation, mean and variance for each layer
                # The data for labeled and unlabeled examples are stored separately
                d['labeled'] = {'z': {}, 'm': {}, 'v': {}, 'h': {}}
                d['unlabeled'] = {'z': {}, 'm': {}, 'v': {}, 'h': {}}
                d['labeled']['z'][0], d['unlabeled']['z'][0] = h_l, h_u
                for l in range(1, L):
                    with tf.variable_scope("encoder-layer-%s" % l):
                        d['labeled']['h'][l - 1], d['unlabeled']['h'][l - 1] = h_l, h_u
                        conv_l = tf.nn.convolution(h_l, weights['W'][l - 1], dilation_rate=[dilation_rates[l]],
                                                   padding="SAME", name="conv-layer-%s-l" % l)
                        z_pre_l = tf.nn.bias_add(conv_l, weights['BW'][l - 1])
                        conv_u = tf.nn.convolution(h_u, weights['W'][l - 1], dilation_rate=[dilation_rates[l]],
                                                   padding="SAME", name="conv-layer-%s-u" % l)
                        z_pre_u = tf.nn.bias_add(conv_u, weights['BW'][l - 1])

                        axis = list(range(len(z_pre_u.get_shape()) - 1))
                        m, v = tf.nn.moments(z_pre_u, axes=axis)

                        if noise_std > 0:
                            z_u = batch_norm_wrapper(z_pre_u, decay=0.95, is_training=set_false,
                                                     name_prefix="unlabeled-corrupt")
                            z_l = batch_norm_wrapper(z_pre_l, decay=0.95, is_training=set_false,
                                                     name_prefix="labeled-corrupt")
                            z_l += tf.random_normal(tf.shape(z_l)) * noise_std
                            z_u += tf.random_normal(tf.shape(z_u)) * noise_std
                        else:
                            z_u = batch_norm_wrapper(z_pre_u, decay=0.95, is_training=training,
                                                     name_prefix="unlabeled-clean")
                            z_l = batch_norm_wrapper(z_pre_l, decay=0.95, is_training=training,
                                                     name_prefix="labeled-clean")
                        z = [z_l, z_u]
                        # # if training:
                        # def training_batch_norm():
                        #     # Training batch normalization
                        #     # batch normalization for labeled and unlabeled examples is performed separately
                        #     if noise_std > 0:
                        #         # Corrupted encoder
                        #         # batch normalization + noise
                        #         z_l = batch_normalization(z_pre_l, keep_dims=keep_dims)
                        #         z_l += tf.random_normal(tf.shape(z_l)) * noise_std
                        #         z_u = batch_normalization(z_pre_u, m, v, keep_dims=keep_dims)
                        #         z_u += tf.random_normal(tf.shape(z_u)) * noise_std
                        #     else:
                        #         # Clean encoder
                        #         # batch normalization + update the average mean and variance using batch mean and variance of labeled examples
                        #         z_l = update_batch_normalization(z_pre_l, l, keep_dims=keep_dims)
                        #         z_u = batch_normalization(z_pre_u, m, v, keep_dims=keep_dims)
                        #     return [z_l, z_u]
                        # # else:
                        # def eval_batch_norm():
                        #     # Evaluation batch normalization
                        #     # obtain average mean and variance and use it to normalize the batch
                        #     mean = ewma.average(running_mean[l - 1])
                        #     var = ewma.average(running_var[l - 1])
                        #     z_l = batch_normalization(z_pre_l, mean, var, keep_dims=keep_dims)
                        #     z_u = batch_normalization(z_pre_u, mean, var, keep_dims=keep_dims)
                        #     # Instead of the above statement, the use of the following 2 statements containing a typo
                        #     # consistently produces a 0.2% higher accuracy for unclear reasons.
                        #     # m_l, v_l = tf.nn.moments(z_pre_l, axes=[0])
                        #     # z = join(batch_normalization(z_pre_l, m_l, mean, var), batch_normalization(z_pre_u, mean, var))
                        #     return [z_l, z_u]
                        # # perform batch normalization according to value of boolean "training" placeholder:
                        # z = tf.cond(training, training_batch_norm, eval_batch_norm)

                        d['labeled']['z'][l], d['unlabeled']['z'][l] = z[0], z[1]
                        d['unlabeled']['m'][l], d['unlabeled']['v'][
                            l] = m, v  # save mean and variance of unlabeled examples for decoding
                        # use ReLU activation in hidden layers
                        # h_l = tf.nn.relu(z[0] + weights["beta"][l - 1])
                        # h_u = tf.nn.relu(z[1] + weights["beta"][l - 1])
                        h_l = tf.nn.relu(z[0])
                        h_u = tf.nn.relu(z[1])
                d['labeled']['h'][l], d['unlabeled']['h'][l] = h_l, h_u
                return [h_l, h_u], d  # wrap labeled and unlabeled hidden value

            y, clean = encoder(inputs, 0.0)  # clean encoder
            y_c, corr = encoder(inputs, noise_std)  # corrupted encoder

            def g_gauss(z_c, u, size):
                "gaussian denoising function proposed in the original paper"
                wi = lambda inits, name: tf.Variable(inits * tf.ones(size), name=name)
                a1 = wi(0., 'a1')
                a2 = wi(1., 'a2')
                a3 = wi(0., 'a3')
                a4 = wi(0., 'a4')
                a5 = wi(0., 'a5')
                a6 = wi(0., 'a6')
                a7 = wi(1., 'a7')
                a8 = wi(0., 'a8')
                a9 = wi(0., 'a9')
                a10 = wi(0., 'a10')

                mu = a1 * tf.sigmoid(a2 * u + a3) + a4 * u + a5
                v = a6 * tf.sigmoid(a7 * u + a8) + a9 * u + a10

                z_est = (z_c - mu) * v + mu
                return z_est

            # Decoder
            z_est = {}
            d_cost = []  # to store the denoising cost of all layers
            for l in range(L - 1, -1, -1):
                with tf.variable_scope("decoder-layer-%s" % l):
                    z, z_c = clean['unlabeled']['z'][l], corr['unlabeled']['z'][l]
                    m, v = clean['unlabeled']['m'].get(l, 0), clean['unlabeled']['v'].get(l, 1 - 1e-10)
                    if l == L - 1:
                        u = y_c[1]  # get the unlabeled corrupted hidden value
                    else:
                        output_shape = tf.stack([tf.shape(z_est[l+1])[0],tf.shape(z_est[l+1])[1],layer_sizes[l]])
                        conv_u = conv1d_transpose(z_est[l + 1], weights['V'][l], output_shape= output_shape, strides=1,
                                                  padding="SAME", name="conv-layer-%s-d" % l)
                        u = tf.nn.bias_add(conv_u, weights['BV'][l])
                        # conv_u = tf.nn.convolution(z_est[l + 1], weights['V'][l],
                        #                            padding="SAME", name="conv-layer-%s-d" % l)
                        # u = tf.nn.bias_add(conv_u, weights['BV'][l])

                    u = batch_norm_wrapper(u, decay=0.95, is_training=set_false)
                    # u = batch_normalization(u)
                    # TODO: modify the size for g_gauss
                    z_est[l] = g_gauss(z_c, u, layer_sizes[l])
                    z_est_bn = (z_est[l] - m) / v
                    # append the cost of this layer to d_cost (normalized by layer size)
                    # TODO: modify layer size normalization
                    # d_cost.append((tf.reduce_mean(tf.reduce_sum(tf.square(z_est_bn - z), 1)) / layer_sizes[l]) *
                    #               denoising_cost[l])
                    d_cost.append((tf.reduce_mean(tf.reduce_sum(tf.square(z_est_bn - z), 1)) /
                         tf.cast(tf.reduce_prod(tf.shape(z_est[l])[1:]), dtype=tf.float32)) * denoising_cost[l])

            # calculate total unsupervised cost by adding the denoising cost of all layers
            unsupervised_cost = tf.add_n(d_cost)

            # build classification block
            y_clean = y[0]
            # Statistic pooling
            tf_mean, tf_var = tf.nn.moments(y_clean, 1)
            h = tf.concat([tf_mean, tf.sqrt(tf_var + VAR2STD_EPSILON)], 1)
            prev_dim = layer_sizes[-1] * 2

            # Embedding layers
            for i, out_dim in enumerate(embedding_sizes):
                with tf.variable_scope("embed_layer-%s" % i):
                    w = tf.Variable(tf.truncated_normal([prev_dim, out_dim], stddev=0.1), name="w")
                    b = tf.Variable(tf.constant(0.1, shape=[out_dim]), name="b")

                    h = tf.nn.xw_plus_b(h, w, b, name="scores")

                    h = tf.nn.relu(h, name="relu")
                    h = batch_norm_wrapper(h, decay=0.95, is_training=self.phase)

                    prev_dim = out_dim

            # Softmax
            with tf.variable_scope("output"):
                w = tf.get_variable("w", shape=[prev_dim, num_classes],
                                    initializer=tf.glorot_uniform_initializer())
                b = tf.Variable(tf.constant(0.1, shape=[num_classes]), name="b")

                scores = tf.nn.xw_plus_b(h, w, b, name="scores")

                predictions = tf.argmax(scores, 1, name="predictions")

            cost = tf.nn.softmax_cross_entropy_with_logits(logits=scores, labels=self.input_y)
            supervised_cost = tf.reduce_mean(cost)

            # self.loss = unsupervised_cost
            self.loss = tf.add(supervised_cost, unsupervised_cost, name="loss")

            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.loss,
                                                                                                   name="optimizer")

            with tf.name_scope("accuracy"):
                correct_predictions = tf.equal(predictions, tf.argmax(self.input_y, 1))
                self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")

        # set_cuda_visible_devices(use_gpu=False, logger=logger)
        set_cuda_visible_devices(use_gpu=False, logger=logger)
        with tf.Session(graph=self.graph, config=tf.ConfigProto(allow_soft_placement=True,
                                                                log_device_placement=False)) as sess:
            if logger is not None:
                logger.info("Start initializing the graph ...")
            sess.run(tf.global_variables_initializer())
            Model.save_model(sess, output_dir, logger)
        if logger is not None:
            logger.info("Building finished.")


# if __name__ == '__main__':
#     model = ModelWithoutDropoutLadderTdnn()
#     model.build_model(7323, 30, '/media/feit/Work/Work/SpeakerID/Kaldi_Voxceleb/exp_ladder/model_0')
#     print("Done")
