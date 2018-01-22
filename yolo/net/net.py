from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import re


class Net(object):
    def __init__(self, common_params, net_params):
        """
        common_params: a params dict
        net_params: a params dict
        """
        # pretrained variable collection
        self.pretrained_collection = []
        # trainable variable collection
        self.trainable_collection = []

    # Helper to create a Variable stored on CPU memory.
    def _variable_on_cpu(self, name, shape, initializer, pretrain=True, train=True):
        with tf.device('/cpu:0'):
            var = tf.get_variable(name, shape, initializer=initializer, dtype=tf.float32)
            if pretrain:
                self.pretrained_collection.append(var)
            if train:
                self.trainable_collection.append(var)
        return var

    def _variable_with_weight_decay(self, name, shape, stddev, weight_decay, pretrain=True, train=True):
        var = self._variable_on_cpu(name, shape,tf.truncated_normal_initializer(stddev=stddev, dtype=tf.float32), pretrain, train)
        if weight_decay is not None:
            weight_loss = tf.multiply(tf.nn.l2_loss(var), weight_decay, name='weight_loss')
            tf.add_to_collection('losses', weight_loss)
        return var

    def conv2d(self, input, kernel_size, stride=1, pretrain=True, train=True):
        self.conv_num += 1
        with tf.variable_scope('conv{}'.format(self.conv_num)):
            kernel = self._variable_with_weight_decay('weights',shape=kernel_size,stddev=5e-2,
                                                      weight_decay=self.weight_decay, pretrain=pretrain, train=train)
            conv = tf.nn.conv2d(input, kernel, [1, stride, stride, 1], padding='SAME')
            biases = self._variable_on_cpu('biases', kernel_size[3:], tf.constant_initializer(0.0), pretrain, train)
            bias = tf.nn.bias_add(conv, biases)
        return self.leaky_relu(bias)

    def max_pool(self, input, kernel_size, stride):
        return tf.nn.max_pool(input, ksize=[1, kernel_size[0], kernel_size[1], 1], strides=[1, stride, stride, 1],padding='SAME')

    def fully_connection(self, scope, input, in_dimension, out_dimension, leaky=True, pretrain=True, train=True):
        with tf.variable_scope(scope) as scope:
            # [batch_size, width * height * dim]
            reshape = tf.reshape(input, [tf.shape(input)[0], -1])

            weights = self._variable_with_weight_decay('weights', shape=[in_dimension, out_dimension],stddev=0.04,
                                                       weight_decay=self.weight_decay, pretrain=pretrain,train=train)
            biases = self._variable_on_cpu('biases', [out_dimension], tf.constant_initializer(0.0), pretrain, train)
            output = tf.matmul(reshape, weights) + biases

        return  self.leaky_relu(output) if leaky else tf.identity(output, name=scope.name)

    def leaky_relu(self, x, alpha=0.1, dtype=tf.float32):
        x = tf.cast(x, dtype=dtype)
        bool_mask = (x > 0)
        mask = tf.cast(bool_mask, dtype=dtype)
        return 1.0 * mask * x + alpha * (1 - mask) * x

    def inference(self, images):
        """Build the yolo model
    
        Args:
          images:  4-D tensor [batch_size, image_height, image_width, channels]
        Returns:
          predicts: 4-D tensor [batch_size, cell_size, cell_size, num_classes + 5 * boxes_per_cell]
        """
        raise NotImplementedError

    def loss(self, predicts, labels, objects_num):
        """Add Loss to all the trainable variables
    
        Args:
          predicts: 4-D tensor [batch_size, cell_size, cell_size, 5 * boxes_per_cell]
          ===> (num_classes, boxes_per_cell, 4 * boxes_per_cell)
          labels  : 3-D tensor of [batch_size, max_objects, 5]
          objects_num: 1-D tensor [batch_size]
        """
        raise NotImplementedError
