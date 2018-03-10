import os
import tvm
import numpy as np
import tensorflow as tf
from scipy import signal
from tvm.contrib import nvcc

import topi
from topi.util import get_const_tuple
from topi.cuda.depthwise_conv2d import schedule_depthwise_conv2d_nchw, schedule_depthwise_conv2d_nhwc


batch = 1
in_channel = 256
in_height = 21
in_width = 21

filter_channel = in_channel
channel_multiplier = 1
filter_height = 3
filter_width = 3

stride_h = 1
stride_w = 1
padding = 'SAME' # or 'VALID'

    # Placeholder
Input = tvm.placeholder((batch, in_channel, in_height, in_width), name='Input')
Filter = tvm.placeholder((filter_channel, channel_multiplier, filter_height, filter_width), name='Filter')

input_np = np.random.uniform(size=get_const_tuple(Input.shape)).astype(Input.dtype)
filter_np = np.random.uniform(size=get_const_tuple(Filter.shape)).astype(Filter.dtype)
 
with tf.device('/gpu:0'):
    filter_tf_np = np.transpose(filter_np, [2,3,0,1])
    input_tf = tf.placeholder(tf.float32, [batch, in_channel, in_height, in_width])
    filter_tf = tf.placeholder(tf.float32, [filter_height, filter_width, in_channel, channel_multiplier])
    depth_conv_out = tf.nn.depthwise_conv2d(input_tf, filter_tf, strides=[1,1,stride_h,stride_w], padding=padding, data_format='NCHW')
    sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
    sess.run(tf.global_variables_initializer())
    output_tf=sess.run(depth_conv_out, feed_dict={input_tf:input_np, filter_tf:filter_tf_np})
    for i in range(5):
        sess.run(depth_conv_out, feed_dict={input_tf:input_np, filter_tf:filter_tf_np})