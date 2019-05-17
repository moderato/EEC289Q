import tvm
import topi
import topi.testing
import numpy as np
import os, logging, sys
from scipy import signal
from topi.util import get_const_tuple
from tvm.contrib.pickle_memoize import memoize
from depth_1by1_schedule import *
from tvm.contrib import nvcc
from tvm import autotvm

# @tvm.register_func
def tvm_callback_cuda_compile(code):
    ptx = nvcc.compile_cuda(code, target="ptx")
    return ptx

def write_code(code, fname):
    with open(fname, "w") as f:
        f.write(code)

# @tvm.register_func
def tvm_callback_cuda_postproc(code):
    if not os.path.exists("perf"):
        os.mkdir("perf")
    write_code(code, "perf/%s_generated.cu" % TASK)
    if USE_MANUAL_CODE:
        code = open("perf/%s_manual.cu" % TASK).read()
    return code

def depthwise_conv2d_with_workload_nhwc(batch, in_channel, in_size, channel_multiplier, kernel, stride, padding="SAME", dtype="float32", default_schedule=False, cudnn_algo=-1, save_data=False, model_name=None, input_feature=None):
    in_width = in_height = in_size
    filter_channel = in_channel
    filter_width = filter_height = kernel
    stride_w = stride_h = stride

    input_shape = (batch, in_channel, in_height, in_width) if default_schedule else (batch, in_height, in_width, in_channel) # in_channel = intermediate_out_channel = group_count
    filter_shape = (filter_channel, channel_multiplier, filter_height, filter_width) if default_schedule else (filter_height, filter_width, filter_channel, channel_multiplier) # channel_multipler = 1

    # placeholder
    Input = tvm.placeholder(input_shape, name='Input')
    Filter = tvm.placeholder(filter_shape, name='Filter')
    # Scale = tvm.placeholder((in_channel * channel_multiplier,), name='Scale')
    # Shift = tvm.placeholder((in_channel * channel_multiplier,), name='Shift')

    # Use memoize, pickle the test data for next time use.
    # @memoize("topi.tests.test_topi_depthwise_conv2d.nhwc")
    def get_ref_data():
        input_np = np.random.uniform(size=input_shape).astype(dtype)
        filter_np = np.random.uniform(size=filter_shape).astype(dtype)
        # scale_np = np.random.uniform(size=scale_shape).astype(dtype)
        # shift_np = np.random.uniform(size=shift_shape).astype(dtype)
        # correctness with scipy
        depthwise_conv2d_scipy = topi.testing.depthwise_conv2d_python_nchw(input_np, filter_np, stride=[stride_h, stride_w], padding=padding) if default_schedule else topi.testing.depthwise_conv2d_python_nhwc(
            input_np, filter_np, stride=[stride_h, stride_w], padding=padding)
        depthwise_conv2d_scipy = np.float32(depthwise_conv2d_scipy)
        # scale_shift_scipy = np.zeros(shape=scale_shift_shape)
        # for c in range(in_channel * channel_multiplier):
        #     scale_shift_scipy[:,:,:,c] = depthwise_conv2d_scipy[:,:,:,c] * scale_np[c] + shift_np[c]
        #     relu_scipy = np.maximum(scale_shift_scipy, 0)
        # return (input_np, filter_np, scale_np, shift_np,
        #         depthwise_conv2d_scipy, scale_shift_scipy, relu_scipy)
        return input_np, filter_np, depthwise_conv2d_scipy
    # Get the test data
    # (input_np, filter_np, scale_np, shift_np,
    #  depthwise_conv2d_scipy, scale_shift_scipy, relu_scipy) = get_ref_data()
    input_np, filter_np, output_np = get_ref_data()
    if input_feature is not None:
        assert input_feature.shape == input_np.shape
        input_np = input_feature

    if save_data:
        filename = "npy/"
        if model_name:
            filename += "%s/" % model_name
        if default_schedule:
            np.save(filename + "depth_input_%d_%d_%d_%d" % (input_np.shape[0], input_np.shape[2], input_np.shape[3], input_np.shape[1]), input_np.transpose((0,2,3,1)))
            # np.save(filename + "depth_weight_%d_%d_%d_%d" % filter_np.shape, filter_np)
            np.save(filename + "depth_weight_%d_%d_%d_%d" % (filter_np.shape[3], filter_np.shape[2], filter_np.shape[0], filter_np.shape[1]), filter_np) # For cudnn benchmark
            np.save(filename + "depth_output_%d_%d_%d_%d" % (output_np.shape[0], output_np.shape[2], output_np.shape[3], output_np.shape[1]), output_np.transpose((0,2,3,1)))
        else:
            np.save(filename + "depth_input_%d_%d_%d_%d" % input_np.shape, input_np)
            # np.save(filename + "depth_weight_%d_%d_%d_%d" % filter_np.shape, filter_np)
            np.save(filename + "depth_weight_%d_%d_%d_%d" % filter_np.shape, filter_np.transpose((3,2,0,1))) # For cudnn benchmark
            np.save(filename + "depth_output_%d_%d_%d_%d" % output_np.shape, output_np)

    # print(output_np[0, 0, 0, 0:100])

    # schedule
    def check_device(device):
        ctx = tvm.context(device, 0)
        if not ctx.exist:
            print("Skip because %s is not enabled" % device)
            return
        if default_schedule:
            device += " -libs=cudnn"
        print("Running on target: %s" % device)

        with tvm.target.create(device):
            # declare
            DepthwiseConv2d = topi.cuda.depthwise_conv2d.depthwise_conv2d_cuda(autotvm.get_config(), Input, Filter, (stride_w, stride_h), (1, 1), dilation=1, algo=0) if default_schedule else topi.nn.depthwise_conv2d_nhwc(Input, Filter, stride=[stride_h, stride_w], padding=padding, dilation=1)
            # ScaleShift = topi.nn.scale_shift_nhwc(DepthwiseConv2d, Scale, Shift)
            # Relu = topi.nn.relu(ScaleShift)

            s1 = topi.cuda.depthwise_conv2d.schedule_depthwise_conv2d_nchw_cuda(autotvm.get_config(), [DepthwiseConv2d]) if default_schedule else schedule_depthwise_conv2d_nhwc_reuse([DepthwiseConv2d], Input)
            # s1 = topi.generic.schedule_depthwise_conv2d_nhwc(DepthwiseConv2d)
            # s2 = topi.generic.schedule_depthwise_conv2d_nhwc(ScaleShift)
            # s3 = topi.generic.schedule_depthwise_conv2d_nhwc(Relu)
            # s3 = schedule_depthwise_conv2d_nhwc_reuse(Relu)
        # build the kernels
        f1 = tvm.build(s1, [Input, Filter, DepthwiseConv2d], device, name="DepthwiseConv2d_%d_%d" % (in_height, in_width))
        # f2 = tvm.build(s2, [Input, Filter, Scale, Shift, ScaleShift], device)
        # f3 = tvm.build(s3, [Input, Filter, Scale, Shift, Relu], device)

        # Prepare pod type for test data closure
        dtype = Input.dtype
        input_shape = get_const_tuple(Input.shape)
        filter_shape = get_const_tuple(Filter.shape)
        # scale_shape = get_const_tuple(Scale.shape)
        # shift_shape = get_const_tuple(Shift.shape)
        # scale_shift_shape = get_const_tuple(ScaleShift.shape)

        # prepare data
        input_tvm = tvm.nd.array(input_np, ctx)
        filter_tvm = tvm.nd.array(filter_np, ctx)
        # scale_tvm = tvm.nd.array(scale_np, ctx)
        # shift_tvm = tvm.nd.array(shift_np, ctx)
        depthwise_conv2d_tvm = tvm.nd.array(np.zeros(shape=get_const_tuple(DepthwiseConv2d.shape), dtype=dtype), ctx)
        # scale_shift_tvm = tvm.nd.array(np.zeros(shape=get_const_tuple(ScaleShift.shape), dtype=dtype), ctx)
        # relu_tvm = tvm.nd.array(np.zeros(shape=get_const_tuple(Relu.shape), dtype=dtype), ctx)
        # launch kernel 1 (depthwise_conv2d)
        timer_1 = f1.time_evaluator(f1.entry_name, ctx, number=1000)
        tcost_1 = timer_1(input_tvm, filter_tvm, depthwise_conv2d_tvm).mean
        # launch kernel 2 (depthwise_conv2d + scale_shift)
        # timer_2 = f2.time_evaluator(f2.entry_name, ctx, number=10)
        # tcost_2 = timer_2(input_tvm, filter_tvm, scale_tvm, shift_tvm, scale_shift_tvm).mean
        # launch kernel 3 (depthwise_conv2d + scale_shift + relu)
        # timer_3 = f3.time_evaluator(f3.entry_name, ctx, number=10)
        # tcost_3 = timer_3(input_tvm, filter_tvm, scale_tvm, shift_tvm, relu_tvm).mean
        # relu_scipy = np.maximum(scale_shift_scipy, 0)
        np.testing.assert_allclose(depthwise_conv2d_tvm.asnumpy(), output_np, rtol=1e-5)
        # np.testing.assert_allclose(scale_shift_tvm.asnumpy(), scale_shift_scipy, rtol=1e-5)
        # np.testing.assert_allclose(relu_tvm.asnumpy(), relu_scipy, rtol=1e-5)
        print("Depthwise convolution: average running time is {:.2f} us.".format(tcost_1 * 1e6))

    check_device("cuda")
    return output_np


def depthwise_conv2d_with_workload_nhwc_auto(batch, in_channel, in_size, channel_multiplier, kernel, stride, padding="SAME", dtype="float32", default_schedule=False):
    in_width = in_height = in_size
    filter_channel = in_channel
    filter_width = filter_height = kernel
    stride_w = stride_h = stride

    def check_device(device):
        ctx = tvm.context(device, 0)
        if not ctx.exist:
            print("Skip because %s is not enabled" % device)
            return
        print("Running on target: %s" % device)

        task = autotvm.task.create(schedule_depthwise_conv2d_nhwc_reuse_auto, args=(batch, in_channel, in_size, channel_multiplier, kernel, stride), target="cuda")
        print(task)
        print(task.config_space)

        # logging config (for printing tuning log to the screen)
        logging.getLogger('autotvm').setLevel(logging.DEBUG)
        logging.getLogger('autotvm').addHandler(logging.StreamHandler(sys.stdout))

        # There are two steps for measuring a config: build and run.
        # By default, we use all cpu cores to compile program. Then measure them sequentially.
        # We measure 5 times and take average to reduce variance.
        measure_option = autotvm.measure_option(
            builder='local',
            runner=autotvm.LocalRunner(number=10))

        tuner = autotvm.tuner.RandomTuner(task)
        tuner.tune(n_trial=25,
                   measure_option=measure_option,
                   callbacks=[autotvm.callback.log_to_file('depthwise_conv2d_nhwc_{}.log'.format(in_size))])

        with autotvm.apply_history_best('depthwise_conv2d_nhwc_{}.log'.format(in_size)):
            with tvm.target.create(device):
                s1, [Input, Filter, DepthwiseConv2d] = schedule_depthwise_conv2d_nhwc_reuse_auto(batch, in_channel, in_size, channel_multiplier, kernel, stride)
                # s3 = schedule_depthwise_conv2d_nhwc_reuse(Relu)
                # build the kernels
                f1 = tvm.build(s1, [Input, Filter, DepthwiseConv2d], device, name="ddd%dddd"%in_size)
                # f2 = tvm.build(s2, [Input, Filter, Scale, Shift, ScaleShift], device)
                # f3 = tvm.build(s3, [Input, Filter, Scale, Shift, Relu], device)

        # Prepare pod type for test data closure
        dtype = Input.dtype
        input_shape = get_const_tuple(Input.shape)
        filter_shape = get_const_tuple(Filter.shape)
        # scale_shape = get_const_tuple(Scale.shape)
        # shift_shape = get_const_tuple(Shift.shape)
        # scale_shift_shape = get_const_tuple(ScaleShift.shape)

        # Use memoize, pickle the test data for next time use.
        @memoize("topi.tests.test_topi_depthwise_conv2d.nhwc")
        def get_ref_data():
            input_np = np.random.uniform(size=input_shape).astype(dtype)
            filter_np = np.random.uniform(size=filter_shape).astype(dtype)
            # scale_np = np.random.uniform(size=scale_shape).astype(dtype)
            # shift_np = np.random.uniform(size=shift_shape).astype(dtype)
            # correctness with scipy
            depthwise_conv2d_scipy = topi.testing.depthwise_conv2d_python_nhwc(input_np, filter_np, stride=[stride_h, stride_w], padding=padding)
            # scale_shift_scipy = np.zeros(shape=scale_shift_shape)
            # for c in range(in_channel * channel_multiplier):
            #     scale_shift_scipy[:,:,:,c] = depthwise_conv2d_scipy[:,:,:,c] * scale_np[c] + shift_np[c]
            #     relu_scipy = np.maximum(scale_shift_scipy, 0)
            # return (input_np, filter_np, scale_np, shift_np, depthwise_conv2d_scipy, scale_shift_scipy, relu_scipy)
            return (input_np, filter_np, depthwise_conv2d_scipy)

        # Get the test data
        # (input_np, filter_np, scale_np, shift_np, depthwise_conv2d_scipy, scale_shift_scipy, relu_scipy) = get_ref_data()
        (input_np, filter_np, depthwise_conv2d_scipy) = get_ref_data()

        # prepare data
        input_tvm = tvm.nd.array(input_np, ctx)
        filter_tvm = tvm.nd.array(filter_np, ctx)
        # scale_tvm = tvm.nd.array(scale_np, ctx)
        # shift_tvm = tvm.nd.array(shift_np, ctx)
        depthwise_conv2d_tvm = tvm.nd.array(np.zeros(shape=get_const_tuple(DepthwiseConv2d.shape), dtype=DepthwiseConv2d.dtype), ctx)
        # scale_shift_tvm = tvm.nd.array(np.zeros(shape=get_const_tuple(ScaleShift.shape), dtype=ScaleShift.dtype), ctx)
        # relu_tvm = tvm.nd.array(np.zeros(shape=get_const_tuple(Relu.shape), dtype=Relu.dtype), ctx)
        # launch kernel 1 (depthwise_conv2d)
        timer_1 = f1.time_evaluator(f1.entry_name, ctx, number=10)
        tcost_1 = timer_1(input_tvm, filter_tvm, depthwise_conv2d_tvm).mean
        # launch kernel 2 (depthwise_conv2d + scale_shift)
        # timer_2 = f2.time_evaluator(f2.entry_name, ctx, number=10)
        # tcost_2 = timer_2(input_tvm, filter_tvm, scale_tvm, shift_tvm, scale_shift_tvm).mean
        # launch kernel 3 (depthwise_conv2d + scale_shift + relu)
        # timer_3 = f3.time_evaluator(f3.entry_name, ctx, number=10)
        # tcost_3 = timer_3(input_tvm, filter_tvm, scale_tvm, shift_tvm, relu_tvm).mean
        # relu_scipy = np.maximum(scale_shift_scipy, 0)
        np.testing.assert_allclose(depthwise_conv2d_tvm.asnumpy(), depthwise_conv2d_scipy, rtol=1e-5)
        # np.testing.assert_allclose(scale_shift_tvm.asnumpy(), scale_shift_scipy, rtol=1e-5)
        # np.testing.assert_allclose(relu_tvm.asnumpy(), relu_scipy, rtol=1e-5)
        print("Depthwise convolution: average running time is {:.2f} us.".format(tcost_1 * 1e6))

    check_device("cuda")

def conv2d_nhwc(batch, in_channel, in_size, num_filter, kernel, stride, padding="SAME", dtype="float32", default_schedule=False, save_data=False, model_name=None, input_feature=None):
    in_height = in_width = in_size

    a_shape = (batch, in_channel, in_height, in_width) if default_schedule else (batch, in_height, in_width, in_channel) # NCHW for cudnn, NHWC for others
    w_shape = (num_filter, in_channel, kernel, kernel) if default_schedule else (kernel, kernel, in_channel, num_filter) # OIHW for cudnn, HWIO for others

    A = tvm.placeholder(a_shape, name='A')
    W = tvm.placeholder(w_shape, name='W')
    dtype = A.dtype

    # @memoize("verify_nhwc")
    def get_ref_data():
        if input_feature is not None:
            assert input_feature.shape == a_shape
            a_np = input_feature
        else:
            a_np = np.random.uniform(size=a_shape).astype(dtype)
        w_np = np.random.uniform(size=w_shape).astype(dtype)
        # Borrow NCHW for cudnn
        b_np = topi.testing.conv2d_nchw_python(a_np, w_np, stride, padding) if default_schedule else topi.testing.conv2d_nhwc_python(a_np, w_np, stride, padding)
        b_np = np.float32(b_np)
        return a_np, w_np, b_np
    a_np, w_np, b_np = get_ref_data()
    
    if save_data:
        filename = "npy/"
        if model_name:
            filename += "%s/" % model_name

        if default_schedule:
            np.save(filename + "conv_input_%d_%d_%d_%d" % (a_np.shape[0], a_np.shape[2], a_np.shape[3], a_np.shape[1]), a_np.transpose((0,2,3,1)))
            # np.save(filename + "depth_weight_%d_%d_%d_%d" % filter_np.shape, filter_np)
            np.save(filename + "conv_weight_%d_%d_%d_%d" % (w_np.shape[3], w_np.shape[2], w_np.shape[0], w_np.shape[1]), w_np) # For cudnn benchmark
            np.save(filename + "conv_output_%d_%d_%d_%d" % (b_np.shape[0], b_np.shape[2], b_np.shape[3], b_np.shape[1]), b_np.transpose((0,2,3,1)))
        else:
            np.save(filename + "conv_input_%d_%d_%d_%d" % a_np.shape, a_np)
            # np.save(filename + "depth_weight_%d_%d_%d_%d" % filter_np.shape, filter_np)
            np.save(filename + "conv_weight_%d_%d_%d_%d" % w_np.shape, w_np.transpose((3,2,0,1))) # For cudnn benchmark
            np.save(filename + "conv_output_%d_%d_%d_%d" % b_np.shape, b_np)

    def check_device(device):
        if not tvm.module.enabled(device):
            print("Skip because %s is not enabled" % device)
            return
        if default_schedule:
            device += " -libs=cudnn"
        print("Running on target: %s" % device)

        with tvm.target.create(device):
            B = topi.cuda.conv2d.conv2d_cuda(autotvm.get_config(), A, W, stride, 0, dilation=1, algo=0) if default_schedule else topi.nn.conv2d_nhwc(A, W, stride, padding, dilation=1)
            s = topi.cuda.schedule_conv2d_nchw_cuda(None, [B]) if default_schedule else schedule_conv2d_nhwc([B], A)

        ctx = tvm.context(device, 0)
        a = tvm.nd.array(a_np, ctx)
        w = tvm.nd.array(w_np, ctx)
        b = tvm.nd.array(np.zeros(get_const_tuple(B.shape), dtype=B.dtype), ctx)
                
        func = tvm.build(s, [A, W, B], device, name=("Conv2d_%d_%d" % (in_height, in_width)))
        # func(a, w, b)
        timer_1 = func.time_evaluator(func.entry_name, ctx, number=10)
        tcost_1 = timer_1(a, w, b).mean
        np.testing.assert_allclose(b.asnumpy(), b_np, rtol=1e-5)
        print("1x1 convolution: average running time is {:.2f} us.".format(tcost_1 * 1e6))

    for device in ["cuda"]:
        check_device(device)

def conv2d_nhwc_auto(batch, in_channel, in_size, num_filter, kernel, stride, padding="SAME", dtype="float32", default_schedule=False):
    in_height = in_width = in_size

    a_shape = (batch, in_height, in_width, in_channel)
    w_shape = (kernel, kernel, in_channel, num_filter)

    def check_device(device):
        if not tvm.module.enabled(device):
            print("Skip because %s is not enabled" % device)
            return
        print("Running on target: %s" % device)

        task = autotvm.task.create(schedule_conv2d_nhwc_auto, args=(batch, in_channel, in_size, num_filter, kernel, stride), target="cuda")
        print(task.config_space)

        # logging config (for printing tuning log to the screen)
        logging.getLogger('autotvm').setLevel(logging.DEBUG)
        logging.getLogger('autotvm').addHandler(logging.StreamHandler(sys.stdout))

        # There are two steps for measuring a config: build and run.
        # By default, we use all cpu cores to compile program. Then measure them sequentially.
        # We measure 5 times and take average to reduce variance.
        measure_option = autotvm.measure_option(
            builder='local',
            runner=autotvm.LocalRunner(number=10))

        tuner = autotvm.tuner.RandomTuner(task)
        tuner.tune(n_trial=25,
                   measure_option=measure_option,
                   callbacks=[autotvm.callback.log_to_file('conv2d_nhwc_{}.log'.format(in_size))])

        with autotvm.apply_history_best('conv2d_nhwc_{}.log'.format(in_size)):
            with tvm.target.create(device):
                s, [A, W, B] = schedule_conv2d_nhwc_auto(batch, in_channel, in_size, num_filter, kernel, stride)
                func = tvm.build(s, [A, W, B], device, name=("ddd%dddd"%in_size))

        @memoize("verify_nhwc")
        def get_ref_data():
            a_np = np.random.uniform(size=a_shape).astype(dtype)
            w_np = np.random.uniform(size=w_shape).astype(dtype)
            b_np = topi.testing.conv2d_nhwc_python(a_np, w_np, stride, padding)
            return a_np, w_np, b_np
        a_np, w_np, b_np = get_ref_data()

        ctx = tvm.context(device, 0)
        a = tvm.nd.array(a_np, ctx)
        w = tvm.nd.array(w_np, ctx)
        b = tvm.nd.array(np.zeros(get_const_tuple(b_np.shape), dtype=dtype), ctx)

        func(a, w, b)
        timer_1 = func.time_evaluator(func.entry_name, ctx, number=10)
        tcost_1 = timer_1(a, w, b).mean
        np.testing.assert_allclose(b.asnumpy(), b_np, rtol=1e-5)
        print("1x1 convolution: average running time is {:.2f} us.".format(tcost_1 * 1e6))

    for device in ['cuda']:
        check_device(device)

def test_depthwise_conv2d():
    # depthwise_1by1_fused(1, 32, 112, 1, 3, 1, 32, layout="NCHW")

    default_schedule = True
    save_data = True
    # output_data = depthwise_conv2d_with_workload_nhwc(1, 32, 112, 1, 3, 1, default_schedule=default_schedule, save_data=save_data) # 51.62us, cudnn 93.53us
    # # depthwise_conv2d_with_workload_nhwc_auto(1, 32, 112, 1, 3, 1, default_schedule=default_schedule) # 51.05us,
    # conv2d_nhwc(1, 32, 112, 32, 1, 1, default_schedule=default_schedule, save_data=save_data, input_feature=output_data) # 53.023us, cudnn 49.10us
    # # conv2d_nhwc_auto(1, 32, 112, 32, 1, 1, default_schedule=default_schedule) # 52.68us


    output_data = depthwise_conv2d_with_workload_nhwc(1, 128, 56, 1, 3, 1, default_schedule=default_schedule, save_data=save_data) # 45.26us, cudnn 61.90us
    # depthwise_conv2d_with_workload_nhwc_auto(1, 128, 56, 1, 3, 1, default_schedule=default_schedule) # 45.08us,
    conv2d_nhwc(1, 128, 56, 128, 1, 1, default_schedule=default_schedule, save_data=save_data, input_feature=output_data) # 132.06us, cudnn 70.42us
    # conv2d_nhwc_auto(1, 128, 56, 128, 1, 1, default_schedule=default_schedule) # 133.17us


    # output_data = depthwise_conv2d_with_workload_nhwc(1, 256, 28, 1, 3, 1, default_schedule=default_schedule, save_data=save_data) # 24.70us, cudnn 44.10us
    # # # depthwise_conv2d_with_workload_nhwc_auto(1, 256, 28, 1, 3, 1, default_schedule=default_schedule) # 24.63us,
    # conv2d_nhwc(1, 256, 28, 256, 1, 1, default_schedule=default_schedule, save_data=save_data, input_feature=output_data) # 134.21us, cudnn 74.89us
    # # # conv2d_nhwc_auto(1, 256, 28, 256, 1, 1, default_schedule=default_schedule) # 149.73us


    # output_data = depthwise_conv2d_with_workload_nhwc(1, 512, 14, 1, 3, 1, default_schedule=default_schedule, save_data=save_data) # 11.48us, cudnn 25.23us
    # # # depthwise_conv2d_with_workload_nhwc_auto(1, 512, 14, 1, 3, 1, default_schedule=default_schedule) # 10.94us,
    # conv2d_nhwc(1, 512, 14, 512, 1, 1, default_schedule=default_schedule, save_data=save_data, input_feature=output_data) # 145.21us, cudnn 90.64us
    # # # conv2d_nhwc_auto(1, 512, 14, 512, 1, 1, default_schedule=default_schedule) # 145.21us

if __name__ == "__main__":
    test_depthwise_conv2d()
