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

def depthwise_conv2d_with_workload_nhwc(batch, in_channel, in_size, channel_multiplier, kernel, stride, padding="SAME", dtype="float32", default_schedule=False):
    in_width = in_height = in_size
    filter_channel = in_channel
    filter_width = filter_height = kernel
    stride_w = stride_h = stride
    # placeholder
    Input = tvm.placeholder((batch, in_height, in_width, in_channel), name='Input')
    Filter = tvm.placeholder((filter_height, filter_width, filter_channel, channel_multiplier), name='Filter')
    Scale = tvm.placeholder((in_channel * channel_multiplier,), name='Scale')
    Shift = tvm.placeholder((in_channel * channel_multiplier,), name='Shift')
    # declare
    DepthwiseConv2d = topi.nn.depthwise_conv2d_nhwc(Input, Filter, stride=[stride_h, stride_w], padding=padding, dilation=1)
    ScaleShift = topi.nn.scale_shift_nhwc(DepthwiseConv2d, Scale, Shift)
    Relu = topi.nn.relu(ScaleShift)

    # schedule
    def check_device(device):
        ctx = tvm.context(device, 0)
        if not ctx.exist:
            print("Skip because %s is not enabled" % device)
            return
        print("Running on target: %s" % device)

        with tvm.target.create(device):
            s1 = schedule_depthwise_conv2d_nhwc_reuse([DepthwiseConv2d], Input)
            # s1 = topi.generic.schedule_depthwise_conv2d_nhwc(DepthwiseConv2d)
            # s2 = topi.generic.schedule_depthwise_conv2d_nhwc(ScaleShift)
            # s3 = topi.generic.schedule_depthwise_conv2d_nhwc(Relu)
            # s3 = schedule_depthwise_conv2d_nhwc_reuse(Relu)
        # build the kernels
        f1 = tvm.build(s1, [Input, Filter, DepthwiseConv2d], device, name="ddd%dddd"%in_width)
        # f2 = tvm.build(s2, [Input, Filter, Scale, Shift, ScaleShift], device)
        # f3 = tvm.build(s3, [Input, Filter, Scale, Shift, Relu], device)

        # Prepare pod type for test data closure
        dtype = Input.dtype
        input_shape = get_const_tuple(Input.shape)
        filter_shape = get_const_tuple(Filter.shape)
        scale_shape = get_const_tuple(Scale.shape)
        shift_shape = get_const_tuple(Shift.shape)
        scale_shift_shape = get_const_tuple(ScaleShift.shape)

        # Use memoize, pickle the test data for next time use.
        # @memoize("topi.tests.test_topi_depthwise_conv2d.nhwc")
        def get_ref_data():
            input_np = np.random.uniform(size=input_shape).astype(dtype)
            filter_np = np.random.uniform(size=filter_shape).astype(dtype)
            scale_np = np.random.uniform(size=scale_shape).astype(dtype)
            shift_np = np.random.uniform(size=shift_shape).astype(dtype)
            # correctness with scipy
            depthwise_conv2d_scipy = topi.testing.depthwise_conv2d_python_nhwc(
                input_np, filter_np, stride=[stride_h, stride_w], padding=padding)
            scale_shift_scipy = np.zeros(shape=scale_shift_shape)
            for c in range(in_channel * channel_multiplier):
                scale_shift_scipy[:,:,:,c] = depthwise_conv2d_scipy[:,:,:,c] * scale_np[c] + shift_np[c]
                relu_scipy = np.maximum(scale_shift_scipy, 0)
            return (input_np, filter_np, scale_np, shift_np,
                    depthwise_conv2d_scipy, scale_shift_scipy, relu_scipy)
        # Get the test data
        (input_np, filter_np, scale_np, shift_np,
         depthwise_conv2d_scipy, scale_shift_scipy, relu_scipy) = get_ref_data()

        # prepare data
        input_tvm = tvm.nd.array(input_np, ctx)
        filter_tvm = tvm.nd.array(filter_np, ctx)
        scale_tvm = tvm.nd.array(scale_np, ctx)
        shift_tvm = tvm.nd.array(shift_np, ctx)
        depthwise_conv2d_tvm = tvm.nd.array(np.zeros(shape=get_const_tuple(DepthwiseConv2d.shape), dtype=DepthwiseConv2d.dtype), ctx)
        scale_shift_tvm = tvm.nd.array(np.zeros(shape=get_const_tuple(ScaleShift.shape), dtype=ScaleShift.dtype), ctx)
        relu_tvm = tvm.nd.array(np.zeros(shape=get_const_tuple(Relu.shape), dtype=Relu.dtype), ctx)
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
        np.testing.assert_allclose(depthwise_conv2d_tvm.asnumpy(), depthwise_conv2d_scipy, rtol=1e-5)
        # np.testing.assert_allclose(scale_shift_tvm.asnumpy(), scale_shift_scipy, rtol=1e-5)
        # np.testing.assert_allclose(relu_tvm.asnumpy(), relu_scipy, rtol=1e-5)
        print("Depthwise convolution: average running time is {:.2f} us.".format(tcost_1 * 1e6))

    check_device("cuda")

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

def verify_conv2d_nhwc(batch, in_channel, in_size, num_filter, kernel, stride, padding="SAME", dtype="float32", default_schedule=False):
    in_height = in_width = in_size

    a_shape = (batch, in_channel, in_height, in_width) if default_schedule else (batch, in_height, in_width, in_channel) # NCHW for cudnn, NHWC for others
    w_shape = (num_filter, in_channel, kernel, kernel) if default_schedule else (kernel, kernel, in_channel, num_filter) # OIHW for cudnn, HWIO for others

    A = tvm.placeholder(a_shape, name='A')
    W = tvm.placeholder(w_shape, name='W')
    dtype = A.dtype

    # @memoize("verify_nhwc")
    def get_ref_data():
        a_np = np.random.uniform(size=a_shape).astype(dtype)
        w_np = np.random.uniform(size=w_shape).astype(dtype)
        # Borrow NCHW for cudnn
        b_np = topi.testing.conv2d_nchw_python(a_np, w_np, stride, padding) if default_schedule else topi.testing.conv2d_nhwc_python(a_np, w_np, stride, padding)
        return a_np, w_np, b_np
    a_np, w_np, b_np = get_ref_data()

    def check_device(device):
        if not tvm.module.enabled(device):
            print("Skip because %s is not enabled" % device)
            return
        if default_schedule:
            device += " -libs=cudnn"
        print("Running on target: %s" % device)

        with tvm.target.create(device):
            B = topi.cuda.conv2d.conv2d_cuda(autotvm.get_config(), A, W, stride, 0, dilation=1) if default_schedule else topi.nn.conv2d_nhwc(A, W, stride, padding, dilation=1)
            s = topi.cuda.schedule_conv2d_nchw_cuda(None, [B]) if default_schedule else schedule_conv2d_nhwc([B], A)

        ctx = tvm.context(device, 0)
        a = tvm.nd.array(a_np, ctx)
        w = tvm.nd.array(w_np, ctx)
        b = tvm.nd.array(np.zeros(get_const_tuple(B.shape), dtype=B.dtype), ctx)
                
        func = tvm.build(s, [A, W, B], device, name=("ddd%dddd"%a.shape[2]))
        # func(a, w, b)
        timer_1 = func.time_evaluator(func.entry_name, ctx, number=1000)
        tcost_1 = timer_1(a, w, b).mean
        np.testing.assert_allclose(b.asnumpy(), b_np, rtol=1e-5)
        print("1x1 convolution: average running time is {:.2f} us.".format(tcost_1 * 1e6))

    for device in ["cuda"]:
        check_device(device)

def verify_conv2d_nhwc_auto(batch, in_channel, in_size, num_filter, kernel, stride, padding="SAME", dtype="float32", default_schedule=False):
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

# def depthwise_1by1_fused(batch, in_channel_depthwise, in_size, channel_multiplier, kernel_depthwise, stride_depthwise, num_filter, padding_depthwise="SAME", dtype="float32", layout="NCHW"):
#     assert layout in ["NCHW", "NHWC"]
#     in_h_d = in_w_d = in_size
#     in_c_d = kernel_c_d = in_channel_depthwise
#     kernel_h_d = kernel_w_d = kernel_depthwise
#     out_c_d = in_c_1 = in_channel_depthwise * channel_multiplier
#     stride_h_d = stride_w_d = stride_depthwise

#     kernel_h_1 = kernel_w_1 = 1
#     stride_h_1 = stride_w_1 = 1
#     padding_h_1 = padding_w_1 = 0
#     out_c_1 = num_filter

#     # placeholder
#     if layout == "NCHW":
#         # Input: NCHW, Kernel: CMHW for depthwise, OIHW for conv2d
#         Input = tvm.placeholder((batch, in_c_d, in_h_d, in_w_d), name='Input')
#         Kernel_d = tvm.placeholder((kernel_c_d, channel_multiplier, kernel_h_d, kernel_w_d), name='Kernel_d')
#         Kernel_1 = tvm.placeholder((out_c_1, in_c_1, kernel_h_1, kernel_w_1), name='Kernel_1')
#     else: # NHWC
#         # Input: NHWC, Kernel: HWCM for depthwise, HWIO for conv2d
#         Input = tvm.placeholder((batch, in_h_d, in_w_d, in_c_d), name='Input')
#         Kernel_d = tvm.placeholder((kernel_h_d, kernel_w_d, kernel_c_d, channel_multiplier), name='Kernel_d')
#         Kernel_1 = tvm.placeholder((kernel_h_1, kernel_w_1, in_c_1, out_c_1), name='Kernel_1')
    
#     Scale_d = tvm.placeholder((out_c_d,), name='Scale_d')
#     Shift_d = tvm.placeholder((out_c_d,), name='Shift_d')
#     Scale_1 = tvm.placeholder((out_c_1,), name='Scale_1')
#     Shift_1 = tvm.placeholder((out_c_1,), name='Shift_1')


#     # declare
#     if layout == "NCHW":
#         Conv_d = topi.nn.depthwise_conv2d_nchw(Input, Kernel_d, stride=[stride_h_d, stride_w_d], padding=padding_depthwise, dilation=1, out_dtype=dtype)
#         ScaleShift_d = topi.nn.scale_shift_nchw(Conv_d, Scale_d, Shift_d)
#         Relu_d= topi.nn.relu(ScaleShift_d)
#         Conv_1 = topi.nn.conv2d(Relu_d, Kernel_1, strides=1, padding=0, dilation=1, layout="NCHW", out_dtype=dtype)
#         ScaleShift_1 = topi.nn.scale_shift_nchw(Conv_1, Scale_1, Shift_1)
#     else: # NHWC
#         Conv_d = topi.nn.depthwise_conv2d_nhwc(Input, Kernel_d, stride=[stride_h_d, stride_w_d], padding=padding_depthwise, dilation=1, out_dtype=dtype)
#         ScaleShift_d = topi.nn.scale_shift_nhwc(Conv_d, Scale_d, Shift_d)
#         Relu_d= topi.nn.relu(ScaleShift_d)
#         Conv_1 = topi.nn.conv2d(Relu_d, Kernel_1, strides=1, padding=0, dilation=1, layout="NHWC", out_dtype=dtype)
#         ScaleShift_1 = topi.nn.scale_shift_nhwc(Conv_1, Scale_1, Shift_1)
    
#     Relu_1 = topi.nn.relu(ScaleShift_1)

#     # Prepare pod type for test data closure
#     dtype = Input.dtype
#     input_shape = get_const_tuple(Input.shape)
#     kernel_d_shape = get_const_tuple(Kernel_d.shape)
#     scale_d_shape = get_const_tuple(Scale_d.shape)
#     shift_d_shape = get_const_tuple(Shift_d.shape)
#     scale_shift_d_shape = get_const_tuple(ScaleShift_d.shape)
#     kernel_1_shape = get_const_tuple(Kernel_1.shape)
#     scale_1_shape = get_const_tuple(Scale_1.shape)
#     shift_1_shape = get_const_tuple(Shift_1.shape)
#     scale_shift_1_shape = get_const_tuple(ScaleShift_1.shape)

#     # @memoize("depthwise_1by1_fused")
#     def get_ref_data():
#         input_np = np.random.uniform(size=input_shape).astype(dtype)

#         kernel_d_np = np.random.uniform(size=kernel_d_shape).astype(dtype)
#         scale_d_np = np.random.uniform(size=scale_d_shape).astype(dtype)
#         shift_d_np = np.random.uniform(size=shift_d_shape).astype(dtype)

#         kernel_1_np = np.random.uniform(size=kernel_1_shape).astype(dtype)
#         scale_1_np = np.random.uniform(size=scale_1_shape).astype(dtype)
#         shift_1_np = np.random.uniform(size=shift_1_shape).astype(dtype)

#         # correctness with scipy
#         if layout == "NCHW":
#             # depthwise
#             conv_d_np = topi.testing.depthwise_conv2d_python_nchw(
#                 input_np, kernel_d_np, stride=stride_depthwise, padding=padding_depthwise)
#             scale_shift_d_np = np.zeros(shape=scale_shift_d_shape)
#             for c in range(out_c_d):
#                 scale_shift_d_np[:,c,:,:] = conv_d_np[:,c,:,:] * scale_d_np[c] + shift_d_np[c]
#                 relu_d_np = np.maximum(scale_shift_d_np, 0)
#             # 1by1
#             conv_1_np = topi.testing.conv2d_nchw_python(relu_d_np, kernel_1_np, 1, 0)
#             scale_shift_1_np = np.zeros(shape=scale_shift_1_shape)
#             for c in range(out_c_1):
#                 scale_shift_1_np[:,c,:,:] = conv_1_np[:,c,:,:] * scale_1_np[c] + shift_1_np[c]
#                 relu_1_np = np.maximum(scale_shift_1_np, 0)
#         else: # NHWC
#             # depthwise
#             conv_d_np = topi.testing.depthwise_conv2d_python_nhwc(
#                 input_np, kernel_d_np, stride=stride_depthwise, padding=padding_depthwise)
#             scale_shift_d_np = np.zeros(shape=scale_shift_d_shape)
#             for c in range(out_c_d):
#                 scale_shift_d_np[:,:,:,c] = conv_d_np[:,:,:,c] * scale_d_np[c] + shift_d_np[c]
#                 relu_d_np = np.maximum(scale_shift_d_np, 0)
#             # 1by1
#             conv_1_np = topi.testing.conv2d_nhwc_python(relu_d_np, kernel_1_np, 1, 0)
#             scale_shift_1_np = np.zeros(shape=scale_shift_1_shape)
#             for c in range(out_c_1):
#                 scale_shift_1_np[:,:,:,c] = conv_1_np[:,:,:,c] * scale_1_np[c] + shift_1_np[c]
#                 relu_1_np = np.maximum(scale_shift_1_np, 0)

#         return (input_np, kernel_d_np, scale_d_np, shift_d_np, 
#                 kernel_1_np, scale_1_np, shift_1_np,
#                 conv_d_np, scale_shift_d_np, relu_d_np,
#                 conv_1_np, scale_shift_1_np, relu_1_np)
#     (input_np, kernel_d_np, scale_d_np, shift_d_np,
#                 kernel_1_np, scale_1_np, shift_1_np,
#                 conv_d_np, scale_shift_d_np, relu_d_np,
#                 conv_1_np, scale_shift_1_np, relu_1_np) = get_ref_data()

#     def check_device(device):
#         ctx = tvm.context(device, 0)
#         if not ctx.exist:
#             print("Skip because %s is not enabled" % device)
#             return
#         print("Running on target: %s" % device)
#         with tvm.target.create(device):
#             # schedule
#             s = schedule_depthwise_1by1_fused([Relu_1], layout)

#         input_tvm = tvm.nd.array(input_np, ctx)
#         kernel_d_tvm = tvm.nd.array(kernel_d_np, ctx)
#         scale_d_tvm = tvm.nd.array(scale_d_np, ctx)
#         shift_d_tvm = tvm.nd.array(shift_d_np, ctx)
#         kernel_1_tvm = tvm.nd.array(kernel_1_np, ctx)
#         scale_1_tvm = tvm.nd.array(scale_1_np, ctx)
#         shift_1_tvm = tvm.nd.array(shift_1_np, ctx)

#         relu_1_tvm = tvm.nd.array(np.zeros(get_const_tuple(Relu_1.shape), dtype=Relu_1.dtype), ctx)
#         with tvm.build_config(auto_unroll_max_step=1400,
#                               unroll_explicit=(device != "cuda")):
#             func = tvm.build(s, [Input,
#                 Kernel_d, Scale_d, Shift_d,
#                 Kernel_1, Scale_1, Shift_1,
#                 Relu_1], target=device, target_host="llvm")
#             func(input_tvm,
#                 kernel_d_tvm, scale_d_tvm, shift_d_tvm,
#                 kernel_1_tvm, scale_1_tvm, shift_1_tvm,
#                 relu_1_tvm)
#             np.testing.assert_allclose(relu_1_tvm.asnumpy(), relu_1_np, rtol=1e-5)

#     check_device("cuda")

def test_depthwise_conv2d():
    # depthwise_1by1_fused(1, 32, 112, 1, 3, 1, 32, layout="NCHW")

    default_schedule = True
    # depthwise_conv2d_with_workload_nhwc(1, 32, 112, 1, 3, 1, default_schedule=default_schedule) # 51.62us,
    # depthwise_conv2d_with_workload_nhwc_auto(1, 32, 112, 1, 3, 1, default_schedule=default_schedule) # 51.05us,
    verify_conv2d_nhwc(1, 32, 112, 32, 1, 1, default_schedule=default_schedule) # 53.023us
    # verify_conv2d_nhwc_auto(1, 32, 112, 32, 1, 1, default_schedule=default_schedule) # 52.68us


    # depthwise_conv2d_with_workload_nhwc(1, 128, 56, 1, 3, 1, default_schedule=default_schedule) # 45.26us,
    # depthwise_conv2d_with_workload_nhwc_auto(1, 128, 56, 1, 3, 1, default_schedule=default_schedule) # 45.08us,
    verify_conv2d_nhwc(1, 128, 56, 128, 1, 1, default_schedule=default_schedule) # 132.06us
    # verify_conv2d_nhwc_auto(1, 128, 56, 128, 1, 1, default_schedule=default_schedule) # 133.17us


    # depthwise_conv2d_with_workload_nhwc(1, 256, 28, 1, 3, 1, default_schedule=default_schedule) # 24.70us,
    # depthwise_conv2d_with_workload_nhwc_auto(1, 256, 28, 1, 3, 1, default_schedule=default_schedule) # 24.63us,
    verify_conv2d_nhwc(1, 256, 28, 256, 1, 1, default_schedule=default_schedule) # 134.21us
    # verify_conv2d_nhwc_auto(1, 256, 28, 256, 1, 1, default_schedule=default_schedule) # 149.73us


    # depthwise_conv2d_with_workload_nhwc(1, 512, 14, 1, 3, 1, default_schedule=default_schedule) # 11.48us,
    # depthwise_conv2d_with_workload_nhwc_auto(1, 512, 14, 1, 3, 1, default_schedule=default_schedule) # 10.94us,
    verify_conv2d_nhwc(1, 512, 14, 512, 1, 1, default_schedule=default_schedule) # 145.21us
    # verify_conv2d_nhwc_auto(1, 512, 14, 512, 1, 1, default_schedule=default_schedule) # 145.21us

if __name__ == "__main__":
    test_depthwise_conv2d()
