import tvm
import topi
import topi.testing
import numpy as np
from scipy import signal
from topi.util import get_const_tuple
from tvm.contrib.pickle_memoize import memoize
from fused_schedule import conv2d_nchw_1by1, schedule_depthwise_1by1_fused

def depthwise(batch, in_channel, in_size, channel_multiplier, kernel, stride, padding):
    in_width = in_height = in_size
    filter_channel = in_channel
    filter_width = filter_height = kernel
    # placeholder
    Input = tvm.placeholder((batch, in_channel, in_height, in_width), name='Input')
    Filter = tvm.placeholder((filter_channel, channel_multiplier, filter_height, filter_width), name='Filter')
    Scale = tvm.placeholder((in_channel * channel_multiplier,), name='Scale')
    Shift = tvm.placeholder((in_channel * channel_multiplier,), name='Shift')

    # declare
    DepthwiseConv2d = topi.nn.depthwise_conv2d_nchw(Input, Filter, stride=stride, padding=padding)
    ScaleShift = topi.nn.scale_shift_nchw(DepthwiseConv2d, Scale, Shift)
    Relu = topi.nn.relu(ScaleShift)

    # Prepare pod type for test data closure
    dtype = Input.dtype
    input_shape = get_const_tuple(Input.shape)
    filter_shape = get_const_tuple(Filter.shape)
    scale_shape = get_const_tuple(Scale.shape)
    shift_shape = get_const_tuple(Shift.shape)
    scale_shift_shape = get_const_tuple(ScaleShift.shape)

    def check_device(device):
        ctx = tvm.context(device, 0)
        if not ctx.exist:
            print("Skip because %s is not enabled" % device)
            return
        print("Running on target: %s" % device)
        with tvm.target.create(device):
            # schedule
            s = topi.generic.schedule_depthwise_conv2d_nchw(Relu)
        # build the kernels
        f = tvm.build(s, [Input, Filter, Scale, Shift, Relu], device, name="depthwise_bn_relu_%d_%d_%d_%d_%d_%d_%s" % (batch, in_channel, in_size, channel_multiplier, kernel, stride, padding))

        # Use memoize, pickle the test data for next time use.
        @memoize("depthwise")
        def get_ref_data():
            input_np = np.random.uniform(size=input_shape).astype(dtype)
            filter_np = np.random.uniform(size=filter_shape).astype(dtype)
            scale_np = np.random.uniform(size=scale_shape).astype(dtype)
            shift_np = np.random.uniform(size=shift_shape).astype(dtype)
            # correctness with scipy
            depthwise_conv2d_scipy = topi.testing.depthwise_conv2d_python_nchw(
                input_np, filter_np, stride=stride, padding=padding)
            scale_shift_scipy = np.zeros(shape=scale_shift_shape)
            for c in range(in_channel * channel_multiplier):
                scale_shift_scipy[:,c,:,:] = depthwise_conv2d_scipy[:,c,:,:] * scale_np[c] + shift_np[c]
                relu_scipy = np.maximum(scale_shift_scipy, 0)
            return (input_np, filter_np, scale_np, shift_np,
                    depthwise_conv2d_scipy, scale_shift_scipy, relu_scipy)
        # Get the test data
        (input_np, filter_np, scale_np, shift_np,
         depthwise_conv2d_scipy, scale_shift_scipy, relu_scipy) = get_ref_data()

        input_tvm = tvm.nd.array(input_np, ctx)
        filter_tvm = tvm.nd.array(filter_np, ctx)
        scale_tvm = tvm.nd.array(scale_np, ctx)
        shift_tvm = tvm.nd.array(shift_np, ctx)
        depthwise_conv2d_tvm = tvm.nd.array(np.zeros(shape=get_const_tuple(DepthwiseConv2d.shape), dtype=DepthwiseConv2d.dtype), ctx)
        scale_shift_tvm = tvm.nd.array(np.zeros(shape=get_const_tuple(ScaleShift.shape), dtype=ScaleShift.dtype), ctx)
        relu_tvm = tvm.nd.array(np.zeros(shape=get_const_tuple(Relu.shape), dtype=Relu.dtype), ctx)
        # launch kernel 3 (depthwise_conv2d + scale_shift + relu)
        timer = f.time_evaluator(f.entry_name, ctx, number=1)
        tcost = timer(input_tvm, filter_tvm, scale_tvm, shift_tvm, relu_tvm).mean
        np.testing.assert_allclose(relu_tvm.asnumpy(), relu_scipy, rtol=1e-5)

    check_device("cuda")

def verify_conv2d_nchw(batch, in_channel, in_size, num_filter, kernel, stride, padding):
    in_height = in_width = in_size

    A = tvm.placeholder((batch, in_channel, in_height, in_width), name='A')
    W = tvm.placeholder((num_filter, in_channel, kernel, kernel), name='W')
    Scale = tvm.placeholder((num_filter,), name='Scale')
    Shift = tvm.placeholder((num_filter,), name='Shift')

    B = topi.nn.conv2d(A, W, stride, padding, layout='NCHW')
    C = topi.nn.scale_shift_nchw(B, Scale, Shift)
    D = topi.nn.relu(C)

    dtype = A.dtype
    a_shape = get_const_tuple(A.shape)
    w_shape = get_const_tuple(W.shape)
    scale_shape = get_const_tuple(Scale.shape)
    shift_shape = get_const_tuple(Shift.shape)
    c_shape = get_const_tuple(C.shape)

    def check_device(device):
        @memoize("conv_1by1")
        def get_ref_data():
            a_np = np.random.uniform(size=a_shape).astype(dtype)
            w_np = np.random.uniform(size=w_shape).astype(dtype)
            scale_np = np.random.uniform(size=scale_shape).astype(dtype)
            shift_np = np.random.uniform(size=shift_shape).astype(dtype)
            b_np = topi.testing.conv2d_nchw_python(a_np, w_np, stride, padding)
            c_np = np.zeros(shape=c_shape)
            for c in range(num_filter):
                c_np[:,c,:,:] = b_np[:,c,:,:] * scale_np[c] + shift_np[c]
                d_np = np.maximum(c_np, 0)
            return (a_np, w_np, b_np, scale_np, shift_np, c_np, d_np)

        (a_np, w_np, b_np, scale_np, shift_np, c_np, d_np) = get_ref_data()

        ctx = tvm.context(device.split(' ')[0], 0)
        if not ctx.exist:
            print("Skip because %s is not enabled" % device)
            return
        print("Running on target: %s" % device.split(' ')[0])
        with tvm.target.create(device):
            s = topi.generic.schedule_conv2d_nchw([D])
        a = tvm.nd.array(a_np, ctx)
        w = tvm.nd.array(w_np, ctx)
        scale = tvm.nd.array(scale_np, ctx)
        shift = tvm.nd.array(shift_np, ctx)
        d = tvm.nd.array(np.zeros(get_const_tuple(D.shape), dtype=D.dtype), ctx)
        with tvm.build_config(auto_unroll_max_step=1400,
                              unroll_explicit=(device.split(' ')[0] != "cuda")):
            func = tvm.build(s, [A, W, Scale, Shift, D], target=device.split(' ')[0], name="conv_bn_relu_%d_%d_%d_%d_%d_%d_%d" % (batch, in_channel, in_size, num_filter, kernel, stride, padding))
            func(a, w, scale, shift, d)
            np.testing.assert_allclose(d.asnumpy(), d_np, rtol=1e-5)

    for device in ['cuda']:
        check_device(device)

def depthwise_1by1_fused(batch, in_channel_depthwise, in_size, channel_multiplier, kernel_depthwise, stride_depthwise, padding_depthwise, num_filter):

    in_h_d = in_w_d = in_size
    in_c_d = kernel_c_d = in_channel_depthwise
    kernel_h_d = kernel_w_d = kernel_depthwise
    out_c_d = in_c_1 = in_channel_depthwise * channel_multiplier
    stride_h_d = stride_w_d = stride_depthwise
    padding_h_1 = padding_w_1 = padding_depthwise

    kernel_h_1 = kernel_w_1 = 1
    stride_h_1 = stride_w_1 = 1
    padding_h_1 = padding_w_1 = 0
    out_c_1 = num_filter

    # placeholder
    Input = tvm.placeholder((batch, in_c_d, in_h_d, in_w_d), name='Input')
    Kernel_d = tvm.placeholder((kernel_c_d, channel_multiplier, kernel_h_d, kernel_w_d), name='Kernel_d')
    Scale_d = tvm.placeholder((out_c_d,), name='Scale_d')
    Shift_d = tvm.placeholder((out_c_d,), name='Shift_d')
    Kernel_1 = tvm.placeholder((out_c_1, in_c_1, kernel_h_1, kernel_w_1), name='Kernel_1')
    Scale_1 = tvm.placeholder((out_c_1,), name='Scale_1')
    Shift_1 = tvm.placeholder((out_c_1,), name='Shift_1')

    # declare
    Conv_d = topi.nn.depthwise_conv2d_nchw(Input, Kernel_d, stride=stride_depthwise, padding=padding_depthwise)
    ScaleShift_d = topi.nn.scale_shift_nchw(Conv_d, Scale_d, Shift_d)
    Relu_d= topi.nn.relu(ScaleShift_d)
    Conv_1 = topi.nn.conv2d(Relu_d, Kernel_1, stride=1, padding=0)
    ScaleShift_1 = topi.nn.scale_shift_nchw(Conv_1, Scale_1, Shift_1)
    Relu_1 = topi.nn.relu(ScaleShift_1)

    # Prepare pod type for test data closure
    dtype = Input.dtype
    input_shape = get_const_tuple(Input.shape)
    kernel_d_shape = get_const_tuple(Kernel_d.shape)
    scale_d_shape = get_const_tuple(Scale_d.shape)
    shift_d_shape = get_const_tuple(Shift_d.shape)
    scale_shift_d_shape = get_const_tuple(ScaleShift_d.shape)
    kernel_1_shape = get_const_tuple(Kernel_1.shape)
    scale_1_shape = get_const_tuple(Scale_1.shape)
    shift_1_shape = get_const_tuple(Shift_1.shape)
    scale_shift_1_shape = get_const_tuple(ScaleShift_1.shape)

    @memoize("depthwise_1by1_fused")
    def get_ref_data():
        input_np = np.random.uniform(size=input_shape).astype(dtype)

        kernel_d_np = np.random.uniform(size=kernel_d_shape).astype(dtype)
        scale_d_np = np.random.uniform(size=scale_d_shape).astype(dtype)
        shift_d_np = np.random.uniform(size=shift_d_shape).astype(dtype)

        kernel_1_np = np.random.uniform(size=kernel_1_shape).astype(dtype)
        scale_1_np = np.random.uniform(size=scale_1_shape).astype(dtype)
        shift_1_np = np.random.uniform(size=shift_1_shape).astype(dtype)

        # correctness with scipy
        # depthwise
        conv_d_np = topi.testing.depthwise_conv2d_python_nchw(
            input_np, kernel_d_np, stride=stride_depthwise, padding=padding_depthwise)
        scale_shift_d_np = np.zeros(shape=scale_shift_d_shape)
        for c in range(out_c_d):
            scale_shift_d_np[:,c,:,:] = conv_d_np[:,c,:,:] * scale_d_np[c] + shift_d_np[c]
            relu_d_np = np.maximum(scale_shift_d_np, 0)
        # 1by1
        conv_1_np = topi.testing.conv2d_nchw_python(relu_d_np, kernel_1_np, 1, 0)
        scale_shift_1_np = np.zeros(shape=scale_shift_1_shape)
        for c in range(out_c_1):
            scale_shift_1_np[:,c,:,:] = conv_1_np[:,c,:,:] * scale_1_np[c] + shift_1_np[c]
            relu_1_np = np.maximum(scale_shift_1_np, 0)

        return (input_np, kernel_d_np, scale_d_np, shift_d_np,
                conv_d_np, scale_shift_d_np, relu_d_np,
                conv_1_np, scale_shift_1_np, relu_1_np)
    (input_np, kernel_d_np, scale_d_np, shift_d_np,
                conv_d_np, scale_shift_d_np, relu_d_np,
                conv_1_np, scale_shift_1_np, relu_1_np) = get_ref_data()

def test_depthwise_conv2d():
    # depthwise(1, 32, 112, 1, 3, 1, "SAME") # 133.0us
    verify_conv2d_nchw(1, 32, 112, 64, 1, 1, 0) # 217.8us
    # depthwise_1by1_fused(1, 32, 112, 1, 3, 1, "SAME", 64)

    # depthwise(1, 128, 56, 1, 3, 1, "SAME") # 166.1us
    # verify_conv2d_nchw(1, 128, 56, 128, 1, 1, 0) # 163.3us

    # depthwise(1, 256, 28, 1, 3, 1, "SAME") # 75.8us
    # verify_conv2d_nchw(1, 256, 28, 256, 1, 1, 0) # 469.21us

    # depthwise(1, 512, 14, 1, 3, 1, "SAME") # 27.8us
    # verify_conv2d_nchw(1, 512, 14, 512, 1, 1, 0) # 254.01us

if __name__ == "__main__":
    test_depthwise_conv2d()
