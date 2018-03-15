import tvm
import topi
import topi.testing
import numpy as np
import os
from scipy import signal
from topi.util import get_const_tuple
from tvm.contrib.pickle_memoize import memoize
from fused_schedule import schedule_conv2d_nhwc, schedule_depthwise_conv2d_nhwc_reuse
from tvm.contrib import nvcc

TASK = "hhhh"
USE_MANUAL_CODE = False

@tvm.register_func
def tvm_callback_cuda_compile(code):
    ptx = nvcc.compile_cuda(code, target="ptx")
    return ptx

def write_code(code, fname):
    with open(fname, "w") as f:
        f.write(code)

@tvm.register_func
def tvm_callback_cuda_postproc(code):
    if not os.path.exists("perf"):
        os.mkdir("perf")
    write_code(code, "perf/%s_generated.cu" % TASK)
    if USE_MANUAL_CODE:
        code = open("perf/%s_manual.cu" % TASK).read()
    return code

def depthwise_conv2d_with_workload_nhwc(batch, in_channel, in_height, channel_multiplier, filter_height, stride_h, padding):
    in_width = in_height
    filter_channel = in_channel
    filter_width = filter_height
    stride_w = stride_h
    # placeholder
    Input = tvm.placeholder((batch, in_height, in_width, in_channel), name='Input')
    Filter = tvm.placeholder((filter_height, filter_width, filter_channel, channel_multiplier), name='Filter')
    Scale = tvm.placeholder((in_channel * channel_multiplier,), name='Scale')
    Shift = tvm.placeholder((in_channel * channel_multiplier,), name='Shift')
    # declare
    DepthwiseConv2d = topi.nn.depthwise_conv2d_nhwc(Input, Filter, stride=[stride_h, stride_w], padding=padding)
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
            s1 = schedule_depthwise_conv2d_nhwc_reuse(Input, [DepthwiseConv2d])
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
        @memoize("topi.tests.test_topi_depthwise_conv2d.nhwc")
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
        timer_1 = f1.time_evaluator(f1.entry_name, ctx, number=1)
        tcost_1 = timer_1(input_tvm, filter_tvm, depthwise_conv2d_tvm).mean
        # launch kernel 2 (depthwise_conv2d + scale_shift)
        # timer_2 = f2.time_evaluator(f2.entry_name, ctx, number=1)
        # tcost_2 = timer_2(input_tvm, filter_tvm, scale_tvm, shift_tvm, scale_shift_tvm).mean
        # launch kernel 3 (depthwise_conv2d + scale_shift + relu)
        # timer_3 = f3.time_evaluator(f3.entry_name, ctx, number=1)
        # tcost_3 = timer_3(input_tvm, filter_tvm, scale_tvm, shift_tvm, relu_tvm).mean
        # relu_scipy = np.maximum(scale_shift_scipy, 0)
        np.testing.assert_allclose(depthwise_conv2d_tvm.asnumpy(), depthwise_conv2d_scipy, rtol=1e-5)
        # np.testing.assert_allclose(scale_shift_tvm.asnumpy(), scale_shift_scipy, rtol=1e-5)
        # np.testing.assert_allclose(relu_tvm.asnumpy(), relu_scipy, rtol=1e-5)

    check_device("cuda")

def verify_conv2d_nhwc(batch, in_channel, in_size, num_filter, kernel, stride, padding):
    in_height = in_width = in_size

    A = tvm.placeholder((batch, in_height, in_width, in_channel), name='A')
    W = tvm.placeholder((kernel, kernel, in_channel, num_filter), name='W')
    B = topi.nn.conv2d_nhwc(A, W, stride, padding)

    a_shape = get_const_tuple(A.shape)
    w_shape = get_const_tuple(W.shape)
    print(a_shape)
    print(w_shape)
    dtype = A.dtype

    @memoize("verify_nhwc")
    def get_ref_data():
        a_np = np.random.uniform(size=a_shape).astype(dtype)
        w_np = np.random.uniform(size=w_shape).astype(dtype)
        b_np = topi.testing.conv2d_nhwc_python(a_np, w_np, stride, padding)
        return a_np, w_np, b_np
    a_np, w_np, b_np = get_ref_data()

    def check_device(device):
        if not tvm.module.enabled(device):
            print("Skip because %s is not enabled" % device)
            return
        print("Running on target: %s" % device)
        with tvm.target.create(device):
            s = schedule_conv2d_nhwc(A, [B])
        ctx = tvm.context(device, 0)
        a = tvm.nd.array(a_np, ctx)
        w = tvm.nd.array(w_np, ctx)
        b = tvm.nd.array(np.zeros(get_const_tuple(B.shape), dtype=B.dtype), ctx)
        func = tvm.build(s, [A, W, B], device, name=("ddd%dddd"%a.shape[2]))
        func(a, w, b)
        np.testing.assert_allclose(b.asnumpy(), b_np, rtol=1e-5)

    for device in ['cuda']:
        check_device(device)

# def depthwise_1by1_nhwc_fused(batch, in_channel_depthwise, in_size, channel_multiplier, kernel_depthwise, stride_depthwise, padding_depthwise, num_filter):

#     in_h_d = in_w_d = in_size
#     in_c_d = kernel_c_d = in_channel_depthwise
#     kernel_h_d = kernel_w_d = kernel_depthwise
#     out_c_d = in_c_1 = in_channel_depthwise * channel_multiplier
#     stride_h_d = stride_w_d = stride_depthwise
#     padding_h_1 = padding_w_1 = padding_depthwise

#     kernel_h_1 = kernel_w_1 = 1
#     stride_h_1 = stride_w_1 = 1
#     padding_h_1 = padding_w_1 = 0
#     out_c_1 = num_filter

#     # placeholder
#     Input = tvm.placeholder((batch, in_c_d, in_h_d, in_w_d), name='Input')
#     Kernel_d = tvm.placeholder((kernel_c_d, channel_multiplier, kernel_h_d, kernel_w_d), name='Kernel_d')
#     Scale_d = tvm.placeholder((out_c_d,), name='Scale_d')
#     Shift_d = tvm.placeholder((out_c_d,), name='Shift_d')
#     Kernel_1 = tvm.placeholder((out_c_1, in_c_1, kernel_h_1, kernel_w_1), name='Kernel_1')
#     Scale_1 = tvm.placeholder((out_c_1,), name='Scale_1')
#     Shift_1 = tvm.placeholder((out_c_1,), name='Shift_1')

#     # declare
#     Conv_d = topi.nn.depthwise_conv2d_nchw(Input, Kernel_d, stride=stride_depthwise, padding=padding_depthwise)
#     ScaleShift_d = topi.nn.scale_shift_nchw(Conv_d, Scale_d, Shift_d)
#     Relu_d= topi.nn.relu(ScaleShift_d)
#     Conv_1 = topi.nn.conv2d(Relu_d, Kernel_1, stride=1, padding=0, layout="NCHW")
#     ScaleShift_1 = topi.nn.scale_shift_nchw(Conv_1, Scale_1, Shift_1)
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

#     @memoize("depthwise_1by1_fused")
#     def get_ref_data():
#         input_np = np.random.uniform(size=input_shape).astype(dtype)

#         kernel_d_np = np.random.uniform(size=kernel_d_shape).astype(dtype)
#         scale_d_np = np.random.uniform(size=scale_d_shape).astype(dtype)
#         shift_d_np = np.random.uniform(size=shift_d_shape).astype(dtype)

#         kernel_1_np = np.random.uniform(size=kernel_1_shape).astype(dtype)
#         scale_1_np = np.random.uniform(size=scale_1_shape).astype(dtype)
#         shift_1_np = np.random.uniform(size=shift_1_shape).astype(dtype)

#         # correctness with scipy
#         # depthwise
#         conv_d_np = topi.testing.depthwise_conv2d_python_nchw(
#             input_np, kernel_d_np, stride=stride_depthwise, padding=padding_depthwise)
#         scale_shift_d_np = np.zeros(shape=scale_shift_d_shape)
#         for c in range(out_c_d):
#             scale_shift_d_np[:,c,:,:] = conv_d_np[:,c,:,:] * scale_d_np[c] + shift_d_np[c]
#             relu_d_np = np.maximum(scale_shift_d_np, 0)
#         # 1by1
#         conv_1_np = topi.testing.conv2d_nchw_python(relu_d_np, kernel_1_np, 1, 0)
#         scale_shift_1_np = np.zeros(shape=scale_shift_1_shape)
#         for c in range(out_c_1):
#             scale_shift_1_np[:,c,:,:] = conv_1_np[:,c,:,:] * scale_1_np[c] + shift_1_np[c]
#             relu_1_np = np.maximum(scale_shift_1_np, 0)

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
#             s = schedule_depthwise_1by1_fused([Relu_1])
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
    depthwise_conv2d_with_workload_nhwc(1, 32, 112, 1, 3, 1, "SAME") # 111.77us, 168.69us
    # verify_conv2d_nhwc(1, 32, 112, 32, 1, 1, "SAME") # 53.023us
    # depthwise_1by1_fused(1, 32, 112, 1, 3, 1, "SAME", 64)

    depthwise_conv2d_with_workload_nhwc(1, 128, 56, 1, 3, 1, "SAME") # 116.77us, 78,03us
    # verify_conv2d_nhwc(1, 128, 56, 128, 1, 1, "SAME") # 132.06us

    depthwise_conv2d_with_workload_nhwc(1, 256, 28, 1, 3, 1, "SAME") # 67.71us, 57.81us
    # verify_conv2d_nhwc(1, 256, 28, 256, 1, 1, "SAME") # 134.21us

    depthwise_conv2d_with_workload_nhwc(1, 512, 14, 1, 3, 1, "SAME") # 24.83us, 30.21us
    # verify_conv2d_nhwc(1, 512, 14, 512, 1, 1, "SAME") # 145.21us

if __name__ == "__main__":
    test_depthwise_conv2d()
