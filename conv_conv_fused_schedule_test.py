import tvm
import topi
import topi.testing
import numpy as np
import os, logging, sys
from scipy import signal
from topi.util import get_const_tuple
from tvm.contrib.pickle_memoize import memoize
from general_fused_compute import *
from tvm.contrib import nvcc
from tvm import autotvm
import topi.tag as tag
np.random.seed(42)

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

# @register_fused.register(["cuda", "gpu"])
def schedule_general_fused_nhwc(outs, nodes, params, NHWC_transpose=False):
    outs = [outs] if isinstance(outs, tvm.tensor.Tensor) else outs
    s = tvm.create_schedule([x.op for x in outs])

    ######################
    PaddedInput = nodes[1]
    Intermediate = nodes[2]
    Out = nodes[3]
    F_1 = params[1]
    F_2 = params[2]

    s[PaddedInput].compute_inline()

    block_x = tvm.thread_axis("blockIdx.x")
    thread_x = tvm.thread_axis("threadIdx.x")

    n, h, w, c = Out.op.axis
    fused = s[Out].fuse(h, w)
    s[Out].bind(fused, block_x)
    s[Out].bind(c, thread_x)

    s[Intermediate].compute_at(s[Out], fused)

    return s

def verify_conv_conv_fused(parameters, dtype="float32", layout="NHWC", NHWC_transpose=False, print_code=True, save_data=False, export_code=False):
    assert layout in ["NHWC", "NCHW"]

    p = parameters
    input_shape = (p[0], p[1], p[1], p[2])
    filter_1_shape = (p[3], p[3], p[4], p[2]) if NHWC_transpose else (p[3], p[3], p[2], p[4])
    filter_2_shape = (p[6], p[6], p[7], p[4]) if NHWC_transpose else (p[6], p[6], p[4], p[7])

    # placeholder (NHWC)
    # Input: NHWC, Kernel: HWIO for both depthwise and conv2d
    Input = tvm.placeholder(input_shape, name='Input')
    Conv2dFilter_1 = tvm.placeholder(filter_1_shape, name='Conv2dFilter_1')
    Conv2dFilter_2 = tvm.placeholder(filter_2_shape, name='Conv2dFilter_2')
    dtype = Input.dtype

    # For getting ref data
    placeholders = []
    placeholders.append(Input)
    placeholders.append(Conv2dFilter_1)
    placeholders.append(Conv2dFilter_2)

    # For getting schedule
    Filters = []
    Filters.append(FilterConstructor(
                    Conv2dFilter_1,
                    depthwise=p[5], kernel=p[3], stride=1, dilation=1, NHWC_transpose=NHWC_transpose))
    Filters.append(FilterConstructor(
                    Conv2dFilter_2,
                    depthwise=p[8], kernel=p[6], stride=1, dilation=1, NHWC_transpose=NHWC_transpose))

    # Get the graph
    # nodes: all nodes in the graph
    # params: inputs & outputs of the graph
    nodes, params = fused_convs(Input, Filters)

    # @memoize("verify_nhwc")
    def get_ref_data():
        # Pretending the input_data is some output_data from stage -1
        output_data = np.random.uniform(size=get_const_tuple(Input.shape)).astype(dtype)
        ref_data = [output_data] 
        
        for idx, f in enumerate(Filters):
            p = f.placeholder
            filter_data = np.random.uniform(size=get_const_tuple(p.shape)).astype(dtype)
            ref_data.append(filter_data)

            if "Depthwise" in p.name:
                output_data = topi.testing.depthwise_conv2d_python_nhwc(output_data, filter_data, stride=[f.stride, f.stride], padding=f.padding, filter_transpose=f.NHWC_transpose)
            else: # Normal convolution
                output_data = topi.testing.conv2d_nhwc_python(output_data, filter_data, f.stride, f.padding, filter_transpose=f.NHWC_transpose)
            if idx == len(Filters) - 1: # At the last stage, append output_data as the final output for reference
                ref_data.append(output_data)

        return ref_data
    ref_data = get_ref_data()

    if save_data:
        # Save ref data
        for i in range(0, len(ref_data)):
            filename = "conv_conv_"
            if i == 0:
                filename += "input_"
            elif i == 1:
                filename += "filter_1_"
            elif i == 2:
                filename += "filter_2_"
            else:
                filename += "output_"
            np.save(filename + "%d_%d_%d_%d" % ref_data[i].shape, ref_data[i])

    # tmp = np.load("output_1_112_112_32.npy")
    # print(tmp[0,0,0,0:100])

    def check_device(device):
        if not tvm.module.enabled(device):
            print("Skip because %s is not enabled" % device)
            return
        print("Running on target: %s" % device)

        ctx = tvm.context(device, 0)

        nd_arrays = []
        for idx, array in enumerate(ref_data):
            if idx == len(ref_data) - 1: # Omit output data here
                break
            nd_arrays.append(tvm.nd.array(array, ctx))
        nd_arrays.append(tvm.nd.array(np.zeros(get_const_tuple(nodes[-1].shape), dtype=nodes[-1].dtype), ctx)) # Append 0 output data

        with tvm.target.create(device):
            s = schedule_general_fused_nhwc([nodes[-1]], nodes, params, NHWC_transpose)
        if print_code:
            print(tvm.lower(s, params, simple_mode=True))

        # with tvm.build_config(  auto_unroll_max_step=16,
        #                         double_buffer_split_loop=8):
        func = tvm.build(s, params, device, name=("ConvConvFused_{}".format(len(Filters))))
        # if print_code:
        #     print(func.imported_modules[0].get_source())
        if export_code:
            cuda_code = func.imported_modules[0].get_source()
            # write_code(cuda_code, "kernel_conv_conv_%s_%s_%s_%s.cu" % nd_arrays[0].asnumpy().shape)
            write_code(cuda_code, "kernel_conv_conv.cu")
        # func(a, w, b)
        timer_1 = func.time_evaluator(func.entry_name, ctx, number=10)
        tcost_1 = timer_1(*nd_arrays).mean
        # np.testing.assert_allclose(nd_arrays[-1].asnumpy(), ref_data[-1], rtol=1e-3)
        d = ~np.isclose(nd_arrays[-1].asnumpy(), ref_data[-1], rtol=1e-3)
        if (np.sum(d) > 0):
            print(nd_arrays[-1].asnumpy()[d])
            print(ref_data[-1][d])
            print(np.where(d))
        # print("Error rate: {:.2f}%".format((len(d) / len(ref_data[-1]) * 100)))
        print("General Fused of {} layers ({}): average running time is {:.2f} us.".format(len(Filters), layout, tcost_1 * 1e6))

    for device in ['cuda']:
        check_device(device)

if __name__ == "__main__":
    parameters = []

    parameters.append([1, 224, 3, 3, 64, False, 3, 64, False]) # 
    # parameters.append([1, 112, 64, 3, 128, False, 3, 128, False]) # 
    # parameters.append([1, 56, 128, 3, 256, False, 3, 256, False]) # 
    # parameters.append([1, 28, 256, 3, 512, False, 3, 512, False]) #

    for p in parameters:
        verify_conv_conv_fused(p, NHWC_transpose=False, print_code=True, save_data=False, export_code=True)
    
