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
def schedule_general_fused_nhwc(outs, nodes, params):
    outs = [outs] if isinstance(outs, tvm.tensor.Tensor) else outs
    s = tvm.create_schedule([x.op for x in outs])

    ############# Schedule begins ##############
    s[nodes[1]].compute_inline()
        
    return s

def verify_general_fused(padding_depthwise="SAME", dtype="float32", layout="NHWC"):
    assert layout in ["NHWC", "NCHW"]

    # placeholder (NHWC)
    # Input: NHWC, Kernel: HWIO for both depthwise and conv2d
    Input = tvm.placeholder((1, 112, 112, 32), name='Input')
    DepthwiseFilter_1 = tvm.placeholder((3, 3, 32, 1), name='DepthwiseFilter_1')
    Conv2dFilter_1 = tvm.placeholder((1, 1, 32, 32), name='Conv2dFilter_1')
    dtype = Input.dtype

    # For getting ref data
    placeholders = []
    placeholders.append(Input)
    placeholders.append(DepthwiseFilter_1)
    placeholders.append(Conv2dFilter_1)

    # For getting schedule
    Filters = []
    Filters.append(FilterConstructor(
                    DepthwiseFilter_1,
                    depthwise=True, kernel=3, stride=1, dilation=1))
    Filters.append(FilterConstructor(
                    Conv2dFilter_1,
                    depthwise=False, kernel=1, stride=1, dilation=1))

    # Get the graph
    nodes, params = fused_two_conv(Input, Filters)

    # @memoize("verify_nhwc")
    def get_ref_data():
        ref_data = [np.random.uniform(size=get_const_tuple(Input.shape)).astype(dtype)] # Input

        for idx, f in enumerate(Filters):
            p = f.placeholder
            filter_data = np.random.uniform(size=get_const_tuple(p.shape)).astype(dtype)
            ref_data.append(filter_data)

            if "Depthwise" in p.name:
                output_data = topi.testing.depthwise_conv2d_python_nhwc(ref_data[-2], filter_data, stride=[f.stride, f.stride], padding=f.padding)
            else: # Normal convolution
                output_data = topi.testing.conv2d_nhwc_python(ref_data[-2], filter_data, f.stride, f.padding)
            ref_data.append(output_data)

        return ref_data
    ref_data = get_ref_data()

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
            s = schedule_general_fused_nhwc([nodes[-1]], nodes, params)
        print(tvm.lower(s, params, simple_mode=True))
                
        func = tvm.build(s, params, device, name=("GeneralFused_{}".format(len(Filters))))
        # func(a, w, b)
        timer_1 = func.time_evaluator(func.entry_name, ctx, number=10)
        tcost_1 = timer_1(nd_arrays).mean
        np.testing.assert_allclose(nd_arrays[-1].asnumpy(), ref_data[-1], rtol=1e-3)
        print("General Fused of {} layers ({}): average running time is {:.2f} us.".format(len(Filters), layout, tcost_1 * 1e6))

    for device in ['cuda']:
        check_device(device)



if __name__ == "__main__":
    verify_general_fused()
    
