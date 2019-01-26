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
def schedule_general_fused_nhwc(outs, nodes, params):
    outs = [outs] if isinstance(outs, tvm.tensor.Tensor) else outs
    s = tvm.create_schedule([x.op for x in outs])

    ############# Schedule begins ##############
    # nodes: input, padded, after_depthwise, after_1by1 (output)
    # params: input, depthwise_filter, 1by1_filter, output

    # s[nodes[1]].compute_inline()

    # block_x = tvm.thread_axis("blockIdx.x")
    # thread_x = tvm.thread_axis("threadIdx.x")

    # n, h, w, c = nodes[3].op.axis
    # fused = s[nodes[3]].fuse(h, w)
    # s[nodes[3]].bind(fused, block_x)
    # s[nodes[3]].bind(c, thread_x)

    # s[nodes[2]].compute_at(s[nodes[3]], fused)

    ############################################
    # PaddedInput = nodes[1]
    # AfterDepthwise = nodes[2]
    # After1by1 = nodes[3]
    # F_d = params[1]
    # F_1 = params[2]

    # s[PaddedInput].compute_inline()

    # FS_d = s.cache_read(F_d, "shared", [AfterDepthwise])

    # Output = outs[0].op.output(0)
    # s[AfterDepthwise].set_scope("local")

    # block_x = tvm.thread_axis("blockIdx.x")
    # thread_x = tvm.thread_axis("threadIdx.x")

    # n, h, w, c = s[Output].op.axis
    # # num_thread here could be 728, it is larger than cuda.max_num_threads
    # num_thread = tvm.ir_pass.Simplify(PaddedInput.shape[3]).value
    # target = tvm.target.current_target()
    # if target and (target.target_name not in ["cuda", "nvptx"]):
    #     num_thread = target.max_num_threads
    # xoc, xic = s[Output].split(c, factor=num_thread)
    # s[Output].reorder(xoc, n, h, w, xic)
    # xo, yo, _, _ = s[Output].tile(h, w, x_factor=2, y_factor=2)
    # fused = s[Output].fuse(yo, xo)
    # fused = s[Output].fuse(fused, n)
    # fused = s[Output].fuse(fused, xoc)

    # s[Output].bind(fused, block_x)
    # s[Output].bind(xic, thread_x)

    # if AfterDepthwise.op in s.outputs:
    #     s[CL].compute_at(s[Output], xic)
    # else:
    #     s[AfterDepthwise].compute_at(s[Output], xic)

    # _, _, ci, fi = s[FS_d].op.axis
    # s[FS_d].compute_at(s[Output], fused)
    # fused = s[FS_d].fuse(fi, ci)
    # s[FS_d].bind(fused, thread_x)

    # ########################################### 72 us for 112
    # PaddedInput = nodes[1]
    # Intermediate = nodes[2]
    # Out = nodes[3]
    # F_d = params[1]
    # F_1 = params[2]

    # s[PaddedInput].compute_inline()

    # num_channel = tvm.ir_pass.Simplify(PaddedInput.shape[3]).value
    # num_thread = num_channel if num_channel <= 128 else int(num_channel / 4)

    # # FS_d = s.cache_read(F_d, "shared", [Intermediate])

    # OL = s.cache_write(Out, "local")
    # s[Intermediate].set_scope("shared")
    # IL = s.cache_write(Intermediate, "local")

    # block_x = tvm.thread_axis("blockIdx.x")
    # thread_x = tvm.thread_axis("threadIdx.x")

    # # Final output
    # cc = s[OL].op.reduce_axis[0]
    # xocc, xicc = s[OL].split(cc, factor=4)
    # CTR = s.rfactor(OL, xicc) # Cross thread reduction
    
    # n, h, w, c = s[Out].op.axis
    # xoc, xic = s[Out].split(c, factor=num_thread)
    # s[Out].reorder(xoc, n, h, w, xic)
    # yo, xo, yi, xi = s[Out].tile(h, w, x_factor=4, y_factor=4)
    # fused = s[Out].fuse(yo, xo)
    # fused = s[Out].fuse(fused, n)
    # fused = s[Out].fuse(fused, xoc)

    # s[Out].bind(fused, block_x)
    # s[Out].bind(xic, thread_x)
    # s[OL].compute_at(s[Out], xic)
    # s[CTR].compute_at(s[Out], xic)
    # s[CTR].vectorize(s[CTR].op.axis[0])

    # # Intermediate output
    # n, h, w, c = s[Intermediate].op.axis
    # xoc, xic = s[Intermediate].split(c, factor=num_thread)
    # # s[Intermediate].reorder(xoc, n, h, w, xic)
    # # yo, xo, yi, xi = s[Intermediate].tile(h, w, x_factor=2, y_factor=2)
    # # fused = s[Intermediate].fuse(yo, xo)
    # # fused = s[Intermediate].fuse(fused, n)
    # # fused = s[Intermediate].fuse(fused, xoc)

    # # s[Intermediate].bind(fused, block_x)
    # s[Intermediate].bind(xic, thread_x)
    # s[IL].compute_at(s[Intermediate], xic)
    # s[Intermediate].compute_at(s[Out], xi)

    # # # Unroll, small improvements without sharing FS_d and local IL
    # s[IL].unroll(s[IL].op.reduce_axis[0])
    # s[IL].unroll(s[IL].op.reduce_axis[1])

    # # # Shared depthwise filter
    # # s[FS_d].compute_at(s[Intermediate], w)
    # # _, _, ci, fi = s[FS_d].op.axis
    # # fused = s[FS_d].fuse(fi, ci)
    # # s[FS_d].bind(fused, thread_x)

    # #################################
    PaddedInput = nodes[1]
    Intermediate = nodes[2]
    Out = nodes[3]
    F_d = params[1]
    F_1 = params[2]

    num_channel = tvm.ir_pass.Simplify(PaddedInput.shape[3]).value
    num_thread = num_channel if num_channel <= 64 else int(num_channel / 4)
    output_num_per_block = 8

    Padded = s.cache_read(PaddedInput, "shared", [Intermediate])
    s[PaddedInput].compute_inline()
    FS_d = s.cache_read(F_d, "shared", [Intermediate])
    FS_1 = s.cache_read(F_1, "shared", [Out])
    # FL_d = s.cache_read(FS_d, "local", [Intermediate])
    # FL_1 = s.cache_read(FS_1, "local", [Out])

    # Output
    Output = Out
    OL = s.cache_write(Out, "shared")
    # Intermediate output
    IS = Intermediate
    s[Intermediate].set_scope("shared")

    block_x = tvm.thread_axis("blockIdx.x")
    thread_x = tvm.thread_axis("threadIdx.x")
    thread_y = tvm.thread_axis("threadIdx.y")
    thread_z = tvm.thread_axis("threadIdx.z")

    # *.local.rf = CTR
    # *.local = OL
    # * = Output

    # OL: 1 14 14 512, 512; Output: 1 14 14 512, na
    # Final output
    cc = s[OL].op.reduce_axis[0]
    xocc, xicc = s[OL].split(cc, factor=num_thread)
    CTR = s.rfactor(OL, xocc) # Cross thread reduction
    # OL: 1 14 14 512, 4; Output: 1 14 14 512, na
    
    n, h, w, c = s[Output].op.axis
    xoc, xic = s[Output].split(c, factor=output_num_per_block)
    s[Output].reorder(xoc, n, h, w, xic)
    yo, xo, yi, xi = s[Output].tile(h, w, x_factor=2, y_factor=2)
    fused = s[Output].fuse(yo, xo)
    # fused = s[Output].fuse(h, w)
    fused = s[Output].fuse(fused, n)
    fused = s[Output].fuse(fused, xoc)
    s[Output].bind(fused, block_x)

    n, h, w, col = s[OL].op.axis
    rc = s[OL].op.reduce_axis[0]
    s[OL].reorder(n, rc, h, w, col)
    s[CTR].bind(s[CTR].op.reduce_axis[0], thread_x)
    s[CTR].compute_at(s[OL], s[OL].op.axis[-1])
    # s[OL].bind(s[OL].op.reduce_axis[0], thread_x)
    s[OL].compute_at(s[Output], fused)

    # # Intermediate
    n, h, w, c = s[IS].op.axis
    co, ci = s[IS].split(c, factor=num_thread)
    s[IS].reorder(co, n, h, w, ci)
    print(s[IS].op.reduce_axis)
    yo, xo, yi, xi = s[IS].tile(h, w, x_factor=2, y_factor=2)
    fused_i = s[IS].fuse(yo, xo)
    s[IS].bind(ci, thread_x)
    ry, rx = s[IS].op.reduce_axis
    # s[IS].bind(ry, thread_z)
    # s[IS].bind(rx, thread_y)
    s[IS].compute_at(s[OL], rc)

    # Shared Input
    s[Padded].compute_at(s[IS], fused_i)
    n, h, w, c = s[Padded].op.axis
    co, ci = s[Padded].split(c, factor=num_thread)
    s[Padded].bind(ci, thread_x)

    # Shared depthwise filter
    s[FS_d].compute_at(s[OL], rc)
    # s[FL_d].compute_at(s[OL], col)
    hd, wd, id, od = s[FS_d].op.axis
    fused_f = s[FS_d].fuse(id, od)
    s[FS_d].bind(fused_f, thread_x)
    # s[FS_d].bind(wd, thread_y)
    # s[FS_d].bind(hd, thread_z)
    # s[FS_d].unroll(h)
    # s[FS_d].unroll(w)

    # Shared 1by1 filter
    s[FS_1].compute_at(s[OL], rc)
    # h1, w1, i1, o1 = s[F1_d].op.axis
    # if output_num_per_block >= num_thread:
    #     o11, o12 = s[F1_d].split(o1, factor=num_thread)
    #     s[F1_d].bind(o12, thread_x)
    # else:
    #     x = int(num_thread / output_num_per_block)
    #     i11, i12 = s[F1_d].split(i1, factor=x)
    #     fused_1 = s[F1_d].fuse(i12, o1)
    #     s[F1_d].bind(fused_1, thread_x)
    h1, w1, o1, i1 = s[FS_1].op.axis
    i11, i12 = s[FS_1].split(i1, factor=num_thread)
    s[FS_1].bind(i12, thread_x)
    # s[FL_1].compute_at(s[OL], col)

    return s

def verify_general_fused(parameters, padding_depthwise="SAME", dtype="float32", layout="NHWC", NHWC_transpose=False):
    assert layout in ["NHWC", "NCHW"]

    p = parameters
    input_shape = (p[0], p[1], p[1], p[2])
    filter_1_shape = (p[3], p[3], p[4], p[2]) if NHWC_transpose else (p[3], p[3], p[2], p[4])
    filter_2_shape = (p[6], p[6], p[7], p[2] * p[4]) if NHWC_transpose else (p[6], p[6], p[2] * p[4], p[7])

    # placeholder (NHWC)
    # Input: NHWC, Kernel: HWIO for both depthwise and conv2d
    Input = tvm.placeholder(input_shape, name='Input')
    DepthwiseFilter_1 = tvm.placeholder(filter_1_shape, name='DepthwiseFilter_1')
    Conv2dFilter_1 = tvm.placeholder(filter_2_shape, name='Conv2dFilter_1')
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
                    depthwise=p[5], kernel=p[3], stride=1, dilation=1, NHWC_transpose=NHWC_transpose))
    Filters.append(FilterConstructor(
                    Conv2dFilter_1,
                    depthwise=p[8], kernel=p[6], stride=1, dilation=1, NHWC_transpose=NHWC_transpose))

    # Get the graph
    # nodes: all nodes in the graph
    # params: inputs & outputs of the graph
    nodes, params = fused_two_conv(Input, Filters)

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
        # print(tvm.lower(s, params, simple_mode=True))

        func = tvm.build(s, params, device, name=("GeneralFused_{}".format(len(Filters))))
        print(func.imported_modules[0].get_source())
        # func(a, w, b)
        timer_1 = func.time_evaluator(func.entry_name, ctx, number=10)
        tcost_1 = timer_1(*nd_arrays).mean
        # np.testing.assert_allclose(nd_arrays[-1].asnumpy(), ref_data[-1], rtol=1e-3)
        d = ~np.isclose(nd_arrays[-1].asnumpy(), ref_data[-1], rtol=1e-3)
        print(nd_arrays[-1].asnumpy()[d])
        print(ref_data[-1][d])
        print(np.where(d))
        # print("Error rate: {:.2f}%".format((len(d) / len(ref_data[-1]) * 100)))
        print("General Fused of {} layers ({}): average running time is {:.2f} us.".format(len(Filters), layout, tcost_1 * 1e6))

    for device in ['cuda']:
        check_device(device)



if __name__ == "__main__":
    # parameters = [1, 112, 32, 3, 1, True, 1, 32, False]
    parameters = [1, 56, 128, 3, 1, True, 1, 256, False]
    # parameters = [1, 28, 256, 3, 1, True, 1, 256, False]
    # parameters = [1, 14, 256, 3, 1, True, 1, 512, False]

    verify_general_fused(parameters, NHWC_transpose=True)
    
