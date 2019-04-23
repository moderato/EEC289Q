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
def schedule_depth_conv_fused_nhwc(outs, nodes, params, NHWC_transpose=False):
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

    # # ########################################### 72 us for 112
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
    # yo, xo, yi, xi = s[Out].tile(h, w, x_factor=2, y_factor=2)
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
    # # s[IL].unroll(s[IL].op.reduce_axis[0])
    # # s[IL].unroll(s[IL].op.reduce_axis[1])

    # # # Shared depthwise filter
    # # s[FS_d].compute_at(s[Intermediate], w)
    # # _, _, ci, fi = s[FS_d].op.axis
    # # fused = s[FS_d].fuse(fi, ci)
    # # s[FS_d].bind(fused, thread_x)

    ################################# Distribute on Output
    PaddedInput = nodes[1]
    Intermediate = nodes[2]
    Out = nodes[3]
    F_d = params[1]
    F_1 = params[2]

    output_tile_size = 2
    num_thread_x = 32 # 64 if tvm.ir_pass.Simplify(PaddedInput.shape[3]).value >= 64 else 32
    num_thread_y = output_tile_size * output_tile_size
    # num_vthread_x = 3
    # num_vthread_y = 3

    s[PaddedInput].compute_inline()
    Padded = s.cache_read(PaddedInput, "shared", [Intermediate])
    # FS_d = s.cache_read(F_d, "shared", [Intermediate])
    # FL_d = s.cache_read(FS_d, "local", [Intermediate])
    FS_1 = s.cache_read(F_1, "shared", [Out])
    # FL_1 = s.cache_read(FS_1, "local", [Out])

    # # Intermediate output
    IS = Intermediate
    s[Intermediate].set_scope("shared")
    ITM = s.cache_write(Intermediate, "local")
    ISL = s.cache_read(Intermediate, "local", [Out])
    # # Output
    Output = Out
    OL = s.cache_write(Out, "local")
    
    block_x = tvm.thread_axis("blockIdx.x")
    block_y = tvm.thread_axis("blockIdx.y")
    thread_x = tvm.thread_axis("threadIdx.x")
    thread_y = tvm.thread_axis("threadIdx.y")

    # vthread_x = tvm.thread_axis((0, num_vthread_x), "vthread", name="vx")
    # vthread_y = tvm.thread_axis((0, num_vthread_y), "vthread", name="vy")

    ######## Final output
    # stride
    # print(s[OL].op.axis, s[OL].op.reduce_axis)
    # print(s[Output].op.axis, s[Output].op.reduce_axis)
    xocc, xicc = s[OL].split(s[OL].op.reduce_axis[0], factor=num_thread_x)
    CTR = s.rfactor(OL, xocc)
    # output
    n, h, w, c = s[Output].op.axis
    rc = s[OL].op.reduce_axis[0]
    xoc, xic = s[Output].split(c, factor=num_thread_x)
    yo, xo, yi, xi = s[Output].tile(h, w, x_factor=output_tile_size, y_factor=output_tile_size)
    s[Output].reorder(n, yo, xo, xoc, yi, xi, xic)
    # thread_x/y
    fused_b = s[Output].fuse(yi, xi)
    s[Output].bind(fused_b, thread_y)
    s[Output].bind(xic, thread_x)
    # thread_block
    fused_tb = s[Output].fuse(n, yo, xo, xoc)
    s[Output].bind(fused_tb, block_x)
    # Write to global
    s[OL].compute_at(s[Output], xic)
    
    ######## CTR
    s[CTR].compute_at(s[OL], rc)
    stride, n, h, w, o = s[CTR].op.axis
    print(s[CTR].op.axis)
    crc = s[CTR].op.reduce_axis[0]
    # # print(crc)
    crc_nparts = 8 # int(num_thread_x / 4 / num_thread_y)
    if crc_nparts > 0:
        crco, crci = s[CTR].split(crc, nparts=crc_nparts)
        s[CTR].unroll(crci)
    else:
        crco = o # No split

    # ######### Intermediate
    s[ISL].compute_at(s[CTR], crco) # s[CTR].op.reduce_axis[0] ############# Read intermediate to local
    # s[ISL].vectorize(s[ISL].op.axis[-1])
    s[IS].compute_at(s[OL], rc)
    n, h, w, c = s[IS].op.axis
    co, ci = s[IS].split(c, factor=num_thread_x)
    hw = s[IS].fuse(h, w)
    s[IS].bind(ci, thread_x)
    s[IS].bind(hw, thread_y)
    # # Unrolling
    ry, rx = s[ITM].op.reduce_axis
    # s[ITM].unroll(ry)
    # s[ITM].unroll(rx)
    s[ITM].compute_at(s[IS], ci)

    ######## Shared Input
    s[Padded].compute_at(s[IS], hw)
    n, h, w, c = s[Padded].op.axis
    co, ci = s[Padded].split(c, factor=num_thread_x)
    s[Padded].reorder(co, n, h, w, ci)
    # # Old binding
    s[Padded].bind(ci, thread_x)
    _, w = s[Padded].split(w, factor=num_thread_y)
    s[Padded].bind(w, thread_y)
    # # # Vectorization, -~10 us in 2nd test case
    # cio, cii = s[Padded].split(ci, factor=4)
    # _, w = s[Padded].split(w, factor=4)
    # _, h = s[Padded].split(h, factor=num_thread_y)
    # fused_p = s[Padded].fuse(w, cio)
    # s[Padded].bind(fused_p, thread_x)
    # s[Padded].bind(h, thread_y)
    # s[Padded].vectorize(cii)

    # ####### Shared depthwise filter
    # # s[FS_d].compute_at(s[IS], hw) # If read all in one time
    # s[FS_d].compute_at(s[ITM], rx)
    # hd, wd, id, od = s[FS_d].op.axis
    # fused_f = s[FS_d].fuse(id, od)
    # s[FS_d].bind(fused_f, thread_x)
    # # s[FS_d].set_store_predicate(thread_y.var.equal(0))

    # If read all in one time
    # _, wd = s[FS_d].split(wd, factor=4)
    # s[FS_d].bind(wd, thread_y)
    # s[FL_d].compute_at(s[IS], s[IS].op.reduce_axis[-1])

    # If read all in one time + vectorization
    # s[FS_d].compute_at(s[IS], hw)
    # hd, wd, id, od = s[FS_d].op.axis
    # ido, idi = s[FS_d].split(id, factor=4)
    # _, wd = s[FS_d].split(wd, factor=4)
    # _, hd = s[FS_d].split(hd, factor=num_thread_y)
    # fused_f = s[FS_d].fuse(wd, ido)
    # s[FS_d].bind(fused_f, thread_x)
    # s[FS_d].bind(hd, thread_y)
    # fused_v = s[FS_d].fuse(idi, od)
    # s[FS_d].vectorize(fused_v)


    ######## Shared 1by1 filter
    if not NHWC_transpose:
        s[FS_1].compute_at(s[CTR], o) # Options: crco/o
        h1, w1, i1, o1 = s[FS_1].op.axis
        # # Old binding
        _, ii = s[FS_1].split(i1, factor=num_thread_y)
        _, oi = s[FS_1].split(o1, factor=num_thread_x)
        s[FS_1].bind(oi, thread_x)
        s[FS_1].bind(ii, thread_y)

        # # # Vectorization, -~18 us in 2nd test case
        # # ioo, ioi(thread_y), (ii, oo)(thread_x), oi(4)
        # io, ii = s[FS_1].split(i1, factor=4)
        # ioo, ioi = s[FS_1].split(io, factor=num_thread_y)
        # oo, oi = s[FS_1].split(o1, factor=4)
        # fused_1 = s[FS_1].fuse(ii, oo)
        # s[FS_1].bind(ioi, thread_y)
        # s[FS_1].bind(fused_1, thread_x)
        # s[FS_1].vectorize(oi)

        # # Read to local
        # s[FL_1].compute_at(s[CTR], s[CTR].op.reduce_axis[0])
        # # Read input to local
        # s[ISL].compute_at(s[CTR], s[CTR].op.reduce_axis[0]) # Options: s[CTR].op.reduce_axis[0] / crco

    else:
        s[FS_1].compute_at(s[CTR], o)
        h1, w1, o1, i1 = s[FS_1].op.axis
        io, ii = s[FS_1].split(i1, factor=4)
        oo, oi = s[FS_1].split(o1, factor=4)
        ooo, ooi = s[FS_1].split(oo, factor=num_thread_y)
        fused_1 = s[FS_1].fuse(oi, io)
        s[FS_1].bind(ooi, thread_y)
        s[FS_1].bind(fused_1, thread_x)
        s[FS_1].vectorize(ii)
        # # Read to local
        # s[FL_1].compute_at(s[CTR], crci)

    # #################################
    # PaddedInput = nodes[1]
    # Intermediate = nodes[2]
    # Out = nodes[3]
    # F_d = params[1]
    # F_1 = params[2]

    # output_tile_size = 2
    # num_thread_x = 32 # 64 if tvm.ir_pass.Simplify(PaddedInput.shape[3]).value >= 64 else 32
    # num_thread_y = output_tile_size * output_tile_size
    # output_num = 32
    # # num_vthread_x = 3
    # # num_vthread_y = 3

    # s[PaddedInput].compute_inline()
    # Padded = s.cache_read(PaddedInput, "shared", [Intermediate])
    # # FS_d = s.cache_read(F_d, "shared", [Intermediate])
    # # FL_d = s.cache_read(FS_d, "local", [Intermediate])
    # FS_1 = s.cache_read(F_1, "shared", [Out])
    # # FL_1 = s.cache_read(FS_1, "local", [Out])

    # # # Intermediate output
    # # IS = Intermediate
    # # s[Intermediate].set_scope("shared")
    # # ITM = s.cache_write(Intermediate, "local")
    # # ISL = s.cache_read(Intermediate, "local", [Out])
    # IS = Intermediate
    # s[Intermediate].set_scope("local")

    # # # Output
    # Output = Out
    # # OL = s.cache_write(Out, "local")
    
    # block_x = tvm.thread_axis("blockIdx.x")
    # thread_x = tvm.thread_axis("threadIdx.x")
    # thread_y = tvm.thread_axis("threadIdx.y")

    # # vthread_x = tvm.thread_axis((0, num_vthread_x), "vthread", name="vx")
    # # vthread_y = tvm.thread_axis((0, num_vthread_y), "vthread", name="vy")

    # ######## Final output
    # # stride
    # # print(s[OL].op.axis, s[OL].op.reduce_axis)
    # # print(s[Output].op.axis, s[Output].op.reduce_axis)
    # xocc, xicc = s[Output].split(s[Output].op.reduce_axis[0], factor=num_thread_x)
    # CTR = s.rfactor(Output, xocc)
    # rc = s[Output].op.reduce_axis[0]
    # s[CTR].set_scope("local")

    # # # output
    # n, h, w, c = s[Output].op.axis
    # xoc, xic = s[Output].split(c, factor=output_num)
    # print(s[Output].op.axis, s[Output].op.reduce_axis)
    # yo, xo, yi, xi = s[Output].tile(h, w, x_factor=output_tile_size, y_factor=output_tile_size)
    # s[Output].reorder(n, yo, xo, xoc, yi, xi, xic)
    # # thread_y
    # fused_b = s[Output].fuse(yi, xi)
    # s[Output].bind(fused_b, thread_y)
    # # thread_block
    # fused_tb = s[Output].fuse(n, yo, xo, xoc)
    # s[Output].bind(fused_tb, block_x)
    # # CTR
    # crc = s[CTR].op.reduce_axis[0]
    # s[CTR].compute_at(s[Output], rc)
    # s[CTR].bind(crc, thread_x)

    # s[Output].bind(xic, thread_x)
    
    # # ######### Intermediate
    # # s[ISL].compute_at(s[CTR], crc) # crco ############# Read intermediate to local
    # s[IS].compute_at(s[Output], rc)
    # # n, h, w, c = s[IS].op.axis
    # # co, ci = s[IS].split(c, factor=num_thread_x)
    # # hw = s[IS].fuse(h, w)
    # # s[IS].bind(ci, thread_x)
    # # s[IS].bind(hw, thread_y)
    # # # Unrolling
    # # ry, rx = s[ITM].op.reduce_axis
    # # s[ITM].unroll(ry)
    # # s[ITM].unroll(rx)
    # # s[ITM].compute_at(s[IS], ci)

    # # ######## Shared Input
    # s[Padded].compute_at(s[Output], rc)
    # # s[Padded].compute_at(s[IS], hw)
    # # n, h, w, c = s[Padded].op.axis
    # # co, ci = s[Padded].split(c, factor=num_thread_x)
    # # s[Padded].reorder(co, n, h, w, ci)
    # # # Old binding
    # # s[Padded].bind(ci, thread_x)
    # # _, w = s[Padded].split(w, factor=num_thread_y)
    # # s[Padded].bind(w, thread_y)

    # # Vectorization, -~10 us in 2nd test case
    # # cio, cii = s[Padded].split(ci, factor=4)
    # # _, w = s[Padded].split(w, factor=4)
    # # _, h = s[Padded].split(h, factor=num_thread_y)
    # # fused_p = s[Padded].fuse(w, cio)
    # # s[Padded].bind(fused_p, thread_x)
    # # s[Padded].bind(h, thread_y)
    # # s[Padded].vectorize(cii)

    # # # ####### Shared depthwise filter
    # # # # s[FS_d].compute_at(s[IS], hw) # If read all in one time
    # # # s[FS_d].compute_at(s[ITM], rx)
    # # # hd, wd, id, od = s[FS_d].op.axis
    # # # fused_f = s[FS_d].fuse(id, od)
    # # # s[FS_d].bind(fused_f, thread_x)
    # # # # s[FS_d].set_store_predicate(thread_y.var.equal(0))

    # # # If read all in one time
    # # # _, wd = s[FS_d].split(wd, factor=4)
    # # # s[FS_d].bind(wd, thread_y)
    # # # s[FL_d].compute_at(s[IS], s[IS].op.reduce_axis[-1])

    # # # If read all in one time + vectorization
    # # # s[FS_d].compute_at(s[IS], hw)
    # # # hd, wd, id, od = s[FS_d].op.axis
    # # # ido, idi = s[FS_d].split(id, factor=4)
    # # # _, wd = s[FS_d].split(wd, factor=4)
    # # # _, hd = s[FS_d].split(hd, factor=num_thread_y)
    # # # fused_f = s[FS_d].fuse(wd, ido)
    # # # s[FS_d].bind(fused_f, thread_x)
    # # # s[FS_d].bind(hd, thread_y)
    # # # fused_v = s[FS_d].fuse(idi, od)
    # # # s[FS_d].vectorize(fused_v)


    # ######## Shared 1by1 filter
    # if not NHWC_transpose:
    #     # s[FS_1].compute_at(s[CTR], o) # Options: crco/o
    #     s[FS_1].compute_at(s[Output], rc)
    #     # h1, w1, i1, o1 = s[FS_1].op.axis
    #     # # Old binding
    #     # _, ii = s[FS_1].split(i1, factor=num_thread_y)
    #     # _, oi = s[FS_1].split(o1, factor=num_thread_x)
    #     # s[FS_1].bind(oi, thread_x)
    #     # s[FS_1].bind(ii, thread_y)

    #     # # Vectorization, -~18 us in 2nd test case
    #     # # ioo, ioi(thread_y), (ii, oo)(thread_x), oi(4)
    #     # oo, oi = s[FS_1].split(o1, factor=4)
    #     # io, ii = s[FS_1].split(i1, factor=int(num_thread_x * 4 / output_num))
    #     # ioo, ioi = s[FS_1].split(io, factor=num_thread_y)
    #     # fused_1 = s[FS_1].fuse(ii, oo)
    #     # s[FS_1].bind(ioi, thread_y)
    #     # s[FS_1].bind(fused_1, thread_x)
    #     # s[FS_1].vectorize(oi)

    # #     # # Read to local
    # #     # s[FL_1].compute_at(s[CTR], s[CTR].op.reduce_axis[0])
    # #     # # Read input to local
    # #     # s[ISL].compute_at(s[CTR], s[CTR].op.reduce_axis[0]) # Options: s[CTR].op.reduce_axis[0] / crco

    return s

def verify_depth_conv_fused(parameters, dtype="float32", layout="NHWC", NHWC_transpose=False, print_code=True, save_data=False, export_code=False):
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
            filename = "npy/depth_conv_"
            if i == 0:
                filename += "input_"
            elif i == 1:
                filename += "filter_d_"
            elif i == 2:
                filename += "filter_1_"
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
            s = schedule_depth_conv_fused_nhwc([nodes[-1]], nodes, params, NHWC_transpose)
        if print_code:
            print(tvm.lower(s, params, simple_mode=True))

        # with tvm.build_config(data_alignment=4):
        func = tvm.build(s, params, device, name=("DepthConvFused_{}".format(len(Filters))))
        if print_code:
            print(func.imported_modules[0].get_source())
        if export_code:
            cuda_code = func.imported_modules[0].get_source()
            write_code(cuda_code, "testbed/kernel_depth_conv.cuh")
        # func(a, w, b)
        timer_1 = func.time_evaluator(func.entry_name, ctx, number=10)
        tcost_1 = timer_1(*nd_arrays).mean
        # np.testing.assert_allclose(nd_arrays[-1].asnumpy(), ref_data[-1], rtol=1e-3)
        d = ~np.isclose(nd_arrays[-1].asnumpy(), ref_data[-1], rtol=1e-3)
        if (np.sum(d) > 0):
            print("# of incorrect numbers: {}".format(len(ref_data[-1][d])))
            print(nd_arrays[-1].asnumpy()[d])
            print(ref_data[-1][d])
            print(np.where(d))
        # print("Error rate: {:.2f}%".format((len(d) / len(ref_data[-1]) * 100)))
        print("Depthwise Conv Fused of {} layers ({}): average running time is {:.2f} us.".format(len(Filters), layout, tcost_1 * 1e6))

    for device in ['cuda']:
        check_device(device)

if __name__ == "__main__":
    parameters = []

    parameters.append([1, 112, 32, 3, 1, True, 1, 32, False]) # 122.78 us
    # parameters.append([1, 56, 128, 3, 1, True, 1, 128, False]) # 398.18 / 456.16 us
    # parameters.append([1, 28, 256, 3, 1, True, 1, 256, False]) # 389.57 / 423.63 us
    # parameters.append([1, 14, 512, 3, 1, True, 1, 512, False]) # 367.71 us, 344.27 us

    for p in parameters:
        verify_depth_conv_fused(p, NHWC_transpose=False, print_code=True, save_data=False, export_code=False)
    
