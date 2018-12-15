import tvm
import topi
import topi.testing
import numpy as np
import os, logging, sys
from scipy import signal
from topi.util import get_const_tuple
from tvm.contrib.pickle_memoize import memoize
from depth_1by1_fused_compute import *
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
def schedule_depth_1by1_fused_nhwc(outs):
    """Schedule for depthwise_conv2d nhwc forward.
    Parameters
    ----------
    outs: Array of Tensor
        The computation graph description of depthwise_conv2d
        in the format of an array of tensors.
    Returns
    -------
    s: Schedule
        The computation schedule for depthwise_conv2d nhwc.
    """
    outs = [outs] if isinstance(outs, tvm.tensor.Tensor) else outs
    s = tvm.create_schedule([x.op for x in outs])
    def _schedule(Padded, F_d, F_1, Out):
        # s[In].compute_inline()

        # num_thread = 256
        # block_x = tvm.thread_axis("blockIdx.x")
        # thread_x = tvm.thread_axis((0, num_thread), "threadIdx.x")

        # ni, hi, wi, ci = s[Out].op.axis
        # wi = s[Out].fuse(hi, wi)
        # s[Out].bind(wi, block_x)
        # s[Out].bind(ci, thread_x)

        s[Padded].compute_inline()

        if Out.op in s.outputs:
            Output = Out
            OL = s.cache_write(Out, "local")
        else:
            Output = outs[0].op.output(0)
            s[Out].set_scope("local")

        IS = s.cache_read(Padded, "shared", [OL])
        FS_d = s.cache_read(F_d, "shared", [OL])
        # FS_1 = s.cache_read(F_1, "shared", [OL])

        in_channel = tvm.ir_pass.Simplify(Padded.shape[3]).value
        num_thread = in_channel
        multiples = 1
        # # multiples
        # while num_thread < 1024 and multiples <= 4:
        #     num_thread *= 2
        #     multiples *= 2

        block_x = tvm.thread_axis("blockIdx.x")
        thread_x = tvm.thread_axis((0, in_channel), "threadIdx.x")
        if multiples != 1:
            thread_y = tvm.thread_axis((0, multiples), "threadIdx.y")

        # Output tiling
        n, h, w, c = s[Output].op.axis
        xoc, xic = s[Output].split(c, factor=num_thread)
        # _, _, cc = s[Out].op.reduce_axis
        # xocc, xicc = s[Out].split(cc, factor=16)
        # cl = s.rfactor(Out, xicc)
        s[Output].reorder(xoc, n, h, w, xic)
        yo, xo, b, a = s[Output].tile(h, w, x_factor=2, y_factor=2)
        fused = s[Output].fuse(yo, xo)
        fused = s[Output].fuse(n, fused)
        fused = s[Output].fuse(xoc, fused)
        s[Output].bind(fused, block_x)
        s[Output].bind(xic, thread_x)
        if Out.op in s.outputs:
            s[OL].compute_at(s[Output], xic)
        else:
            s[Out].compute_at(s[Output], xic)

        # n, h, w, c = s[Output].op.axis
        # yo, xo, b, a = s[Output].tile(h, w, x_factor=2, y_factor=2)
        # if multiples != 1:
        #     x2, x1 = s[Output].split(xo, factor=multiples)
        #     fused = s[Output].fuse(yo, x2)
        #     fused = s[Output].fuse(n, fused)
        #     s[Output].reorder(fused, b, a, x1, c)
        #     s[Output].bind(x1, thread_y)
        # else:
        #     fused = s[Output].fuse(yo, xo)
        #     fused = s[Output].fuse(n, fused)
        # s[Output].bind(fused, block_x)
        # s[Output].bind(c, thread_x)
        # if Out.op in s.outputs:
        #     s[OL].compute_at(s[Output], c)
        # else:
        #     s[Out].compute_at(s[Output], c)

        # # Input reuse
        s[IS].compute_at(s[Output], fused)
        n, h, w, c = s[IS].op.axis
        s[IS].reorder(h, w, n, c)
        fused_is = s[IS].fuse(n, c)
        s[IS].bind(fused_is, thread_x)
        # # multiples
        # w2, w1 = s[IS].split(w, factor=multiples)
        # s[IS].bind(w1, thread_y)

        # Filter_d reuse
        s[FS_d].compute_at(s[Output], fused) # Necessary!!
        fy, fx, c, f = s[FS_d].op.axis
        fused_fs_d = s[FS_d].fuse(c, f)
        s[FS_d].bind(fused_fs_d, thread_x)

        # # Filter_1 reuse
        # s[FS_1].compute_at(s[Output], fused)
        # h, w, o, i = s[FS_1].op.axis
        # s[FS_1].reorder(h, w, i, o)
        # s[FS_1].bind(i, thread_x)


        # 1 SM multiple warps: hide latency
        # Don't use too much registers in one thread (limiting the number of threads/warps on 1 SM and resulting in worse latency hiding)
        # Increase occupancy
        


    def traverse(OP):
        print("***********")
        print(OP.tag)

        # inline all one-to-one-mapping operators except the last stage (output)
        if tag.is_broadcast(OP.tag):
            print("is broadcast")
            if OP not in s.outputs:
                s[OP].compute_inline()
            for tensor in OP.input_tensors:
                if tensor.op.input_tensors:
                    traverse(tensor.op)
        elif tag.is_injective(OP.tag):
            print("is injective")
            for tensor in OP.input_tensors:
                if tensor.op.input_tensors:
                    traverse(tensor.op)
        # schedule depthwise_conv2d
        elif OP.tag == 'depthwise_1by1_fused_nhwc':
            PaddedInput = OP.input_tensors[0]
            Filter_d = OP.input_tensors[1]
            Filter_1 = OP.input_tensors[2]
            Depthwise1by1Fused = OP.output(0)
            _schedule(PaddedInput, Filter_d, Filter_1, Depthwise1by1Fused)

    traverse(outs[0].op)
    return s

# @register_fused.register(["cuda", "gpu"])
def schedule_depth_1by1_fused_nchw(outs):
    """Schedule for depthwise_conv2d nhwc forward.
    Parameters
    ----------
    outs: Array of Tensor
        The computation graph description of depthwise_conv2d
        in the format of an array of tensors.
    Returns
    -------
    s: Schedule
        The computation schedule for depthwise_conv2d nhwc.
    """
    outs = [outs] if isinstance(outs, tvm.tensor.Tensor) else outs
    s = tvm.create_schedule([x.op for x in outs])
    def _schedule(Padded, F_d, F_1, Out):
        # s[Padded].compute_inline()

        # num_thread = 256
        # block_x = tvm.thread_axis("blockIdx.x")
        # thread_x = tvm.thread_axis((0, num_thread), "threadIdx.x")

        # ni, ci, hi, wi = s[Out].op.axis
        # ha, hb = s[Out].split(hi, factor=16)
        # wa, wb = s[Out].split(wi, factor=16)
        # s[Out].reorder(ha, wa, hb, wb)
        # inner = s[Out].fuse(hb, wb)
        # outer = s[Out].fuse(ha, wa)
        # outer = s[Out].fuse(ci, outer)
        # s[Out].bind(outer, block_x)
        # s[Out].bind(inner, thread_x)

        s[Padded].compute_inline()
        IS = s.cache_read(Padded, "shared", [Out])
        FS_d = s.cache_read(F_d, "shared", [Out])
        # FS_1 = s.cache_read(F_1, "shared", [Out])

        if Out.op in s.outputs:
            Output = Out
            OL = s.cache_write(Out, "local")
        else:
            Output = outs[0].op.output(0)
            s[Out].set_scope("local")

        in_channel = tvm.ir_pass.Simplify(Padded.shape[1]).value
        num_thread = in_channel
        # 
        # while num_thread < 128:
        #     num_thread *= 2
        # multiples = int(num_thread / in_channel)

        block_x = tvm.thread_axis("blockIdx.x")
        thread_x = tvm.thread_axis((0, num_thread), "threadIdx.x")

        # # Output tiling
        n, c, h, w = s[Output].op.axis
        s[Out].reorder(n, h, w, c)
        yo, xo, b, a = s[Output].tile(h, w, x_factor=2, y_factor=2)
        fused = s[Output].fuse(yo, xo)
        fused = s[Output].fuse(n, fused)
        # if multiples != 1:
        #     fused, tmp = s[Out].split(fused, factor=multiples)
        #     s[Out].reorder(fused, b, a, tmp, c)
        #     c = s[Out].fuse(tmp, c)
        s[Output].bind(fused, block_x)
        s[Output].bind(c, thread_x)
        if Out.op in s.outputs:
            s[OL].compute_at(s[Output], c)
        else:
            s[Out].compute_at(s[Output], c)

        # # Input reuse
        s[IS].compute_at(s[Output], fused)
        n, c, h, w = s[IS].op.axis
        fused_is = s[IS].fuse(n, c)
        s[IS].bind(fused_is, thread_x)

        # # Filter_d reuse
        # s[FS_d].compute_at(s[Out], fused) # Not necessary!
        c, _, fy, fx = s[FS_d].op.axis
        s[FS_d].bind(c, thread_x)

        # # Filter_1 reuse
        # s[FS_1].compute_at(s[Output], fused)
        # i, o, h, w = s[FS_1].op.axis
        # s[FS_1].reorder(h, w, i, o)
        # s[FS_1].bind(o, thread_x)

        
    def traverse(OP):
        print("***********")
        print(OP.tag)

        # inline all one-to-one-mapping operators except the last stage (output)
        if tag.is_broadcast(OP.tag):
            # print("is broadcast")
            if OP not in s.outputs:
                s[OP].compute_inline()
            for tensor in OP.input_tensors:
                if tensor.op.input_tensors:
                    traverse(tensor.op)
        elif tag.is_injective(OP.tag):
            # print("is injective")
            for tensor in OP.input_tensors:
                if tensor.op.input_tensors:
                    traverse(tensor.op)
        # schedule depthwise_conv2d
        elif OP.tag == 'depthwise_1by1_fused_nchw':
            PaddedInput = OP.input_tensors[0]
            Filter_d = OP.input_tensors[1]
            Filter_1 = OP.input_tensors[2]
            Depthwise1by1Fused = OP.output(0)
            _schedule(PaddedInput, Filter_d, Filter_1, Depthwise1by1Fused)

    traverse(outs[0].op)
    return s

def verify_depth_1by1_fused(batch, in_channel_depthwise, in_size, channel_multiplier, kernel_depthwise, stride_depthwise, num_filter, padding_depthwise="SAME", dtype="float32", layout="NHWC"):
    assert layout in ["NHWC", "NCHW"]

    in_height = in_width = in_size
    in_c_d = in_channel_depthwise
    out_c_d = channel_multiplier
    in_c_1 = in_channel_depthwise * channel_multiplier
    out_c_1 = num_filter
    kernel_h_d = kernel_w_d = kernel_depthwise
    kernel_h_1 = kernel_w_1 = 1
    stride_d = stride_depthwise
    stride_1 = 1

    # placeholder
    if layout == "NCHW":
        # Input: NCHW, Kernel: IOHW for depthwise, OIHW for conv2d
        Input = tvm.placeholder((batch, in_c_d, in_height, in_width), name='Input')
        Filter_d = tvm.placeholder((in_c_d, out_c_d, kernel_h_d, kernel_w_d), name='Filter_d')
        Filter_1 = tvm.placeholder((out_c_1, in_c_1, kernel_h_1, kernel_w_1), name='Filter_1')
    else: # NHWC
        # Input: NHWC, Kernel: HWIO for both depthwise and conv2d
        Input = tvm.placeholder((batch, in_height, in_width, in_c_d), name='Input')
        Filter_d = tvm.placeholder((kernel_h_d, kernel_w_d, in_c_d, out_c_d), name='Filter_d')
        Filter_1 = tvm.placeholder((kernel_h_1, kernel_w_1, in_c_1, out_c_1), name='Filter_1')
        
    Output = depth_1by1_fused(Input, Filter_d, Filter_1, stride_d, layout=layout)

    input_shape = get_const_tuple(Input.shape)
    filter_shape_d = get_const_tuple(Filter_d.shape)
    filter_shape_1 = get_const_tuple(Filter_1.shape)
    dtype = Input.dtype

    # @memoize("verify_nhwc")
    def get_ref_data():
        input_np = np.random.uniform(size=input_shape).astype(dtype)
        filter_np_d = np.random.uniform(size=filter_shape_d).astype(dtype)
        filter_np_1 = np.random.uniform(size=filter_shape_1).astype(dtype)

        if layout == "NCHW":
            intermediate = topi.testing.depthwise_conv2d_python_nchw(input_np, filter_np_d, stride=[stride_d, stride_d], padding=padding_depthwise)
            output_np = topi.testing.conv2d_nchw_python(intermediate, filter_np_1, stride_1, padding_depthwise)
        else: 
            intermediate = topi.testing.depthwise_conv2d_python_nhwc(input_np, filter_np_d, stride=[stride_d, stride_d], padding=padding_depthwise)
            output_np = topi.testing.conv2d_nhwc_python(intermediate, filter_np_1, stride_1, padding_depthwise)

        return input_np, filter_np_d, filter_np_1, output_np
    input_np, filter_np_d, filter_np_1, output_np = get_ref_data()

    def check_device(device):
        if not tvm.module.enabled(device):
            print("Skip because %s is not enabled" % device)
            return
        print("Running on target: %s" % device)

        ctx = tvm.context(device, 0)
        input = tvm.nd.array(input_np, ctx)
        filter_d = tvm.nd.array(filter_np_d, ctx)
        filter_1 = tvm.nd.array(filter_np_1, ctx)
        output = tvm.nd.array(np.zeros(get_const_tuple(Output.shape), dtype=Output.dtype), ctx)

        with tvm.target.create(device):
            if layout == "NCHW":
                s = schedule_depth_1by1_fused_nchw([Output])
            else:
                s = schedule_depth_1by1_fused_nhwc([Output])
        # print(tvm.lower(s, [Input, Filter_d, Filter_1, Output], simple_mode=True))
                
        func = tvm.build(s, [Input, Filter_d, Filter_1, Output], device, name=("Depthwise1by1Fused_%d_%d" % (Input.shape[1], Input.shape[2])))
        # func(a, w, b)
        timer_1 = func.time_evaluator(func.entry_name, ctx, number=1000)
        tcost_1 = timer_1(input, filter_d, filter_1, output).mean
        np.testing.assert_allclose(output.asnumpy(), output_np, rtol=1e-5)
        print("Depthwise & 1by1 Fused ({}): average running time is {:.2f} us.".format(layout, tcost_1 * 1e6))

    for device in ['cuda']:
        check_device(device)

if __name__ == "__main__":
    # verify_depth_1by1_fused(1, 32, 112, 1, 3, 1, 32, layout="NHWC")
    verify_depth_1by1_fused(1, 128, 56, 1, 3, 1, 128, layout="NHWC")
    # verify_depth_1by1_fused(1, 256, 28, 1, 3, 1, 256, layout="NHWC")
    # verify_depth_1by1_fused(1, 512, 14, 1, 3, 1, 512, layout="NHWC")
    # verify_depth_1by1_fused(1, 32, 112, 1, 3, 1, 32, layout="NCHW")
