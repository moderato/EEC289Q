from __future__ import absolute_import as _abs
import tvm
import topi
import topi.tag as tag
import topi.util as util
from topi.nn.pad import pad
from topi.nn.util import get_pad_tuple
from topi.util import simplify
from tvm import autotvm
import topi.generic as generic
import numpy as np
import scipy

@tvm.target.generic_func
def register_fused(outs, auto_inline=False):
    """Default schedule for llvm."""
    target = tvm.target.current_target(allow_none=False)
    outs = [outs] if isinstance(outs, tvm.tensor.Tensor) else outs
    if target.target_name != "llvm":
        raise RuntimeError("schedule not registered for '%s'" % target)
    s = tvm.create_schedule([x.op for x in outs])
    if auto_inline:
        x = outs[0]
        tvm.schedule.AutoInlineInjective(s)
        s[x].fuse(s[x].op.axis)
    return s

# def schedule_fused_nchw(outs):
#     cfg = autotvm.get_config()

#     """Create schedule for tensors or return error if batch size is larger than 1"""
#     s = tvm.create_schedule([x.op for x in outs])

#     def schedule_conv2d(temp, Filter, Output):
#         """Schedule conv2d_nchw"""

#         flag = util.get_const_int(Filter.shape[0])+util.get_const_int(Filter.shape[1])

#         if flag > 768:
#             temp_G = s.cache_read(temp, "global", [Output])
#             s[temp_G].compute_inline()
#             i, ic, h, w = s[temp_G].op.axis
#             oic, iic = s[temp_G].split(ic, factor=4)
#             s[temp_G].reorder(i, h, w, oic, iic)
#             temp_R = s.cache_write(temp_G, "global")
#             temp_S = s.cache_read(temp_R, "shared", [temp_G])
#         elif 128 < flag < 512:
#             temp_G = s.cache_read(temp, "global", [Output])
#             s[temp_G].compute_inline()
#             i, ic, h, w = s[temp_G].op.axis
#             oic, iic = s[temp_G].split(ic, factor=4)
#             s[temp_G].reorder(i, oic, h, w, iic)
#             temp_R = s.cache_write(temp_G, "global")
#             temp_S = s.cache_read(temp_R, "shared", [temp_G])
#         elif util.get_const_int(Filter.shape[3]) == 7 or (util.get_const_int(Output.shape[2] == 224) and flag < 128):
#             temp_G = s.cache_read(temp, "global", [Output])
#             s[temp_G].compute_inline()
#             i, ic, h, w = s[temp_G].op.axis
#             s[temp_G].split(w, factor=4)
#             temp_R = s.cache_write(temp_G, "global")
#             temp_S = s.cache_read(temp_R, "shared", [temp_G])
#         else:
#             s[temp].compute_inline()
#             temp_S = s.cache_read(temp, "shared", [Output])
#             temp_R = temp_S

#         Filter_S = s.cache_read(Filter, "shared", [Output])

#         if Output.op in s.outputs:
#             Out = Output
#             Out_L = s.cache_write(Out, "local")
#         else:
#             Out = outs[0].op.output(0)
#             s[Output].set_scope("local")
#             Out_L = Output

#         if util.get_const_int(Filter.shape[3]) == 7 or (util.get_const_int(Output.shape[2] == 224) and flag < 128):
#             print("224_3_64")
#             conv2d_224_3_64(s, temp, temp_R, temp_S, Filter_S, Out, Out_L, flag)
#         elif 128 < flag < 512:
#             print("56_64_128")
#             conv2d_56_64_128(s, temp, temp_R, temp_S, Filter_S, Out, Out_L, flag)
#         elif flag >= 512:
#             print("14_256_256")
#             conv2d_14_256_256(s, temp, temp_R, temp_S, Filter, Filter_S, Out, Out_L)
#         else:
#             print("56_64_64")
#             conv2d_56_64_64(s, Filter, temp_S, Filter_S, Out, Out_L)

#     def schedule_depth(PaddedInput, Filter, DepthwiseConv2d):
#         in_shape = util.get_const_tuple(PaddedInput.shape)
#         out_shape = util.get_const_tuple(DepthwiseConv2d.shape)
#         in_height = in_shape[2]
#         in_width = in_shape[3]
#         out_height = out_shape[2]
#         out_width = out_shape[3]
#         channel_multiplier = util.get_const_tuple(Filter.shape)[1]
#         s[PaddedInput].compute_inline()
#         IS = s.cache_read(PaddedInput, "shared", [DepthwiseConv2d])
#         FS = s.cache_read(Filter, "shared", [DepthwiseConv2d])
#         IL = s.cache_read(IS, "local", [DepthwiseConv2d])
#         FL = s.cache_read(FS, "local", [DepthwiseConv2d])
#         if DepthwiseConv2d.op in s.outputs:
#             Output = DepthwiseConv2d
#             CL = s.cache_write(DepthwiseConv2d, "local")
#         else:
#             Output = outs[0].op.output(0)
#             s[DepthwiseConv2d].set_scope("local")
#         # schedule parameters
#         num_thread_y = 8
#         num_thread_x = 8
#         num_vthread_y = 1
#         num_vthread_x = 1
#         blocking_h = out_height
#         blocking_w = out_width
#         if out_height % 32 == 0 or in_height >= 108:
#             blocking_h = 32
#         if out_width % 32 == 0:
#             blocking_w = 32
#             num_thread_x = 16
#             num_vthread_x = 2
#         elif in_width >= 108:
#             blocking_w = 32
#         block_y = tvm.thread_axis("blockIdx.y")
#         block_x = tvm.thread_axis("blockIdx.x")
#         thread_y = tvm.thread_axis((0, num_thread_y), "threadIdx.y")
#         thread_x = tvm.thread_axis((0, num_thread_x), "threadIdx.x")
#         thread_vy = tvm.thread_axis((0, num_vthread_y), "vthread", name="vy")
#         thread_vx = tvm.thread_axis((0, num_vthread_x), "vthread", name="vx")
#         # split and bind
#         by, byi = s[Output].split(Output.op.axis[1], factor=channel_multiplier)
#         s[Output].reorder(Output.op.axis[2], Output.op.axis[3], byi)
#         by = s[Output].fuse(Output.op.axis[0], by)
#         s[Output].bind(by, block_y)
#         bx1, x1i = s[Output].split(Output.op.axis[2], factor=blocking_h)
#         tvy, vyi = s[Output].split(x1i, nparts=num_vthread_y)
#         ty, yi = s[Output].split(vyi, nparts=num_thread_y)
#         bx2, x2i = s[Output].split(Output.op.axis[3], factor=blocking_w)
#         tvx, vxi = s[Output].split(x2i, nparts=num_vthread_x)
#         tx, xi = s[Output].split(vxi, nparts=num_thread_x)
#         s[Output].reorder(bx1, bx2, tvy, tvx, ty, tx, yi, xi)
#         bx = s[Output].fuse(bx1, bx2)
#         s[Output].bind(bx, block_x)
#         s[Output].bind(tvy, thread_vy)
#         s[Output].bind(tvx, thread_vx)
#         s[Output].bind(ty, thread_y)
#         s[Output].bind(tx, thread_x)
#         # local memory load
#         s[IL].compute_at(s[Output], tx)
#         s[FL].compute_at(s[Output], tx)
#         if DepthwiseConv2d.op in s.outputs:
#             s[CL].compute_at(s[Output], tx)
#         else:
#             s[DepthwiseConv2d].compute_at(s[Output], tx)
#         # input's shared memory load
#         s[IS].compute_at(s[Output], bx)
#         ty, yi = s[IS].split(IS.op.axis[2], nparts=num_thread_y)
#         tx, xi = s[IS].split(IS.op.axis[3], nparts=num_thread_x)
#         s[IS].bind(ty, thread_y)
#         s[IS].bind(tx, thread_x)
#         # filter's shared memory load
#         s[FS].compute_at(s[Output], bx)
#         s[FS].reorder(FS.op.axis[2], FS.op.axis[3], FS.op.axis[1])
#         ty, yi = s[FS].split(FS.op.axis[2], nparts=num_thread_y)
#         tx, xi = s[FS].split(FS.op.axis[3], nparts=num_thread_x)
#         s[FS].bind(ty, thread_y)
#         s[FS].bind(tx, thread_x)

#     def traverse(OP, cfg):
#         """Traverse operators from computation graph"""
#         # inline all one-to-one-mapping operators except the last stage (output)
#         print("**********")
#         print(OP.input_tensors)
#         print(tag.is_broadcast(OP.tag))
#         print(OP.tag)

#         if tag.is_broadcast(OP.tag):
#             print("is broadcast")
#             if OP not in s.outputs:
#                 s[OP].compute_inline()
#             for tensor in OP.input_tensors:
#                 if tensor.op.input_tensors:
#                     traverse(tensor.op, cfg)
#         elif tag.is_injective(OP.tag):
#             print("is injective")
#             for tensor in OP.input_tensors:
#                 if tensor.op.input_tensors:
#                     traverse(tensor.op, cfg)

#         # schedule conv2d
#         if 'conv2d_nchw' in OP.tag:
#             # temp = OP.input_tensors[0]
#             # Filter = OP.input_tensors[1]
#             # Output = OP.output(0)
#             # schedule_conv2d(temp, Filter, Output)
#             topi.cuda.conv2d.schedule_conv2d_nchw_cuda(cfg, OP.output(0))

#         # schedule depthwise_conv2d
#         if 'depthwise_conv2d_nchw' in OP.tag:
#             # PaddedInput = OP.input_tensors[0]
#             # Filter = OP.input_tensors[1]
#             # DepthwiseConv2d = OP.output(0)
#             # schedule_depth(PaddedInput, Filter, DepthwiseConv2d)
#             topi.cuda.depthwise_conv2d.schedule_depthwise_conv2d_nchw_cuda(cfg, OP.output(0))

#     traverse(outs[0].op, cfg)
#     return s

# def schedule_fused_nhwc(global_outs):
#     cfg = autotvm.get_config()

#     """Create schedule for tensors or return error if batch size is larger than 1"""
#     global_outs = [global_outs] if isinstance(global_outs, tvm.tensor.Tensor) else global_outs
#     s = tvm.create_schedule([x.op for x in global_outs])

#     def schedule_conv2d(OP):
#         Apad = OP.input_tensors[0]
#         W = OP.input_tensors[1]
#         B = OP.output(0)

#         s[Apad].compute_inline()
#         AA = s.cache_read(Apad, "shared", [B])
#         # WW = s.cache_read(W, "shared", [B])
#         # AL = s.cache_read(AA, "local", [B])
#         # WL = s.cache_read(WW, "local", [B])

#         if B.op in s.outputs:
#             Out = B
#             BL = s.cache_write(Out, "local")
#         else:
#             Out = s.outputs[0].output(0)
#             s[B].set_scope("local")
#             BL = B

#         assert tvm.ir_pass.Simplify(W.shape[0]).value == 1 and \
#                 tvm.ir_pass.Simplify(W.shape[1]).value == 1

#         num_thread = tvm.ir_pass.Simplify(Apad.shape[3]).value

#         ######################## 1.06 ms
#         # block_x = tvm.thread_axis("blockIdx.x")
#         # thread_x = tvm.thread_axis((0, num_thread), "threadIdx.x")

#         # ni, hi, wi, ci = s[Out].op.axis
#         # s[BL].compute_at(s[Out], ci)

#         # wi = s[Out].fuse(hi, wi)

#         # s[Out].bind(wi, block_x)
#         # s[Out].bind(ci, thread_x)

#         ####################### 145.21
#         vthread = 2

#         block_x = tvm.thread_axis("blockIdx.x")
#         thread_x = tvm.thread_axis((0, num_thread), "threadIdx.x")
#         thread_xz = tvm.thread_axis((0, vthread), "vthread", name="vx")
#         thread_yz = tvm.thread_axis((0, vthread), "vthread", name="vy")

#         ni, hi, wi, ci = s[Out].op.axis
#         hia, hib = s[Out].split(hi, factor=vthread)
#         wia, wib = s[Out].split(wi, factor=vthread)
#         s[Out].reorder(ni, hia, wia, hib, wib, ci)
#         hw = s[Out].fuse(hia, wia)
#         s[Out].bind(hw, block_x)
#         s[Out].bind(hib, thread_yz)
#         s[Out].bind(wib, thread_xz)
#         s[Out].bind(ci, thread_x)

#         s[BL].compute_at(s[Out], ci)
#         s[AA].compute_at(s[Out], ci)

#         ni, hi, wi, ci = s[AA].op.axis
#         s[AA].bind(ci, thread_x)

#     def schedule_depth(OP):
#         temp = OP.input_tensors[0]
#         Filter = OP.input_tensors[1]
#         DepthwiseConv2d = OP.output(0)

#         # num_thread here could be 728, it is larger than cuda.max_num_threads
#         target = tvm.target.current_target()
#         if target and target.target_name != "cuda":
#             num_thread = target.max_num_threads
#         num_thread = tvm.ir_pass.Simplify(temp.shape[3]).value

#         s[temp].compute_inline()
#         # # input reuse
#         AS = s.cache_read(temp, "shared", [DepthwiseConv2d])
#         # # filter reuse
#         # if num_thread < 512:
#         #     FS = s.cache_read(Filter, "shared", [DepthwiseConv2d])

#         if DepthwiseConv2d.op in s.outputs:
#             Output = DepthwiseConv2d
#             CL = s.cache_write(DepthwiseConv2d, "local")
#         else:
#             Output = outs[0].op.output(0)
#             s[DepthwiseConv2d].set_scope("local")

#         block_x = tvm.thread_axis("blockIdx.x")
#         thread_x = tvm.thread_axis("threadIdx.x")

#         b, h, w, c = s[Output].op.axis

#         xoc, xic = s[Output].split(c, factor=num_thread)
#         s[Output].reorder(xoc, b, h, w, xic)
#         xo, yo, _, _ = s[Output].tile(h, w, x_factor=2, y_factor=2)
#         fused = s[Output].fuse(yo, xo)
#         fused = s[Output].fuse(fused, b)
#         fused = s[Output].fuse(fused, xoc)

#         s[Output].bind(fused, block_x)
#         s[Output].bind(xic, thread_x)

#         if DepthwiseConv2d.op in s.outputs:
#             s[CL].compute_at(s[Output], xic)
#         else:
#             s[DepthwiseConv2d].compute_at(s[Output], xic)

#         # # input reuse
#         s[AS].compute_at(s[Output], fused)
#         b, h, w, c = s[AS].op.axis
#         s[AS].reorder(h, w, b, c)
#         fused_as = s[AS].fuse(b, c)
#         s[AS].bind(fused_as, thread_x)

#         # filter reuse
#         # if num_thread < 512:
#         #     _, _, ci, fi = s[FS].op.axis
#         #     s[FS].compute_at(s[Output], fused)
#         #     fused = s[FS].fuse(fi, ci)
#         #     s[FS].bind(fused, thread_x)

#     def traverse(OP, cfg):
#         """Traverse operators from computation graph"""
#         # inline all one-to-one-mapping operators except the last stage (output)
#         print("**********")
#         print(OP.input_tensors)
#         print(tag.is_broadcast(OP.tag))
#         print(OP.tag)

#         if tag.is_broadcast(OP.tag):
#             print("is broadcast")
#             if OP not in s.outputs:
#                 s[OP].compute_inline()
#         elif tag.is_injective(OP.tag):
#             print("is injective")

#         # schedule depthwise_conv2d
#         if 'depthwise_conv2d_nhwc' in OP.tag:
#             schedule_depth(OP)
#         # schedule conv2d
#         elif 'conv2d_nhwc' in OP.tag:
#             schedule_conv2d(OP)

#         for tensor in OP.input_tensors:
#             if tensor.op.input_tensors:
#                 traverse(tensor.op, cfg)

#     traverse(global_outs[0].op, cfg)
#     return s

# # @register_fused.register(["cuda", "gpu"])
# def schedule_depthwise_1by1_fused(outs, layout="NCHW"):
#     """Schedule for depthwise + 1by1 fused operator.

#     Parameters
#     ----------
#     outs: Array of Tensor
#         The computation graph description of depthwise + 1by1
#         in the format of an array of tensors.

#     Returns
#     -------
#     s: Schedule
#         The computation schedule for depthwise + 1by1.
#     """
#     target = tvm.target.current_target()
#     assert target.target_name == "cuda"

#     outs = [outs] if isinstance(outs, tvm.tensor.Tensor) else outs
#     batch_size = util.get_const_int(outs[0].op.output(0).shape[0])
#     if batch_size > 1:
#         raise RuntimeError("Batch size: %d is too large for this schedule" % batch_size)
#     return schedule_fused_nchw(outs) if layout == "NCHW" else schedule_fused_nhwc(outs)


# @register_fused.register(["cuda", "gpu"])
def schedule_depthwise_conv2d_nhwc_reuse(outs, A=None):
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
    def _schedule(temp, Filter, DepthwiseConv2d):
        # num_thread here could be 728, it is larger than cuda.max_num_threads
        target = tvm.target.current_target()
        if target and target.target_name != "cuda":
            num_thread = target.max_num_threads
        num_thread = tvm.ir_pass.Simplify(temp.shape[3]).value

        s[temp].compute_inline()
        # # input reuse
        AS = s.cache_read(temp, "shared", [DepthwiseConv2d])
        # # filter reuse
        # if num_thread < 512:
        #     FS = s.cache_read(Filter, "shared", [DepthwiseConv2d])

        if DepthwiseConv2d.op in s.outputs:
            Output = DepthwiseConv2d
            CL = s.cache_write(DepthwiseConv2d, "local")
        else:
            Output = outs[0].op.output(0)
            s[DepthwiseConv2d].set_scope("local")

        block_x = tvm.thread_axis("blockIdx.x")
        thread_x = tvm.thread_axis("threadIdx.x")

        b, h, w, c = s[Output].op.axis

        xoc, xic = s[Output].split(c, factor=num_thread)
        s[Output].reorder(xoc, b, h, w, xic)
        xo, yo, _, _ = s[Output].tile(h, w, x_factor=2, y_factor=2)
        fused = s[Output].fuse(yo, xo)
        fused = s[Output].fuse(fused, b)
        fused = s[Output].fuse(fused, xoc)

        s[Output].bind(fused, block_x)
        s[Output].bind(xic, thread_x)

        if DepthwiseConv2d.op in s.outputs:
            s[CL].compute_at(s[Output], xic)
        else:
            s[DepthwiseConv2d].compute_at(s[Output], xic)

        # # input reuse
        s[AS].compute_at(s[Output], fused)
        b, h, w, c = s[AS].op.axis
        s[AS].reorder(h, w, b, c)
        fused_as = s[AS].fuse(b, c)
        s[AS].bind(fused_as, thread_x)

        # filter reuse
        # if num_thread < 512:
        #     _, _, ci, fi = s[FS].op.axis
        #     s[FS].compute_at(s[Output], fused)
        #     fused = s[FS].fuse(fi, ci)
        #     s[FS].bind(fused, thread_x)

        if A is not None:
            print(tvm.lower(s, [A, Filter, Output], simple_mode=True))

    def traverse(OP):
        # inline all one-to-one-mapping operators except the last stage (output)
        if tag.is_broadcast(OP.tag):
            if OP not in s.outputs:
                s[OP].compute_inline()
            for tensor in OP.input_tensors:
                if tensor.op.input_tensors:
                    traverse(tensor.op)
        # schedule depthwise_conv2d
        if OP.tag == 'depthwise_conv2d_nhwc':
            PaddedInput = OP.input_tensors[0]
            Filter = OP.input_tensors[1]
            DepthwiseConv2d = OP.output(0)
            _schedule(PaddedInput, Filter, DepthwiseConv2d)

    traverse(outs[0].op)
    return s

@autotvm.template
def schedule_depthwise_conv2d_nhwc_reuse_auto(batch, in_channel, in_size, channel_multiplier, kernel, stride, padding="SAME", dtype="float32", input_reuse=True, filter_reuse=False):
    in_height = in_width = in_size
    filter_channel = in_channel
    filter_width = filter_height = kernel
    stride_w = stride_h = stride

    # Placeholders
    Input = tvm.placeholder((batch, in_height, in_width, in_channel), name='Input')
    Filter = tvm.placeholder((filter_height, filter_width, filter_channel, channel_multiplier), name='Filter')
    # Scale = tvm.placeholder((in_channel * channel_multiplier,), name='Scale')
    # Shift = tvm.placeholder((in_channel * channel_multiplier,), name='Shift')

    # Computes
    DepthwiseConv2d = topi.nn.depthwise_conv2d_nhwc(Input, Filter, stride=[stride_h, stride_w], padding=padding, dilation=1)
    # ScaleShift = topi.nn.scale_shift_nhwc(DepthwiseConv2d, Scale, Shift)
    # Relu = topi.nn.relu(ScaleShift)
    temp = DepthwiseConv2d.op.input_tensors[0]

    # Create schedule
    s = tvm.create_schedule([DepthwiseConv2d.op])

    # schedule
    # num_thread here could be 728, it is larger than cuda.max_num_threads
    target = tvm.target.current_target()
    if target and target.target_name != "cuda":
        num_thread = target.max_num_threads
    num_thread = tvm.ir_pass.Simplify(temp.shape[3]).value

    s[temp].compute_inline()
    # Input reuse and filter reuse
    if input_reuse:
        AS = s.cache_read(temp, "shared", [DepthwiseConv2d])
    if filter_reuse and num_thread < 512:
        FS = s.cache_read(Filter, "shared", [DepthwiseConv2d])

    if DepthwiseConv2d.op in s.outputs:
        Output = DepthwiseConv2d
        CL = s.cache_write(DepthwiseConv2d, "local")
    else:
        Output = outs[0].op.output(0)
        s[DepthwiseConv2d].set_scope("local")

    block_x = tvm.thread_axis("blockIdx.x")
    thread_x = tvm.thread_axis("threadIdx.x")

    b, h, w, c = s[Output].op.axis

    cfg = autotvm.get_config()
    # Tune splitting of hi and wi
    cfg.define_split("split_c", c, num_outputs=2)

    # Apply splitting
    xoc, xic = cfg["split_c"].apply(s, Output, c)
    s[Output].reorder(xoc, b, h, w, xic)
    xo, yo, _, _ = s[Output].tile(h, w, x_factor=2, y_factor=2)
    fused = s[Output].fuse(yo, xo)
    fused = s[Output].fuse(fused, b)
    fused = s[Output].fuse(fused, xoc)

    s[Output].bind(fused, block_x)
    s[Output].bind(xic, thread_x)

    if DepthwiseConv2d.op in s.outputs:
        s[CL].compute_at(s[Output], xic)
    else:
        s[DepthwiseConv2d].compute_at(s[Output], xic)

    if input_reuse:
        s[AS].compute_at(s[Output], fused)
        b, h, w, c = s[AS].op.axis
        s[AS].reorder(h, w, b, c)
        fused_as = s[AS].fuse(b, c)
        s[AS].bind(fused_as, thread_x)

    if filter_reuse and num_thread < 512:
        _, _, ci, fi = s[FS].op.axis
        s[FS].compute_at(s[Output], fused)
        fused = s[FS].fuse(fi, ci)
        s[FS].bind(fused, thread_x)

    # print(tvm.lower(s, [A, Filter, Output], simple_mode=True))

    return s, [Input, Filter, DepthwiseConv2d]

# @register_fused.register(["cuda", "gpu"])
def schedule_conv2d_nhwc(outs, A=None):
    """Schedule for conv2d_nhwc and any element-wise operations.

    Parameters
    ----------
    outs: Array of Tensor
        The computation graph description of conv2d_hwcn in the format
        of an array of tensors.

    Returns
    -------
    s: Schedule
        The computation schedule for conv2d_nhwc.
    """

    # Apad: NHWC, Filter: HWIO

    outs = [outs] if isinstance(outs, tvm.tensor.Tensor) else outs
    s = tvm.create_schedule([x.op for x in outs])
    def _schedule(Apad, W, B):
        """Schedule conv2d_hwcn"""
        s[Apad].compute_inline()
        AA = s.cache_read(Apad, "shared", [B])
        WW = s.cache_read(W, "shared", [B])
        AL = s.cache_read(AA, "local", [B])
        WL = s.cache_read(WW, "local", [B])

        if B.op in s.outputs:
            Out = B
            BL = s.cache_write(Out, "local")
        else:
            Out = s.outputs[0].output(0)
            s[B].set_scope("local")
            BL = B

        assert tvm.ir_pass.Simplify(W.shape[0]).value == 1 and \
                tvm.ir_pass.Simplify(W.shape[1]).value == 1

        ######################## 1.06 ms
        # num_thread = tvm.ir_pass.Simplify(Apad.shape[3]).value
        # block_x = tvm.thread_axis("blockIdx.x")
        # thread_x = tvm.thread_axis((0, num_thread), "threadIdx.x")

        # ni, hi, wi, ci = s[Out].op.axis
        # s[BL].compute_at(s[Out], ci)

        # wi = s[Out].fuse(hi, wi)
        # # wi = s[Out].fuse(ni, wi)

        # s[Out].bind(wi, block_x)
        # s[Out].bind(ci, thread_x)

        ####################### 145.21
        # num_thread = tvm.ir_pass.Simplify(Apad.shape[3]).value

        # vthread = 2

        # block_x = tvm.thread_axis("blockIdx.x")
        # thread_x = tvm.thread_axis((0, num_thread), "threadIdx.x")
        # thread_xz = tvm.thread_axis((0, vthread), "vthread", name="vx")
        # thread_yz = tvm.thread_axis((0, vthread), "vthread", name="vy")

        # ni, hi, wi, ci = s[Out].op.axis
        # hia, hib = s[Out].split(hi, factor=vthread)
        # wia, wib = s[Out].split(wi, factor=vthread)
        # s[Out].reorder(ni, hia, wia, hib, wib, ci)
        # hw = s[Out].fuse(hia, wia)
        # s[Out].bind(hw, block_x)
        # s[Out].bind(hib, thread_yz)
        # s[Out].bind(wib, thread_xz)
        # s[Out].bind(ci, thread_x)

        # s[BL].compute_at(s[Out], ci)
        # s[AA].compute_at(s[Out], ci)

        # ni, hi, wi, ci = s[AA].op.axis
        # s[AA].bind(ci, thread_x)

        #######################
        output_tile = 2
        num_thread_x = 32
        num_thread_y = output_tile * output_tile
        # vthread = 2

        block_x = tvm.thread_axis("blockIdx.x")
        thread_x = tvm.thread_axis((0, num_thread_x), "threadIdx.x")
        thread_y = tvm.thread_axis((0, num_thread_y), "threadIdx.y")
        # thread_xz = tvm.thread_axis((0, vthread), "vthread", name="vx")
        # thread_yz = tvm.thread_axis((0, vthread), "vthread", name="vy")

        ni, hi, wi, ci = s[Out].op.axis
        cio, cii = s[Out].split(ci, factor=num_thread_x)
        yo, xo, yi, xi = s[Out].tile(hi, wi, x_factor=output_tile, y_factor=output_tile)
        s[Out].reorder(ni, yo, xo, cio, yi, xi, cii)
        fused_o = s[Out].fuse(yi, xi)
        s[Out].bind(fused_o, thread_y)
        s[Out].bind(cii, thread_x)
        fused_b = s[Out].fuse(ni, yo, xo, cio)
        s[Out].bind(fused_b, block_x)

        # # Local
        s[BL].compute_at(s[Out], cii)
        hl, wl, il, ol = s[BL].op.axis
        ry, rx, rc = s[BL].op.reduce_axis
        rco, rci = s[BL].split(rc, factor=64)
        s[BL].reorder(hl, wl, rco, ry, rx, rci, il, ol)

        # Input
        s[AA].compute_at(s[BL], rx)
        s[AL].compute_at(s[BL], rci)
        n, h, w, c = s[AA].op.axis
        hw = s[AA].fuse(h, w)
        tx, c = s[AA].split(c, nparts=num_thread_x)
        ty, hw = s[AA].split(hw, nparts=num_thread_y)
        _, c = s[AA].split(c, factor=4)
        s[AA].reorder(ty, tx, n, hw, c)
        s[AA].bind(ty, thread_y)
        s[AA].bind(tx, thread_x)
        s[AA].vectorize(c)

        # Filter
        s[WW].compute_at(s[BL], rx)
        s[WL].compute_at(s[BL], rci)
        h, w, ic, oc = s[WW].op.axis
        tx, oc = s[WW].split(oc, nparts=num_thread_x)
        ty, ic = s[WW].split(ic, nparts=num_thread_y)
        _, oc = s[WW].split(oc, factor=4)
        s[WW].reorder(ty, tx, h, w, ic, oc)
        s[WW].bind(ty, thread_y)
        s[WW].bind(tx, thread_x)
        s[WW].vectorize(oc)

        if A is not None:
            print(tvm.lower(s, [A, W, Out], simple_mode=True))

    def traverse(operator):
        """Traverse operators from computation graph"""
        if tag.is_broadcast(operator.tag):
            if operator not in s.outputs:
                s[operator].compute_inline()
            for tensor in operator.input_tensors:
                if tensor.op.input_tensors:
                    traverse(tensor.op)
        elif operator.tag == 'conv2d_nhwc':
            Apad = operator.input_tensors[0]
            W = operator.input_tensors[1]
            B = operator.output(0)
            print(Apad, W, B)
            _schedule(Apad, W, B)
        else:
            raise RuntimeError("Unsupported operator: %s" % operator.tag)

    traverse(outs[0].op)
    return s

@autotvm.template
def schedule_conv2d_nhwc_auto(batch, in_channel, in_size, num_filter, kernel, stride, padding="SAME", dtype="float32"):
    in_height = in_width = in_size

    A = tvm.placeholder((batch, in_height, in_width, in_channel), name='Apad')
    W = tvm.placeholder((kernel, kernel, in_channel, num_filter), name='W')
    B = topi.nn.conv2d_nhwc(A, W, stride, padding, dilation=1)
    Apad = B.op.input_tensors[0]

    # Create schedule
    sch = tvm.create_schedule([B.op])

    sch[Apad].compute_inline()
    AA = sch.cache_read(Apad, "shared", [B])

    if B.op in sch.outputs:
        Out = B
        BL = sch.cache_write(Out, "local")
    else:
        Out = sch.outputs[0].output(0)
        sch[B].set_scope("local")
        BL = B

    assert tvm.ir_pass.Simplify(W.shape[0]).value == 1 and \
            tvm.ir_pass.Simplify(W.shape[1]).value == 1

    ni, hi, wi, ci = sch[Out].op.axis

    # Auto tuning configuration
    cfg = autotvm.get_config()
    # Tune splitting of hi and wi
    cfg.define_split("tile_hi", hi, num_outputs=2)
    cfg.define_split("tile_wi", wi, num_outputs=2)

    thread_x = tvm.thread_axis("threadIdx.x")

    # Apply splitting
    hia, hib = cfg["tile_hi"].apply(sch, Out, hi)
    wia, wib = cfg["tile_wi"].apply(sch, Out, wi)

    # cfg.define_reorder("reorder", [ni, hia, wia, hib, wib, ci], "all")
    # reordered = cfg["reorder"].apply(sch, Out, [ni, hia, wia, hib, wib, ci])
    # print(reordered)
    sch[Out].reorder(ni, hia, wia, hib, wib, ci)
    hw = sch[Out].fuse(hia, wia)
    sch[Out].bind(hw, tvm.thread_axis("blockIdx.x"))
    sch[Out].bind(hib, tvm.thread_axis("vthread"))
    sch[Out].bind(wib, tvm.thread_axis("vthread"))
    sch[Out].bind(ci, thread_x)

    sch[BL].compute_at(sch[Out], ci)
    
    sch[AA].compute_at(sch[Out], ci)
    ni, hi, wi, ci = sch[AA].op.axis
    sch[AA].bind(ci, thread_x)

    # print(tvm.lower(sch, [A, W, Out], simple_mode=True))

    return sch, [A, W, B]

