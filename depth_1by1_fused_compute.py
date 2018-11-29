# pylint: disable=invalid-name, unused-variable, too-many-locals
"""Depthwise convolution operators"""
from __future__ import absolute_import as _abs
import tvm

from topi.nn.dilate import dilate
from topi.nn.pad import pad
from topi.nn.util import get_pad_tuple
from topi.util import simplify

# def depth_1by1_fused_nhwc(  Input, 
#                             Filter_d, Filter_1, stride_d, stride_1, 
#                             padding_d='SAME', padding_1='SAME', dilation_d=1, dilation_1=1, out_dtype=None):
#     """Fused depthwise convolution + 1x1 convolution forward operator (NHWC).

#     Parameters
#     ----------
#     Input : tvm.Tensor
#         4-D with shape [batch, in_height, in_width, in_channel]

#     Filter_d : tvm.Tensor
#         4-D with shape [filter_height, filter_width, in_channel, in_channel * channel_multiplier]

#     Filter_1 : tvm.Tensor
#         4-D with shape [filter_height, filter_width, in_channel * channel_multiplier, out_channel]

#     stride_d : tuple of two ints
#         The spatial stride along height and width

#     stride_1 : int or a list/tuple of two ints
#         Stride size, or [stride_height, stride_width]

#     padding_d : int or str
#         Padding size, or ['VALID', 'SAME']

#     padding_1 : int or str
#         Padding size, or ['VALID', 'SAME']

#     dilation_d: int or a list/tuple of two ints
#         dilation size, or [dilation_height, dilation_width]

#     dilation_1: int or a list/tuple of two ints
#         dilation size, or [dilation_height, dilation_width]

#     out_dtype: str, optional
#         Output data type

#     Returns
#     -------
#     output : tvm.Tensor
#         4-D with shape [batch, out_height, out_width, out_channel]

#     Returns
#     -------
#     Output : tvm.Tensor
#         4-D with shape [batch, out_height, out_width, out_channel]
#     """

#     """ Depthwise Convolution """
#     out_dtype = Input.dtype if out_dtype is None else out_dtype

#     if isinstance(stride_d, int):
#         stride_h = stride_w = stride_d
#     else:
#         stride_h, stride_w = stride_d

#     if isinstance(dilation_d, int):
#         dilation_h = dilation_w = dilation_d
#     else:
#         dilation_h, dilation_w = dilation_d

#     if dilation_h != 1 or dilation_w != 1:
#         Filter_d = dilate(Filter_d, (dilation_h, dilation_w, 1, 1))

#     batch, in_height, in_width, in_channel = Input.shape
#     # shape of dilated kernel
#     filter_height, filter_width, filter_channel, channel_multiplier = Filter_d.shape

#     pad_top, pad_left, pad_down, pad_right = get_pad_tuple(
#         padding_d, (filter_height, filter_width))
#     out_channel = simplify(in_channel * channel_multiplier)
#     out_height = simplify((in_height - filter_height + pad_top + pad_down) // stride_h + 1)
#     out_width = simplify((in_width - filter_width + pad_left + pad_right) // stride_w + 1)

#     # padding stage
#     pad_before = [0, pad_top, pad_left, 0]
#     pad_after = [0, pad_down, pad_right, 0]
#     PaddedInput = pad(Input, pad_before, pad_after, name="PaddedInput")
#     # depthconv stage
#     di = tvm.reduce_axis((0, filter_height), name='di')
#     dj = tvm.reduce_axis((0, filter_width), name='dj')
#     Output_d = tvm.compute(
#         (batch, out_height, out_width, out_channel),
#         lambda b, i, j, c: tvm.sum(
#             (PaddedInput[b, i*stride_h + di, j*stride_w + dj, c/channel_multiplier].astype(
#                 out_dtype) *
#              Filter_d[di, dj, c/channel_multiplier, c%channel_multiplier].astype(out_dtype)),
#             axis=[di, dj]),
#         name='DepthwiseConv2d', tag="depthwise_conv2d_nhwc")
    

#     """ 1x1 convolution """
#     assert isinstance(stride_1, int) or len(stride_1) == 2
#     assert isinstance(dilation_1, int) or len(dilation_1) == 2

#     if isinstance(stride_1, int):
#         stride_h = stride_w = stride_1
#     else:
#         stride_h, stride_w = stride_1

#     if isinstance(dilation_1, int):
#         dilation_h = dilation_w = dilation_1
#     else:
#         dilation_h, dilation_w = dilation_1

#     if dilation_h != 1 or dilation_w != 1:
#         Filter_1 = dilate(Filter_1, (dilation_h, dilation_w, 1, 1))

#     batch, in_height, in_width, in_channel = Output_d.shape
#     kernel_h, kernel_w, channel, num_filter = Filter_1.shape
#     pad_top, pad_left, pad_down, pad_right = get_pad_tuple(
#         padding_1, (kernel_h, kernel_w))
#     # compute the output shape
#     out_channel = num_filter
#     out_height = simplify((in_height - kernel_h + pad_top + pad_down) // stride_h + 1)
#     out_width = simplify((in_width - kernel_w + pad_left + pad_right) // stride_w + 1)
#     pad_before = [0, pad_top, pad_left, 0]
#     pad_after = [0, pad_down, pad_right, 0]

#     PaddedInput = pad(Output_d, pad_before, pad_after, name="PaddedInput")
#     rc = tvm.reduce_axis((0, in_channel), name='rc')
#     ry = tvm.reduce_axis((0, kernel_h), name='ry')
#     rx = tvm.reduce_axis((0, kernel_w), name='rx')
#     Output = tvm.compute(
#         (batch, out_height, out_width, out_channel),
#         lambda nn, yy, xx, ff: tvm.sum(
#             PaddedInput[nn, yy * stride_h + ry, xx * stride_w + rx, rc].astype(out_dtype) *
#             Filter_1[ry, rx, rc, ff].astype(out_dtype), axis=[ry, rx, rc]),
#         name="Conv2dOutput", tag="conv2d_nhwc")
#     return Output

def depth_1by1_fused_nhwc(  Input, 
                            Filter_d, Filter_1, stride_d, 
                            padding_d='SAME', dilation_d=1, out_dtype=None):
    """Fused depthwise convolution + 1x1 convolution forward operator (NHWC).

    Parameters
    ----------
    Input : tvm.Tensor
        4-D with shape [batch, in_height, in_width, in_channel]

    Filter_d : tvm.Tensor
        4-D with shape [filter_height, filter_width, in_channel, in_channel * channel_multiplier]

    stride_d : tuple of two ints
        The spatial stride along height and width

    padding_d : int or str
        Padding size, or ['VALID', 'SAME']

    dilation_d: int or a list/tuple of two ints
        dilation size, or [dilation_height, dilation_width]

    Filter_1 : tvm.Tensor
        4-D with shape [filter_height, filter_width, in_channel * channel_multiplier, out_channel]

    out_dtype: str, optional
        Output data type

    Returns
    -------
    output : tvm.Tensor
        4-D with shape [batch, out_height, out_width, out_channel]

    Returns
    -------
    Output : tvm.Tensor
        4-D with shape [batch, out_height, out_width, out_channel]
    """

    """ Depthwise Convolution """
    out_dtype = Input.dtype if out_dtype is None else out_dtype

    if isinstance(stride_d, int):
        stride_h_d = stride_w_d = stride_d
    else:
        stride_h_d, stride_w_d = stride_d

    if isinstance(dilation_d, int):
        dilation_h_d = dilation_w_d = dilation_d
    else:
        dilation_h_d, dilation_w_d = dilation_d

    if dilation_h_d != 1 or dilation_w_d != 1:
        Filter_d = dilate(Filter_d, (dilation_h_d, dilation_w_d, 1, 1))

    batch, in_height_d, in_width_d, in_channel_d = Input.shape
    # shape of dilated kernel
    filter_height, filter_width, filter_channel, _ = Filter_d.shape

    pad_top, pad_left, pad_down, pad_right = get_pad_tuple(
        padding_d, (filter_height, filter_width))
    out_channel = simplify(in_channel_d)
    out_height = simplify((in_height_d - filter_height + pad_top + pad_down) // stride_h_d + 1)
    out_width = simplify((in_width_d - filter_width + pad_left + pad_right) // stride_w_d + 1)

    # padding stage
    pad_before = [0, pad_top, pad_left, 0]
    pad_after = [0, pad_down, pad_right, 0]
    PaddedInput = pad(Input, pad_before, pad_after, name="PaddedInput")
    
    """ 1x1 convolution """
    # batch, in_height, in_width, in_channel = Output_d.shape
    _, _, channel, num_filter = Filter_1.shape
    out_channel = num_filter

    # depthconv stage
    di = tvm.reduce_axis((0, filter_height), name='di')
    dj = tvm.reduce_axis((0, filter_width), name='dj')
    # 1by1 stage
    c = tvm.reduce_axis((0, in_channel_d), name='c')

    Output_d = tvm.compute(
    (batch, out_height, out_width, out_channel),
    lambda b, i, j, f: tvm.sum(
        (PaddedInput[b, i*stride_h_d + di, j*stride_w_d + dj, c].astype(
            out_dtype) * Filter_d[di, dj, c, 0].astype(out_dtype)) * Filter_1[0, 0, c, f],
        axis=[di, dj, c]),
    name='Depthwise1by1Fused', tag="depthwise_1by1_fused_nhwc")

    return Output_d