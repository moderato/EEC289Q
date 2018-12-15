# pylint: disable=invalid-name, unused-variable, too-many-locals
"""Depthwise convolution operators"""
from __future__ import absolute_import as _abs
import tvm

from topi.nn.dilate import dilate
from topi.nn.pad import pad
from topi.nn.util import get_pad_tuple
from topi.util import simplify

def depth_1by1_fused(   Input, 
                        Filter_d, Filter_1, stride_d, 
                        padding_d='SAME', dilation_d=1, out_dtype=None, layout="NCHW"):
    """Fused depthwise convolution + 1x1 convolution forward operator (NCHW & NHWC).

    Parameters
    ----------
    Input : tvm.Tensor
        4-D with shape [batch, in_channel, in_height, in_width] (NCHW)
                    or [batch, in_height, in_width, in_channel] (NHWC)

    Filter_d : tvm.Tensor
        4-D with shape [in_channel, in_channel * channel_multiplier, filter_height, filter_width]
                    or [filter_height, filter_width, in_channel, in_channel * channel_multiplier]

    Filter_1 : tvm.Tensor
        4-D with shape [out_channel, in_channel * channel_multiplier, 0, 0]
                    or [0, 0, out_channel, in_channel * channel_multiplier]

    stride_d : tuple of two ints
        The spatial stride along height and width

    padding_d : int or str
        Padding size, or ['VALID', 'SAME']

    dilation_d: int or a list/tuple of two ints
        dilation size, or [dilation_height, dilation_width]

    out_dtype: str, optional
        Output data type

    Returns
    -------
    output : tvm.Tensor
        4-D with shape [batch, out_height, out_width, out_channel]
    """

    assert layout in ["NCHW", "NHWC"]

    out_dtype = Input.dtype if out_dtype is None else out_dtype

    if isinstance(stride_d, int):
        stride_h_d = stride_w_d = stride_d
    else:
        stride_h_d, stride_w_d = stride_d

    if isinstance(dilation_d, int):
        dilation_h_d = dilation_w_d = dilation_d
    else:
        dilation_h_d, dilation_w_d = dilation_d

    if layout == "NCHW":
        if dilation_h_d != 1 or dilation_w_d != 1:
            Filter_d = dilate(Filter_d, (1, 1, dilation_h_d, dilation_w_d))
        batch, in_channel_d, in_height_d, in_width_d = Input.shape
        filter_channel, _, filter_height, filter_width = Filter_d.shape
        num_filter, channel, _, _ = Filter_1.shape
    else: # NHWC
        if dilation_h_d != 1 or dilation_w_d != 1:
            Filter_d = dilate(Filter_d, (dilation_h_d, dilation_w_d, 1, 1))
        batch, in_height_d, in_width_d, in_channel_d = Input.shape
        filter_height, filter_width, filter_channel, _ = Filter_d.shape
        _, _, num_filter, channel= Filter_1.shape

    pad_top, pad_left, pad_down, pad_right = get_pad_tuple(
        padding_d, (filter_height, filter_width))
    out_channel = simplify(in_channel_d)
    out_height = simplify((in_height_d - filter_height + pad_top + pad_down) // stride_h_d + 1)
    out_width = simplify((in_width_d - filter_width + pad_left + pad_right) // stride_w_d + 1)
    out_channel = num_filter

    # padding stage
    if layout == "NCHW":
        pad_before = [0, 0, pad_top, pad_left]
        pad_after = [0, 0, pad_down, pad_right]
    else: # NHWC
        pad_before = [0, pad_top, pad_left, 0]
        pad_after = [0, pad_down, pad_right, 0]
        
    PaddedInput = pad(Input, pad_before, pad_after, name="PaddedInput")

    # depthconv stage
    di = tvm.reduce_axis((0, filter_height), name='di')
    dj = tvm.reduce_axis((0, filter_width), name='dj')
    # 1by1 stage
    c = tvm.reduce_axis((0, out_channel), name='c')

    if layout == "NCHW":
        Output = tvm.compute(
            (batch, out_channel, out_height, out_width),
            lambda b, f, i, j: tvm.sum(
                (PaddedInput[b, c, i*stride_h_d+di, j*stride_w_d+dj].astype(out_dtype) *
                 Filter_d[c, 0, di, dj].astype(out_dtype) * Filter_1[f, c, 0, 0].astype(out_dtype)),
                axis=[di, dj, c]),
            name='Depthwise1by1Fused', tag="depthwise_1by1_fused_nchw")
    else: # NHWC
        Output = tvm.compute(
            (batch, out_height, out_width, out_channel),
            lambda b, i, j, f: tvm.sum(
                (PaddedInput[b, i*stride_h_d + di, j*stride_w_d + dj, c].astype(
                    out_dtype) * Filter_d[di, dj, c, 0].astype(out_dtype) * Filter_1[0, 0, c, f].astype(out_dtype)),
                axis=[di, dj, c]),
            name='Depthwise1by1Fused', tag="depthwise_1by1_fused_nhwc")
    return Output
    