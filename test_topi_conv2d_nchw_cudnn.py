"""Example code to do convolution."""
import os
import numpy as np
import tvm
import topi
import topi.testing
from tvm.contrib.pickle_memoize import memoize
from topi.util import get_const_tuple


def verify_conv2d_nhwc(batch, in_channel, in_size, num_filter, kernel, stride, padding):
    in_height = in_width = in_size

    # Read as NHWC and convert to NCHW for existing schedule
    A = tvm.placeholder((batch, in_channel, in_height, in_width), name='A')
    W = tvm.placeholder((num_filter, in_channel, kernel, kernel), name='W')
    
    a_shape = get_const_tuple(A.shape)
    w_shape = get_const_tuple(W.shape)
    dtype = A.dtype

    # @memoize("topi.tests.test_topi_conv2d_nhwc.verify_nhwc")
    def get_ref_data():
        a_np = np.random.uniform(size=a_shape).astype(dtype)
        w_np = np.random.uniform(size=w_shape).astype(dtype)
        b_np = topi.testing.conv2d_nchw_python(a_np, w_np, stride, padding)
        return a_np, w_np, b_np
    a_np, w_np, b_np = get_ref_data()

    def check_device(device):
        if not tvm.module.enabled(device.split(' ')[0]):
            print("Skip because %s is not enabled" % device)
            return
        print("Running on target: %s" % device)
        ctx = tvm.context(device.split(' ')[0], 0)
        with tvm.target.create(device):
            B = topi.cuda.conv2d_cuda(A, W, stride, 0, layout='NCHW') # Return NCHW
            s = topi.cuda.schedule_conv2d_nchw([B]) # Borrow NCHW schedule
        
        a = tvm.nd.array(a_np, ctx)
        w = tvm.nd.array(w_np, ctx)
        b = tvm.nd.array(np.zeros(get_const_tuple(B.shape), dtype=B.dtype), ctx)
        func = tvm.build(s, [A, W, B], target="cuda", target_host="llvm")
        func(a, w, b)
        
        np.testing.assert_allclose(b.asnumpy(), b_np, rtol=1e-5)

    for device in ["cuda -libs=cudnn"]:
        check_device(device)


def test_conv2d_nhwc():
    verify_conv2d_nhwc(1, 32, 112, 64, 1, 1, "SAME")
    # verify_conv2d_nhwc(1, 128, 56, 128, 1, 1, "SAME")
    # verify_conv2d_nhwc(1, 256, 28, 256, 1, 1, "SAME")
    # verify_conv2d_nhwc(1, 512, 14, 512, 1, 1, "SAME")

if __name__ == "__main__":
    test_conv2d_nhwc()
