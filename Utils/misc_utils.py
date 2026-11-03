# Helper functions copied from the official CMRx4DFlow2026 challenge code
# (CMRx4DFlowReconDemo/FlowVN/utils/misc_utils.py): https://github.com/CmrxRecon/CMRx4DFlow2026
# Only the functions needed by CLEAR are kept.
import numpy as np
import torch


def mriAdjointOp(rawdata, sens, mask):
    """ Adjoint operation that convert kspace to coil-combined under-sampled image """
    coil_sens = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(rawdata * mask), norm="ortho"))
    img = np.sum(coil_sens*np.conj(sens), axis=1)
    return img


def fftc2d(x):
    x = torch.fft.ifftshift(x, dim=(-2,-1))
    x = torch.fft.fft2(x, dim=(-2, -1), norm="ortho") 
    x = torch.fft.fftshift(x, dim=[-1,-2])
    return x


def ifftc2d(x):
    x = torch.fft.ifftshift(x, dim=(-2,-1))
    x = torch.fft.ifft2(x, dim=(-2, -1), norm="ortho") 
    x = torch.fft.fftshift(x, dim=[-1,-2])
    return x
