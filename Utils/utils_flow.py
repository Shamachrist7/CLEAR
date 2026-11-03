# Helper functions copied from the official CMRx4DFlow2026 challenge code
# (CMRx4DFlowReconDemo/Utils/utils_flow.py): https://github.com/CmrxRecon/CMRx4DFlow2026
# Only the functions needed by CLEAR are kept.
import numpy as np
import scipy


def k2i_numpy(x, ax=None):
    """
    Convert k-space into image with ifft.

    Args:
        x (np.ndarray):
            K-space array (typically complex-valued).
        ax (list[int] | tuple[int] | None):
            Axes along which to apply the inverse FFT. If None, defaults to [-2, -1].

    Returns:
        np.ndarray:
            Image-domain array with the same shape as x.

    """
    if ax is None:
        ax = [-2, -1]
    return scipy.fft.fftshift(
        scipy.fft.ifftn(
            scipy.fft.ifftshift(x, axes=ax),
            axes=ax,
            norm="ortho",
            workers=-1,
        ),
        axes=ax,
    )
