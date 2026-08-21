import math
from typing import Sequence, Tuple, List, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.parametrize as P

from deepinv.optim import Prior
from deepinv.optim import L2
import torchcde


class LinearSpline(nn.Module):
    """
    Learn N functions alpha_i(R) = exp( s_i(1/R) ) * R,
    where s_i is a natural cubic spline over 1/R  (when potential_params=True).
    
    Else, gives out softplus applied to the raw output of the spline values (when potential_params=False).

    Args
    ----
    N : int
        Number of alphas.
    K : int
        Number of spline knots (>=2).
    usrate_min, usrate_max : float
        Range of R where knots are placed.
    init : float
        Initial value for s_i at all knots.
    potential_params: bool
        Wether we want the potential parameters alphas or just a general linear spline utility.
    """

    def __init__(self, N: int = 12, K: int = 5,
                 usrate_min: float = 10.0, usrate_max: float = 50.0,
                 init: float = 0.0, potential_params=True):
        super().__init__()
        assert K >= 2, "Need at least 2 knots."
        assert usrate_max > usrate_min > 0.0
    
        self.potential_params = potential_params
        self.N = N
        self.K = K

        # --- fixed knot locations (registered buffers, no grads) ---
        t = torch.linspace(1.0 / usrate_max, 1.0 / usrate_min, K)
        self.register_buffer("t_knots", t)  # shape [K]

        # --- learnable spline values at knots: s_i(t_k) ---
        # Shape [1, K, N]: batch=1, time=K, channels=N
        self.s_at_knots = nn.Parameter(torch.full((1, K, N), float(init)))

    def _build_spline(self):
        """
        Build a LinearSpline object from current knot values.
        torchcde expects data of shape [B, T, C] with strictly increasing T.
        """
        # coefficients are computed with gradients flowing to s_at_knots
        coeffs = torchcde.linear_interpolation_coeffs(
            self.s_at_knots, t=self.t_knots
        )
        return torchcde.LinearInterpolation(coeffs)

    def forward(self, usrate: torch.Tensor) -> torch.Tensor:
        """
        Evaluate all N alphas at a batch of undersampling rates.

        Parameters
        ----------
        usrate : Tensor, shape [B] or [B,1]
            Per-sample undersampling rates (acceleration factors) (must be > 0).

        Returns
        -------
        alphas : Tensor, shape [B, N]
        """
        usrate = usrate.view(-1)  # [B]
        
        spline = self._build_spline()              # linear spline s(t)
        s_vals = spline.evaluate(1.0 / usrate)           # shape [1, B, N]
        s_vals = s_vals.squeeze(0)                 # -> [B, N]

        if self.potential_params:
            alphas = torch.exp(s_vals) * usrate.view(-1, 1)
            out = alphas.view(len(usrate), self.N, 1, 1, 1) # -> [B, N, 1, 1, 1] for the 3D parameters potentials
        else:
            out = F.softplus(s_vals).view(len(usrate), self.N, 1, 1, 1, 1) # -> [B, N, 1, 1, 1, 1] for flow data
        return out


class ComplexZeroMean3D(nn.Module):
    """
    Enforces zero mean on each complex 3D filter.

    Input weight shape:
        [out_channels, in_channels, kT, kH, kW]
    """

    def forward(self, w: torch.Tensor) -> torch.Tensor:
        return w - w.mean(dim=(1, 2, 3, 4), keepdim=True)


class ComplexConv3dBuiltIn(nn.Module):
    """
    Complex-linear 3D convolution using PyTorch's built-in complex Conv3d.

    If W is complex and x is complex, PyTorch directly computes

        y = W x

    using native complex tensors.

    The adjoint is implemented explicitly as

        W^* y

    using conv_transpose3d with the conjugated complex weight.

    This is the correct complex adjoint for the Hilbert-space inner product.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int, int, int]],
        padding: Union[int, Tuple[int, int, int]],
        bias: bool = False,
        dtype: torch.dtype = torch.complex64,
    ):
        super().__init__()

        if bias:
            raise NotImplementedError("Bias is intentionally disabled for WCRR filters.")

        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size, kernel_size)
        elif not (isinstance(kernel_size, tuple) and len(kernel_size) == 3):
            raise ValueError("kernel_size must be an int or a tuple of length 3.")

        if isinstance(padding, int):
            padding = (padding, padding, padding)
        elif not (isinstance(padding, tuple) and len(padding) == 3):
            raise ValueError("padding must be an int or a tuple of length 3.")

        if any(k % 2 == 0 for k in kernel_size):
            raise ValueError(
                "Use odd kernel sizes so that conv_transpose3d preserves shape cleanly."
            )

        if dtype not in (torch.complex64, torch.complex128):
            raise ValueError(
                "ComplexConv3dBuiltIn requires dtype=torch.complex64 or torch.complex128."
            )

        self.conv = nn.Conv3d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            padding=padding,
            bias=False,
            dtype=dtype,
        )

        self.reset_parameters()

    def reset_parameters(self):
        """
        Complex He-like initialization.

        PyTorch supports complex Conv3d, but it is safer to explicitly initialize
        the real and imaginary parts in a controlled way.
        """

        with torch.no_grad():
            wr = torch.empty_like(self.conv.weight.real)
            wi = torch.empty_like(self.conv.weight.real)

            nn.init.kaiming_normal_(wr, a=math.sqrt(5))
            nn.init.kaiming_normal_(wi, a=math.sqrt(5))

            wr.mul_(1.0 / math.sqrt(2.0))
            wi.mul_(1.0 / math.sqrt(2.0))

            self.conv.weight.copy_(torch.complex(wr, wi))

    @property
    def in_channels(self):
        return self.conv.in_channels

    @property
    def out_channels(self):
        return self.conv.out_channels

    @property
    def kernel_size(self):
        return self.conv.kernel_size

    @property
    def padding(self):
        return self.conv.padding

    @property
    def weight(self):
        return self.conv.weight

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x:
            complex tensor [B, Cin, D, H, W]

        returns:
            complex tensor [B, Cout, D, H, W]
        """

        if not torch.is_complex(x):
            raise TypeError("ComplexConv3dBuiltIn expects a complex input tensor.")

        return self.conv(x)

    def adjoint(self, y: torch.Tensor) -> torch.Tensor:
        """
        Explicit complex adjoint of the convolution.

        y:
            complex tensor [B, Cout, D, H, W]

        returns:
            complex tensor [B, Cin, D, H, W]
        """

        if not torch.is_complex(y):
            raise TypeError("ComplexConv3dBuiltIn.adjoint expects a complex input tensor.")

        return F.conv_transpose3d(
            y,
            self.weight.conj(),
            padding=self.padding,
        )


class ComplexRadialWCRR3D(Prior):
    """
    Complex-radial 3D WCRR.

    Input:
        x complex tensor [B, C, D, H, W]

    Regularizer:

        R(x) = sum_j sum_n 1/s_j^2 [
                    smooth_l1(beta s_j |K_j x|_eps) / beta
                    - weak_cvx smooth_l1(s_j |K_j x|_eps)
               ]

    where

        |z|_eps = sqrt(Re(z)^2 + Im(z)^2 + eps^2).

    The explicit gradient is

        grad R(x)
        =
        K^* [
            ((h'(beta s r) - weak_cvx h'(s r)) / (s r)) * Kx
        ]

    with

        r = sqrt(|Kx|^2 + eps^2).
    """

    def __init__(
        self,
        weak_convexity: float,
        nb_channels: Sequence[int] = (1, 8, 16, 32),
        filter_sizes: Sequence[int] = [(5, 5, 5), (5, 5, 5), (5, 5, 5)], # Layer-wise
        radial_eps: float = 1e-6,
        detach_lipschitz: bool = True,
        cache_lipschitz: bool = True,
        dtype: torch.dtype = torch.complex64,
        usrate_min: float = 10.0,
        usrate_max: float = 50.0,
        nknots: int = 5,
    ):
        super().__init__()

        if len(nb_channels) != len(filter_sizes) + 1:
            raise ValueError("Expected len(nb_channels) == len(filter_sizes) + 1.")

        if dtype not in (torch.complex64, torch.complex128):
            raise ValueError("dtype must be torch.complex64 or torch.complex128.")
        
        if type(filter_sizes[0]) == int:
            if any(k % 2 == 0 for k in filter_sizes):
                raise ValueError("All filter sizes should be odd.")
            self.filter_size = (sum(filter_sizes) - len(filter_sizes) + 1,)*(len(nb_channels)-1)
        else:
            if any(k % 2 == 0 for k in filter_sizes[0]):
                raise ValueError("All filter sizes should be odd.")
            self.filter_size = tuple(sum(filter_sizes[i][j] for i in range(len(filter_sizes))) - len(filter_sizes) + 1 for j in range(len(filter_sizes[0])))


        self.nb_filters = nb_channels[-1]
        self.filter_sizes = tuple(filter_sizes)
        self.radial_eps = radial_eps
        self.detach_lipschitz = detach_lipschitz
        self.cache_lipschitz = cache_lipschitz
        self.complex_dtype = dtype

        layers = []
        for i, k in enumerate(filter_sizes):
            padding = k//2 if type(k)==int else tuple(ki//2 for ki in k)
            layers.append(
                ComplexConv3dBuiltIn(
                    in_channels=nb_channels[i],
                    out_channels=nb_channels[i + 1],
                    kernel_size=k,
                    padding=padding,
                    bias=False,
                    dtype=dtype,
                )
            )

        self.filters = nn.ModuleList(layers)

        # Zero-mean constraint on the first complex filter.
        P.register_parametrization(self.filters[0].conv, "weight", ComplexZeroMean3D())

        # Same role as your original alphas/scaling.
        # scale = exp(alphas), one scale per output feature channel.
        self.scaling = LinearSpline(N = self.nb_filters, K = nknots, usrate_min = usrate_min, usrate_max = usrate_max) #nn.Parameter(torch.zeros(1, self.nb_filters, 1, 1, 1))

        # Same beta parameterization as your original class.
        self.beta = nn.Parameter(torch.tensor(1.0)) 

        self.register_buffer("weak_cvx", torch.tensor(float(weak_convexity)))

        # Cached Lipschitz constant.
        # This is deliberately non-persistent because it can be recomputed after loading weights.
        self.register_buffer("_cached_lip", torch.tensor(float("nan")), persistent=False)

        # Python-side cache flag avoids a CUDA synchronization from torch.isfinite(...).item()
        # inside every reconstruction iteration.
        self._lip_cache_valid = False
        
        # Dirac impulse for computing the Lipschitz constant.
        cin = self.filters[0].in_channels
        sz = tuple(self.filter_size[i] for i in range(len(self.filter_size)))
        center = tuple(self.filter_size[i] // 2 + 1 for i in range(len(self.filter_size)))
        # Batch dimension enumerates input channels.
        # dirac[m, m, center, center, center] = 1
        self.dirac = torch.zeros(
            cin,
            cin,
            sz[0],
            sz[1],
            sz[2],
            device=self.filters[0].weight.device,
            dtype=torch.complex64,
        )

        for c in range(cin):
            self.dirac[c, c, center[0], center[1], center[2]] = 1.0 + 0.0j

    @staticmethod
    def smooth_l1(x: torch.Tensor) -> torch.Tensor:
        """
        Standard smooth-L1 / Huber-like penalty:

            0.5 x^2        if |x| <= 1
            |x| - 0.5      otherwise
        """

        abs_x = x.abs()
        return torch.where(abs_x <= 1.0, 0.5 * x.square(), abs_x - 0.5)

    @staticmethod
    def grad_smooth_l1(x: torch.Tensor) -> torch.Tensor:
        return torch.clamp(x, min=-1.0, max=1.0)

    def _apply_filters(self, x: torch.Tensor) -> torch.Tensor:
        out = x
        for layer in self.filters:
            out = layer(out)
        return out

    def _apply_adjoint_filters(self, x: torch.Tensor) -> torch.Tensor:
        out = x
        for layer in reversed(self.filters):
            out = layer.adjoint(out)
        return out

    def _real_dtype_for_lip(self) -> torch.dtype:
        dtype = self.filters[0].weight.real.dtype
        if dtype in (torch.float16, torch.bfloat16):
            return torch.float32
        return dtype

    def _complex_dtype_for_lip(self) -> torch.dtype:
        dtype = self._real_dtype_for_lip()
        if dtype == torch.float64:
            return torch.complex128
        return torch.complex64

    def _compute_conv_lip(self) -> torch.Tensor:
        """
        Computes the complex convolutional Lipschitz constant.

        For the full cascade K, at each spatial frequency omega we build the
        complex matrix H(omega) with shape

            [Cout, Cin].

        Then

            ||K||^2 = max_omega sigma_max(H(omega))^2.

        This is the appropriate complex multi-channel analogue of the scalar FFT
        normalization. It is stricter and cleaner than taking abs().max() of the
        impulse response.
        """

        device = self.filters[0].weight.device
        complex_dtype = self._complex_dtype_for_lip()

        cin = self.filters[0].in_channels

        sz = tuple(self.filter_size[i] for i in range(len(self.filter_size)))
        center = tuple(self.filter_size[i]//2 + 1 for i in range(len(self.filter_size)))

        # Batch dimension enumerates input channels.
        # dirac[m, m, center, center, center] = 1
        dirac = torch.zeros(
            cin,
            cin,
            sz[0],
            sz[1],
            sz[2],
            device=device,
            dtype=complex_dtype,
        )

        for c in range(cin):
            dirac[c, c, center[0], center[1], center[2]] = 1.0 + 0.0j

        impulse = self._apply_filters(dirac)
        # impulse shape: [Cin, Cout, sz, sz, sz]
        # impulse[input_channel, output_channel, ...]

        H = torch.fft.fftn(impulse, dim=(-3, -2, -1))
        # [Cin, Cout, D, H, W] -> [D, H, W, Cout, Cin]
        H = H.permute(2, 3, 4, 1, 0).contiguous()

        # Batched singular values over all spatial frequencies.
        svals = torch.linalg.svdvals(H)
        lip = svals.amax().square().real

        return lip.clamp_min(1e-12)

    def has_lipschitz_cache(self) -> bool:
        """
        Returns True if the cached Lipschitz constant is currently valid.
        """

        return self._lip_cache_valid

    def clear_lipschitz_cache(self):
        """
        Clears the cached Lipschitz constant.

        Use this if the filters have changed and you want the next call to recompute it.
        """

        self._cached_lip = torch.tensor(
            float("nan"),
            device=self._cached_lip.device,
            dtype=self._cached_lip.dtype,
        )
        self._lip_cache_valid = False

    def update_lipschitz_cache(self) -> torch.Tensor:
        """
        Recompute and cache the convolutional Lipschitz constant.

        Call this after changing/training/loading the filters.
        During reconstruction with frozen filters, do not recompute it every iteration.
        """

        with torch.no_grad():
            lip = self._compute_conv_lip()

        self._cached_lip = lip.detach()
        self._lip_cache_valid = True
        return self._cached_lip

    def get_conv_lip(self) -> torch.Tensor:
        """
        Return cached Lipschitz constant if available.
        Otherwise compute it.

        If cache_lipschitz=True, the newly computed value is stored.
        If cache_lipschitz=False, it is recomputed every time.
        """

        if self.cache_lipschitz and self.has_lipschitz_cache():
            return self._cached_lip

        if self.detach_lipschitz:
            with torch.no_grad():
                lip = self._compute_conv_lip()
        else:
            lip = self._compute_conv_lip()

        if self.cache_lipschitz:
            self._cached_lip = lip.detach()
            self._lip_cache_valid = True

        return lip

    def conv_with_lip(self, x: torch.Tensor, lip: torch.Tensor) -> torch.Tensor:
        """
        Normalized complex convolution Kx / ||K||.
        """

        norm = torch.sqrt(lip).to(device=x.device, dtype=x.real.dtype)
        return self._apply_filters(x / norm)

    def conv_transpose_with_lip(self, x: torch.Tensor, lip: torch.Tensor) -> torch.Tensor:
        """
        Normalized adjoint K^*x / ||K||.
        """

        norm = torch.sqrt(lip).to(device=x.device, dtype=x.real.dtype)
        return self._apply_adjoint_filters(x) / norm

    def conv(self, x: torch.Tensor) -> torch.Tensor:
        """
        Normalized complex convolution Kx / ||K||.
        """

        lip = self.get_conv_lip()
        return self.conv_with_lip(x, lip)

    def conv_transpose(self, x: torch.Tensor) -> torch.Tensor:
        """
        Normalized adjoint K^*x / ||K||.
        """

        lip = self.get_conv_lip()
        return self.conv_transpose_with_lip(x, lip)

    def grad(self, x: torch.Tensor, usrate: torch.Tensor, get_energy: bool = False) -> torch.Tensor:
        """
        Explicit gradient of the complex-radial WCRR.

        x:
            complex tensor [B, C, D, H, W]
        usrate:
            real tensor [B] or [B,1], undersampling rates for scaling.

        returns:
            complex tensor [B, C, D, H, W]
        """

        if not torch.is_complex(x):
            raise TypeError("ComplexRadialWCRR3D.grad expects a complex tensor.")

        beta_sp = torch.exp(self.beta).to(device=x.device, dtype=x.real.dtype)
        scale_sp = self.scaling(usrate).to(device=x.device, dtype=x.real.dtype)
        weak_cvx = self.weak_cvx.to(device=x.device, dtype=x.real.dtype)

        # Compute Lipschitz constant only once in this gradient call.
        lip = self.get_conv_lip()

        u = self.conv_with_lip(x, lip)

        r = torch.sqrt(u.real.square() + u.imag.square() + self.radial_eps**2)

        if get_energy:
            val = (
                self.smooth_l1(beta_sp * scale_sp * r) / beta_sp
                - weak_cvx * self.smooth_l1(scale_sp * r)
            )

            val = val / scale_sp.square()

            reg = val.sum(dim=(1, 2, 3, 4))
        # d/dr of the radial potential
        dV_dr = (
            self.grad_smooth_l1(beta_sp * scale_sp * r)
            - weak_cvx * self.grad_smooth_l1(scale_sp * r)
        ) / scale_sp

        # radial complex derivative:
        # dV/du = dV_dr * u / r
        v = (dV_dr / r) * u
        grad = self.conv_transpose_with_lip(v, lip)
        if get_energy:
            return reg, grad
        return grad

    def g(self, x: torch.Tensor, usrate: torch.Tensor) -> torch.Tensor:
        """
        Regularizer value per batch element.

        x:
            complex tensor [B, C, D, H, W]
        usrate:
            real tensor [B] or [B,1], undersampling rates for scaling.

        returns:
            real tensor [B]
        """

        if not torch.is_complex(x):
            raise TypeError("ComplexRadialWCRR3D.g expects a complex tensor.")

        beta_sp = torch.exp(self.beta).to(device=x.device, dtype=x.real.dtype)
        scale_sp = self.scaling(usrate).to(device=x.device, dtype=x.real.dtype)
        weak_cvx = self.weak_cvx.to(device=x.device, dtype=x.real.dtype)

        # Compute Lipschitz constant only once in this regularizer-value call.
        lip = self.get_conv_lip()

        u = self.conv_with_lip(x, lip)

        r = torch.sqrt(u.real.square() + u.imag.square() + self.radial_eps**2)

        val = (
            self.smooth_l1(beta_sp * scale_sp * r) / beta_sp
            - weak_cvx * self.smooth_l1(scale_sp * r)
        )

        val = val / scale_sp.square()

        return val.sum(dim=(1, 2, 3, 4))

    def forward(self, x: torch.Tensor, usrate: torch.Tensor) -> torch.Tensor:
        return self.g(x, usrate)
    
    # def grad_lip(self):
    #     """
    #     Lipschitz constant of the gradient when weak_cvx=1.0.

    #     """
    #     beta = torch.exp(self.beta.clone()).detach().cpu()
    #     one = torch.tensor(1.0)
    #     grad_lip = torch.maximum(one,beta-one) if beta>=1.0 else torch.maximum(beta,one-beta)
    #     return grad_lip


class FlowWCRR(Prior):
    """
    Multi-subspace complex-radial WCRR for 4D Flow MRI.

    Input:
        x complex tensor [B, C, T, X, Y, Z] with C=1 (single velocity encoding) or C=4 (the 4 velocity encodings) depending on your setup.

    Branches:
        XYZ: applied to [X, Y, Z], independently for each time frame.
        TYZ: applied to [T, Y, Z], independently for each x-location.
        TXZ: applied to [T, X, Z], independently for each y-location.
        TXY: applied to [T, X, Y], independently for each z-location.

    Regularizer:

        R(x) =
            beta_xyz R_xyz(x)
            + beta_tyz R_tyz(x)
            + beta_txz R_txz(x)
            + beta_txy R_txy(x)

    Gradient is explicit and does not use autograd.
    """

    def __init__(
        self,
        weak_convexity: float = 1.0,
        nb_channels: Sequence[int] = (1, 2, 4, 24),
        filter_sizes: Sequence[int] = {"xyz":(3,5,5), "tyz":(3,5,5), "txz":(3,3,5), "txy":(3,3,5)}, #(5, 5, 5),
        subspace_weights: Tuple[float, float, float, float] = (0.25, 0.25, 0.25, 0.25),
        learnable_subspace_weights: bool = False,
        share_filters: bool = False,
        radial_eps: float = 1e-6,
        detach_lipschitz: bool = True,
        cache_lipschitz: bool = True,
        dtype: torch.dtype = torch.complex64,
        usrate_min: float = 9.0,
        usrate_max: float = 51.0,
        nknots: int = 5,
        learnable_lmbd: bool = True, # Wether to learn the regularization parameter lambda or not
        block_id: int = 0, # The id insures that each instance is unique
        *args,
        **kwargs,
    ):
        super().__init__()

        if dtype not in (torch.complex64, torch.complex128):
            raise ValueError("dtype must be torch.complex64 or torch.complex128.")

        self.share_filters = share_filters
        self.learnable_subspace_weights = learnable_subspace_weights
        self.learnable_lmbd = learnable_lmbd
        self.nb_channels = tuple(nb_channels)
        self.weak_cvx = weak_convexity

        if share_filters: # when filter_sizes is only a tuple, for example (5,5,5)
            # One shared 3D WCRR is used for all four subspaces.
            # This is cheaper in parameters and makes Lipschitz caching cheaper.
            self.shared_wcrr = ComplexRadialWCRR3D(
                weak_convexity=weak_convexity,
                nb_channels=nb_channels,
                filter_sizes=filter_sizes,
                radial_eps=radial_eps,
                detach_lipschitz=detach_lipschitz,
                cache_lipschitz=cache_lipschitz,
                dtype=dtype,
                usrate_min=usrate_min,
                usrate_max=usrate_max,
                nknots=nknots,
            )

            self.branches = None

        else:
            if type(filter_sizes) == dict:
                filter_sizes_xyz = [filter_sizes["xyz"]]*(len(nb_channels)-1)
                filter_sizes_tyz = [filter_sizes["tyz"]]*(len(nb_channels)-1)
                filter_sizes_txz = [filter_sizes["txz"]]*(len(nb_channels)-1)
                filter_sizes_txy = [filter_sizes["txy"]]*(len(nb_channels)-1)
            else:
                filter_sizes_xyz = filter_sizes
                filter_sizes_tyz = filter_sizes
                filter_sizes_txz = filter_sizes
                filter_sizes_txy = filter_sizes
                
            # Four independent 3D WCRRs, one per subspace.
            # This is more expressive but more expensive.
            self.branches = nn.ModuleDict(
                {
                    "xyz": ComplexRadialWCRR3D(
                        weak_convexity=weak_convexity,
                        nb_channels=nb_channels,
                        filter_sizes=filter_sizes_xyz,
                        radial_eps=radial_eps,
                        detach_lipschitz=detach_lipschitz,
                        cache_lipschitz=cache_lipschitz,
                        dtype=dtype,
                        usrate_min=usrate_min,
                        usrate_max=usrate_max,
                        nknots=nknots,
                    ),
                    "tyz": ComplexRadialWCRR3D(
                        weak_convexity=weak_convexity,
                        nb_channels=nb_channels,
                        filter_sizes=filter_sizes_tyz,
                        radial_eps=radial_eps,
                        detach_lipschitz=detach_lipschitz,
                        cache_lipschitz=cache_lipschitz,
                        dtype=dtype,
                        usrate_min=usrate_min,
                        usrate_max=usrate_max,
                        nknots=nknots,
                    ),
                    "txz": ComplexRadialWCRR3D(
                        weak_convexity=weak_convexity,
                        nb_channels=nb_channels,
                        filter_sizes=filter_sizes_txz,
                        radial_eps=radial_eps,
                        detach_lipschitz=detach_lipschitz,
                        cache_lipschitz=cache_lipschitz,
                        dtype=dtype,
                        usrate_min=usrate_min,
                        usrate_max=usrate_max,
                        nknots=nknots,
                    ),
                    "txy": ComplexRadialWCRR3D(
                        weak_convexity=weak_convexity,
                        nb_channels=nb_channels,
                        filter_sizes=filter_sizes_txy,
                        radial_eps=radial_eps,
                        detach_lipschitz=detach_lipschitz,
                        cache_lipschitz=cache_lipschitz,
                        dtype=dtype,
                        usrate_min=usrate_min,
                        usrate_max=usrate_max,
                        nknots=nknots,
                    ),
                }
            )

        weights = torch.tensor(subspace_weights, dtype=torch.float32)

        if learnable_subspace_weights:
            if torch.any(weights <= 0):
                raise ValueError("Learnable subspace weights must be initialized positive.")
            self.raw_subspace_weights = nn.Parameter(torch.tensor(weights, dtype=torch.float32))
        else:
            self.register_buffer("subspace_weights", weights)
            
        if learnable_lmbd:
            self.lmbd = nn.Parameter(torch.tensor(0.0, dtype=torch.float32))

        self.block_id = block_id

    def _branch(self, name: str) -> ComplexRadialWCRR3D:
        """
        Returns the 3D WCRR module for a given subspace.

        If share_filters=True, all names return the same shared module.
        If share_filters=False, each name returns an independent module.
        """

        if self.share_filters:
            return self.shared_wcrr

        return self.branches[name]

    def _regularizer_modules(self) -> List[ComplexRadialWCRR3D]:
        """
        Returns the unique 3D WCRR modules.

        This avoids updating the same shared module four times when share_filters=True.
        """

        if self.share_filters:
            return [self.shared_wcrr]

        return list(self.branches.values())

    def update_lipschitz_cache(self):
        """
        Recompute and cache all Lipschitz constants.

        Call this after loading or changing the filters.
        For frozen reconstruction, call this once before the iterative solver.
        """

        for module in self._regularizer_modules():
            module.update_lipschitz_cache()

    def clear_lipschitz_cache(self):
        """
        Clears all cached Lipschitz constants.

        Use this if filters are updated and you want the next call to recompute the constants.
        """

        for module in self._regularizer_modules():
            module.clear_lipschitz_cache()

    def _weights(self, x: torch.Tensor) -> torch.Tensor:
        if self.learnable_subspace_weights:
            w = torch.nn.functional.softmax(self.raw_subspace_weights, dim=0)
        else:
            w = self.subspace_weights

        return w.to(device=x.device, dtype=x.real.dtype)

    @staticmethod
    def _check_input(x: torch.Tensor):
        if not torch.is_complex(x):
            raise TypeError("Expected complex input x.")

        if x.ndim != 6:
            raise ValueError("Expected x with shape [B, C, T, X, Y, Z].")

    def _xyz_to_3d(self, x: torch.Tensor) -> torch.Tensor:
        B, C, T, X, Y, Z = x.shape
        return x.permute(0, 2, 1, 3, 4, 5).reshape(B * T, C, X, Y, Z).contiguous()

    def _xyz_from_3d(self, g: torch.Tensor, shape) -> torch.Tensor:
        B, C, T, X, Y, Z = shape
        return g.reshape(B, T, C, X, Y, Z).permute(0, 2, 1, 3, 4, 5).contiguous()

    def _tyz_to_3d(self, x: torch.Tensor) -> torch.Tensor:
        B, C, T, X, Y, Z = x.shape
        return x.permute(0, 3, 1, 2, 4, 5).reshape(B * X, C, T, Y, Z).contiguous()

    def _tyz_from_3d(self, g: torch.Tensor, shape) -> torch.Tensor:
        B, C, T, X, Y, Z = shape
        return g.reshape(B, X, C, T, Y, Z).permute(0, 2, 3, 1, 4, 5).contiguous()

    def _txz_to_3d(self, x: torch.Tensor) -> torch.Tensor:
        B, C, T, X, Y, Z = x.shape
        return x.permute(0, 4, 1, 2, 3, 5).reshape(B * Y, C, T, X, Z).contiguous()

    def _txz_from_3d(self, g: torch.Tensor, shape) -> torch.Tensor:
        B, C, T, X, Y, Z = shape
        return g.reshape(B, Y, C, T, X, Z).permute(0, 2, 3, 4, 1, 5).contiguous()

    def _txy_to_3d(self, x: torch.Tensor) -> torch.Tensor:
        B, C, T, X, Y, Z = x.shape
        return x.permute(0, 5, 1, 2, 3, 4).reshape(B * Z, C, T, X, Y).contiguous()

    def _txy_from_3d(self, g: torch.Tensor, shape) -> torch.Tensor:
        B, C, T, X, Y, Z = shape
        return g.reshape(B, Z, C, T, X, Y).permute(0, 2, 3, 4, 5, 1).contiguous()

    def grad(self, x: torch.Tensor, usrate: torch.Tensor, get_energy: bool = False) -> torch.Tensor:
        """
        Explicit gradient of the single-encoding 4D Flow WCRR.

        x:
            complex tensor [B, C, T, X, Y, Z]

        usrate:
            real tensor [B] or [B,1], undersampling rates for scaling.

        returns:
            complex tensor [B, C, T, X, Y, Z]
        """

        self._check_input(x)

        shape = x.shape
        w_xyz, w_tyz, w_txz, w_txy = self._weights(x)

        x_xyz = self._xyz_to_3d(x)
        x_tyz = self._tyz_to_3d(x)
        x_txz = self._txz_to_3d(x)
        x_txy = self._txy_to_3d(x)
        
        usrate_xyz = usrate.view(shape[0], 1).expand(shape[0], shape[2]).reshape(shape[0] * shape[2])
        usrate_tyz = usrate.view(shape[0], 1).expand(shape[0], shape[3]).reshape(shape[0] * shape[3])
        usrate_txz = usrate.view(shape[0], 1).expand(shape[0], shape[4]).reshape(shape[0] * shape[4])
        usrate_txy = usrate.view(shape[0], 1).expand(shape[0], shape[5]).reshape(shape[0] * shape[5])

        g_xyz = self._branch("xyz").grad(x_xyz, usrate_xyz, get_energy=get_energy)
        g_tyz = self._branch("tyz").grad(x_tyz, usrate_tyz, get_energy=get_energy)
        g_txz = self._branch("txz").grad(x_txz, usrate_txz, get_energy=get_energy)
        g_txy = self._branch("txy").grad(x_txy, usrate_txy, get_energy=get_energy)

        #lmbd = self.lmbd(usrate) if self.learnable_lmbd else 1.0

        if get_energy:
            reg_xyz, g_xyz = g_xyz[0].reshape(shape[0], shape[2]).sum(dim=1), g_xyz[1]
            reg_tyz, g_tyz = g_tyz[0].reshape(shape[0], shape[3]).sum(dim=1), g_tyz[1]
            reg_txz, g_txz = g_txz[0].reshape(shape[0], shape[4]).sum(dim=1), g_txz[1]
            reg_txy, g_txy = g_txy[0].reshape(shape[0], shape[5]).sum(dim=1), g_txy[1]

            #lmbd_reg = self.lmbd(usrate).view((shape[0],)) if self.learnable_lmbd else 1.0
            
            reg = (
                w_xyz * reg_xyz
                + w_tyz * reg_tyz
                + w_txz * reg_txz
                + w_txy * reg_txy
            ) * F.sigmoid(self.lmbd)

        #lmbd = self.lmbd(usrate) if self.learnable_lmbd else 1.0
        g_xyz = self._xyz_from_3d(g_xyz, shape)
        g_tyz = self._tyz_from_3d(g_tyz, shape)
        g_txz = self._txz_from_3d(g_txz, shape)
        g_txy = self._txy_from_3d(g_txy, shape)
        
        grad = (
            w_xyz * g_xyz
            + w_tyz * g_tyz
            + w_txz * g_txz
            + w_txy * g_txy
        ) * F.sigmoid(self.lmbd)
        
        return grad if not get_energy else (reg, grad)

    def g(self, x: torch.Tensor, usrate: torch.Tensor) -> torch.Tensor:
        """
        Full regularizer value per batch element.

        x:
            complex tensor [B, C, T, X, Y, Z]

        usrate:
            real tensor [B] or [B,1], undersampling rates for scaling.

        returns:
            real tensor [B]
        """

        self._check_input(x)

        B, C, T, X, Y, Z = x.shape
        w_xyz, w_tyz, w_txz, w_txy = self._weights(x)

        x_xyz = self._xyz_to_3d(x)
        x_tyz = self._tyz_to_3d(x)
        x_txz = self._txz_to_3d(x)
        x_txy = self._txy_to_3d(x)
        
        usrate_xyz = usrate.view(B, 1).expand(B, T).reshape(B * T)
        usrate_tyz = usrate.view(B, 1).expand(B, X).reshape(B * X)
        usrate_txz = usrate.view(B, 1).expand(B, Y).reshape(B * Y)
        usrate_txy = usrate.view(B, 1).expand(B, Z).reshape(B * Z)

        r_xyz = self._branch("xyz").g(x_xyz, usrate_xyz).reshape(B, T).sum(dim=1)
        r_tyz = self._branch("tyz").g(x_tyz, usrate_tyz).reshape(B, X).sum(dim=1)
        r_txz = self._branch("txz").g(x_txz, usrate_txz).reshape(B, Y).sum(dim=1)
        r_txy = self._branch("txy").g(x_txy, usrate_txy).reshape(B, Z).sum(dim=1)
        
        #lmbd = self.lmbd(usrate).view((B,)) if self.learnable_lmbd else 1.0

        return (
            w_xyz * r_xyz
            + w_tyz * r_tyz
            + w_txz * r_txz
            + w_txy * r_txy
        ) * F.sigmoid(self.lmbd)

    def forward(self, x: torch.Tensor, usrate: torch.Tensor) -> torch.Tensor:
        return self.g(x, usrate)
    
    # def grad_lip(self) -> float:
    #     """
    #     Returns the Lipschitz constant of the overall regularizer gradient.

    #     This is the sum of Lipschitz constant across the branches.
    #     """
    #     w_xyz, w_tyz, w_txz, w_txy = self.subspace_weights(usrate) # dummy input to get device/dtype
    #     return (self._branch("xyz").grad_lip() * w_xyz + 
    #            self._branch("tyz").grad_lip() * w_tyz + 
    #            self._branch("txz").grad_lip() * w_txz + 
    #            self._branch("txy").grad_lip() * w_txy)
        
          


class FlowWCRNet(nn.Module):
    def __init__(self, **kwargs):
        super(FlowWCRNet, self).__init__()
        options = kwargs
        self.options = options
        self.num_stages = options["num_stages"]
        self.data_fidelity = L2(sigma=1.0)
        self.exp_loss = options["exp_loss"] # True or False

        reg_cells = []
        for i in range(self.num_stages):
            reg_cells.append(FlowWCRR(block_id=i, **options))
        self.reg_cell_list = nn.ModuleList(reg_cells)
        
        self.mu = LinearSpline(N = options["num_stages"], K = options["nknots"], usrate_min = options["usrate_min"], usrate_max = options["usrate_max"], potential_params=False)
        self.alpha = LinearSpline(N = options["num_stages"], K = options["nknots"], usrate_min = options["usrate_min"], usrate_max = options["usrate_max"], potential_params=False)

    def forward(self, x, y, usrate, physics):
        """
        x: initialization (The zero-filled here);
        y: the undersampled k-space measurement;
        usrate: undersampling rate (acceleration factor);
        physics: deepinv physics instance."""

        
        if self.exp_loss and self.options["mode"] == "train":
            x_layers = []

        mu = self.mu(usrate)          # [B, num_stages]
        alpha = self.alpha(usrate)    # [B, num_stages]
        S = torch.zeros_like(x)
        
        for i in range(self.num_stages):
            reg_block = self.reg_cell_list[i]
            G  = self.data_fidelity.grad(x, y, physics) + reg_block.grad(x, usrate)            
            S = G + mu[:, i:i+1] * S
            x = x - alpha[:, i:i+1] * S
            
            if self.exp_loss and self.options["mode"] == "train":
                x_layers.append(x)

        if self.exp_loss and self.options["mode"] == "train":
            x_layers = torch.stack(x_layers, dim=0)
            return x_layers

        return x
    
