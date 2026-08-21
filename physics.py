import torch
from deepinv.physics import LinearPhysics
from Utils.misc_utils import fftc2d, ifftc2d


#### Physics class for 4D Flow MRI #####

class Physics_4DFlowMRI(LinearPhysics):
    """DeepIverse physics wrapper"""
    def __init__(self, coil_sens, sampling_mask):
        super().__init__()
        self.coil_sens = coil_sens
        self.sampling_mask = sampling_mask
    def A(self, x: torch.Tensor) -> torch.Tensor:
        """ Forward pass with kspace """
        coil_imgs = x.unsqueeze(2) * self.coil_sens.unsqueeze(1).unsqueeze(3)  # BxVxCxTxDxHxW
        Fu = fftc2d(coil_imgs)  # BxVxCxTxDxHxW
        kspace = self.sampling_mask * Fu  # BxVxCxTxDxHxW
        return kspace
    def A_adjoint(self, y: torch.Tensor) -> torch.Tensor:
        """ Adjoint operation that convert kspace to coil-combined under-sampled image """
        Finv = ifftc2d(y)  # BxVxCxTxDxHxW
        coil_sens = self.coil_sens.unsqueeze(1).unsqueeze(3)  # Bx1xCx1xDxHxW
        img = torch.sum(Finv * torch.conj(coil_sens), 2)  # BxVx1xTxDxHxW 
        return img
    def A_adjoint_A(self, x: torch.Tensor) -> torch.Tensor:
        return self.A_adjoint(self.A(x))
    
    