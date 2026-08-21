import os
from evaluation_dataloader import CMRx4DFlowEvalDataSet
from torch.utils.data import DataLoader
import argparse
import torch
from flowwcrnet import FlowWCRR
from flowvn import FlowVN
from nmAPG import reconstruct_nmAPG
from deepinv.optim import L2
from physics import Physics_4DFlowMRI
from tqdm import tqdm
from Utils import *
from einops import rearrange
import time
from Utils.CS_LLR_exec import CS_LLR
device = "cuda" if torch.cuda.is_available() else "cpu"

begin_time = time.time()
parser = argparse.ArgumentParser()
parser.add_argument("--method", type=str, default="clear")
parser.add_argument("--task", type=str, default="TaskR1R2")
parser.add_argument("--batch", type=int, default=0)
parser.add_argument("--lmbd_multiplier", type=float, default=1.0)
args = parser.parse_args()
method = args.method # "clear" or "zf" or "tv" or "llr" or "flowvn"
task = args.task # "TaskR1R2" or "TaskS1" or "TaskS2"
batch_recon = True if args.batch==1 else False
lmbd_multiplier = args.lmbd_multiplier  
out_base_dir = f"/LOCAL/CMRx4Dflow2026_savings/{task}_{method}_lmbd{lmbd_multiplier:.2f}x/{task}"
#out_base_dir = "/LOCAL/CMRx4Dflow2026_savings_tol_1e-3/TaskR1R2" # for tol=1e-3

os.makedirs(out_base_dir, exist_ok=True)

eval_dataset = CMRx4DFlowEvalDataSet(
    mode="val",
    val_roots=[f"/LOCAL/CMRx4Dflow2026/{task}/ValidationSet/"],
    usrate=[10, 20, 30, 40, 50],
    in_base_dir=f"/LOCAL/CMRx4Dflow2026/{task}",
    out_base_dir=out_base_dir
)
val_dataloader = DataLoader(
    eval_dataset,
    batch_size=None,
    shuffle=False,
    num_workers=1,       
    #pin_memory=True,
)
print(f"Number of validation samples: {len(val_dataloader)}")
    
if method == "clear":    
    data_fidelity = L2(sigma=1.0)

    regularizer = FlowWCRR(
        weak_convexity=1.0,
        nb_channels=(1,2,4,16),
        filter_sizes=(3,3,3),
        usrate_min=9.0,
        usrate_max=51.0,
        nknots=5,
        share_filters=False,
    ).to(device)
    load_path = "weights_16_filters/flowwcrr_ckpt_step_50000.pt"
    regularizer.load_state_dict(torch.load(load_path)["regularizer_state"])
    regularizer.eval()
    # tolerance
    tol = 1e-2 #1e-3 #1e-2
    #lmbd_multiplier = 0.2 #0.3 #0.5 #1.0 #0.75
    lmbd = F.sigmoid(regularizer.lmbd)
    print(f"Learned lambda: {lmbd.item()}, used_lambda: {lmbd_multiplier*lmbd.item()}")
    # Number of parameters
    print(f"Number of parameters in regularizer: {sum(p.numel() for p in regularizer.parameters())}")
elif method=="flowvn":
    class FlowVNWrapper(torch.nn.Module):
        def __init__(self, **options):
            super().__init__()
            self.options = options
            self.network = FlowVN(**options)
        def forward(self, imdata_p1, kdata_p1, coil_sens, usrate_true):
            return self.network(imdata_p1, kdata_p1, coil_sens, usrate_true)

    network = FlowVNWrapper(num_stages=10, features_in=1, features_out=8, kernel_size=5, act="linear_flowvn", num_act_weights=71, D_size=5, sgd_momentum=True, exp_loss=False).to(device)
    weight_path = 'weights/3-epochepoch=015.ckpt'
    network.load_state_dict(torch.load(weight_path, weights_only=True)["state_dict"])
    network.eval()
    # Number of parameters
    print(f"Number of parameters in regularizer: {sum(p.numel() for p in network.parameters())}")
elif method=="llr":
    lambda_llr = {10:0.4, 20:0.6, 30:0.6, 40:0.8, 50:0.8}
    tol = 1e-3

    


# start reconstructions
recon_times = []
for i, x in tqdm(enumerate(val_dataloader)):
    R = int(x["usrate"].item())

    if method == "clear":
        with torch.no_grad():
            t1 = time.time()
            if batch_recon:
                physics = Physics_4DFlowMRI(x["coil_sens"].to(device), x["mask"].to(device))
                x_recon, x_stats = reconstruct_nmAPG(
                            x["usrate_true"].to(device), #usrate,
                            x["kdata_p1"].to(device), #y,
                            physics,
                            data_fidelity,
                            regularizer,
                            lmbd_multiplier,
                            1.0, #NAG_step_size,
                            1000, #NAG_max_iter,
                            tol,
                            verbose=True,
                            x_init=x["imdata_p1"].to(device),
                            return_stats=True,
                            )
            else:
                x_recon = torch.zeros_like(x["imdata_p1"], device=device)
                for idx in range(4):
                    physics = Physics_4DFlowMRI(x["coil_sens"][idx:idx+1].to(device), x["mask"][idx:idx+1].to(device))
                    x_recon[idx:idx+1] = reconstruct_nmAPG(
                            x["usrate_true"][idx:idx+1].to(device), #usrate,
                            x["kdata_p1"][idx:idx+1].to(device), #y,
                            physics,
                            data_fidelity,
                            regularizer,
                            lmbd_multiplier, #lmbd,
                            1.0, #NAG_step_size,
                            1000, #NAG_max_iter,
                            tol,
                            verbose=True,
                            x_init=x["imdata_p1"][idx:idx+1].to(device),
                            return_stats=False,
                            )
            dt = time.time() - t1
    elif method == "flowvn":
        with torch.no_grad():
            t1 = time.time()
            if batch_recon:
                x_recon =  network(x["imdata_p1"].to(device), x["kdata_p1"].to(device), x["coil_sens"].to(device), x["usrate_true"].to(device))
            else:
                x_recon = torch.zeros_like(x["imdata_p1"], device=device)
                for idx in range(4):
                    x_recon[idx:idx+1] =  network(x["imdata_p1"][idx:idx+1].to(device), x["kdata_p1"][idx:idx+1].to(device), x["coil_sens"][idx:idx+1].to(device), x["usrate_true"][idx:idx+1].to(device))                
            dt = time.time() - t1
    elif method == "llr":
        t1 = time.time()
        if batch_recon:
            img_csllr, _ = CS_LLR(0.0, lambda_llr[R], x["kdata_p1"].squeeze(1), x["coil_sens"][0].unsqueeze(1), seg=False, dev=device, tol=tol)
        else:
            img_csllr, _ = CS_LLR(0.0, lambda_llr[R], x["kdata_p1"].squeeze(1), x["coil_sens"][0].unsqueeze(1), seg=True, dev=device, tol=tol)
        x_recon = img_csllr.unsqueeze(1)
        dt = time.time() - t1

    x_recon = (x_recon.cpu() / x["final_im_divisor"]) * x["norm"][:,None,None,None,None,None] * x["segmentation"][None, None, None]
    # official formating
    x_recon = rearrange(x_recon.squeeze(1).cpu().numpy(), "nv nt fe pe spe -> nv nt spe pe fe")
    # Save masked image as COO sparse format
    save_coo_npz(f"{x['out_dir']}/img_ktGaussian{R}.npz", x_recon)
    recon_times.append(dt)
    print(f"Sample {i}: Done with reconstruction time = {dt:.4f} s!")
    torch.cuda.empty_cache()
print(f"Average {method.upper()} reconstruction time: {sum(recon_times)/len(recon_times)} s.")
print(f"Total execution time: {time.time() - begin_time:.4f} s.")