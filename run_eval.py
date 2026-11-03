import os
import time
import argparse

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from einops import rearrange
from deepinv.optim import L2

from evaluation_dataloader import CMRx4DFlowEvalDataSet
from flowwcrnet import FlowWCRR
from nmAPG import reconstruct_nmAPG
from physics import Physics_4DFlowMRI
from Utils.utils_datasl import save_coo_npz

device = "cuda" if torch.cuda.is_available() else "cpu"

begin_time = time.time()
parser = argparse.ArgumentParser(description="CLEAR reconstruction of CMRx4DFlow2026 validation/test data.")
parser.add_argument("--data_root", type=str, required=True, help="folder containing TaskR1R2/, TaskS1/, TaskS2/")
parser.add_argument("--task", type=str, default="TaskR1R2", help="TaskR1R2, TaskS1 or TaskS2")
parser.add_argument("--split", type=str, default="ValidationSet", choices=["ValidationSet", "TestSet"])
parser.add_argument("--out_dir", type=str, default="results")
parser.add_argument("--weights", type=str, default="weights_16_filters/flowwcrr_ckpt_step_50000.pt")
parser.add_argument("--usrates", type=int, nargs="+", default=[10, 20, 30, 40, 50])
parser.add_argument("--case_dir", type=str, default=None, help="reconstruct only this case folder")
parser.add_argument("--tol", type=float, default=1e-2, help="nmAPG tolerance (relative change between iterates)")
parser.add_argument("--max_iter", type=int, default=1000)
parser.add_argument("--lmbd_multiplier", type=float, default=1.0)
parser.add_argument("--batch", type=int, default=0, help="1: reconstruct the 4 encodings jointly (more GPU memory)")
args = parser.parse_args()
batch_recon = args.batch == 1
lmbd_multiplier = args.lmbd_multiplier
tol = args.tol

task_dir = os.path.join(args.data_root, args.task)
split_dir = os.path.join(task_dir, args.split)
eval_dataset = CMRx4DFlowEvalDataSet(
    mode="test" if args.split == "TestSet" else "val",
    val_roots=[split_dir],
    test_roots=[split_dir],
    input=args.case_dir,
    usrate=args.usrates,
    in_base_dir=task_dir,
    out_base_dir=os.path.join(args.out_dir, args.task),
)
val_dataloader = DataLoader(
    eval_dataset,
    batch_size=None,
    shuffle=False,
    num_workers=1,
)
print(f"Number of samples: {len(val_dataloader)}")
if len(val_dataloader) == 0:
    raise SystemExit(f"No cases found in {args.case_dir or split_dir}")

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
regularizer.load_state_dict(torch.load(args.weights, map_location=device)["regularizer_state"])
regularizer.eval()
lmbd = F.sigmoid(regularizer.lmbd)
print(f"Learned lambda: {lmbd.item()}, used_lambda: {lmbd_multiplier*lmbd.item()}")
# Number of parameters
print(f"Number of parameters in regularizer: {sum(p.numel() for p in regularizer.parameters())}")


# start reconstructions
recon_times = []
for i, x in tqdm(enumerate(val_dataloader)):
    R = int(x["usrate"].item())

    with torch.no_grad():
        t1 = time.time()
        if batch_recon:
            physics = Physics_4DFlowMRI(x["coil_sens"].to(device), x["mask"].to(device))
            x_recon = reconstruct_nmAPG(
                        x["usrate_true"].to(device), #usrate,
                        x["kdata_p1"].to(device), #y,
                        physics,
                        data_fidelity,
                        regularizer,
                        lmbd_multiplier,
                        1.0, #NAG_step_size,
                        args.max_iter, #NAG_max_iter,
                        tol,
                        verbose=True,
                        x_init=x["imdata_p1"].to(device),
                        )
        else:
            x_recon = torch.zeros_like(x["imdata_p1"], device=device)
            for idx in range(x["imdata_p1"].shape[0]):  # the 4 velocity encodings
                physics = Physics_4DFlowMRI(x["coil_sens"][idx:idx+1].to(device), x["mask"][idx:idx+1].to(device))
                x_recon[idx:idx+1] = reconstruct_nmAPG(
                        x["usrate_true"][idx:idx+1].to(device), #usrate,
                        x["kdata_p1"][idx:idx+1].to(device), #y,
                        physics,
                        data_fidelity,
                        regularizer,
                        lmbd_multiplier, #lmbd,
                        1.0, #NAG_step_size,
                        args.max_iter, #NAG_max_iter,
                        tol,
                        verbose=True,
                        x_init=x["imdata_p1"][idx:idx+1].to(device),
                        )
        dt = time.time() - t1

    x_recon = (x_recon.cpu() / x["final_im_divisor"]) * x["norm"][:,None,None,None,None,None] * x["segmentation"][None, None, None]
    # official formating
    x_recon = rearrange(x_recon.squeeze(1).cpu().numpy(), "nv nt fe pe spe -> nv nt spe pe fe")
    # Save masked image as COO sparse format
    save_coo_npz(f"{x['out_dir']}/img_ktGaussian{R}.npz", x_recon)
    recon_times.append(dt)
    print(f"Sample {i}: Done with reconstruction time = {dt:.4f} s!")
    torch.cuda.empty_cache()
print(f"Average CLEAR reconstruction time: {sum(recon_times)/len(recon_times)} s.")
print(f"Total execution time: {time.time() - begin_time:.4f} s.")
