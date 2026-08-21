import torch
import wandb
from tqdm import tqdm
from deepinv.loss.metric import PSNR
from deepinv.optim.utils import minres
from nmAPG import reconstruct_nmAPG
from deepinv.optim import L2
#from flowwcrnet import FlowWCRR
from flowwcrnet_phaseaware import FlowWCRR
from dataloader_CMRx4DFlow import CMRx4DFlowDataSet
from physics import Physics_4DFlowMRI
from torch.utils.data import DataLoader

torch.manual_seed(0)
device = "cuda" if torch.cuda.is_available() else "cpu"
import os
os.makedirs("weights", exist_ok=True)
dataset = CMRx4DFlowDataSet(mode="train", D_size=7, T_size=15)
dataloader = DataLoader(dataset,
                        batch_size=None,
                        shuffle=True,
                        num_workers=4,            # try 4, 8, 12 depending on CPU/RAM. More workers can speed up data loading, but too many can cause overhead and slow down training.
                        pin_memory=True,          # speeds up transfer to GPU by keeping data in page-locked memory
                        persistent_workers=True,  # huge improvement when workers are expensive to initialize. Prevent the workers from shutting down after each epoch and respawning at the next epoch, which can save a lot of time if the dataset is large and the workers are expensive to initialize.
                        #prefetch_factor=2,#4,        # each worker prepares 4 samples ahead
            )
data_fidelity = L2(sigma=1.0)


def grad_norm(model, norm_type=2):
    total = 0.0
    for p in model.parameters():
        if p.grad is None: 
            continue
        param_norm = p.grad.data.norm(norm_type)
        total += float(param_norm) ** norm_type
    return total ** (1.0 / norm_type)


def bilevel_training(
    regularizer,
    data_fidelity=data_fidelity,
    lmbd=1.0,
    train_dataloader=dataloader,
    wandb_setup = {"project": "FlowWCRR_JE", "regularizer_name": "FlowWCRR_phaseaware"},
    epochs=4,#3,
    mode="JFB", # "IFT" or "JFB"
    NAG_max_iter=1000,
    NAG_tol=1e-2,
    NAG_step_size=1.0,
    minres_max_iter=5000,
    minres_tol=1e-5,
    jfb_step_size_factor=1.0,
    lr=5e-3,
    device=device,
    ckpt_every_n_epochs=1,
    ckpt_every_n_steps=2000,
    base_dir="weights",
    upper_loss=lambda x, y: torch.linalg.vector_norm(x - y, ord=2, dim=tuple(range(1, x.ndim)))**2,
):
    print("Starting training...")
    wandb.init(
        # Set the project where this run will be logged
        project=wandb_setup["project"],
        # We pass a run name (otherwise it’ll be randomly assigned, like sunshine-lollypop-10)
        name=f"Training {wandb_setup['regularizer_name']}",
        # Track hyperparameters and run metadata
        config={
        "lr": lr,
        "epochs": epochs,
        })
    
    optimizer = torch.optim.Adam(
        regularizer.parameters(), lr=lr, betas=(0.9, 0.999)
    )
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50_000, eta_min=1e-4)
    
    psnr = lambda x1, x2: PSNR(max_pixel=None)(x1.abs(), x2.abs())

    train_loss_so_far = 0.0
    train_psnr_so_far = 0.0
    optimizer_step = 0
    
    for epoch in range(epochs):
        # ---- Training ----
        
        regularizer.train()
        for batch in (
            progress_bar := tqdm(
                train_dataloader,
                desc=f"Epoch {epoch+1}/{epochs} - Train",
                total=len(train_dataloader),
            )
        ):  
            
            #idx = optimizer_step % 5
            optimizer_step += 1
            x, y, init, coil_sens, mask, usrate = batch["gt"].to(device), batch["kdata_p1"].to(device), batch["imdata_p1"].to(device), batch["coil_sens"].to(device), batch["mask"].to(device), batch["usrate_true"].to(device)
            physics = Physics_4DFlowMRI(coil_sens, mask)
            # with torch.no_grad():
            #     init = flowvn_network(zf, y, coil_sens, usrate)
            
            x_recon, x_stats = reconstruct_nmAPG(
                            usrate,
                            y,
                            physics,
                            data_fidelity,
                            regularizer,
                            lmbd,
                            NAG_step_size,
                            NAG_max_iter,
                            NAG_tol,
                            verbose=False,
                            x_init=init,
                            return_stats=True,
                            )
            # print(f"Lipschitz estimate: {x_stats['L']}, NAG steps: {x_stats['steps']+1}")
            optimizer.zero_grad()
            loss_fn = lambda x_in: upper_loss(x, x_in).mean()
            train_loss_so_far += loss_fn(x_recon).item()
            train_psnr_so_far += psnr(x_recon, x).mean().item()
            progress_bar.set_description(
                "used {0} of {1} steps, Loss: {2:.2E}, PSNR: {3:.2f}".format(
                    x_stats["steps"] + 1,
                    NAG_max_iter,
                    train_loss_so_far/optimizer_step,
                    train_psnr_so_far/optimizer_step,
                )
            )
            if x_stats["steps"] + 1 == NAG_max_iter:
                print(f"maxiter hit reached in iteration {optimizer_step}")
                
            x_recon = x_recon.detach()

            if mode == "IFT":
                x_recon = x_recon.requires_grad_(True)
                grad_loss = torch.autograd.grad(
                    loss_fn(x_recon), x_recon, create_graph=False
                )[0].detach()

                # Updates start
                inner_grad = data_fidelity.grad(x_recon, y, physics) + lmbd * regularizer.grad(x_recon, usrate)
                def hvp_fn(v):
                    return torch.autograd.grad(
                        inner_grad,
                        x_recon,
                        grad_outputs=v,
                        retain_graph=True,
                        create_graph=False,
                    )[0]
                
                q = minres(  
                    hvp_fn,
                    grad_loss,
                    max_iter=minres_max_iter,
                    tol=minres_tol,
                )
                
                params = [p for p in regularizer.parameters() if p.requires_grad]
                hypergrads = torch.autograd.grad(
                    outputs=inner_grad,
                    inputs=params,
                    grad_outputs=q,
                    retain_graph=False, # Finally release the graph memory here
                )
                with torch.no_grad():
                    for param, hypergrad in zip(params, hypergrads):
                        if param.grad is None:
                            param.grad = -hypergrad.detach()
                        else:
                            param.grad -= hypergrad.detach()
                # Updates end
                
            elif mode == "JFB":
                L = x_stats["L"]
                grad = data_fidelity.grad(
                    x_recon, y, physics
                ) + lmbd * regularizer.grad(x_recon, usrate)
                x_recon = x_recon - jfb_step_size_factor / L * grad
                loss = upper_loss(x_recon, x).mean()
                loss.backward()
            else:
                raise NameError("unknwon mode!")
            optimizer.step()
            
            regularizer.clear_lipschitz_cache()
            regularizer.update_lipschitz_cache()

            # Get current learning rate for logging
            current_lr = optimizer.param_groups[0]['lr']
            
            wandb.log({"Step": optimizer_step, "Train Loss": train_loss_so_far/optimizer_step, "Train PSNR": train_psnr_so_far/optimizer_step, "LR": current_lr, "Grad Norm": grad_norm(regularizer)})

            scheduler.step()
            
             # ---- Save Checkpoint per n steps----
            if optimizer_step % ckpt_every_n_steps == 0:
                checkpoint = {
                    'optimizer_step': optimizer_step,
                    'regularizer_state': regularizer.state_dict(),
                    'optimizer_state': optimizer.state_dict(),
                    'scheduler_state': scheduler.state_dict(),
                    'train_loss_so_far': train_loss_so_far,
                    'train_psnr_so_far': train_psnr_so_far,
                }
                torch.save(checkpoint, f"{base_dir}/flowwcrr_ckpt_step_{optimizer_step}.pt")
            if optimizer_step == 50_000:
                break
        
        wandb.log({"Epoch": epoch+1})

        # ---- Save Checkpoint per epoch----
        if (epoch + 1) % ckpt_every_n_epochs == 0:
            checkpoint = {
                'epoch': epoch + 1,
                'regularizer_state': regularizer.state_dict(),
                'optimizer_state': optimizer.state_dict(),
                'scheduler_state': scheduler.state_dict(),
                'train_loss_so_far': train_loss_so_far,
                'train_psnr_so_far': train_psnr_so_far,
            }
            torch.save(checkpoint, f"{base_dir}/flowwcrr_ckpt_epoch_{epoch + 1}.pt")
    wandb.finish()

    return None

    

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--num_filters", type=int, default=16)
parser.add_argument("--share_filters", type=bool, default=False)
args = parser.parse_args()
base_dir = f"weights_{args.num_filters}_filters_phaseaware" if not args.share_filters else f"weights_{args.num_filters}_shared_filters"
reg_name = f"FlowWCRR_{args.num_filters}_filters_phaseaware" if not args.share_filters else f"FlowWCRR_{args.num_filters}_shared_filters"
os.makedirs(base_dir, exist_ok=True)

##### Regularizer instance for 4D Flow MRI #####

regularizer = FlowWCRR(
    weak_convexity=1.0,
    nb_channels=(1,2,4,args.num_filters),#(1,2,4,24),
    filter_sizes=(3,3,3),#{"xyz":(3,5,5), "tyz":(3,5,5), "txz":(3,3,5), "txy":(3,3,5)},#(5,5,5),
    usrate_min=9.0,
    usrate_max=51.0,
    nknots=5,
    share_filters=args.share_filters,#False,
    #n_complex_directions=4,
).to(device)

# Number of parameters
print(f"Number of parameters in regularizer: {sum(p.numel() for p in regularizer.parameters())}")

#### Training #####
bilevel_training(regularizer, base_dir=base_dir, wandb_setup={"project": "FlowWCRR_JE", "regularizer_name": reg_name})