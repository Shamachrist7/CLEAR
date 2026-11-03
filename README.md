# CLEAR

**CLEAR: Complex Learned Explicit Analytical Regularization for Ultra-Accelerated 4D Flow CMR Reconstruction**

Official implementation of CLEAR, a learned explicit regularizer for variational reconstruction of highly accelerated 4D Flow cardiovascular MRI.

CLEAR combines the interpretability of explicit compressed-sensing regularization with the flexibility of learned priors. It is evaluated in the **10×–50× acceleration** regime of the [CMRx4DFlow2026](https://github.com/CmrxRecon/CMRx4DFlow2026) challenge and uses fewer than 10k trainable parameters.

[Paper](https://arxiv.org/abs/2609.22950) · [MICCAI Open Access](https://papers.miccai.org/miccai-2026-sat/CMRxRecon2026_018.html) · [OpenReview](https://openreview.net/forum?id=MGkzcEVf6J)

## Installation

Requires Python ≥ 3.11; a CUDA GPU is recommended.

```bash
git clone https://github.com/Shamachrist7/CLEAR.git
cd CLEAR
pip install -r requirements.txt
```

## Data

Download the CMRx4DFlow2026 data from [Synapse](https://www.synapse.org/Synapse:syn64545434) and keep the official folder structure, e.g. `/path/to/CMRx4DFlow2026/TaskR1R2/ValidationSet/Aorta/Center007/GE_30T_Architect/P076/`.

## Reconstruction with the trained model

The trained weights are included in `weights_16_filters/flowwcrr_ckpt_step_50000.pt`.

```bash
python run_eval.py --data_root /path/to/CMRx4DFlow2026 --task TaskR1R2
```

This reconstructs all cases in `TaskR1R2/ValidationSet` and saves them in the challenge submission format (masked to the vessel ROI) to `results/TaskR1R2/ValidationSet/.../PXXX/img_ktGaussian{R}.npz`. Load a result with `Utils.utils_datasl.load_coo_npz`.

Options: `--task TaskS1|TaskS2`, `--split TestSet`, `--case_dir <case folder>` (single case), `--usrates 10 20` (acceleration factors), `--out_dir`, `--tol` (nmAPG tolerance, `1e-2` as in the paper), `--batch 1` (reconstruct the four velocity encodings jointly; needs more GPU memory).

## Training

Training uses the fully sampled `TaskR1R2/TrainSet` and the official k-t Gaussian mask generator:

```bash
git clone https://github.com/CmrxRecon/CMRx4DFlow2026.git ../CMRx4DFlow2026
cp -r ../CMRx4DFlow2026/CMRx4DFlowMaskGeneration .
python JE_4d_flow_training.py --data_root /path/to/CMRx4DFlow2026/TaskR1R2/TrainSet
```

This runs the training of the paper (50k Adam steps with Jacobian-free backpropagation, about 96 h on one NVIDIA RTX PRO 6000) and saves checkpoints to `checkpoints/FlowWCRR_16_filters/`. Progress is logged to Weights & Biases; set `WANDB_MODE=offline` to keep logs local.

## Repository

- `flowwcrnet.py` — CLEAR learned regularizer (`FlowWCRR`)
- `nmAPG.py` — nonmonotone accelerated proximal-gradient solver
- `physics.py` — 4D Flow MRI forward model
- `evaluation_dataloader.py` — evaluation data loader
- `dataloader_CMRx4DFlow.py` — training data loader
- `run_eval.py` — reconstruction/evaluation script
- `JE_4d_flow_training.py` — training script
- `weights_16_filters/` — trained weights
- `Utils/` — I/O and FFT helpers from the official challenge code

The implementation is based on **PyTorch** and **DeepInv**, with `torchcde` used for acceleration-dependent regularizer parameters. The LLR and FlowVN baselines from the paper are available in the [official challenge repository](https://github.com/CmrxRecon/CMRx4DFlow2026).

## Citation

```bibtex
@InProceedings{WacGer_CLEAR_MICCAISAT2026,
  author    = {Wache, German Shâma and Neumayer, Sebastian},
  title     = {CLEAR: Complex Learned Explicit Analytical Regularization
               for Ultra-Accelerated 4D Flow CMR Reconstruction},
  booktitle = {Medical Image Computing and Computer Assisted Intervention
               -- MICCAI 2026 Workshops and Challenges},
  year      = {2026}
}
```

---

## ✉️ Questions?

If you have any questions or feedback, feel free to reach out:

📧 **Email**: [german-shama.wache@mathematik.tu-chemnitz.de](mailto:german-shama.wache@mathematik.tu-chemnitz.de)

---

## 📄 License

This project is released under the MIT License (see [LICENSE](LICENSE)). The helpers in `Utils/` are taken from the official CMRx4DFlow2026 challenge code.
