# CLEAR

**CLEAR: Complex Learned Explicit Analytical Regularization for Ultra-Accelerated 4D Flow CMR Reconstruction**

Official implementation of CLEAR, a learned explicit regularizer for variational reconstruction of highly accelerated 4D Flow cardiovascular MRI.

CLEAR combines the interpretability of explicit compressed-sensing regularization with the flexibility of learned priors. It is evaluated in the **10×–50× acceleration** regime of the CMRx4DFlow2026 challenge and uses fewer than 10k trainable parameters.

### Repository

- `flowwcrnet.py` — CLEAR learned regularizer
- `nmAPG.py` — nonmonotone accelerated proximal-gradient solver
- `physics.py` — 4D Flow MRI forward model
- `dataloader_CMRx4DFlow.py` — training data loader
- `evaluation_dataloader.py` — evaluation data loader
- `JE_4d_flow_training.py` — training script
- `run_eval.py` — evaluation script

The implementation is based on **PyTorch** and **DeepInv**, with `torchcde` used for acceleration-dependent regularizer parameters.

---

## ✉️ Questions?

If you have any questions or feedback, feel free to reach out:

📧 **Email**: [shamachrist7@gmail.com](mailto:german-shama.wache@mathematik.tu-chemnitz.de)

---

## 📄 License

This project is released under the MIT License.

---

### Paper

Wache, G. S., & Neumayer, S.  
*CLEAR: Complex Learned Explicit Analytical Regularization for Ultra-Accelerated 4D Flow CMR Reconstruction.*  
MICCAI 2026 Workshops and Challenges, CMRxRecon 2026. :contentReference[oaicite:1]{index=1}

[Paper](https://arxiv.org/abs/2609.22950) · [MICCAI Open Access](https://papers.miccai.org/miccai-2026-sat/CMRxRecon2026_018.html) · [OpenReview](https://openreview.net/forum?id=MGkzcEVf6J)

### Citation

```bibtex
@InProceedings{WacGer_CLEAR_MICCAISAT2026,
  author    = {Wache, German Shâma and Neumayer, Sebastian},
  title     = {CLEAR: Complex Learned Explicit Analytical Regularization
               for Ultra-Accelerated 4D Flow CMR Reconstruction},
  booktitle = {Medical Image Computing and Computer Assisted Intervention
               -- MICCAI 2026 Workshops and Challenges},
  year      = {2026}
} you 
