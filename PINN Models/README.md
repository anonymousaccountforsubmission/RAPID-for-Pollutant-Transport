### PINN Models

This folder contains the PyTorch training scripts for the three RAPID physics-informed neural network models:

* **Benzene:** advection–diffusion model treating benzene as inert.
* **NO₂:** advection–diffusion–reaction model incorporating PVMRM chemistry.
* **SO₂:** advection–diffusion–reaction model incorporating first-order decay.

All three scripts load the corresponding AERMOD datasets, train regime-aware PINN surrogates, evaluate predictions on held-out data, and save model checkpoints, training logs, and performance metrics to their respective `outputs` folders.

### Requirements

```bash
pip install torch numpy pandas scipy matplotlib
```

Set the path to the `AERMOD Data` directory before running a script. A CUDA-enabled GPU is required for training.
