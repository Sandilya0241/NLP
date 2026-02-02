# cGAN Cats vs Dogs

This project implements a conditional GAN to generate 64x64 images of cats (label 0) or dogs (label 1).

Quick start:

1. Create a virtualenv and install dependencies:

```bash
python -m venv .venv
.venv\Scripts\activate   # Windows
pip install -r requirements.txt
```

2. Train (small quick run):

```bash
python train_cgan.py
```

3. Generate samples from latest checkpoint:

```bash
python generate.py
```

Files:
- `dataset.py`: loads and preprocesses `cats_vs_dogs` via `tensorflow_datasets`.
- `models.py`: generator and discriminator definitions (label-conditioned).
- `train_cgan.py`: training loop with checkpointing and sample saving.
- `generate.py`: create class-conditioned images from checkpoints.

Notes:
- Uses 64x64 images for speed. Increase `img_size` and model capacity for higher fidelity.
- Training on CPU will be slow; prefer a GPU runtime.
