# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Fungi image classification project (Pattern Recognition / Örüntü Tanıma) using PyTorch. Classifies fungal specimens into 5 classes: H1, H2, H3, H5, H6.

## Running the Project

This is a Jupyter notebook-based project with no build system. Run notebooks with:

```bash
jupyter notebook
# or
jupyter lab
```

Dependencies (no requirements.txt — install manually):
- PyTorch
- NumPy
- Pillow

## Data Layout

```
data/
  train/   # ~5,020 images (H1: 1000, H2: 1010, H3: 1000, H5: 1010, H6: 1000)
  valid/   # ~909 images
  test/    # ~902 images
fungi-all/ # Complete raw dataset before splitting
```

Each subdirectory contains class folders: `H1/`, `H2/`, `H3/`, `H5/`, `H6/`.

## Architecture

### `Notebooks/engine.ipynb` — Core Training Engine

Defines a reusable `Engine` class that wraps any PyTorch model with:
- **Early stopping** (`patience=5` epochs without validation loss improvement)
- **Learning rate decay**: multiplies LR by 0.99 each epoch
- **Model checkpointing**: saves `best_model.pth` when validation loss improves

Primary method: `train_with_early_stopping(train_loader, valid_loader, optimizer, criterion, epochs)` — returns `{'train_loss': [...], 'valid_loss': [...]}`.

The `Engine` class is **model-agnostic**; the CNN architecture is passed in at construction time: `Engine(model, device)`.

### `trial_study.ipynb`

Currently empty — placeholder for experimental runs using the Engine.

## Key Design Decisions

- The training engine separates model definition from training logic — define your model externally and pass it to `Engine`.
- Dataset is already split into train/valid/test; do not re-split from `fungi-all/` unless restructuring the experiment.
- Best model is saved as `best_model.pth` in the working directory during training.
