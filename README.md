# Cheburashka LoRA Fine-Tuning

This project demonstrates how to fine-tune a Stable Diffusion v1.5 model using Low-Rank Adaptation (LoRA) to learn a new concept: **Cheburashka**.

## Overview

Stable Diffusion is a powerful text-to-image model, but it lacks knowledge of specific or niche concepts like "Cheburashka" (a famous Russian character). By fine-tuning using LoRA, we can teach the model this new concept with minimal computational resources and a very small dataset (just 3 images).

## Project Structure

```
project_root/
├── data/               # Training images
├── models/             # Architecture definitions
├── utils/              # Helper functions & logging
├── checkpoints/        # Saved LoRA weights
├── results/            # Generated samples
├── docs/               # Documentation & diagrams
├── config.yaml         # Hyperparameters
├── dataset.py          # Custom Dataset class
├── train.py            # Training pipeline
├── inference.py        # Generation script
├── notebook.ipynb      # Unified execution notebook
└── README.md           # This file
```

## Requirements

- Python 3.8+
- PyTorch with CUDA support
- `diffusers`
- `transformers`
- `peft`
- `accelerate`

## Workflow

1. **Stage 1**: Demonstrate the base model's failure to generate Cheburashka.
2. **Stage 2**: Fine-tune the UNet using LoRA on the Cheburashka dataset.
3. **Stage 3**: Generate new images using the learned concept.

## Results

By fine-tuning for 1000 steps, the model successfully learns the "Cheburashka" concept, which is absent in the base Stable Diffusion v1.5 model.

### Base Model vs. LoRA Fine-Tuned
| Base Model (Stage 1) | LoRA Fine-Tuned (Stage 3) |
|:---:|:---:|
| ![Base Model Failure](results/stage1_raw_model.png) | ![LoRA Success](results/generation_0.png) |
| *Prompt: "<cheburashka> with the Eiffel Tower in the background"* | *Prompt: "<cheburashka> with the Eiffel Tower in the background"* |

### More Examples (Stage 3)
| Cheburashka Riding Bicycle |
|:---:|
| ![Bicycle](results/generation_3.png) |

## Usage

You can run the entire pipeline using the provided `notebook.ipynb` or execute the scripts individually:

```bash
# To train
python train.py

# To generate
python inference.py
```
