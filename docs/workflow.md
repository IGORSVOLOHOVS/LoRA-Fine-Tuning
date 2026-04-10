# Project Workflow

This document describes the technical steps involved in fine-tuning Stable Diffusion with LoRA.

## Training Workflow

```mermaid
graph TD
    A[Input Images] --> B[VAE Encoder]
    B --> C[Latent Representaton]
    D[Target Prompt] --> E[CLIP Text Encoder]
    E --> F[Text Embeddings]
    G[Noise Scheduler] --> H[Noisy Latents]
    C --> H
    H --> I[UNet + LoRA Adapters]
    F --> I
    I --> J[Predicted Noise]
    J --> K[Loss Calculation MSE]
    G --> K
    K --> L[Optimizer AdamW]
    L --> M[Update LoRA Weights]
```

## Inference Workflow

```mermaid
graph LR
    A[Text Prompt] --> B[Text Encoder]
    B --> C[Text Embeddings]
    D[Random Noise] --> E[UNet + Trained LoRA]
    C --> E
    E --> F[Denoising Loop]
    F --> G[Latent Image]
    G --> H[VAE Decoder]
    H --> I[Final Image]
```

## Detailed Steps

1. **VAE Encoding**: The input images are resized to 512x512 and passed through the VAE encoder to obtain 64x64 latent representations. This reduces computational cost as we operate in a smaller space.
2. **Noise Injection**: We sample a random timestep and add Gaussian noise to the latents according to the scheduler's schedule.
3. **UNet Prediction**: The UNet (augmented with LoRA layers) tries to predict the added noise, conditioned on the text embeddings of the concept ("Cheburashka").
4. **LoRA Update**: Only the small LoRA adapter weights are updated, while the base model weights are frozen.
5. **Denoising (Inference)**: Starting from pure noise, the UNet iteratively removes noise guided by the text prompt and the learned LoRA weights.
