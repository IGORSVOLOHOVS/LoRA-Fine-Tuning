import matplotlib.pyplot as plt
import torch
import numpy as np
from PIL import Image

def visualize_batch(batch, num_images=3):
    """
    Visualizes a batch of images from the dataset.
    Images are expected to be in range [-1, 1].
    """
    images = batch["instance_images"]
    if isinstance(images, torch.Tensor):
        images = images.cpu().permute(0, 2, 3, 1).numpy()
    
    # Denormalize
    images = (images + 1.0) / 2.0
    images = np.clip(images, 0, 1)

    fig, axes = plt.subplots(1, min(len(images), num_images), figsize=(15, 5))
    if not isinstance(axes, np.ndarray):
        axes = [axes]
        
    for i, ax in enumerate(axes):
        ax.imshow(images[i])
        ax.set_title(batch["instance_prompt"])
        ax.axis("off")
    plt.tight_layout()
    plt.show()

def show_results(results_dir, prompt):
    """
    Shows generated images from the results directory.
    """
    import os
    files = [f for f in os.listdir(results_dir) if f.endswith(('.png', '.jpg'))]
    if not files:
        print("No results found.")
        return

    images = [Image.open(os.path.join(results_dir, f)) for f in files[:4]]
    fig, axes = plt.subplots(1, len(images), figsize=(20, 5))
    if not isinstance(axes, np.ndarray):
        axes = [axes]
        
    for i, ax in enumerate(axes):
        ax.imshow(images[i])
        ax.axis("off")
    plt.suptitle(f"Results for: {prompt}")
    plt.tight_layout()
    plt.show()

def save_image(tensor, path):
    """
    Saves a normalized torch tensor as an image.
    """
    img = (tensor + 1.0) / 2.0
    img = img.clamp(0, 1)
    img = img.cpu().permute(1, 2, 0).numpy()
    img = (img * 255).astype(np.uint8)
    Image.fromarray(img).save(path)
