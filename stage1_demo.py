import torch
import os
from diffusers import StableDiffusionPipeline
from PIL import Image

def run_stage1():
    model_id = "runwayml/stable-diffusion-v1-5"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # 1. Load Original Model
    print("Loading Original Stable Diffusion Pipeline...")
    pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16 if device == "cuda" else torch.float32)
    pipe.to(device)

    # 2. Demonstrate that the model doesn't know Cheburashka
    prompt = "<cheburashka> with the Eiffel Tower in the background"
    print(f"Generating image for prompt: '{prompt}'")
    
    # Set seed for reproducibility
    generator = torch.Generator(device).manual_seed(42)
    image = pipe(prompt, num_inference_steps=30, generator=generator).images[0]
    
    os.makedirs("./results", exist_ok=True)
    image.save("./results/stage1_raw_model.png")
    print("Saved results/stage1_raw_model.png")

    # 3. Encode prompt and save embeddings
    prompt_to_encode = "<cheburashka> plushie"
    print(f"Encoding prompt: '{prompt_to_encode}'")
    
    # We use pipe.encode_prompt to get text embeddings
    # This is useful because it handles the tokenizer and the text encoder
    prompt_embeds, negative_prompt_embeds = pipe.encode_prompt(
        prompt=prompt_to_encode,
        device=device,
        num_images_per_prompt=1,
        do_classifier_free_guidance=False
    )
    
    # Save embeddings for Stage 2
    os.makedirs("./data", exist_ok=True)
    torch.save(prompt_embeds, "./data/prompt_embeds.pt")
    print("Saved encoded prompt embeddings to data/prompt_embeds.pt")

    # 4. Cleanup to save VRAM
    import gc
    del pipe
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()
    print("Memory cleaned up.")

if __name__ == "__main__":
    run_stage1()
