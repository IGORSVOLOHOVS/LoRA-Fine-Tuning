import torch
import os
import yaml
from diffusers import StableDiffusionPipeline
from peft import PeftModel

def generate(config_path="configs/train_config_v2.yaml"):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_id = config["model"]["pretrained_model_name_or_path"]
    lora_path = os.path.join(config["output"]["output_dir"], "lora_final")

    print(f"Loading base model: {model_id}")
    pipe = StableDiffusionPipeline.from_pretrained(
        model_id, 
        torch_dtype=torch.float16 if device == "cuda" else torch.float32
    )
    pipe.to(device)

    if os.path.exists(lora_path):
        print(f"Loading LoRA weights from: {lora_path}")
        pipe.unet = PeftModel.from_pretrained(pipe.unet, lora_path)
    else:
        print("LoRA weights not found. Generating with base model.")

    prompts = [
        "<cheburashka> with the Eiffel Tower in the background",
        "<cheburashka> plushie",
        "<cheburashka> in sketch style",
        "<cheburashka> riding a bycycle"
    ]

    os.makedirs(config["output"]["results_dir"], exist_ok=True)
    
    inference_config = config.get("inference", {})
    guidance_scale = inference_config.get("guidance_scale", 7.5)
    num_inference_steps = inference_config.get("num_inference_steps", 30)
    
    for i, prompt in enumerate(prompts):
        print(f"Generating image {i+1}/{len(prompts)} for prompt: {prompt}")
        image = pipe(
            prompt, 
            num_inference_steps=num_inference_steps, # denoising steps; 30 is balanced for v1.5
            guidance_scale=guidance_scale # prompt fidelity; 7.5 provides good variety/accuracy
        ).images[0]
        image.save(os.path.join(config["output"]["results_dir"], f"generation_{i}.png"))

    print(f"All images saved to {config['output']['results_dir']}")

if __name__ == "__main__":
    generate()
