import os
import yaml
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from diffusers import DDPMScheduler, UNet2DConditionModel, AutoencoderKL
from diffusers.optimization import get_scheduler
from transformers import CLIPTextModel, CLIPTokenizer
from peft import LoraConfig, get_peft_model

from dataset import CheburashkaDataset

def train(config_path="configs/train_config_v2.yaml"):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    os.makedirs(config["output"]["output_dir"], exist_ok=True)
    os.makedirs(config["output"]["results_dir"], exist_ok=True)
    writer = SummaryWriter(log_dir=os.path.join(config["output"]["output_dir"], "logs"))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    weight_dtype = torch.float16 if device == "cuda" else torch.float32

    model_id = config["model"]["pretrained_model_name_or_path"]
    
    tokenizer = CLIPTokenizer.from_pretrained(model_id, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(model_id, subfolder="text_encoder").to(device, dtype=weight_dtype)
    vae = AutoencoderKL.from_pretrained(model_id, subfolder="vae").to(device, dtype=weight_dtype)
    unet = UNet2DConditionModel.from_pretrained(model_id, subfolder="unet").to(device, dtype=weight_dtype)
    noise_scheduler = DDPMScheduler.from_pretrained(model_id, subfolder="scheduler")

    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)

    lora_config = LoraConfig(
        r=config["training"]["lora_rank"], # capacity of adapters; 128 is high but good for specific concepts
        lora_alpha=config["training"]["lora_rank"], # scale factor; matching rank simplifies tuning
        init_lora_weights="gaussian", # random starts; better for convergence than zeros
        target_modules=["to_k", "to_q", "to_v", "to_out.0"], # attention projections; standard LoRA targets for SD
    )
    unet = get_peft_model(unet, lora_config)
    unet.print_trainable_parameters()

    dataset = CheburashkaDataset(
        data_dir=config["data"]["instance_data_dir"],
        instance_prompt=config["data"]["instance_prompt"],
        size=config["data"]["resolution"],
        center_crop=config["data"]["center_crop"]
    )
    train_dataloader = DataLoader(
        dataset, 
        batch_size=config["training"]["train_batch_size"], 
        shuffle=True
    )

    optimizer = torch.optim.AdamW(
        unet.parameters(), 
        lr=float(config["training"]["learning_rate"]), # 1e-4 is standard for LoRA fine-tuning
        weight_decay=1e-2
    )
    
    lr_scheduler = get_scheduler(
        config["training"]["lr_scheduler"],
        optimizer=optimizer,
        num_warmup_steps=config["training"]["lr_warmup_steps"],
        num_training_steps=config["training"]["max_train_steps"],
    )


    progress_bar = tqdm(range(config["training"]["max_train_steps"]), desc="Training")
    global_step = 0
    
    unet.train()
    
    while global_step < config["training"]["max_train_steps"]:
        for step, batch in enumerate(train_dataloader):
            if global_step >= config["training"]["max_train_steps"]:
                break
                
            # Convert images to latents
            pixel_values = batch["instance_images"].to(device, dtype=weight_dtype)
            latents = vae.encode(pixel_values).latent_dist.sample()
            latents = latents * vae.config.scaling_factor

            bsz = latents.shape[0]
            timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device)
            timesteps = timesteps.long()
            noise = torch.randn_like(latents)
            noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

            inputs = tokenizer(
                batch["instance_prompt"], 
                padding="max_length", 
                max_length=tokenizer.model_max_length, 
                truncation=True, 
                return_tensors="pt"
            ).to(device)
            encoder_hidden_states = text_encoder(inputs.input_ids)[0]

            model_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample

            target = noise

            snr_gamma = config["training"].get("snr_gamma") # 5.0 caps max loss contribution from early steps
            if snr_gamma is not None:
                # Compute SNR
                alphas_cumprod = noise_scheduler.alphas_cumprod.to(device)
                sigmas = ((1 - alphas_cumprod) / alphas_cumprod) ** 0.5
                snr = 1.0 / (sigmas ** 2)
                
                # Compute weighting for the specific timesteps
                mse_loss_weights = (
                    torch.stack([snr[t] for t in timesteps]).to(device)
                )
                mse_loss_weights = torch.minimum(mse_loss_weights, torch.ones_like(mse_loss_weights) * snr_gamma) / mse_loss_weights
                
                loss = F.mse_loss(model_pred.float(), target.float(), reduction="none")
                loss = loss.mean(dim=list(range(1, len(loss.shape)))) * mse_loss_weights
                loss = loss.mean()
            else:
                loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
            
            # Optimization
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(unet.parameters(), config["training"]["max_grad_norm"])
            optimizer.step()
            lr_scheduler.step()

            # Update progress
            progress_bar.update(1)
            global_step += 1
            
            # Logging
            writer.add_scalar("Loss/train", loss.item(), global_step)
            writer.add_scalar("LR", lr_scheduler.get_last_lr()[0], global_step)
            
            if global_step % 100 == 0:
                progress_bar.set_postfix({"loss": loss.item()})

            # Checkpointing
            if global_step % config["training"]["checkpointing_steps"] == 0:
                save_path = os.path.join(config["output"]["output_dir"], f"checkpoint-{global_step}")
                unet.save_pretrained(save_path)
                print(f"Saved checkpoint to {save_path}")

    # Final save
    unet.save_pretrained(os.path.join(config["output"]["output_dir"], "lora_final"))
    writer.close()
    print("Training completed!")

if __name__ == "__main__":
    train()