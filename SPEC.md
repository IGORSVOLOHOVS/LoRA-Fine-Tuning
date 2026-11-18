# Software requirements specification: lo-ra-fine-tuning

Reverse-engineered from the code at commit `62b7b11` on 2026-11-18. Every requirement
describes behaviour found in the code and cites the code that implements it.
References to notebook cells use 0-based cell indices. The hub document is
[docs/README.md](docs/README.md).

## 1. Purpose and scope

The project fine-tunes Stable Diffusion v1.5 with LoRA adapters on the UNet so that
the token `<cheburashka>` produces one concept learned from a small image set
(`README.md:3-6`).

In scope:

- `stage1_demo.py`: what the base model draws for the prompt
- `dataset.py`: image loading and preprocessing
- `train.py`: LoRA training
- `inference.py`: generation with the trained adapter
- `utils/`: TensorBoard launcher and plotting helpers
- `configs/*.yaml`, `notebook.ipynb`, and the committed result images

Out of scope:

- the base weights and the training images, which are not in the repository
  (`README.md:44-47`, `.gitignore:31`)
- training the text encoder or the VAE, which are frozen (`train.py:35-36`)
- installation and packaging (`recipe.ps1:16`)
- quantitative evaluation of generated images, for which no code exists
- the repository tooling, which is described by repo-quality-template

## 2. Context

| Item | Description | Source |
|---|---|---|
| Actor | A person who runs the three scripts or the notebook on a machine with the data and the packages. Who these users are is not stated | `README.md:38-57` |
| External system | Hugging Face Hub, model `runwayml/stable-diffusion-v1-5`, loaded with `from_pretrained` | `configs/train_config_v2.yaml:3`, `stage1_demo.py:7`, `train.py:29-33`, `inference.py:16` |
| External tool | TensorBoard, which reads `<output_dir>/logs` | `train.py:22`, `utils/open_tensorboard.py:25` |
| Runtime | Python with PyTorch. Uses CUDA when available, otherwise the CPU. The notebook kernel records Python 3.10.0 | `train.py:24-25`, `notebook.ipynb` metadata |
| Working directory | The repository root. All paths are relative to it, and `train.py` imports `dataset` from it | `train.py:14`, `configs/train_config_v2.yaml:20,30-31` |

## 3. Functional requirements

### 3.1 Stage 1: base model demonstration (`stage1_demo.py`)

- **FR-1** The system shall load `StableDiffusionPipeline` from the hard-coded id `runwayml/stable-diffusion-v1-5`, in float16 on CUDA and float32 otherwise. Source: `stage1_demo.py:7-14`. Verification: no test.
- **FR-2** The system shall generate one image for `<cheburashka> with the Eiffel Tower in the background` with 30 inference steps and a generator seeded with 42, and save it as `./results/stage1_raw_model.png`. Source: `stage1_demo.py:17-26`. Verification: no test. The notebook output in cell 2 records the file being saved.
- **FR-3** The system shall encode `<cheburashka> plushie` with `pipe.encode_prompt` without classifier-free guidance and save the tensor to `./data/prompt_embeds.pt`. Source: `stage1_demo.py:29-44`. Verification: no test.
- **FR-4** The system shall delete the pipeline, run garbage collection and, on CUDA, empty the CUDA cache. Source: `stage1_demo.py:47-52`. Verification: no test.

### 3.2 Dataset (`dataset.py`)

- **FR-5** The system shall raise `ValueError` when the data directory does not exist. Source: `dataset.py:15-16`. Verification: no test.
- **FR-6** The system shall treat as images the files in the data directory whose names end in `.png`, `.jpg` or `.jpeg` (case-insensitive), and report their count as the dataset length. Source: `dataset.py:18-25,34-35`. Verification: no test.
- **FR-7** The system shall resize each image (bilinear) so that its shorter side is `size`, then centre-crop it to `size` by `size` (random crop when `center_crop` is false), convert it to a tensor and normalise it to [-1, 1]. Source: `dataset.py:27-32`. Verification: no test.
- **FR-8** The system shall convert non-RGB images to RGB and return each item as `{"instance_images": tensor, "instance_prompt": str}`. Source: `dataset.py:37-45`. Verification: no test.
- **FR-9** When run as a script, the system shall load `./data/cheburashka` with prompt `<cheburashka> plushie` and print the length, the keys, the image shape and the prompt of item 0. Source: `dataset.py:47-53`. Verification: no test.

### 3.3 Training (`train.py`)

- **FR-10** The system shall read a YAML config from `config_path` (default `configs/train_config_v2.yaml`) and create `output.output_dir` and `output.results_dir`. Source: `train.py:16-21`. Verification: no test.
- **FR-11** The system shall load the CLIP tokenizer, CLIP text encoder, VAE, UNet and `DDPMScheduler` from the model id's subfolders, and cast the models to float16 on CUDA or float32 otherwise. Source: `train.py:24-33`. Verification: no test.
- **FR-12** The system shall freeze the VAE and the text encoder, and wrap the UNet with a PEFT LoRA config: `r = lora_alpha = training.lora_rank`, Gaussian initialisation, and target modules `to_k`, `to_q`, `to_v`, `to_out.0`. It then prints the trainable parameter count. Source: `train.py:35-45`. Verification: no test. The notebook output in cell 4 records 25,509,888 trainable of 885,030,852 parameters.
- **FR-13** The system shall iterate over the dataset with batch size `training.train_batch_size` and shuffling. Source: `train.py:47-57`. Verification: no test.
- **FR-14** The system shall optimise with AdamW (learning rate `training.learning_rate`, weight decay 0.01) under the diffusers scheduler named by `training.lr_scheduler` with `training.lr_warmup_steps`. Source: `train.py:59-70`. Verification: no test.
- **FR-15** For each batch, the system shall encode the images to VAE latents scaled by `vae.config.scaling_factor`, sample one uniform timestep per sample in [0, `num_train_timesteps`), and add Gaussian noise with the scheduler. Source: `train.py:84-92`. Verification: no test.
- **FR-16** The system shall tokenise the instance prompt padded and truncated to the tokenizer's maximum length, take the text encoder's hidden states, and train the UNet to predict the added noise. Source: `train.py:94-105`. Verification: no test.
- **FR-17** When `training.snr_gamma` is set, the system shall compute a per-sample MSE weighted by min(SNR(t), gamma) / SNR(t), with SNR(t) = alphas_cumprod / (1 - alphas_cumprod). Otherwise it shall use the mean MSE. Source: `train.py:107-124`. Verification: no test.
- **FR-18** The system shall zero the gradients, backpropagate, clip the gradient norm to `training.max_grad_norm`, and step the optimiser and the scheduler, until `training.max_train_steps` steps are done, passing over the data repeatedly. Source: `train.py:73-81,127-135`. Verification: no test. The notebook output in cell 4 records 1000 of 1000 steps.
- **FR-19** The system shall write the scalars `Loss/train` and `LR` to TensorBoard in `<output_dir>/logs` at every step, and show the loss in the progress bar every 100 steps. Source: `train.py:22,138-142`. Verification: no test.
- **FR-20** The system shall save the adapter with `save_pretrained` to `<output_dir>/checkpoint-<step>` every `training.checkpointing_steps` steps, and to `<output_dir>/lora_final` after the loop. Source: `train.py:145-151`. Verification: no test. The notebook output in cell 4 records `checkpoint-500` and `checkpoint-1000`.

### 3.4 Inference (`inference.py`)

- **FR-21** The system shall read the YAML config (default `configs/train_config_v2.yaml`) and load `StableDiffusionPipeline` from the model id in float16 on CUDA or float32 otherwise. Source: `inference.py:7-20`. Verification: no test.
- **FR-22** When `<output_dir>/lora_final` exists, the system shall replace `pipe.unet` with `PeftModel.from_pretrained(pipe.unet, lora_path)`. Otherwise it shall print `LoRA weights not found. Generating with base model.` and continue with the base model. Source: `inference.py:13,22-26`. Verification: no test. The notebook output in cell 6 records `Loading LoRA weights from: ./checkpoints_v2\lora_final`.
- **FR-23** The system shall generate one image per fixed prompt (four prompts) using `inference.guidance_scale` (default 7.5) and `inference.num_inference_steps` (default 30), and save them as `<results_dir>/generation_<i>.png` for i = 0 to 3. Source: `inference.py:28-50`. Verification: no test.

### 3.5 Utilities (`utils/`)

- **FR-24** The TensorBoard launcher shall read `configs/train_config_v2.yaml` relative to its own file. It prints an error and returns if the file is missing, warns if `<repo>/<output_dir>/logs` does not exist (`output_dir` default `./checkpoints`), and runs `python -m tensorboard.main --logdir <dir>` with the current interpreter. Source: `utils/open_tensorboard.py:6-30`. Verification: no test.
- **FR-25** `visualize_batch` shall map a batch from [-1, 1] to [0, 1] and plot up to `num_images` images titled with the prompt. Source: `utils/visualization.py:6-28`. Verification: no test.
- **FR-26** `show_results` shall plot the first four `.png` or `.jpg` files listed in a directory, or print `No results found.` when there are none. Source: `utils/visualization.py:30-50`. Verification: no test.
- **FR-27** `save_image` shall write a CHW tensor in [-1, 1] as an 8-bit image file. Source: `utils/visualization.py:52-60`. Verification: no test.

### 3.6 Notebook (`notebook.ipynb`)

- **FR-28** The notebook shall run `run_stage1()` and display `results/stage1_raw_model.png` (cell 2), run `train("configs/train_config_v2.yaml")` (cell 4), and run `generate("configs/train_config_v2.yaml")` followed by `show_results('results', '<cheburashka>')` (cell 6). Source: `notebook.ipynb` cells 2, 4, 6. Verification: no test. The outputs are saved in the file.

### 3.7 Declared but not implemented

| Declaration | Source | What the code does |
|---|---|---|
| A fixed seed of 42 makes noise patterns identical between runs | `configs/train_config_v2.yaml:16`, `configs/train_config.yaml:17` | No `.py` file reads `training.seed`. Training and inference set no seed |
| `training.gradient_accumulation_steps: 1` | `configs/train_config_v2.yaml:17` | Never read. Every step updates the weights directly (`train.py:127-131`) |
| `model.revision: null` | `configs/train_config_v2.yaml:4`, `configs/train_config.yaml:5` | Never read. `from_pretrained` gets no revision |
| Prompt embeddings saved "for Stage 2" | `stage1_demo.py:41-43` | `train.py` does not load `data/prompt_embeds.pt` |
| `accelerate` as a requirement | `README.md:50` | No file imports it |
| `data/cheburashka` holds three training images | `notebook.ipynb` cell 0, `README.md:4` | The directory is gitignored and not in any commit (`.gitignore:31`, `README.md:46-47`) |
| VRAM stays under 8-10 GB; about 10 GB needed | `configs/train_config_v2.yaml:9`, `README.md:49` | Stated only. No measurement is recorded |

## 4. Interfaces

### 4.1 Command line

No script parses arguments or environment variables.

| Command | Effect | Source |
|---|---|---|
| `python stage1_demo.py` | FR-1 to FR-4 | `README.md:54`, `stage1_demo.py:54-55` |
| `python train.py` | FR-10 to FR-20 with the v2 config | `README.md:55`, `train.py:155-156` |
| `python inference.py` | FR-21 to FR-23 with the v2 config | `README.md:56`, `inference.py:52-53` |
| `python dataset.py` | FR-9 | `dataset.py:47-53` |
| `python utils/open_tensorboard.py` | FR-24 | `utils/open_tensorboard.py:32-33` |

### 4.2 Python functions

| Function | Signature | Source |
|---|---|---|
| `run_stage1` | `run_stage1()` | `stage1_demo.py:6` |
| `CheburashkaDataset` | `CheburashkaDataset(data_dir, instance_prompt, size=512, center_crop=True)` | `dataset.py:9` |
| `train` | `train(config_path="configs/train_config_v2.yaml")` | `train.py:16` |
| `generate` | `generate(config_path="configs/train_config_v2.yaml")` | `inference.py:7` |
| `open_tensorboard` | `open_tensorboard()` | `utils/open_tensorboard.py:6` |
| `visualize_batch`, `show_results`, `save_image` | `(batch, num_images=3)`, `(results_dir, prompt)`, `(tensor, path)` | `utils/visualization.py:6,30,52` |

### 4.3 Config file

The config is YAML with sections `model`, `training`, `data`, `inference` and `output`
(`configs/train_config_v2.yaml`). The two committed files differ only in
`training.learning_rate` (2e-5 against 1e-4), `output.output_dir` and
`output.results_dir`. The key table, with readers and defaults, is in
[docs/README.md, Configuration](docs/README.md#configuration).

### 4.4 Outputs

| Path | Written by | Format |
|---|---|---|
| `results/stage1_raw_model.png` | `stage1_demo.py:25` | PNG, 512 by 512 RGB (decoded copy in the repository) |
| `data/prompt_embeds.pt` | `stage1_demo.py:43` | `torch.save` of a tensor |
| `<output_dir>/checkpoint-<step>/`, `<output_dir>/lora_final/` | `train.py:147,151` | PEFT `save_pretrained` directory |
| `<output_dir>/logs/` | `train.py:22` | TensorBoard event files, scalars `Loss/train` and `LR` |
| `<results_dir>/generation_<i>.png` | `inference.py:48` | PNG, one per prompt, i = 0 to 3 |

## 5. Data

| Entity | Storage | Notes |
|---|---|---|
| Training images | `data/cheburashka/` | Not in git (`.gitignore:31`) |
| Base model | Hugging Face cache | Downloaded on first use (`README.md:44-45`) |
| LoRA adapter | `checkpoints/`, `checkpoints_v2/` | Not in git (`.gitignore:33-38`) |
| Result images | `results/` (5 files), `results_v2/` (4 files) | Git LFS pointers in the commits (`.gitattributes:9`). Four files are 512 by 512 and entirely black |
| Recorded run | `notebook.ipynb` outputs | Covers the v2 config only. No run log exists for `results/` |

## 6. Non-functional requirements

- **NFR-1** The scripts shall run on CUDA in float16 when CUDA is available, and on the CPU in float32 otherwise. Source: `train.py:24-25`, `inference.py:11,18`, `stage1_demo.py:8,13`.
- **NFR-2** Stage 1 shall release the pipeline and the CUDA cache before it returns, so that the notebook can go on to training. Source: `stage1_demo.py:46-52`.
- **NFR-3** Training images shall be processed at the configured resolution, 512 in both configs. Source: `configs/train_config_v2.yaml:22`, `configs/train_config.yaml:23`, `dataset.py:27-29`.
- **NFR-4** Stage 1 sampling shall be seeded (seed 42). This is the only seeded step. Source: `stage1_demo.py:21`.
- **NFR-5** Data, weights, checkpoints and tensors (`*.pt`, `*.bin`, `*.ckpt`, `*.safetensors`) shall stay out of git, and PNG and JPG files shall go through Git LFS. Source: `.gitignore:31-38`, `.gitattributes:9-10`.
- **NFR-6** The repository shall comply with revision 4.0.0 of the lab requirements, except R-02, R-09, R-13, R-41 and R-42, through `recipe.ps1` and the lab tooling it shares with repo-quality-template. Source: `README.md:74`, `recipe.ps1:1-74`.

The recorded timings are observations, not requirements: 1000 training steps in
5 min 24 s at about 3.05 steps/s, and 3 to 4 s per 30-step image (`notebook.ipynb`,
outputs of cells 4 and 6). The device model is not recorded.

## 7. Constraints and dependencies

No lock or requirements file exists, and no version is pinned (`README.md:59`).

| Package | Used for | Source |
|---|---|---|
| torch, torchvision | tensors, DataLoader, AdamW, TensorBoard writer, transforms | `train.py:3-7`, `dataset.py:2-5` |
| diffusers | `DDPMScheduler`, `UNet2DConditionModel`, `AutoencoderKL`, `get_scheduler`, `StableDiffusionPipeline` | `train.py:9-10`, `inference.py:4`, `stage1_demo.py:3` |
| transformers | `CLIPTextModel`, `CLIPTokenizer` | `train.py:11` |
| peft | `LoraConfig`, `get_peft_model`, `PeftModel` | `train.py:12`, `inference.py:5` |
| PyYAML, tqdm, Pillow, numpy, matplotlib, tensorboard | config, progress bar, images, plots, log viewer | `train.py:2,5`, `dataset.py:3`, `utils/visualization.py:1-4`, `utils/open_tensorboard.py:25` |

Further constraints:

- The pipeline is built without arguments for the safety checker, so the diffusers
  default applies. The notebook output in cell 6 records it replacing two images
  with black ones (`inference.py:16-19`).
- The code is MIT-licensed (`LICENSE`). The licence of the training images and of
  the base model is not stated in the repository.

## 8. Acceptance

There are no automated tests. A run meets this specification when, from the
repository root and with images in `data/cheburashka/`:

1. `python dataset.py` prints a non-zero length and the shape `torch.Size([3, 512, 512])` (FR-5 to FR-9).
2. `python stage1_demo.py` writes `results/stage1_raw_model.png` and `data/prompt_embeds.pt` (FR-1 to FR-4).
3. `python train.py` prints the trainable parameter count and finishes 1000 steps. Afterwards `checkpoints_v2/` holds `checkpoint-500`, `checkpoint-1000`, `lora_final` and `logs` (FR-10 to FR-20).
4. `python utils/open_tensorboard.py` shows `Loss/train` and `LR` for 1000 steps (FR-19, FR-24).
5. `python inference.py` prints `Loading LoRA weights from` and writes `results_v2/generation_0.png` to `generation_3.png` (FR-21 to FR-23). Check that each image is not entirely black.
6. `.\recipe.ps1 test` exits with 0 (NFR-6).

## 9. Open questions

- Who uses this project and why (coursework, a demo, research)? The repository does not say.
- Which package versions and which GPU produced the recorded run and the committed images? The notebook's log records only the base model's Hub snapshot, `451f4fe16113bff5a5d2269ed5ad43b0592e9a14` (outputs of cells 2 and 6).
- Is `runwayml/stable-diffusion-v1-5` still downloadable from the Hub? This was not checked: no network calls were made.
- Should `training.seed`, `training.gradient_accumulation_steps` and `model.revision` take effect, or be removed?
- What is `data/prompt_embeds.pt` meant to be used for in stage 2?
- Are the black images from the safety checker accepted, or should prompts, seeds or the checker change? Was the safety checker also the cause in `results/`?
- Which run produced `results/`? `README.md:19-20` attributes it to `configs/train_config.yaml`, but the notebook records only the v2 run.
- Where do the three training images come from, and under what licence?
- What do the exceptions R-13, R-41 and R-42 cover? They are tracked as Gitea issues #9, #11 and #12, which this clone cannot see. Partly answered elsewhere: `docs/BEST_REQUIREMENTS-4.0.0-gap-report.md` on repo-quality-template's `release` branch ties R-13 to ctest and clang-tidy, R-41 to issue fields (a `Priority/*` label, an assignee, a milestone) and R-42 to the README line that states the revision. Why this repository is excepted from each is not recorded.
- Has a Gitea runner run the `propose` and `release-gate` workflows yet? `CLAUDE.md:68-70` makes branch protection depend on it.
