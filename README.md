# lo-ra-fine-tuning

Fine-tuning Stable Diffusion v1.5 with Low-Rank Adaptation (LoRA) to learn one
new concept, **Cheburashka**, from a dataset of three images. The base model has
never seen the character; after 1000 training steps on the UNet's LoRA adapters it
draws it on request.

## Results

| Base model (stage 1) | LoRA fine-tuned (stage 3) |
|:---:|:---:|
| ![Base model failure](results/stage1_raw_model.png) | ![LoRA success](results/generation_0.png) |
| *Prompt: "&lt;cheburashka&gt; with the Eiffel Tower in the background"* | *Same prompt* |

| Cheburashka riding a bicycle |
|:---:|
| ![Bicycle](results/generation_3.png) |

`results/` holds the run with `configs/train_config.yaml` and `results_v2/` the
run with `configs/train_config_v2.yaml`. Four of the nine PNG files there are
843 bytes each, too small to be a real 512 by 512 sample.

## What is here

- `notebook.ipynb` runs the three stages in order, and its outputs are saved in it.
- `stage1_demo.py`: stage 1, the base model fails to draw the concept, and the
  prompt `<cheburashka> plushie` is encoded and saved to `data/prompt_embeds.pt`.
- `train.py`: stage 2, LoRA adapters of rank 128 on the UNet, trained for 1000
  steps with Min-SNR weighting (gamma 5.0), TensorBoard logs and a checkpoint every
  500 steps.
- `inference.py`: stage 3, four prompts through the base model with the trained
  adapters loaded.
- `dataset.py`: `CheburashkaDataset`, the images resized and centre-cropped to 512.
- `configs/`: two sets of hyperparameters, each value commented with why it was
  chosen. The scripts read `train_config_v2.yaml` by default.
- `utils/`: TensorBoard launcher and plotting helpers. `docs/workflow.md`
  describes the training and inference pipelines with diagrams.

## Running it

This repository ships no program, and nothing in it runs without two things it
does not contain:

- **The weights.** Every script downloads `runwayml/stable-diffusion-v1-5` from
  Hugging Face on first use, several gigabytes.
- **The data.** Training reads its images from `data/cheburashka/`, which is
  gitignored and in no commit. Put your own images there.

With both, a CUDA GPU with about 10 GB of memory, Python with PyTorch,
`diffusers`, `transformers`, `peft`, `accelerate`, `torchvision`, `pyyaml` and
`tensorboard`:

```bash
python stage1_demo.py
python train.py
python inference.py
```

The repository pins none of these packages.

## Working on it

`.\recipe.ps1` is the whole interface: `.\recipe.ps1 test` runs pytest and ruff
over `.claude`, and nothing over the scripts above, which would have to download
the weights first. `.\recipe.ps1 info` prints the metadata. The other verbs print
`nothing to <verb>`. `CLAUDE.md` holds the rules.

## Licence

MIT. See [LICENSE](LICENSE).

## Lab requirements

This repository complies with [revision 4.0.0](http://127.0.0.1:3000/igors-lab/software-requirements/src/tag/v4.0.0/BEST_REQUIREMENTS.md) of the lab requirements, except R-02 (#7), R-09 (#8), R-13 (#9), R-37 (#10), R-41 (#11), R-42 (#12).
