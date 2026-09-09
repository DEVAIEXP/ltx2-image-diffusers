# LTX 2.3 Image Diffusers Runners

Local runners and Gradio apps for the pruned LTX 2.3 image-only Diffusers implementation.

## Online Demos

[![LTX 2.3 Image Base](https://img.shields.io/badge/Online%20Demo-LTX%202.3%20Image%20Base-ffcc4d?logo=huggingface&logoColor=black)](https://huggingface.co/spaces/elismasilva/ltx2.3-image)
[![LTX 2.3 Image Distilled](https://img.shields.io/badge/Online%20Demo-LTX%202.3%20Image%20Distilled-ffcc4d?logo=huggingface&logoColor=black)](https://huggingface.co/spaces/elismasilva/ltx2.3-image-distilled)

The command-line runners are intentionally thin: they load the requested model variant, encode the prompt, call the Diffusers pipeline, save the image, and write benchmark metrics. The Gradio apps expose the same T2I and optional I2I flow with the defaults selected during local testing.

## Installation

Clone this repository:

```powershell
git clone https://github.com/DEVAIEXP/ltx2-image-diffusers.git
cd ltx2-image-diffusers
```

Create and sync the Python environment with `uv`:

```powershell
uv sync
```

Activate the environment:

```powershell
.\.venv\Scripts\Activate.ps1
```

Note: this project currently depends on a temporary Diffusers branch (`DEVAIEXP/diffusers`, branch `ltx-image`) while the LTX 2.3 image pipelines are not available in the official upstream package yet. This dependency is configured in `pyproject.toml` and is installed by `uv sync`. Windows uses `triton-windows` and a pinned FlashAttention wheel. Linux/WSL uses the regular `triton` package, but FlashAttention is intentionally not installed by `uv sync` because it often needs to be built against the same CUDA/Torch environment already present in the virtualenv.

Optional Linux/WSL FlashAttention install, after `uv sync` and after activating the environment:

```bash
uv pip install flash-attn==2.8.3.post1 --no-build-isolation
```

`psutil` and `wheel` are included in the environment because FlashAttention commonly expects them during this manual build path. If a compatible prebuilt Linux wheel exists for your Python, Torch, and CUDA combination, prefer installing that wheel instead of building locally.

Run the distilled app locally with:

```powershell
python app_distilled.py
```

Run the base app locally with:

```powershell
python app_base.py
```

## Model Paths

The runners and apps use these Hugging Face model repositories by default:

- distilled bf16: `elismasilva/ltx2.3-image-distilled-1.1`
- distilled SDNQ int4/uint4: `elismasilva/ltx2.3-image-distilled-1.1-sdnq-int4`
- distilled SDNQ int8: `elismasilva/ltx2.3-image-distilled-1.1-sdnq-int8`
- base bf16: `elismasilva/ltx2.3-image-base`
- base SDNQ int4/uint4: `elismasilva/ltx2.3-image-base-sdnq-int4`
- base SDNQ int8: `elismasilva/ltx2.3-image-base-sdnq-int8`

The Gradio apps can override these defaults with environment variables: `LTX_DISTILLED_MODEL_PATH`, `LTX_DISTILLED_SDNQ_INT4_PATH`, `LTX_DISTILLED_SDNQ_INT8_PATH`, `LTX_BASE_MODEL_PATH`, `LTX_BASE_SDNQ_INT4_PATH`, and `LTX_BASE_SDNQ_INT8_PATH`.

Update the `MODEL_PATH` / `SDNQ_MODEL_PATHS` constants in the scripts if your folders or repository ids are different.

## Quick Start

Traditional Diffusers distilled T2I:

```powershell
python run_distilled.py
```

Modular custom-block distilled T2I:

```powershell
python run_modular_distilled.py
```

Modular custom-block distilled I2I:

```powershell
python run_modular_distilled_img2img.py --input-image path\to\input.png
```

These commands use the tested defaults. Every `run*.py` script now accepts CLI arguments; use `--help` on any runner to see the supported prompt, size, seed, step, PAG, LoRA, metrics, and I2I options.

## Traditional Runners

These scripts use the regular Diffusers pipeline classes.

| Script | Mode | Model family | Quantization |
| --- | --- | --- | --- |
| `run_distilled.py` | T2I | distilled | bf16 |
| `run_sdnq_distilled.py` | T2I | distilled | SDNQ int8 by default, `--bits 4` optional |
| `run_base.py` | T2I | base | bf16 |
| `run_sdnq_base.py` | T2I | base | SDNQ int8 by default, `--bits 4` optional |
| `run_distilled_img2img.py` | I2I | distilled | bf16 |
| `run_sdnq_distilled_img2img.py` | I2I | distilled | bf16 by default, `--bits 4` / `--bits 8` optional |
| `run_base_img2img.py` | I2I | base | bf16 |
| `run_sdnq_base_img2img.py` | I2I | base | SDNQ int8 by default, `--bits 4` optional |

Traditional examples:

```powershell
python run_distilled.py --prompt "a cinematic portrait of a robot barista"
python run_sdnq_distilled.py --bits 8
python run_base.py --steps 28 --guidance-scale 3.0
python run_sdnq_base.py --bits 4
python run_distilled_img2img.py --input-image path\to\input.png --strength 0.2
python run_sdnq_distilled_img2img.py --input-image path\to\input.png --bits 8 --soft-lora
python run_base_img2img.py --input-image path\to\input.png
python run_sdnq_base_img2img.py --input-image path\to\input.png --bits 4
```

## Modular Runners

These scripts use the custom Modular Diffusers blocks in `custom_blocks/ltx2_image` and load components step by step to keep memory pressure lower during local experiments.

| Script | Mode | Model family | Quantization |
| --- | --- | --- | --- |
| `run_modular_distilled.py` | T2I | distilled | bf16 |
| `run_modular_sdnq_distilled.py` | T2I | distilled | SDNQ int8 by default, `--bits 4` optional |
| `run_modular_base.py` | T2I | base | bf16 |
| `run_modular_sdnq_base.py` | T2I | base | SDNQ int8 by default, `--bits 4` optional |
| `run_modular_distilled_img2img.py` | I2I | distilled | bf16 |
| `run_modular_sdnq_distilled_img2img.py` | I2I | distilled | SDNQ int8 by default, `--bits 4` optional |
| `run_modular_base_img2img.py` | I2I | base | bf16 |
| `run_modular_sdnq_base_img2img.py` | I2I | base | SDNQ int8 by default, `--bits 4` optional |

Modular examples:

```powershell
python run_modular_distilled.py
python run_modular_sdnq_distilled.py --bits 8
python run_modular_base.py --steps 28 --guidance-scale 3.0
python run_modular_sdnq_base.py --bits 4
python run_modular_distilled_img2img.py --input-image path\to\input.png --strength 0.2
python run_modular_sdnq_distilled_img2img.py --input-image path\to\input.png --bits 8 --soft-lora
python run_modular_base_img2img.py --input-image path\to\input.png
python run_modular_sdnq_base_img2img.py --input-image path\to\input.png --bits 4
```

## Apps

| Script | Mode | Model family | Quantization |
| --- | --- | --- | --- |
| `app_distilled.py` | T2I + optional I2I | distilled | bf16 app defaults |
| `app_base.py` | T2I + optional I2I | base | bf16 app defaults |

## App Defaults

The Gradio apps use these UI defaults:

- resolution: `1920x1056`
- seed: `43`
- I2I: disabled by default
- I2I accordion: collapsed by default
- I2I preprocess: enabled by default
- I2I `strength=0.20`
- I2I `input_noise_sigma=5.0`
- I2I `input_sharpen=1.3`
- I2I structured noise cutoff: enabled by default, with `phase_cutoff_radius=9.0`
- PAG: disabled by default
- LoRAs: disabled by default

If `Input Preprocess` is disabled, the app sends `input_noise_sigma=0.0` and `input_sharpen=1.0` to the pipeline, regardless of the slider values. This makes it easy to compare raw I2I against the tested preprocessing pass. CLI I2I runners expose the same behavior with `--no-input-preprocess`.

## Runner Defaults

Distilled T2I defaults:

- resolution: `1280x704`
- seed: `43`
- steps: `8`
- guidance scale: `1.0`
- PAG: disabled
- negative prompt: empty
- text encoder: original bf16 unless `--text-encoder-bits 8` is passed on SDNQ scripts

Base T2I defaults:

- resolution: `1280x704`
- seed: `43`
- steps: `28`
- guidance scale: `3.0`
- guidance rescale: `0.7`
- PAG: disabled
- negative prompt: enabled

I2I runner defaults:

- resolution: `1280x704`
- seed: `43`
- strength: `0.20`
- `input_noise_sigma=5.0`
- `input_sharpen=1.3`
- `phase_cutoff=9.0` by default; pass `--phase-cutoff 0` to disable structured noise
- PAG: disabled
- LoRAs: disabled unless explicitly requested

## Pipelines

`LTX2ImagePipeline` is the T2I pipeline for the pruned image-only LTX 2.x transformer.

`LTX2ImageImg2ImgPipeline` is the I2I pipeline. It takes an input image, resizes it, optionally applies lightweight pixel preprocessing, encodes it with the LTX VAE as a single frame, adds scheduler-consistent FlowMatch noise, runs a partial denoising pass, and decodes the output image.

This is different from video image-conditioning. The image is not used as a first-frame condition; it is used as the initial latent state for an img2img trajectory.

## I2I Parameters

`strength` controls drift. Lower values preserve identity, crop, geometry, and background layout. Higher values can create more texture but also increase scene changes.

`Input Preprocess` controls whether pixel-space preprocessing is applied before VAE encoding. When disabled, preprocessing is neutral: `input_noise_sigma=0.0` and `input_sharpen=1.0`.

`input_sharpen` is a simple PIL sharpness pre-pass before VAE encoding. In recent tests, `1.3` kept useful leaf/edge definition without making the image look too processed.

`input_noise_sigma` applies deterministic pixel-space noise before VAE encoding. In recent tests, `5.0` was a better default than `7.0`: it kept most of the detail gain with less artificial grain.

`phase_cutoff` enables Frequency Selective Structured noise from Phase-Preserving Diffusion. The current implementation preserves low-frequency latent phase from the input image inside the cutoff radius and uses noise phase outside it. A radius around `9.0` gave a good balance between structure preservation and detail reconstruction in the butterfly/leaf tests.

`pag_scale` enables Perturbed Attention Guidance. The default runners keep PAG disabled because it was not the selected baseline.

## Prompt Guidance

For I2I, using the same prompt as T2I is usually safer when the output image will be fed back into a video workflow. It keeps the model aligned with the same scene semantics.

A useful I2I prompt suffix is:

`Preserve the exact same composition, camera framing, subject identity, pose, lighting, background layout, and geometry. Restore crisp fine detail, natural micro texture, clean edges, realistic material detail, and high-resolution sharpness. Do not change the scene.`

For distilled runs, negative prompts are normally left empty. For base runs, the runners and base app keep a standard negative prompt enabled.

## LoRA Options

The runners and apps only keep the two LoRAs that were useful in the current tests:

- `vrgamedevgirl84/LTX_2.3_Soft_Enhance_Style_LoRa` exposed as Enhance LoRA
- `vrgamedevgirl84/LTX_2.3_Crisp_Enhance_Style_LoRa` exposed as Crisp LoRA

Enhance is the safer natural/detail option. Crisp is the stronger detail option and should be checked for style drift.

Use Enhance in I2I:

```powershell
python run_sdnq_distilled_img2img.py --input-image path\to\input.png --soft-lora
```

Use Crisp in I2I:

```powershell
python run_sdnq_distilled_img2img.py --input-image path\to\input.png --crisp-lora
```

Use Enhance lightly in distilled T2I when matching a video pipeline that also uses the same LoRA family:

```powershell
python run_sdnq_distilled.py --soft-lora
```

## SDNQ Notes

Two transformer variants are worth keeping:

- `uint4/int4`: best for smaller distribution size. The current policy gave the best balance between file size and visual closeness to bf16.
- `int8`: best visual fidelity. The default/hadamard int8 output was very close to bf16, while the current int8 policy reduced runtime in local tests and remained visually close.

On Windows, runtime varied a lot depending on other GPU users and SDNQ version/kernel warmup. Measure at least one warm run before comparing policies.

## Tested Environment

Tests were run on an NVIDIA RTX 3060 Ti 8 GB system with 64 GB RAM on Windows.

Most image quality comparisons used the default/original bf16 text encoder unless explicitly testing an SDNQ text encoder. Text encoder quantization can slightly change prompt interpretation and composition.

## Quality Checks

When comparing outputs, inspect both detail and drift:

- subject identity
- eye and face geometry
- crop and camera framing
- background object stability
- edge stability around hair, fur, leaves, whiskers, and thin structures
- whether detail is truly recovered or only sharpened/noisier
