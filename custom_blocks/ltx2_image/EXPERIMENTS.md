# LTX 2.3 Image Modular Experiments

This file tracks local benchmark results and implementation notes for the experimental LTX 2.3 image-only Modular Diffusers pipeline.

The goal is to keep enough context in the repository so experiments can continue even if chat context is lost.

## Test Setup

Known setup from the current experiments:

- Repository: `E:\ProjetosIA\ltx2.3-image-diffusers`
- Branch: `experiment/dynamic-vram-manager`
- Runner: `run_modular_distilled.py`
- Model path and precision: distilled image-only BF16 path used by the runner
- Resolution: `1280x704`
- Steps: `8`
- Seed: `43`
- Transformer blocks: `48`
- Prompt embeds shape: `[1, 1024, 188160]`
- CUDA fallback memory: tested both enabled and disabled in NVIDIA settings
- Page file: Windows-managed, about `46 GB`
- System RAM: `64 GB`

## Important Baselines

| Mode | Encode | Transformer setup/load | Denoise | Pass 1 total | Torch alloc during denoise | Peak RAM | Notes |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Old `run_distilled.py`, good run | `58.0s` | `~1.5s` | `341.9s` | `346.4s` | not logged per step | `54.9 GB` | Standard Diffusers-style pipeline with group offload. |
| Old `run_distilled.py`, later slow run | `30.4s` | `~1.8s` | `594.2s` | `601.8s` | not logged per step | `55.4 GB` | Likely affected by Windows memory pressure or paging. |
| Modular Diffusers, leaf/group offload | `47.4s` | `~0.3s` | `423.5s` | `444.3s` | `0.39 GiB` | `28.8 GB` | Lower RAM than old runner, slower denoise. |
| Modular Diffusers, best group-offload outlier | `16.4s` | `~0.1s` | `263.4s` | `274.6s` | `0.39 GiB` | `28.8 GB` | Good run, but not stable. |
| Transformer fully on CUDA/fallback, manager off | `95.9s` | `110.4s` | `129.4s` | `251.6s` | `24.84 GiB` | `33.2 GB` | Fast denoise, expensive full transformer move. |
| Experimental `manual_linear`, no pinned CPU | `95.7s` | `~0.6s` | `277.6s` | `288.6s` | `0.80 GiB` | `28.2 GB` | Copy dominated; profile reported `192 GB` tensor movement. |
| Experimental `manual_linear` + pinned CPU, confirmed | `102.7-103.8s` | `103.4-114.6s` | `97.1-97.9s` | `210.9-222.5s` | `0.80 GiB` | `32.9 GB` | Best confirmed Diffusers-side denoise so far. Setup cost moved into pinning/staging. |
| ComfyUI BF16 | not directly isolated | staged dynamically | `~26.7s` | total prompt `63.0s` | not PyTorch-comparable | not logged | `8/8` at `3.34s/it`; no flash-attn or xformers installed. |
| ComfyUI quantized | not directly isolated | staged dynamically | `~10.9s` | total prompt `45.0s` | not PyTorch-comparable | not logged | `8/8` at `1.36s/it`. |

## Current Best Diffusers-Side Command

Use this as the current best experimental baseline:

```powershell
$env:LTX_IMAGE_TRANSFORMER_MEMORY_MANAGER="manual_linear"
$env:LTX_IMAGE_TRANSFORMER_GROUP_OFFLOAD="0"
$env:LTX_IMAGE_TRANSFORMER_PIN_CPU_MEMORY="1"
$env:LTX_IMAGE_TRANSFORMER_MANAGER_PROFILE="1"
$env:LTX_IMAGE_TRANSFORMER_MANAGER_PROFILE_SYNC_COPIES="0"
$env:LTX_IMAGE_ATTENTION_BACKEND="native"
python run_modular_distilled.py
```

Expected behavior from the confirmed runs:

- `setup_transformer_memory_manager` around `103-115s`
- `denoise_modular_pipe_call` around `97-98s`
- `torch_alloc` during denoise around `0.80 GiB`
- `Pass 1 total` around `211-223s`
- `Peak RAM` around `33 GB`

## Attention Backend Findings

Flash attention was installed in the project environment, but it did not improve the bottleneck.

Observed behavior:

- Diffusers `flash` and native flash backends reject non-None `attn_mask`.
- The LTX attention mask reaching the transformer was trivial/all-zero:
  - Raw mask shape: `[1, 1, 1024]`
  - Prepared mask shape: `[1, 32, 1, 1024]`
  - `nonzero: 0`
  - `trivial: true`
- Dropping the trivial mask allows flash to run.
- Flash still produced denoise times around `272s`, similar to native/manual manager results at that time.
- ComfyUI environment had no `flash-attn` and no `xformers`, so its large speed advantage is probably not from flash attention.

Conclusion: attention backend is not the primary bottleneck. Memory movement and staging are the bottleneck.

## Memory Manager Findings

The original experimental manager copied tensors too granularly.

A representative profile showed:

```text
[manager-profile] copy_runtime:
tensor_to_input: calls=10752 seconds=273.4668 gb=192.2374
```

With `PROFILE_SYNC_COPIES=0`, copy timings measure enqueue cost rather than synchronized copy duration. Even so, the volume is useful: one denoise pass can schedule about `192 GB` of CPU/GPU tensor movement.

Pinned CPU memory is now confirmed by repeated runs as the current best Diffusers-side mode:

| Run | Setup | Denoise | Pass 1 total | Torch alloc | Peak RAM |
| --- | ---: | ---: | ---: | ---: | ---: |
| Run 1 | `103.3507s` | `97.1237s` | `210.9s` | `0.80 GiB` | `32.92 GB` |
| Run 2 | `114.6131s` | `97.3091s` | `222.5s` | `0.80 GiB` | `32.92 GB` |
| Run 3 | `113.6660s` | `97.9024s` | `222.2s` | `0.80 GiB` | `32.93 GB` |

The denoise loop is stable. Most variance moved to setup/pinning.

A synchronized copy-profile run confirmed where the remaining cost lives:

```text
setup_transformer_memory_manager: 113.1982s
denoise_modular_pipe_call: 125.0540s
Pass 1 total: 248.7s
```

Copy profile totals from that run:

| Tensor type | Calls | Seconds | GB copied |
| --- | ---: | ---: | ---: |
| `linear_weight` | `4608` | `63.4258s` | `192.1872 GB` |
| `linear_bias` | `4608` | `3.5571s` | `0.0384 GB` |
| `rms_norm_weight` | `1536` | `0.7538s` | `0.0096 GB` |

The synchronized block runtime sum was `124.9315s`, matching the denoise event. This makes `linear_weight` transfer the clear optimization target.

Pinned CPU memory changed the result significantly:

```text
setup_transformer_memory_manager: 103.3507s
denoise_modular_pipe_call: 97.1237s
copy_runtime: calls=10752 seconds=0.2230 gb=192.2374
```

Interpretation:

- The setup cost increased because CPU weights are pinned/staged.
- The denoise became much faster.
- Asynchronous transfer overlap or driver behavior is hiding much more copy cost.
- This is currently the best Diffusers-side path, but still far behind ComfyUI.

## ComfyUI Findings

ComfyUI appears to use a stronger dynamic VRAM strategy rather than simple PyTorch module `.to()` offload.

Relevant code areas inspected:

- `E:\ProjetosIA\ComfyUI\comfy\model_patcher.py`
- `E:\ProjetosIA\ComfyUI\comfy\ops.py`
- `E:\ProjetosIA\ComfyUI\comfy\model_management.py`
- `E:\ProjetosIA\ComfyUI\comfy\ldm\lightricks\model.py`

Observed concepts:

- Models are prepared for dynamic VRAM loading.
- Large host/device buffers are used for staged casting/copying.
- Layer operations use cast-on-demand wrappers for weights and bias.
- Logs mention staged sizes such as:
  - Text encoder: `11200MB Staged`
  - BF16 LTX image transformer: `28101MB Staged`
  - Quantized LTX image transformer: `16846MB Staged`
- ComfyUI likely relies on SDPA for attention in this environment.

Important caveat: ComfyUI code is GPL. The experimental manager should remain clean-room and use only concepts, not copied implementation.

## Working Hypothesis

Current Diffusers-side performance is limited by explicit Python/PyTorch tensor movement.

ComfyUI is faster because it stages model weights and executes layer-level dynamic VRAM loading with:

- Larger reusable buffers
- Pinned host memory
- Cast-on-demand layer operations
- Better overlap between CPU/GPU transfers and compute
- Possibly NVIDIA shared-memory fallback behavior used more effectively

The next improvement should avoid moving whole `nn.Module` objects or thousands of individual tensors with repeated `.to()` calls during the hot path.

## Next Experiments

1. Keep `manual_linear + pinned CPU` as the current baseline.
2. Fix manager profiling so copy time is broken down by tensor/module type instead of only `tensor_to_input`.
3. Add a clean-room `manual_buffer_pool` or `manual_cast_buffer` mode:
   - Keep master weights on CPU.
   - Use reusable CUDA buffers for weight/bias casting.
   - Patch `Linear`, `RMSNorm`, and `LayerNorm` forwards to consume staged tensors.
   - Avoid repeated module-level `.to()`.
4. Try prefetching the next block/group with a CUDA stream after the buffer-pool mode works correctly.
5. Continue micro-committing every working state.

## Current Verdict

The modular pipeline plus custom dynamic manager is already more memory-stable than the old runner and can beat the old Diffusers-side denoise time in the best pinned-memory configuration.

However, ComfyUI remains much faster. Matching it likely requires moving from block/module offload to layer-level staged weight casting with reusable buffers.