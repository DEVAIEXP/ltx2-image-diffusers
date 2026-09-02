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

A `manual_block_staged` mode was tested after this. It stages all linear weights for the active block before yielding the block forward. Result:

```text
setup_transformer_memory_manager: 113.5052s
denoise_modular_pipe_call: 97.3465s
Pass 1 total: 222.1s
torch_alloc: 0.80 GiB
torch_reserved: 1.46 GiB
Peak RAM: 32.93 GB
```

This is effectively tied with `manual_linear + pinned CPU`. The profile still reports `96` linear weight staging calls per block over `8` steps, or `4608` total linear weight transfers. The staging location changed, but the number of weight transfers did not, so it does not address the real bottleneck.

`manual_hot_blocks` keeps selected transformer blocks resident on the execution device while the remaining blocks use the manual linear CPU-pinned path.

| Hot blocks | Setup | Denoise | Pass 1 total | Torch alloc | Peak RAM | Copied GB | Notes |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `4` | `104.1339s` | `88.1385s` | `207.2s` | `2.81 GiB` | `30.86 GB` | `176.2156 GB` | Clear improvement over baseline. |
| `8` | `99.1513s` | `76.8898s` | `195.2s` | `4.81 GiB` | `28.86 GB` | `160.1960 GB` | Better again; still no obvious memory pressure. |
| `12` | `94.2222s` | `71.3681s` | `188.8s` | `6.81 GiB` | `27.75 GB` | `144.1764 GB` | Best observed hot-block result so far. |
| `16` | `99.8078s` | `201.2456s` | `329.8s` | `8.81 GiB` | `29.85 GB` | `128.1568 GB` | Regression despite fewer copies; likely memory pressure/fallback contention. |

This confirms that eliminating repeated transfers for resident blocks improves denoise time until memory pressure starts to dominate. The curve bent hard at `16` hot blocks: copies dropped to `128.1568 GB`, but denoise regressed to `201.2456s`. The current sweet spot is `12` hot blocks for this machine/resolution.

An automatic hot-block budget selector was added after the explicit hot-block tests. Explicit `LTX_IMAGE_TRANSFORMER_HOT_BLOCKS` still wins; when it is unset, `LTX_IMAGE_TRANSFORMER_HOT_BLOCK_BUDGET_GB` selects every `LTX_IMAGE_TRANSFORMER_HOT_BLOCK_STRIDE` block until the memory budget is consumed.

The first budget run used:

```powershell
$env:LTX_IMAGE_TRANSFORMER_MEMORY_MANAGER="manual_hot_blocks"
$env:LTX_IMAGE_TRANSFORMER_GROUP_OFFLOAD="0"
$env:LTX_IMAGE_TRANSFORMER_PIN_CPU_MEMORY="1"
$env:LTX_IMAGE_TRANSFORMER_HOT_BLOCKS=""
$env:LTX_IMAGE_TRANSFORMER_HOT_BLOCK_BUDGET_GB="6"
$env:LTX_IMAGE_TRANSFORMER_HOT_BLOCK_STRIDE="3"
$env:LTX_IMAGE_ATTENTION_BACKEND="native"
```

Result:

| Mode | Budget | Setup | Denoise | Pass 1 total | Torch alloc | Peak VRAM | Peak RAM | Copied GB | Notes |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `manual_hot_blocks` auto budget | `6 GB` | `97.9940s` | `70.1449s` | `185.3s` | `6.31 GiB` | `6.71 GB` | `27.67 GB` | `148.1813 GB` | New best observed Diffusers-side result. |

Offset tests with the same `6 GB` budget showed that changing the hot-block phase can move the denoise number slightly, but did not beat the total pass time enough to replace offset `0` as the practical default:

| Offset | Setup | Denoise | Pass 1 total | Torch alloc | Peak VRAM | Peak RAM | Copied GB | Notes |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `0` | `97.9940s` | `70.1449s` | `185.3s` | `6.31 GiB` | `6.71 GB` | `27.67 GB` | `148.1813 GB` | Best total pass time. |
| `1` | `97.7758s` | `69.1332s` | `188.7s` | `6.31 GiB` | `6.88 GB` | `27.67 GB` | `148.1813 GB` | Best denoise time, but total pass was slower. |
| `2` | `94.6457s` | `70.6406s` | `186.8s` | `6.31 GiB` | `6.85 GB` | `27.67 GB` | `148.1813 GB` | Similar to offset `0`, slightly slower. |

Insight: the `6 GB` budget beat the explicit `12` hot-block run even though it copied slightly more data (`148.1813 GB` vs `144.1764 GB`). The lower resident allocation appears to keep the run in a better memory-pressure range, so the best point is not strictly the lowest copy volume. The budget selector is a better default interface than hard-coding a block list because it lets each machine/resolution find a similar pressure window.

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

### Instrumented ComfyUI Baseline

A temporary local ComfyUI patch added `[bench]` timing logs around node execution, model loading, and `ModelPatcherDynamic.load`. A representative BF16 run produced:

| Stage | Time | Notes |
| --- | ---: | --- |
| Checkpoint loader node | `0.9103s` | Loads model metadata/checkpoint wrapper. |
| Text encoder loader node | `0.4107s` | Loads LTX AV text encoder wrapper. |
| Text encoder dynamic prepare | `0.1115s` | `25440.5 MB` staged, `1745.2 KB` preloaded. |
| Text encode node | `19.2612s` | Prompt encoding runtime. |
| Transformer dynamic prepare | `0.1609s` | `28101.7 MB` staged, `1703.1 KB` preloaded. |
| Sampler node | `49.6648s` | Includes model initialization plus denoise. |
| VAE dynamic prepare | `0.0114s` | `1385.0 MB` staged. |
| VAE decode node | `1.0294s` | Decode runtime. |
| Preview image node | `0.0533s` | UI output. |
| Prompt total | `72.11s` | End-to-end ComfyUI execution. |

The sampler progress bar showed:

```text
Model Initializing ... -> Model Initialization complete!: about 24s
8 denoise steps after initialization: about 22s total, about 2.80-3.13s/it
```

Key insight: ComfyUI does **not** spend tens of seconds in its dynamic prepare step. `ModelPatcherDynamic.load` only reserves/stages metadata/buffers and wires cast-on-demand behavior. The expensive model initialization is deferred into the sampler's first iteration, but even with that cost included, the sampler node finishes in `49.6648s`.

This changes the optimization target for the Diffusers-side manager:

- Our `setup_transformer_memory_manager` spends about `82.56s` just pinning CPU blocks.
- ComfyUI's transformer dynamic prepare is about `0.16s`.
- Our denoise is about `68-70s` after setup, while ComfyUI's sampler node is `49.66s` including about `24s` of initialization.

The next clean-room experiments should therefore avoid eager CPU pinning/staging during setup and move toward lazy dynamic staging closer to ComfyUI's model: prepare metadata quickly, initialize buffers lazily, and keep hot execution state reusable across steps.

## Working Hypothesis

Current Diffusers-side performance is limited by explicit Python/PyTorch tensor movement.

ComfyUI is faster because it stages model weights and executes layer-level dynamic VRAM loading with:

- Larger reusable buffers
- Pinned host memory
- Cast-on-demand layer operations
- Better overlap between CPU/GPU transfers and compute
- Possibly NVIDIA shared-memory fallback behavior used more effectively

The next improvement should avoid moving whole `nn.Module` objects or thousands of individual tensors with repeated `.to()` calls during the hot path.

## Streamed Small Tensors Resident

The `manual_hot_blocks` path was updated so streamed blocks keep tiny tensors resident on the execution device while only large linear weights are copied on demand. This keeps `Linear.bias`, `RMSNorm.weight`, and `LayerNorm` weight/bias/buffers out of the repeated copy path.

Test configuration:

```powershell
$env:LTX_IMAGE_TRANSFORMER_MEMORY_MANAGER="manual_hot_blocks"
$env:LTX_IMAGE_TRANSFORMER_GROUP_OFFLOAD="0"
$env:LTX_IMAGE_TRANSFORMER_PIN_CPU_MEMORY="1"
$env:LTX_IMAGE_TRANSFORMER_HOT_BLOCKS=""
$env:LTX_IMAGE_TRANSFORMER_HOT_BLOCK_BUDGET_GB="6"
$env:LTX_IMAGE_TRANSFORMER_HOT_BLOCK_STRIDE="3"
$env:LTX_IMAGE_TRANSFORMER_HOT_BLOCK_OFFSET="0"
$env:LTX_IMAGE_TRANSFORMER_STREAMED_COPY_MODE="direct"
$env:LTX_IMAGE_TRANSFORMER_KEEP_STREAMED_SMALL_TENSORS_RESIDENT="1"
$env:LTX_IMAGE_ATTENTION_BACKEND="native"
```

Result:

| Mode | Budget | Setup | Denoise | Pass 1 total | Torch alloc | Torch reserved | Peak VRAM | Peak RAM | Copied GB | Copy time | Notes |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `manual_hot_blocks` + small tensors resident | `6 GB` | `94.2435s` | `69.5991s` | `185.4s` | `6.32 GiB` | `6.63 GiB` | `6.78 GB` | `27.66 GB` | `148.1443 GB` | `0.6358s` | Keeps only `linear_weight` in repeated copies. |

Compared with the previous `6 GB` budget baseline:

| Metric | Baseline | Small tensors resident | Change |
| --- | ---: | ---: | ---: |
| Setup | `97.9940s` | `94.2435s` | `-3.7505s` |
| Denoise | `70.1449s` | `69.5991s` | `-0.5458s` |
| Pass 1 total | `185.3s` | `185.4s` | `+0.1s` |
| Copy time | `1.2608s` | `0.6358s` | `-0.6250s` |
| Copied GB | `148.1813 GB` | `148.1443 GB` | `-0.0370 GB` |

Insight: this is a small but correct improvement. It removes unnecessary repeated tiny-tensor movement without meaningfully increasing VRAM/RAM pressure. The main remaining cost is now setup time, especially CPU pin/staging, rather than denoise transfer volume.

## Lazy CPU Pinning

A `lazy_pin_cpu_memory` experiment tried to avoid the eager `pin_cpu_blocks` setup cost by pinning each CPU tensor the first time it is copied.

Configuration difference from the best `manual_hot_blocks` baseline:

```powershell
$env:LTX_IMAGE_TRANSFORMER_PIN_CPU_MEMORY="0"
$env:LTX_IMAGE_TRANSFORMER_LAZY_PIN_CPU_MEMORY="1"
```

Result:

| Mode | Setup | Denoise | Torch alloc | Torch reserved | Copied GB | Copy time | Key setup runtime | Notes |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| Eager pin baseline | `95.3611s` | `68.4448s` | `6.32 GiB` | `6.63 GiB` | `148.1443 GB` | `0.1080s` | `pin_cpu_blocks=82.5627s` | Best stable path so far. |
| Lazy pin | `11.0458s` | `243.7794s` | `6.32 GiB` | `6.63 GiB` | `148.1443 GB` | `125.0563s` | `lazy_pin_linear_weight=121.4471s` | Regressed heavily. |

Insight: lazy pinning lowered setup by about `84s`, but moved `121s` of pinning into the denoise path. This confirms that the simple lazy approach is not enough. ComfyUI is not merely pinning later; it uses a different dynamic staging model with host/device buffers and cast-on-demand execution that avoids this kind of per-tensor hot-path penalty.

## Host Buffered Copy

A `streamed_copy_mode=host_buffered` experiment tried to create persistent CPU pinned host-buffer copies on first tensor use, without pinning the original CPU tensors eagerly.

Configuration difference from the best `manual_hot_blocks` baseline:

```powershell
$env:LTX_IMAGE_TRANSFORMER_PIN_CPU_MEMORY="0"
$env:LTX_IMAGE_TRANSFORMER_LAZY_PIN_CPU_MEMORY="0"
$env:LTX_IMAGE_TRANSFORMER_STREAMED_COPY_MODE="host_buffered"
```

Result:

| Mode | Setup | Denoise | Pass 1 total | Torch alloc | Torch reserved | Peak VRAM | Peak RAM | Copied GB | Copy time | Key setup runtime | Notes |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| Eager pin baseline | `95.3611s` | `68.4448s` | `185.5s` | `6.32 GiB` | `6.63 GiB` | `6.78 GB` | `27.66 GB` | `148.1443 GB` | `0.1080s` | `pin_cpu_blocks=82.5627s` | Best stable path so far. |
| Host buffered | `7.3320s` | `248.2392s` | `289.3s` | `6.32 GiB` | `6.63 GiB` | `6.82 GB` | `50.55 GB` | `148.1443 GB` | `3.5008s` | `host_buffer_linear_weight=124.3054s` | Regressed heavily and doubled host RAM pressure. |

Insight: host-buffered staging moved the expensive CPU copy/pinning work into denoise and duplicated about `18.5 GB` of linear weights in RAM. It is worse than eager pinning for this process-per-run benchmark. This is still not equivalent to ComfyUI's VBAR/host-buffer behavior, which prepares dynamic metadata quickly without duplicating the whole streamed weight set during the hot path.

## Hot Linear Weight Budget

A `hot_linear_weight_budget_gb` experiment tried to keep selected `Linear.weight` tensors resident on the execution device without promoting entire transformer blocks.

Configuration difference from the best `manual_hot_blocks` baseline:

```powershell
$env:LTX_IMAGE_TRANSFORMER_HOT_LINEAR_WEIGHT_BUDGET_GB="1"
$env:LTX_IMAGE_TRANSFORMER_HOT_LINEAR_WEIGHT_STRIDE="3"
$env:LTX_IMAGE_TRANSFORMER_HOT_LINEAR_WEIGHT_OFFSET="1"
```

Result:

| Mode | Setup | Denoise | Pass 1 total | Torch alloc | Torch reserved | Peak VRAM | Peak RAM | Copied GB | Copy time | Key setup runtime | Notes |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| Eager pin baseline | `95.3611s` | `68.4448s` | `185.5s` | `6.32 GiB` | `6.63 GiB` | `6.78 GB` | `27.66 GB` | `148.1443 GB` | `0.1080s` | `pin_cpu_blocks=82.5627s` | Best stable path so far. |
| Hot linear weights `1 GB` | `87.3497s` | `203.9284s` | `314.6s` | `7.19 GiB` | `7.50 GiB` | `6.97 GB` | `28.18 GB` | `141.1365 GB` | `0.1382s` | `hot_linear_weights_to_device=2.4773s`, `pin_cpu_blocks=78.0482s` | Regressed despite fewer repeated copies. |

Insight: keeping isolated `Linear.weight` tensors resident reduced repeated copy volume by about `7 GB`, but made block runtime much worse. The likely issue is not copy bandwidth alone; partial per-weight residency may create worse execution locality or memory-pressure behavior than keeping whole blocks resident. This result reinforces that the next useful step should model ComfyUI-style staged layer execution more directly, instead of mixing resident and streamed weights inside otherwise streamed blocks.

## Buffered Copy With Resident Small Tensors

A `streamed_copy_mode=buffered` run was repeated after keeping streamed small tensors resident. This checks whether the reusable device-side copy buffer helps once `Linear.bias`, `RMSNorm.weight`, and similar tiny tensors are no longer copied repeatedly.

Configuration difference from the best `manual_hot_blocks` baseline:

```powershell
$env:LTX_IMAGE_TRANSFORMER_STREAMED_COPY_MODE="buffered"
$env:LTX_IMAGE_TRANSFORMER_KEEP_STREAMED_SMALL_TENSORS_RESIDENT="1"
```

Result:

| Mode | Setup | Denoise | Pass 1 total | Torch alloc | Torch reserved | Peak VRAM | Peak RAM | Copied GB | Copy time | Key setup runtime | Notes |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| Eager pin baseline | `95.3611s` | `68.4448s` | `185.5s` | `6.32 GiB` | `6.63 GiB` | `6.78 GB` | `27.66 GB` | `148.1443 GB` | `0.1080s` | `pin_cpu_blocks=82.5627s` | Best stable path so far. |
| Buffered + small tensors resident | `93.2441s` | `70.4581s` | `185.8s` | `6.44 GiB` | `6.63 GiB` | `6.76 GB` | `27.66 GB` | `148.1443 GB` | `1.6734s` | `pin_cpu_blocks=82.3629s`, `blocks_to_target_devices=9.9902s` | Neutral/slightly worse than direct copy. |

Insight: buffering is no longer catastrophic once tiny tensors stay resident, but it still does not beat direct pinned CPU to device copies. The current best path remains `streamed_copy_mode=direct` with small tensors resident and a `6 GB` hot block budget.

## Parallel CPU Pinning

The CPU pinning setup was parallelized with `LTX_IMAGE_TRANSFORMER_PIN_CPU_WORKERS`. This targets the largest remaining setup cost in the best `manual_hot_blocks` path: pinning about `18.5 GB` of CPU transformer block tensors before denoise.

Configuration difference from the best `manual_hot_blocks` baseline:

```powershell
$env:LTX_IMAGE_TRANSFORMER_PIN_CPU_WORKERS="4"
```

Result:

| Mode | Pin workers | Setup | Pin CPU blocks | Denoise | Step avg | Pass 1 total | Torch alloc | Torch reserved | Peak VRAM | Peak RAM | Copied GB | Copy time | Notes |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Eager pin baseline | `1` | `95.3611s` | `82.5627s` | `68.4448s` | n/a | `185.5s` | `6.32 GiB` | `6.63 GiB` | `6.78 GB` | `27.66 GB` | `148.1443 GB` | `0.1080s` | Best previous stable path. |
| Parallel CPU pinning | `2` | `65.8865s` | `60.6064s` | `69.2341s` | `8.6495s/it` | `157.4s` | `6.32 GiB` | `6.63 GiB` | `6.69 GB` | `46.60 GB` | `148.1443 GB` | `0.6513s` | Slightly best Pass 1 total so far. |
| Parallel CPU pinning | `4` | `66.6127s` | `55.3440s` | `68.5952s` | `8.5691s/it` | `157.9s` | `6.32 GiB` | `6.63 GiB` | `6.74 GB` | `46.61 GB` | `148.1443 GB` | `1.7122s` | Best Pass 1 total so far, but with higher peak RAM. |

Insight: parallel CPU pinning reduced setup by about `29s` and Pass 1 by about `28s` while keeping denoise essentially unchanged. The new per-step timing shows denoise is uniform at about `8.6s/it`; unlike ComfyUI, there is no hidden first-step initialization spike in this path. `2` and `4` workers are effectively tied, so `2` is the better conservative default candidate unless repeated runs prove otherwise. The tradeoff is higher reported peak RAM.

## Layer Runtime Profile

Layer-level profiling was added to identify whether the denoise gap is dominated by attention, feed-forward, or modulation. This run used the best `manual_hot_blocks` baseline with synchronized layer timing enabled.

Configuration difference from the best baseline:

```powershell
$env:LTX_IMAGE_TRANSFORMER_PROFILE_LAYERS="1"
$env:LTX_IMAGE_TRANSFORMER_PROFILE_SYNC_LAYERS="1"
```

Result:

| Mode | Setup | Denoise | Step avg | Pass 1 total | Resident block time | Streamed block time | Cross-attn | Self-attn | FF | AdaLN |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `manual_hot_blocks` + layer sync profile | `71.9572s` | `82.5902s` | `10.3239s/it` | `176.8s` | `13.4310s` | `69.0724s` | `30.8499s` | `27.6515s` | `23.3325s` | `0.6158s` |

Insight: synchronized layer profiling slows the run, but the distribution is useful. The denoise gap is not isolated to one layer type. `cross_attn`, `self_attn`, and `ff` are all significant, which points back to the dynamic weight execution path used by all of their `Linear` modules. The next useful comparison is a block-staged mode that copies a block's linear weights once before the block forward rather than staging each linear call independently.

## Block Staged Linear Weights

A `manual_block_staged` experiment tried to stage every streamed block's linear weights before each block forward, instead of staging each `Linear` call independently.

Configuration difference from the best `manual_hot_blocks` baseline:

```powershell
$env:LTX_IMAGE_TRANSFORMER_MEMORY_MANAGER="manual_block_staged"
$env:LTX_IMAGE_TRANSFORMER_HOT_BLOCK_BUDGET_GB="0"
```

Result:

| Mode | Setup | Denoise | Step avg | Pass 1 total | Torch alloc | Torch reserved | Peak VRAM | Peak RAM | Resident blocks | Streamed blocks | Copied GB | Copy time | Key setup runtime |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Best hot-block baseline | `68.6324s` | `70.5502s` | `8.8141s/it` | `161.6s` | `6.32 GiB` | `6.68 GiB` | `6.73 GB` | `46.60 GB` | `11` | `37` | `148.1443 GB` | `0.1107s` | `pin_cpu_blocks=60.9529s` |
| Block staged | `169.5978s` | `99.1882s` | `12.1739s/it` | `279.9s` | `0.81 GiB` | `1.46 GiB` | `6.38 GB` | `51.86 GB` | `0` | `48` | `192.1872 GB` | `0.0658s` | `pin_cpu_blocks=169.2436s` |

Insight: block-staged execution is not useful in this shape. It loses all resident hot blocks, copies more total weight volume, and greatly increases setup time. Even though measured copy time is low, block runtime gets worse, so this path should be discarded for now.

## Stable Hot Blocks With Parallel Pinning

The latest stable `manual_hot_blocks` run used the current best configuration with a `6 GB` hot-block budget, direct streamed copies, resident small tensors, and parallel CPU pinning.

Configuration:

```powershell
$env:LTX_IMAGE_TRANSFORMER_MEMORY_MANAGER="manual_hot_blocks"
$env:LTX_IMAGE_TRANSFORMER_GROUP_OFFLOAD="0"
$env:LTX_IMAGE_TRANSFORMER_PIN_CPU_MEMORY="1"
$env:LTX_IMAGE_TRANSFORMER_PIN_CPU_WORKERS="2"
$env:LTX_IMAGE_TRANSFORMER_KEEP_STREAMED_SMALL_TENSORS_RESIDENT="1"
$env:LTX_IMAGE_TRANSFORMER_HOT_BLOCK_BUDGET_GB="6"
$env:LTX_IMAGE_TRANSFORMER_STREAMED_COPY_MODE="direct"
$env:LTX_IMAGE_ATTENTION_BACKEND="native"
```

Result:

| Metric | Value |
| --- | ---: |
| Setup memory manager | `68.6324s` |
| Pin CPU blocks | `60.9529s` for `18.5212 GB` |
| Blocks to target devices | `7.3261s` |
| Denoise | `70.5502s` |
| Step average | `8.8141s/it` |
| Pass 1 total | `161.6s` |
| Resident block runtime | `3.3982s` across `11` blocks |
| Streamed block runtime | `65.6985s` across `37` blocks |
| Copied weight volume | `148.1443 GB` |
| Measured copy time | `0.1107s` |
| Torch alloc/reserved | `6.32 GiB` / `6.68 GiB` |
| Peak VRAM/RAM | `6.73 GB` / `46.60 GB` |

Insight: this confirms the current best path is setup-bound plus streamed-block-runtime-bound, not copy-time-bound. Resident blocks are extremely fast relative to streamed blocks, but increasing the hot-block budget to `8 GB` previously caused a severe slowdown. The next promising direction is better hot-block selection within the same memory budget, guided by the slowest streamed block profile, rather than more aggressive staging.

## Profile-Guided Hot Block Candidates

`LTX_IMAGE_TRANSFORMER_HOT_BLOCK_CANDIDATES` was added so a profile-derived block priority list can be budgeted safely. Unlike explicit `LTX_IMAGE_TRANSFORMER_HOT_BLOCKS`, candidates are selected in order only until `LTX_IMAGE_TRANSFORMER_HOT_BLOCK_BUDGET_GB` is reached.

Configuration difference from the best baseline:

```powershell
$env:LTX_IMAGE_TRANSFORMER_HOT_BLOCKS=""
$env:LTX_IMAGE_TRANSFORMER_HOT_BLOCK_CANDIDATES="36,2,29,32,12,19,9,22,16,28,4,1,7,13,25,31,35,38,40,43,44,47"
$env:LTX_IMAGE_TRANSFORMER_HOT_BLOCK_BUDGET_GB="6"
```

Result:

| Mode | Setup | Denoise | Step avg | Pass 1 total | Torch alloc | Torch reserved | Peak VRAM | Peak RAM | Resident block time | Streamed block time | Copied GB | Copy time |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Stride/budget baseline | `68.6324s` | `70.5502s` | `8.8141s/it` | `161.6s` | `6.32 GiB` | `6.68 GiB` | `6.73 GB` | `46.60 GB` | `3.3982s` | `65.6985s` | `148.1443 GB` | `0.1107s` |
| Profile-guided candidates | `69.1140s` | `70.0661s` | `8.7537s/it` | `161.6s` | `6.32 GiB` | `6.68 GiB` | `6.85 GB` | `46.60 GB` | `3.3800s` | `65.2779s` | `148.1443 GB` | `0.1106s` |

Insight: profile-guided candidates did not materially improve the result, but they also did not regress it. The similar copied volume suggests the `6 GB` budget still selects roughly the same number of blocks. The changing slowest-block list across runs means single-run slowest-block ordering is noisy; future selection should be based on averaged profiles or structural knowledge, not one trace.

## Next Experiments

1. Keep `manual_hot_blocks + pinned CPU + resident small tensors` as the current baseline.
2. Move new experiments toward a Diffusers-style `Config + apply_* + ModelHook` API.
3. Build a generic dynamic weight plan before changing execution behavior.
4. Reduce setup cost by avoiding repeated full CPU pinning when possible.
5. Investigate a ComfyUI-like dynamic staging path that avoids per-run heavyweight setup while preserving fast streamed execution.
6. Continue micro-committing every working state.

## Diffusers-Friendly Dynamic Weights Direction

The ComfyUI Dynamic VRAM blog clarified the target behavior: a lightweight load plan, on-demand weight materialization, a cache that keeps successfully materialized weights resident when memory allows, and a temporary path when memory pressure prevents residency.

To keep this experiment portable to a future Diffusers-style implementation, new work should follow these rules:

1. Prefer `Config` dataclasses plus `apply_*` functions.
2. Use `ModelHook` and `HookRegistry` for attach/detach behavior.
3. Keep the dynamic weight core model-agnostic.
4. Keep LTX-specific knowledge in adapter/planning code only.
5. Avoid custom devices, global PyTorch monkeypatching, and loader replacement until the execution model proves useful.
6. Bring in one behavior at a time and benchmark each step.

A planner-only module now exists behind:

```powershell
$env:LTX_IMAGE_DYNAMIC_WEIGHTS_PLAN="1"
```

This builds a generic `nn.Linear` weight plan and records module count, total planned weight size, and placement buckets. It does not change the forward path yet.

## Dynamic Planner Baseline Runs

Two runs after a terminal/system restart used the same best `manual_hot_blocks` baseline with the planner enabled. The planner is expected to be execution-neutral.

Planner summary:

| Metric | Value |
| --- | ---: |
| Linear modules | `584` |
| Planned weight size | `24.4404 GB` |
| Resident module weights | `0.4122 GB` |
| Resident small tensors | `0.0033 GB` |
| Streamed large weights | `24.0249 GB` |
| Planner build time | `0.0156s` to `0.0654s` |

Runtime samples:

| Run | Setup | Pin CPU blocks | Denoise | Step avg | Pass 1 total | Peak VRAM | Peak RAM | Resident block time | Streamed block time | Copy time | Hot blocks |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Restart sample 1 | `53.8306s` | `45.3256s` | `48.6036s` | `6.0708s/it` | `117.9s` | `6.68 GB` | `46.60 GB` | `5.3954s` | `41.8514s` | `0.7978s` | `[0, 3, 6, 9, 12, 15, 18, 21, 24, 27, 30]` |
| Restart sample 2 | `52.3666s` | `47.4792s` | `58.3191s` | `7.2376s/it` | `128.8s` | `6.74 GB` | `46.60 GB` | `0.6015s` | `55.4244s` | `2.1821s` | `[0, 3, 6, 9, 12, 15, 18, 21, 24, 27, 30]` |

Insight: the planner itself is cheap and is seeing the expected large-weight universe, close to the ComfyUI staged model size. The major speedup compared to earlier `~70s` denoise runs likely comes from system/driver/cache state after restart rather than the planner changing execution. Still, these are the best observed Diffusers-side samples and are now the performance band to preserve while implementing the next dynamic-weight features.

## Windows Standby Cache Control

Windows memory state appears to materially affect repeatability. After boot, the system showed only about `6 GB` of RAM cache/standby memory. After the first run, standby/cache grew to about `32 GB+` and later runs tended to degrade. Clearing the standby list before the next run produced the best observed result so far. The Windows pagefile was also changed from system-managed to a fixed `32 GB` size to avoid pagefile resizing during inference.

Configuration stayed on the same best baseline:

```powershell
$env:LTX_IMAGE_TRANSFORMER_MEMORY_MANAGER="manual_hot_blocks"
$env:LTX_IMAGE_TRANSFORMER_PIN_CPU_MEMORY="1"
$env:LTX_IMAGE_TRANSFORMER_PIN_CPU_WORKERS="2"
$env:LTX_IMAGE_TRANSFORMER_HOT_BLOCK_BUDGET_GB="6"
$env:LTX_IMAGE_TRANSFORMER_KEEP_STREAMED_SMALL_TENSORS_RESIDENT="1"
$env:LTX_IMAGE_TRANSFORMER_STREAMED_COPY_MODE="direct"
$env:LTX_IMAGE_DYNAMIC_WEIGHTS_PLAN="1"
```

Result after clearing standby cache:

| Setup | Pin CPU blocks | Denoise | Step avg | Pass 1 total | Peak VRAM | Peak RAM | Resident block time | Streamed block time | Copy time | Hot blocks |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `44.4104s` | `35.9688s` | `39.3172s` | `4.9083s/it` | `97.9s` | `6.68 GB` | `46.60 GB` | `2.9649s` | `35.2593s` | `2.1533s` | `[0, 3, 6, 9, 12, 15, 18, 21, 24, 27, 30]` |

Insight: this strongly suggests that Windows standby/cache pressure was a hidden benchmark variable. Clearing standby cache reduced both setup and denoise. The denoise path is now much closer to ComfyUI's total sampler timing, though still slower than ComfyUI's steady per-step rate. Future benchmark comparisons should record whether standby cache was cleared, whether NVIDIA system fallback was enabled, and whether pagefile was fixed or system-managed.

## Current Verdict

The modular pipeline plus custom dynamic manager is already more memory-stable than the old runner and can beat the old Diffusers-side denoise time in the best pinned-memory configuration.

However, ComfyUI remains much faster. Matching it likely requires moving from block/module offload to layer-level staged weight casting with reusable buffers.

## Low CPU Memory and Cleanup A/B

Question: whether `low_cpu_mem_usage` contributes to Windows standby/cache growth and whether an internal cleanup can reduce the need for an external standby-list clear.

Code change:

```powershell
$env:LTX_IMAGE_LOW_CPU_MEM_USAGE="0"              # applies to transformer/connectors/VAE only
$env:LTX_IMAGE_RESET_DYNAMIC_MEMORY_AFTER_RUN="1" # optional Python/CUDA cleanup at script end
```

The text encoder intentionally remains fixed at `low_cpu_mem_usage=True` so the A/B test isolates the image-side components. The runner now records `text_encoder_low_cpu_mem_usage`, `model_low_cpu_mem_usage`, and `reset_dynamic_memory_after_run` in the metrics JSON.

Interpretation before testing:

- `low_cpu_mem_usage=True` is not proven to be the direct cause of standby cache growth, but it may influence how shards/memmaps/file cache are touched during loading.
- `low_cpu_mem_usage=False` may increase temporary RAM pressure because weights can be materialized more directly in host memory before placement.
- ComfyUI does not appear to call a global Windows standby-list cleaner. Its `reset_cast_buffers()` resets ComfyUI dynamic VRAM state, stream cast buffers, dirty mmap state, dynamic pin activity, and CUDA cache. Our new reset flag is closer to that kind of local cleanup, not to `EmptyStandbyList`.

Recommended protocol:

1. Run the current baseline with `LTX_IMAGE_LOW_CPU_MEM_USAGE="1"` and note setup, denoise, RAM cache/standby, and pass total.
2. Clear Windows standby cache externally if you want a cold comparable run.
3. Run with `LTX_IMAGE_LOW_CPU_MEM_USAGE="0"` using the same env vars and compare setup/pin time, denoise time, peak RAM, and standby cache growth.
4. Repeat with `LTX_IMAGE_RESET_DYNAMIC_MEMORY_AFTER_RUN="1"` and check whether the next run degrades less without external cache clearing.

Update: Diffusers rejects `low_cpu_mem_usage=False` together with `device_map="cpu"` during `from_pretrained`. The runner now keeps `device_map="cpu"` only when `LTX_IMAGE_LOW_CPU_MEM_USAGE="1"`. When testing `LTX_IMAGE_LOW_CPU_MEM_USAGE="0"`, transformer loading uses the regular CPU loading path before the custom manager is attached. Treat this as a loading-path A/B, not as a perfectly isolated boolean change.

## Windows Standby Purge Helper

The runner now has an optional Windows-only benchmark helper that calls `NtSetSystemInformation(SystemMemoryListInformation=80, MemoryPurgeStandbyList=4)` through Python `ctypes`. It enables `SeProfileSingleProcessPrivilege` on the current process token first, so the Python process must run from an elevated/admin terminal.

```powershell
$env:LTX_IMAGE_PURGE_WINDOWS_STANDBY_BEFORE_RUN="1"
$env:LTX_IMAGE_PURGE_WINDOWS_STANDBY_AFTER_RUN="1"
```

Smoke test result from an elevated terminal:

```text
{'system_information_class': 80, 'command': 4, 'privilege': 'SeProfileSingleProcessPrivilege'}
```

Use `BEFORE_RUN=1` for cold benchmark control. Use `AFTER_RUN=1` to test whether back-to-back runs stay stable without calling an external executable. This remains a Windows benchmark hygiene tool, not part of the portable dynamic-memory manager.

## Purge Before Transformer Result

A second Windows standby purge point was added after prompt/connectors cleanup and immediately before transformer loading:

```powershell
$env:LTX_IMAGE_PURGE_WINDOWS_STANDBY_BEFORE_TRANSFORMER="1"
```

Observed comparison with the same baseline and `LTX_IMAGE_LOW_CPU_MEM_USAGE="1"`:

| Purge placement | Denoise | Step avg | Torch alloc/reserved |
| --- | ---: | ---: | ---: |
| Before full run only | `52.4843s` | `6.5116s/it` | `6.32 GiB` / `6.63 GiB` |
| Before transformer | `38.7616s` | `4.8405s/it` | `6.32 GiB` / `6.63 GiB` |

Insight: standby/cache pressure can build during the prompt/connectors stage before transformer setup. Purging only at process start is helpful but not sufficient for the fastest run. The new pre-transformer purge is a benchmark hygiene control for Windows and should be kept separate from portable manager behavior.

## Hot Block Selection Policy

A new optional hot-block selection policy was added for the dynamic manager:

```powershell
$env:LTX_IMAGE_TRANSFORMER_HOT_BLOCK_SELECTION="stride" # default, previous behavior
$env:LTX_IMAGE_TRANSFORMER_HOT_BLOCK_SELECTION="spread" # experimental
```

`stride` preserves the current baseline: candidates are visited by `hot_block_stride` and selection stops when the VRAM budget is exhausted. With a `6 GB` block budget this produced the current best stable pattern:

```text
[0, 3, 6, 9, 12, 15, 18, 21, 24, 27, 30]
```

`spread` keeps the same budget but distributes the resident blocks across the full transformer depth. The intent is to test whether broader coverage reduces streamed work in later blocks without increasing the memory budget. This is a controlled A/B against the current best baseline and should be run with the Windows standby purge controls enabled both before the run and before transformer setup.

### Spread Selection Result

With Windows standby purge enabled before the run and before transformer setup, `spread` produced the best denoise sample so far:

| Policy | Setup | Pin CPU blocks | Denoise | Step avg | Pass 1 total | Peak VRAM | Peak RAM | Hot blocks |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `stride` baseline | `44.4104s` | `35.9688s` | `39.3172s` | `4.9083s/it` | `97.9s` | `6.68 GB` | `46.60 GB` | `[0, 3, 6, 9, 12, 15, 18, 21, 24, 27, 30]` |
| `spread` | `53.6983s` | `45.1439s` | `36.1682s` | `4.5145s/it` | `106.5s` | `6.69 GB` | `46.60 GB` | `[0, 5, 9, 14, 19, 24, 28, 33, 38, 42, 47]` |

Insight: distributed residency improved the denoise loop but setup regressed because CPU pinning dominated. Next step is to reduce pinning setup pressure while keeping the same resident block policy.

### Streaming CPU Pin Assignment

`_pin_cpu_blocks` now pins and assigns tensors through a bounded worker queue instead of collecting all source tensor references and pinned results before assignment. This keeps the same pinned tensor set, but should reduce temporary host-memory pressure during setup.

Test with the same stable baseline and `LTX_IMAGE_TRANSFORMER_HOT_BLOCK_SELECTION="spread"`.

### Spread + Bounded Pinning Result

After switching CPU pinning to a bounded assignment queue, `spread` improved again under the two-purge benchmark protocol:

| Change | Setup | Pin CPU blocks | Denoise | Step avg | Pass 1 total | Peak VRAM | Peak RAM | Copy time |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `spread` before bounded pinning | `53.6983s` | `45.1439s` | `36.1682s` | `4.5145s/it` | `106.5s` | `6.69 GB` | `46.60 GB` | `1.0667s` |
| `spread` + bounded pinning | `40.2046s` | `31.4450s` | `34.6251s` | `4.3215s/it` | `91.9s` | `6.71 GB` | `27.65 GB` | `0.3420s` |

Insight: avoiding a full temporary collection of pinned tensors reduced setup, denoise, and peak RAM. The next setup target is `blocks_to_target_devices` around `8s`, which likely includes recursive `.to(cpu)` calls for streamed blocks already loaded on CPU.

### Skip Redundant Block Device Moves

The manager now checks whether a block actually has tensors outside the target device before calling `.to(target_device)`. This should preserve resident CUDA block placement while avoiding recursive no-op `.to(cpu)` traversal for streamed CPU blocks.

### Skip Redundant Device Move Result

The redundant `.to()` guard was neutral in the stable spread baseline:

| Change | Setup | Pin CPU blocks | Blocks to target devices | Denoise | Step avg | Pass 1 total | Peak VRAM | Peak RAM |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `spread` + bounded pinning | `40.2046s` | `31.4450s` | `8.0990s` | `34.6251s` | `4.3215s/it` | `91.9s` | `6.71 GB` | `27.65 GB` |
| + skip redundant block moves | `38.8747s` | `30.1105s` | `8.0700s` | `35.6465s` | `4.4229s/it` | `90.5s` | `6.68 GB` | `27.65 GB` |

Insight: `blocks_to_target_devices` did not materially change, so the remaining setup problem is still CPU pinning. The next controlled A/B should keep the same two-purge spread baseline and vary only `LTX_IMAGE_TRANSFORMER_PIN_CPU_WORKERS`.

### Pin CPU Workers Result

With bounded pin assignment, increasing pin workers from `2` to `4` improved setup without hurting denoise:

| Pin workers | Setup | Pin CPU blocks | Denoise | Step avg | Pass 1 total | Peak VRAM | Peak RAM | Copy time |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `2` | `38.8747s` | `30.1105s` | `35.6465s` | `4.4229s/it` | `90.5s` | `6.68 GB` | `27.65 GB` | `0.3428s` |
| `4` | `32.3166s` | `23.5359s` | `34.8546s` | `4.3236s/it` | `83.8s` | `6.68 GB` | `27.65 GB` | `0.1083s` |

Insight: pinning parallelism is now productive with the bounded queue. Next A/B should test `LTX_IMAGE_TRANSFORMER_PIN_CPU_WORKERS="8"` under the same two-purge spread baseline.

### Pin CPU Workers 8 Result

With bounded pin assignment and `spread` hot block selection, `8` pin workers reduced setup further but made denoise slower than `4` workers:

| Pin workers | Setup | Pin CPU blocks | Denoise | Step avg | Pass 1 total | Peak VRAM | Peak RAM | Copy time |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `4` | `32.3166s` | `23.5359s` | `34.8546s` | `4.3236s/it` | `83.8s` | `6.68 GB` | `27.65 GB` | `0.1083s` |
| `8` | `29.2975s` | `20.5812s` | `37.2691s` | `4.6274s/it` | `83.3s` | `6.68 GB` | `27.66 GB` | `1.2723s` |

Insight: `8` workers buys setup time but appears to disturb the denoise path enough that total Pass 1 is only marginally better. Treat `4` as the safer balanced baseline and `8` as the faster-setup variant. The next A/B should reduce hot block budget with `spread` to see whether fewer resident blocks lower setup enough without giving back too much denoise speed.

### Hot Block Budget 5 GB Result

With `spread` selection and `4` pin workers, reducing the hot block budget from `6 GB` to `5 GB` reduced VRAM but did not improve runtime:

| Hot block budget | Hot blocks | Setup | Pin CPU blocks | Blocks to target devices | Denoise | Step avg | Pass 1 total | Torch alloc/reserved | Peak VRAM | Peak RAM | Copy time |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `6 GB` | `[0, 5, 9, 14, 19, 24, 28, 33, 38, 42, 47]` | `32.3166s` | `23.5359s` | `8.1473s` | `34.8546s` | `4.3236s/it` | `83.8s` | `6.32/6.63 GiB` | `6.68 GB` | `27.65 GB` | `0.1083s` |
| `5 GB` | `[0, 6, 12, 18, 24, 29, 35, 41, 47]` | `31.8124s` | `24.9162s` | `6.3818s` | `37.8765s` | `4.7343s/it` | `85.0s` | `5.32/5.63 GiB` | `6.31 GB` | `28.35 GB` | `1.5416s` |

Insight: `5 GB` is useful as a lower-VRAM profile, but not as the fastest profile. The current speed baseline remains `6 GB`, `spread`, and `4` pin workers.

## Dynamic Weights Linear Runtime

The `dynamic_weights` module now has an experimental execution mode:

```powershell
$env:LTX_IMAGE_DYNAMIC_WEIGHTS_EXECUTION_MODE="linear_runtime"
$env:LTX_IMAGE_DYNAMIC_WEIGHTS_PIN_CPU_MEMORY="1"
$env:LTX_IMAGE_DYNAMIC_WEIGHTS_PIN_CPU_WORKERS="4"
$env:LTX_IMAGE_TRANSFORMER_MEMORY_MANAGER="off"
$env:LTX_IMAGE_TRANSFORMER_GROUP_OFFLOAD="0"
```

This is the first real dynamic-weight execution path, separate from the LTX-specific block manager. It patches `nn.Linear` modules generically, keeps configured resident modules on the execution device, keeps small non-linear local tensors on the execution device, and streams large linear weights from CPU to the input device during forward.

Important: this is not expected to beat `manual_hot_blocks` yet. Its purpose is to move the architecture toward a portable Diffusers-style dynamic weight runtime, so future iterations can add residency budgets, reusable device buffers, and loader-backed storage without baking LTX transformer details into the core mechanism.

Smoke test result: a tiny CUDA `nn.Sequential(nn.RMSNorm, nn.Linear)` model successfully patched, executed, and restored through the hook.

### Pin CPU Workers 6 Result

With `spread`, `6 GB` hot block budget, and the two-purge benchmark protocol, `6` pin workers did not beat the `4` worker baseline:

| Pin workers | Setup | Pin CPU blocks | Denoise | Step avg | Pass 1 total | Peak VRAM | Peak RAM | Copy time |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `4` | `32.3166s` | `23.5359s` | `34.8546s` | `4.3236s/it` | `83.8s` | `6.68 GB` | `27.65 GB` | `0.1083s` |
| `6` | `34.0698s` | `25.4730s` | `38.0033s` | `4.7176s/it` | `89.0s` | `6.73 GB` | `27.66 GB` | `1.5200s` |
| `8` | `29.2975s` | `20.5812s` | `37.2691s` | `4.6274s/it` | `83.3s` | `6.68 GB` | `27.66 GB` | `1.2723s` |

Insight: `4` workers remains the best balanced setting. Higher worker counts can improve or vary setup, but they increase denoise/copy disturbance in this benchmark band.

### Dynamic Weights Store Runtime

The `dynamic_weights` module now has a second execution mode:

```powershell
$env:LTX_IMAGE_DYNAMIC_WEIGHTS_EXECUTION_MODE="linear_store_runtime"
$env:LTX_IMAGE_DYNAMIC_WEIGHTS_PIN_CPU_MEMORY="1"
$env:LTX_IMAGE_DYNAMIC_WEIGHTS_PIN_CPU_WORKERS="4"
$env:LTX_IMAGE_TRANSFORMER_MEMORY_MANAGER="off"
$env:LTX_IMAGE_TRANSFORMER_GROUP_OFFLOAD="0"
```

This mode moves large `nn.Linear.weight` tensors into an internal runtime store and replaces the active module parameter with a `meta` placeholder. The patched linear forward reads from the store and copies the weight to the input device when needed.

This is closer to the intended dynamic-weight architecture than `linear_runtime`, because it separates module execution from weight storage. It is still not ComfyUI-equivalent yet: weights are still first materialized by Diffusers `from_pretrained`, and the store is not loaded directly from checkpoint shards. The next architectural target is a loader-backed store that can avoid the expensive eager pin/materialization setup path.

Smoke test result: a tiny CUDA `nn.Sequential(nn.RMSNorm, nn.Linear)` model successfully entered `linear_store_runtime`, executed with a `meta` linear weight placeholder, and restored the original parameter after hook removal.

First full transformer result with every non-resident linear streamed:

| Mode | Setup | Denoise | Step avg | Pass 1 total | Torch alloc/reserved | Peak VRAM | Peak RAM | Notes |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `linear_store_runtime` all streamed | `45.2711s` | `127.1036s` | `15.8834s/it` | `182.9s` | `0.81/1.18 GiB` | `6.32 GB` | `52.16 GB` | Works, but is a VRAM-minimum profile rather than a speed profile. |

Insight: patching `576` linear modules and streaming all large weights is too slow. The next runtime change is a generic resident-weight budget so `linear_store_runtime` can keep a budgeted spread of `Linear.weight` tensors on the execution device while still storing the rest outside the module.

### Dynamic Weights Resident Weight Budget

`linear_store_runtime` now supports a generic resident weight budget:

```powershell
$env:LTX_IMAGE_DYNAMIC_WEIGHTS_RESIDENT_WEIGHT_BUDGET_GB="6"
$env:LTX_IMAGE_DYNAMIC_WEIGHTS_RESIDENT_WEIGHT_SELECTION="spread"
```

This keeps a spread of large `nn.Linear.weight` tensors resident on the execution device while the remaining linear weights stay in the internal store with `meta` placeholders in the active modules. This is still model-agnostic: selection is based on module order and byte budget, not LTX block names.

Smoke test result: a CUDA sequential model with three large linears successfully ran with one resident linear weight and two store-backed `meta` weights.

### Dynamic Weights Linear Weight Budget Result

First full transformer result with `linear_store_runtime`, `6 GB` resident linear weight budget, and `spread` selection:

| Mode | Patched linears | Resident linear weights | Pin stored weights | Denoise | Step avg | Pass 1 total | Torch alloc/reserved | Peak VRAM | Peak RAM |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `linear_store_runtime` + resident linear budget | `361` | `5.9934 GB` | `29.6968s / 18.0312 GB` | `96.1833s` | `12.0193s/it` | `145.9s` | `6.80/7.17 GiB` | `6.97 GB` | `46.25 GB` |

Insight: per-linear residency is not coherent enough for this transformer. It reduces repeated streamed linears, but leaves many blocks with mixed resident/streamed weights and remains far slower than `manual_hot_blocks`.

### Dynamic Weights Resident Module Budget

`linear_store_runtime` now supports a generic resident module budget:

```powershell
$env:LTX_IMAGE_DYNAMIC_WEIGHTS_RESIDENT_WEIGHT_BUDGET_GB="0"
$env:LTX_IMAGE_DYNAMIC_WEIGHTS_RESIDENT_MODULE_BUDGET_GB="6"
$env:LTX_IMAGE_DYNAMIC_WEIGHTS_RESIDENT_MODULE_PATTERNS="^transformer_blocks\.\d+$"
$env:LTX_IMAGE_DYNAMIC_WEIGHTS_RESIDENT_MODULE_SELECTION="spread"
```

This keeps whole matching modules resident on the execution device and skips patching their descendants. For LTX this lets the portable dynamic weights runtime treat `transformer_blocks.N` as the budget unit without hardcoding LTX-specific names in the manager.

Smoke test result: a CUDA sequential model with two block-like submodules successfully ran with one resident block and remaining linears store-backed through `meta` placeholders.

First full transformer result with `linear_store_runtime`, `6 GB` resident module budget, and `spread` selection:

| Mode | Patched linears | Resident modules | Pin stored weights | Denoise | Step avg | Pass 1 total | Torch alloc/reserved | Peak VRAM | Peak RAM |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `linear_store_runtime` + resident module budget | `444` | `5.5077 GB` | `31.4762s / 18.5181 GB` | `92.9394s` | `11.5235s/it` | `143.7s` | `6.32/6.63 GiB` | `6.69 GB` | `46.66 GB` |

Insight: module residency is structurally closer to `manual_hot_blocks`, but the store/meta path is still much slower than the LTX-specific direct-parameter path. The next A/B enables the same resident module budget for `linear_runtime` so the module selection is generic while streamed linears keep regular CPU parameters instead of store-backed `meta` placeholders.

### Dynamic Weights Linear Runtime Module Budget

`resident_module_budget_gb` now also works with `linear_runtime`. This tests generic module-level residency without the store/meta indirection:

```powershell
$env:LTX_IMAGE_DYNAMIC_WEIGHTS_EXECUTION_MODE="linear_runtime"
$env:LTX_IMAGE_DYNAMIC_WEIGHTS_RESIDENT_WEIGHT_BUDGET_GB="0"
$env:LTX_IMAGE_DYNAMIC_WEIGHTS_RESIDENT_MODULE_BUDGET_GB="6"
$env:LTX_IMAGE_DYNAMIC_WEIGHTS_RESIDENT_MODULE_PATTERNS="^transformer_blocks\.\d+$"
$env:LTX_IMAGE_DYNAMIC_WEIGHTS_RESIDENT_MODULE_SELECTION="spread"
```

Smoke test result: a CUDA sequential model with block-like submodules successfully ran with one resident block and remaining linears patched through regular CPU parameters.

First full transformer result with `linear_runtime`, `6 GB` resident module budget, and `spread` selection:

| Mode | Patched linears | Resident modules | Pin linear weights | Denoise | Step avg | Pass 1 total | Torch alloc/reserved | Peak VRAM | Peak RAM |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `linear_runtime` + resident module budget | `444` | `5.5077 GB` | `31.7174s / 18.5181 GB` | `93.9007s` | `11.7310s/it` | `144.7s` | `6.32/6.63 GiB` | `6.77 GB` | `46.66 GB` |

Insight: removing the `store/meta` indirection did not recover speed. The generic dynamic path is still far slower than `manual_hot_blocks`, so the next diagnostic is runtime profiling inside `dynamic_weights` itself: copy totals, resident module names, and whether input/device transfers are happening during the transformer forward.

Follow-up profiling showed the selected resident modules matched the fastest manual baseline exactly:

```text
[dynamic-weights-profile] summary: mode=linear_runtime modules=584 patched=444 copy_seconds=4.5617 copy_gb=148.1445
[dynamic-weights-profile] resident_modules=['transformer_blocks.0', 'transformer_blocks.5', 'transformer_blocks.9', 'transformer_blocks.14', 'transformer_blocks.19', 'transformer_blocks.24', 'transformer_blocks.28', 'transformer_blocks.33', 'transformer_blocks.38', 'transformer_blocks.42', 'transformer_blocks.47']
[dynamic-weights-profile] copy_runtime_by_type:
  linear_weight: calls=3552 seconds=4.5617 gb=148.1445
```

Insight: the module selection is not the problem. One remaining difference from the manual manager is that non-linear block-local parameters, such as modulation tables used through inline `.to(temb.device)` calls in the block forward, were neither pinned nor made resident by the generic dynamic runtime. The runner now exposes `LTX_IMAGE_DYNAMIC_WEIGHTS_SMALL_TENSOR_THRESHOLD_KB` and defaults it to `1024` for dynamic weights so these small/medium local tensors can stay resident instead of being copied implicitly during every block forward.

Result after increasing the dynamic small tensor resident threshold to `1024 KB`:

| Mode | Resident small | Patched linears | Pin linear weights | Denoise | Step avg | Pass 1 total | Copy time | Torch alloc/reserved | Peak VRAM | Peak RAM |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `linear_runtime` + module budget + `16 KB` small tensors | `0.0033 GB` | `444` | `31.7174s / 18.5181 GB` | `93.9007s` | `11.7310s/it` | `144.7s` | `4.5617s` | `6.32/6.63 GiB` | `6.77 GB` | `46.66 GB` |
| `linear_runtime` + module budget + `1024 KB` small tensors | `0.0282 GB` | `444` | `23.3566s / 18.5181 GB` | `13.9789s` | `1.5810s/it` | `56.1s` | `0.3306s` | `6.32/6.63 GiB` | `6.69 GB` | `27.36 GB` |

Insight: this is the first generic dynamic weights path to beat the ComfyUI denoise loop in isolation on this test. The remaining gap is setup/model initialization: Diffusers still loads the transformer then spends tens of seconds pinning and placing CPU/GPU tensors, while the ComfyUI dynamic loader reports sub-second dynamic preparation plus a larger initialization phase inside the sampler.

### Dynamic VRAM Portability Notes

Comfy-Org/comfy-aimdo issue #25 reports that `ModelVBAR` allocation can fail on NVIDIA GRID / vGPU environments and should degrade gracefully instead of hard-crashing. This does not affect the current `dynamic_weights` implementation because it uses regular CPU/CUDA tensors and pinned memory, not `comfy-aimdo`/VBAR. It is still an important design constraint if we later add a VBAR-like allocator or virtual device:

- Detect allocator support at initialization time, not during the first model forward.
- Treat VBAR/managed host buffer allocation failure as an optional acceleration failure, not a fatal model-load failure.
- Keep a plain pinned-memory path as the portable fallback for vGPU, Linux setups without the allocator, or Windows systems where privilege/driver behavior differs.
- Log the selected dynamic weight backend clearly so benchmark results show whether the run used VBAR-like staging, pinned host tensors, or regular pageable CPU tensors.

The open `comfy-aimdo` issue list reinforces the same boundary: reported failures include `cuMemSetAccess` device-not-ready errors on Blackwell, `cuGetProcAddress` / driver symbol mismatches, ROCm owner-device accounting bugs, VBAR allocation failures on older AMD cards, address-space exhaustion on ROCm/Windows, host buffer free crashes during RAM-pressure eviction, and unexpected system throughput drops after generation. These are allocator/driver/OS failure modes, not ordinary PyTorch module-placement bugs. Any future VBAR-like backend should therefore be optional, capability-probed up front, and paired with the current pinned-memory fallback.

### Dynamic Weights Lazy Pin Candidate

A control run with `linear_runtime`, `6 GB` resident module budget, `1024 KB` small tensor residency, but without pinned CPU memory reduced setup while making denoise much slower:

| Mode | Build dynamic plan | Denoise | Step profile | Copy time | Pass 1 total | Peak VRAM | Peak RAM |
| --- | ---: | ---: | --- | ---: | ---: | ---: | ---: |
| `linear_runtime` + module budget + no pin | `9.0407s` | `121.8672s` | first step `44.7585s`, later `10-13s/it` | `118.7173s` | `142.1s` | `6.79 GB` | `28.22 GB` |

Insight: pageable CPU copies are the wrong tradeoff. The next test adds `LTX_IMAGE_DYNAMIC_WEIGHTS_LAZY_PIN_CPU_MEMORY=1`, which should avoid the eager `pin_linear_weights` setup cost while pinning each streamed CPU weight on first use so later copies can use the faster pinned path.

Test command delta:

```powershell
$env:LTX_IMAGE_DYNAMIC_WEIGHTS_PIN_CPU_MEMORY="1"
$env:LTX_IMAGE_DYNAMIC_WEIGHTS_LAZY_PIN_CPU_MEMORY="1"
```

If this works, `build_dynamic_weights_plan` should stay closer to the no-pin run while `dynamic-weights-profile` reports `lazy_pin_linear_weight` under setup runtime and copy time should drop after the first use of each weight.

First full result:

| Mode | Build dynamic plan | Lazy pin during denoise | Denoise | Step profile | Copy time | Pass 1 total | Peak VRAM | Peak RAM |
| --- | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: |
| `linear_runtime` + module budget + lazy pin | `9.2746s` | `54.8884s / 18.5181 GB` | `71.3503s` | first step `58.8127s`, later mostly `~1.7s/it` | `0.4201s` | `91.5s` | `6.79 GB` | `27.40 GB` |

Insight: lazy pin is technically working, but for an 8-step image run it moves the eager pin cost into the first denoise step instead of eliminating it. After the first-step pinning, later steps are very fast, so this may be useful for long-running or persistent-process workloads, but the best single-run preset remains eager pinned dynamic weights with resident modules and resident small tensors.

### Warm Generation Benchmark

The runner now supports repeated generation with a single loaded/configured transformer:

```powershell
$env:LTX_IMAGE_GENERATION_REPEATS="2"
```

This keeps the default single-run path unchanged when unset. With repeats enabled, the runner rebuilds latents and runs denoise multiple times before VAE decode, saving the last latent/image. This is intended to separate cold setup cost from warm execution cost and to compare more fairly against server-style systems such as ComfyUI.

Preset implication: eager pinned dynamic weights are still best for short one-shot runs. Lazy pin may become useful only when the same process/model performs additional warm generations, because the first generation pays the pinning cost and later generations can reuse pinned CPU weights.

First warm benchmark with the current best eager-pinned dynamic preset:

| Mode | Build dynamic plan | Denoise repeat 1 | Denoise repeat 2 | Avg step | Copy time | Pass 1 total | Peak VRAM | Peak RAM |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `linear_runtime` + module budget + eager pin + repeats `2` | `32.2961s` | `14.4916s` | `14.5287s` | `~1.63s/it` | `0.7809s / 296.2891 GB` | `72.4s` | `6.69 GB` | `27.35 GB` |

Insight: this is the strongest parity signal so far. In a warm/server-style run, the PyTorch-only dynamic weights path is already close to the ComfyUI reference (`72.11s` full prompt execution) while avoiding the VBAR/driver-level risk class. The remaining optimization target is cold setup, mainly `pin_linear_weights` (`23.4006s`) plus resident module placement (`8.3050s`), not the steady denoise loop.

### Preset Planning Notes

The current best results depend on the machine having enough host RAM headroom for pinned streamed weights plus resident modules. On the 64 GB RAM test system, the best dynamic preset used roughly:

- `~27-28 GB` peak process RAM.
- `~6.7 GB` peak VRAM.
- `18.5 GB` pinned streamed linear weights.
- `5.5 GB` resident transformer block budget.
- `0.028 GB` resident small tensors.

This suggests different default guidance by machine class:

| Machine class | Likely safe starting point | Expected tradeoff |
| --- | --- | --- |
| `32 GB RAM` / limited shared memory | Lower resident budget (`3-4 GB`), consider fewer pinned weights, keep standby purge optional on Windows | More streaming and slower denoise, but lower risk of RAM pressure/pagefile stalls. |
| `64 GB RAM` / `~32 GB` shared memory | Current best preset: `6 GB` resident module budget, eager pinned CPU weights, small tensors `1024 KB` | Best observed balance for 8-step image runs. |
| `>64 GB RAM` | Test `6 GB`, then carefully A/B `8 GB` and `12 GB` budgets | More budget is not automatically faster; earlier `16 GB` tests regressed badly, likely due pressure/placement effects. |

Important observed behavior: increasing resident/hot budget past the sweet spot can slow the run. In manual tests, `12 GB` improved over `8 GB` in one phase, but `16 GB` became much worse (`denoise_modular_pipe_call: 201.2456s`). The advisor should therefore prefer a conservative budget ladder and stop increasing when step time or setup worsens.

Windows-specific note: standby cache state can dominate benchmark variance. On the test system, purging standby before the transformer stabilized the best dynamic runs and brought denoise down to the `~34-38s` range in manual mode and `~14-15s` in warm dynamic mode. Presets should describe this as an optional elevated Windows optimization, not as a required cross-platform feature.

### Dynamic Weights Presets

The runner now supports `LTX_IMAGE_DYNAMIC_WEIGHTS_PRESET` as a convenience layer over the existing environment variables. Explicit environment variables still override preset values. Presets intentionally do not enable Windows standby purge so they remain portable to Linux/WSL.

| Preset | Intended use | Main resolved settings |
| --- | --- | --- |
| `off` | Disable dynamic execution | `execution_mode=plan` |
| `one_shot_fast` | Best current one-shot baseline on the 64 GB Windows test system | `linear_runtime`, eager pinned CPU weights, `6 GB` resident module budget, `1024 KB` small tensors, native attention |
| `warm_server` | ComfyUI-like warm/server benchmark | Same as `one_shot_fast` plus `generation_repeats=2` |
| `low_ram` | Conservative start for systems around `32 GB` RAM | `linear_runtime`, eager pin, `3 GB` resident module budget, `2` pin workers |
| `compat` | Highest portability baseline for driver/OS-sensitive machines | `linear_runtime`, no pinned CPU memory, `3 GB` resident module budget |
| `long_steps` | Experimental profile for many steps or persistent reuse | `linear_runtime`, lazy pin enabled, `6 GB` resident module budget |

WSL/Linux test note: run the same preset without `LTX_IMAGE_PURGE_WINDOWS_STANDBY_*`. This will tell us whether the strong warm result is mostly from generic pinned-memory/module-residency behavior or from Windows WDDM/shared-memory behavior. Key comparison fields are `build_dynamic_weights_plan`, repeated denoise times, process RAM, and VRAM.

First WSL isolation found two portability issues before denoise:

- Text encoder group offload failed while calling Diffusers `tensor.pin_memory()` (`CUDA error: out of memory`).
- With fake prompt enabled, the transformer reached dynamic weights setup but failed on our eager `linear.weight.data.pin_memory()` path.

The dynamic weights hook now has `allow_pin_memory_fallback`, exposed as `LTX_IMAGE_DYNAMIC_WEIGHTS_ALLOW_PIN_MEMORY_FALLBACK` and enabled by presets. If large pinned memory allocation fails, the hook logs `pin_linear_weights_failed` or `pin_stored_linear_weights_failed`, disables further pin attempts, and continues with pageable CPU tensors. This keeps WSL/Linux compatibility testing moving, even though performance may drop.
