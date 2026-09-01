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
