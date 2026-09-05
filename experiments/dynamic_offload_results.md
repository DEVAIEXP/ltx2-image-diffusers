# Dynamic Offload Experiment Notes

Baseline unless noted:
- Model: LTX 2.3 distilled image modular runner
- Resolution: 1280x704
- Steps: 8
- Seed: 43
- Dtype: bfloat16
- Prompt mode: fake prompt for transformer-only comparisons
- Attention backend: native
- Windows standby purge before transformer enabled for Windows tests

## Windows Transformer Comparisons

| Scenario | Preset / route | RAM policy | Setup seconds | Denoise seconds | Copy seconds | Peak VRAM | Peak RAM | Notes |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| Dynamic offload, enough RAM | `one_shot_fast` | detected available RAM 46.84 GB, full pin selected | 34.67 | 14.38 | 0.56 | 6.97 GB | 25.15 GB | Correct fast path: `snap_to_full_pin`, 444 patched modules, 11 resident blocks. |
| Dynamic offload, simulated 32 GB RAM | `one_shot_fast` | pin skipped, insufficient RAM | 7.72 | 198.2 total pass, denoise dominated by copies | 174.12 | 6.97 GB | 25.87 GB | Correct safe path: avoids pinning when usable RAM is below required RAM. |
| Low RAM fallback | `low_ram_safe` | no pin, smaller resident budget | 3.78 | 194.10 | 187.81 | 3.28 GB reserved during denoise | not captured in pasted slice | Lower VRAM, but slower than `one_shot_fast` simulated 32 GB because fewer resident modules and more runtime copies. |
| Diffusers group offload | `off` + transformer `leaf_level`, stream, record stream, low CPU mem usage off | official Diffusers group offload | 0.07 group setup | 315.29 | not tracked by dynamic offloads | 6.49 GB | 26.60 GB | Very compatible/official path, but much slower in this low-VRAM 8-step transformer scenario. |
| Diffusers group offload | `off` + transformer `leaf_level`, stream, record stream, low CPU mem usage on | official Diffusers group offload | 0.05 group setup | 320.10 | not tracked by dynamic offloads | 6.56 GB | 26.60 GB | Same shape as low CPU off; setup is tiny, but denoise remains copy/offload bound. |
| Diffusers group offload | `off` + transformer `leaf_level`, no stream, no record stream, low CPU mem usage on | official Diffusers group offload | 0.03 group setup | 507.83 | not tracked by dynamic offloads | 6.58 GB | 39.37 GB | Worst official offload probe so far; disabling stream made denoise much slower and increased peak RAM. |
| Diffusers group offload | `off` + transformer `block_level`, `num_blocks_per_group=1`, stream, record stream, low CPU mem usage on | official Diffusers group offload | 0.02 group setup | 265.47 | not tracked by dynamic offloads | 6.49 GB | 26.79 GB | Better than leaf, but still far from dynamic offloads fast path. `torch_alloc` stayed near 0.03 GiB while reserved grew to 2.11 GiB. |
| Diffusers group offload | `off` + transformer `block_level`, `num_blocks_per_group=1`, stream, no record stream, low CPU mem usage on | official Diffusers group offload | 0.01 group setup | 312.03 | not tracked by dynamic offloads | 6.51 GB | 26.78 GB | Worse than record stream on; reserved VRAM was lower, but denoise returned to leaf-level timing. |

## Current Interpretation

`one_shot_fast` is the default general preset. It pins CPU weights only when the measured or supplied available RAM can cover the model weight copy, resident GPU modules, and configured system headroom. On machines with enough RAM, it pays setup time once and keeps denoise fast. On constrained RAM, it chooses safety over speed.

`low_ram_safe` is not the recommended 32 GB preset for this model. It is a fallback for tighter VRAM cases where the user accepts slow denoise to reduce accelerator memory pressure.

The Diffusers group offload path is useful as an official compatibility baseline, but in this test it is not close to
the dynamic offloads fast path. `block_level` with one block per group and `record_stream=1` was the best official
offload probe, but it still remained dominated by repeated offload movement. `leaf_level` without stream is not viable
for this model/shape.

## Preset Direction

Recommended defaults:
- `auto` -> `one_shot_fast`
- `one_shot_fast` -> balanced planner, RAM-aware pinning
- `low_ram_safe` -> explicit fallback only
- `wsl_compat` -> explicit WSL fallback when stream/pin behavior is unstable
- `warm_process` -> process-lifetime cache comparisons and server-like usage
- `diffusers_offload_compat` -> official Diffusers group-offload compatibility fallback
- `diffusers_leaf_offload_compat` -> official Diffusers leaf-level fallback for models where finer granularity may fit better

Inspect the current preset contract without loading model weights:

```powershell
$env:DDO_RUNNER_PRINT_DYNAMIC_OFFLOAD_PRESETS="1"
python run_modular_distilled.py
Remove-Item Env:DDO_RUNNER_PRINT_DYNAMIC_OFFLOAD_PRESETS -ErrorAction SilentlyContinue
```

Potential report sections:
- one-shot setup cost vs denoise throughput
- RAM-aware pinning behavior
- Windows standby cache impact
- Diffusers official group offload baseline
- WSL/Linux validation matrix
- Quantized model compatibility risks

## Validation Matrix

Use these as the next closed test ladder. Keep Windows as the development gate; repeat WSL/Ubuntu only after a Windows
change produces a useful result.

Windows baseline:

```powershell
$env:DDO_PRESET="auto"
$env:DDO_RUNNER_WIDTH="1280"
$env:DDO_RUNNER_HEIGHT="704"
$env:DDO_RUNNER_STEPS="8"
$env:DDO_RUNNER_SEED="43"
$env:DDO_RUNNER_GENERATION_REPEATS="1"
$env:DDO_RUNNER_FAKE_PROMPT="0"
$env:DDO_RUNNER_METRICS_LEVEL="1"
$env:DDO_SHOW_PROFILE="0"
$env:DDO_RUNNER_PURGE_WINDOWS_STANDBY_BEFORE_RUN="1"
$env:DDO_RUNNER_PURGE_WINDOWS_STANDBY_AFTER_TEXT_ENCODER="1"
$env:DDO_RUNNER_PURGE_WINDOWS_STANDBY_BEFORE_TRANSFORMER="1"
python run_modular_distilled.py
```

Windows RAM-constrained planner check:

```powershell
$env:DDO_AVAILABLE_SYSTEM_RAM_GB="32"
python run_modular_distilled.py
Remove-Item Env:DDO_AVAILABLE_SYSTEM_RAM_GB -ErrorAction SilentlyContinue
```

Linux/Ubuntu baseline:

```bash
export DDO_PRESET=auto
export DDO_RUNNER_WIDTH=1280
export DDO_RUNNER_HEIGHT=704
export DDO_RUNNER_STEPS=8
export DDO_RUNNER_SEED=43
export DDO_RUNNER_GENERATION_REPEATS=1
export DDO_RUNNER_FAKE_PROMPT=0
export DDO_RUNNER_METRICS_LEVEL=1
export DDO_SHOW_PROFILE=0
unset DDO_RUNNER_PURGE_WINDOWS_STANDBY_BEFORE_RUN
unset DDO_RUNNER_PURGE_WINDOWS_STANDBY_AFTER_TEXT_ENCODER
unset DDO_RUNNER_PURGE_WINDOWS_STANDBY_BEFORE_TRANSFORMER
python run_modular_distilled.py
```

WSL compatibility fallback:

```bash
export DDO_PRESET=wsl_compat
export DDO_RUNNER_WIDTH=1280
export DDO_RUNNER_HEIGHT=704
export DDO_RUNNER_STEPS=8
export DDO_RUNNER_SEED=43
export DDO_RUNNER_GENERATION_REPEATS=1
export DDO_RUNNER_FAKE_PROMPT=0
export DDO_RUNNER_METRICS_LEVEL=1
export DDO_SHOW_PROFILE=0
python run_modular_distilled.py
```

Official Diffusers fallback comparison:

```powershell
$env:DDO_PRESET="diffusers_offload_compat"
python run_modular_distilled.py
```
