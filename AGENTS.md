# Repository Guidance

This repository is currently focused on a generic Diffusers-style dynamic weights manager for low-VRAM inference.

## Project Direction

- Keep the dynamic weights implementation model-agnostic. It must not depend on LTX-specific class names or component types.
- Prefer Diffusers conventions and architecture. Code should be easy to upstream or adapt into Diffusers internals later.
- Keep runtime logic self-contained in the memory manager. Runners should mainly parse/pass parameters and orchestrate components.
- Avoid native VBAR or hardware-specific low-level paths unless explicitly chosen later. The current priority is compatibility first, then performance.
- Keep Windows standby purge as a Windows-only runner/system helper, not as a generic Linux requirement.

## Experiment Tracking

- `custom_blocks/ltx2_image/EXPERIMENTS.md` is the canonical experiment log. Update it whenever a benchmark changes the current recommendation or invalidates an older assumption.
- `experiments/dynamic_weights_results.md` is a compact sidecar for current comparison tables and report-ready notes.
- Historical results may stay in `EXPERIMENTS.md`, but stale recommendations must be marked as historical or corrected.
- Use generic environment names in new docs and commands:
  - `DIFFUSERS_DYNAMIC_WEIGHTS_*`
  - `DIFFUSERS_RUNNER_*`
- Legacy `LTX_IMAGE_*` aliases may remain supported in code, but should not be the default in new experiment notes.

## Current Preset Meaning

- `auto` resolves to `one_shot_fast` on Windows, Linux, and WSL.
- `one_shot_fast` is the default benchmark path: RAM-aware balanced planner, up to 6 GB resident modules, and pinned CPU weights only when enough usable system RAM is available.
- `low_ram_safe` is an explicit low-VRAM fallback. It reduces accelerator pressure but can make denoise copy-bound and very slow.
- `wsl_compat` is an explicit WSL/driver fallback for cases where pinned-memory or stream behavior is unstable.
- `warm_process` is for process-lifetime cache/server-like comparisons.
- `diffusers_offload_compat` uses official Diffusers block-level group offload as a compatibility baseline, not the current performance baseline.
- `diffusers_leaf_offload_compat` keeps the official Diffusers leaf-level group offload path available for smaller/different models.

## Working Rules

- Do not revert unrelated dirty files. `app_distilled.py`, `run_distilled.py`, local Ubuntu logs, and local helper binaries may be user-owned unless the task explicitly targets them.
- Use `apply_patch` for manual edits.
- Before changing experiment conclusions, check the latest pasted benchmark numbers and avoid resurrecting older assumptions.
