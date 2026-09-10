# Repository Guidance

This repository is the LTX 2.3 image host project: local apps, traditional Diffusers runners, Modular Diffusers custom-block runners, SDNQ runners, and comparison runners used to validate memory/offload behavior.

## Project Layout

- `app_base.py` and `app_distilled.py` are the Gradio apps.
- `run_*.py` files are local CLI runners. Keep them thin, explicit, and runnable with `--help`.
- `run_modular_*.py` files use the custom Modular Diffusers blocks in `custom_blocks/ltx2_image`.
- `run_dynamic_*.py` files exercise the external `diffusers-dynamic-offloader` package against this LTX workload.
- `run_diffusers_mm_modular_distilled.py` is only a comparison runner for `diffusers-mm`.
- `custom_blocks/ltx2_image` contains the LTX 2.3 image custom blocks, connectors, and local transformer code needed before upstream Diffusers support is complete.

## External DDO Repository

The generic dynamic offload implementation lives in:

`E:\ProjetosIA\diffusers-dynamic-offloader`

This project should depend on DDO through `pyproject.toml` and import from `diffusers_dynamic_offloader`.

DDO documentation and report-ready benchmark results belong in the DDO repo, especially:

- `README.md`
- `docs/`
- `experiments/dynamic_offload_results.md`

Local exploratory notes such as `ltx2_image_experiments.md` should remain local and ignored unless the user explicitly asks to publish them.

## Runner Conventions

- Prefer `argparse` for runner parameters. Avoid adding new environment-variable-only controls in this repo.
- Keep module-level defaults visible near the top of each runner.
- Keep metrics consistent across runners: total time, pass timing, peak VRAM/RAM, and optional per-denoise-step timing.
- Use staged component loading when memory comparison matters: prompt encoding, connector/transformer denoise, VAE decode, then cleanup.
- Do not hide DDO behavior inside LTX-specific helper code. The runner may choose a preset, but DDO should own offload policy decisions.
- For Windows benchmark runners, standby purge should be used only when intentionally comparing that behavior. Do not add purge calls to unrelated comparison runners.

## Current Runner Groups

Traditional Diffusers runners:

- `run_distilled.py`
- `run_base.py`
- `run_distilled_img2img.py`
- `run_base_img2img.py`
- `run_sdnq_distilled.py`
- `run_sdnq_base.py`
- `run_sdnq_distilled_img2img.py`
- `run_sdnq_base_img2img.py`

Modular custom-block runners:

- `run_modular_distilled.py`
- `run_modular_base.py`
- `run_modular_distilled_img2img.py`
- `run_modular_base_img2img.py`
- `run_modular_sdnq_distilled.py`
- `run_modular_sdnq_base.py`
- `run_modular_sdnq_distilled_img2img.py`
- `run_modular_sdnq_base_img2img.py`

DDO runners:

- `run_dynamic_minimal.py`
- `run_dynamic_modular_distilled.py`
- `run_dynamic_old_distilled.py`
- `run_dynamic_old_staged_distilled.py`

Comparison runners:

- `run_diffusers_mm_modular_distilled.py`

## DDO Benchmark Interpretation

- `auto` currently resolves to the preferred DDO path for the detected platform and model/backend.
- `one_shot_fast` is the primary BF16 performance preset for staged LTX tests.
- `diffusers_offload_compat` and `diffusers_leaf_offload_compat` are compatibility baselines using official Diffusers group offload paths.
- For SDNQ or other quantized layers, DDO should preserve backend-specific modules. Prefer SDNQ runners or Diffusers-compatible offload presets unless a dedicated quantized adapter is intentionally being tested.
- `diffusers-mm` results are comparison data only. Keep them separate from the normal runner recommendations.

## Working Rules

- Do not revert unrelated dirty files or local user experiments.
- Use `apply_patch` for manual edits.
- Prefer `Get-ChildItem` and `Select-String` on this Windows environment if `rg` fails.
- Before changing README runner lists or benchmark conclusions, inspect the current files and latest pasted benchmark numbers.
- Keep docs model-agnostic when describing DDO. Keep LTX-specific behavior in this host repo's README or runner comments.
