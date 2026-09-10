"""
Modular LTX 2.3 distilled image runner using diffusers-mm.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("HF_MODULES_CACHE", str((Path(__file__).parent / ".hf_modules").resolve()))

import torch
from diffusers import AutoencoderKLLTX2Video, FlowMatchEulerDiscreteScheduler
from diffusers.loaders.lora_pipeline import LTX2LoraLoaderMixin
from transformers import Gemma3ForConditionalGeneration, GemmaTokenizerFast

from custom_blocks.ltx2_image import LTX2ImageDistilledBlocks, LTX2ImageTextEncoderStep
from custom_blocks.ltx2_image.connectors_ltx2_image import LTX2ImageTextConnectors
from custom_blocks.ltx2_image.modular_blocks_ltx2_image import (
    LTX2ImageConnectorStep,
    LTX2ImageDenoiseStep,
    LTX2ImagePrepareLatentsStep,
)
from custom_blocks.ltx2_image.transformer_ltx2_image import LTX2ImageTransformer2DModel
from inference_utils import RunTracker, flush

DEVICE = "cuda:0"
OFFLOAD_DEVICE = "cpu"
DTYPE = torch.bfloat16

MODEL_TAG = "distilled_modular_diffusers_mm"
MODEL_PATH = r"elismasilva/ltx2.3-image-distilled-1.1"
LOW_CPU_MEM_USAGE = True

DIFFUSERS_MM_STRATEGY = "auto"
TEXT_ENCODER_MM_STRATEGY = "group_offload"
GROUP_OFFLOAD_USE_STREAM = True
GROUP_OFFLOAD_LOW_CPU_MEM = True
BLOCK_PIN_COUNT = None
BLOCK_PIN_AUTO_EVICT = True
BLOCK_PIN_SPILL_AWARE = True
BLOCK_PIN_SPILL_MARGIN_GB = 0.5
BLOCK_PIN_WORKLOAD_PROBE = True
BLOCK_PIN_CALL_WORKLOAD = True
AUTO_BLOCK_PIN_WORKING_SET_GB = None
AUTO_BLOCK_PIN_WORKING_SET_WINDOWS_GB = None
AUTO_BLOCK_PIN_ALLOCATOR_INFLATION = None
AUTO_BLOCK_PIN_ALLOCATOR_INFLATION_WINDOWS = None
AUTO_BLOCK_PIN_ALLOCATOR_POOL_OVERHEAD_GB = None
AUTO_BLOCK_PIN_ALLOCATOR_POOL_OVERHEAD_WINDOWS_GB = None
MM_VRAM_RESERVE_GB = None
MM_VRAM_RESERVE_WINDOWS_GB = None
MM_VRAM_RESERVE_WINDOWS_LARGE_CARD_EXTRA_GB = None
WIDTH = 1280
HEIGHT = 704
SEED = 43
NUM_INFERENCE_STEPS = 8
GUIDANCE_SCALE = 1.0
GUIDANCE_RESCALE = 0.7
DECODE_TIMESTEP = 0.0
DECODE_NOISE_SCALE = None
PAG_ENABLED = False
PAG_SCALE = 0.2
PAG_APPLIED_LAYERS = [28]
GENERATION_REPEATS = 1

SHOW_METRICS = True
SAVE_METRICS = True
SHOW_DENOISE_STEPS = True

CRISP_LORA_ENABLED = False
CRISP_LORA_PATH = "vrgamedevgirl84/LTX_2.3_Crisp_Enhance_Style_LoRa"
CRISP_LORA_WEIGHT_NAME = "LTX2.3_Crisp_Enhance.safetensors"
CRISP_LORA_ADAPTER_NAME = "crisp"
CRISP_LORA_SCALE = 0.3
SOFT_LORA_ENABLED = False
SOFT_LORA_PATH = "vrgamedevgirl84/LTX_2.3_Soft_Enhance_Style_LoRa"
SOFT_LORA_WEIGHT_NAME = "LTX2.3_Soft_Enhance.safetensors"
SOFT_LORA_ADAPTER_NAME = "soft"
SOFT_LORA_SCALE = 0.8

prompt = "Fisheye close-up of a calico cat wearing a tiny flower crown, sniffing the camera lens in a sunny park, with bright colors, realistic fur detail, and playful viral-pet energy."
negative_prompt = ""


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompt", default=prompt)
    parser.add_argument("--negative-prompt", default=negative_prompt)
    parser.add_argument("--width", type=int, default=WIDTH)
    parser.add_argument("--height", type=int, default=HEIGHT)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--steps", type=int, default=NUM_INFERENCE_STEPS)
    parser.add_argument("--guidance-scale", type=float, default=GUIDANCE_SCALE)
    parser.add_argument("--guidance-rescale", type=float, default=GUIDANCE_RESCALE)
    parser.add_argument("--decode-timestep", type=float, default=DECODE_TIMESTEP)
    parser.add_argument("--decode-noise-scale", type=float, default=DECODE_NOISE_SCALE)
    parser.add_argument("--pag", action="store_true", default=PAG_ENABLED)
    parser.add_argument("--pag-scale", type=float, default=PAG_SCALE)
    parser.add_argument("--pag-layers", default=",".join(map(str, PAG_APPLIED_LAYERS)))
    parser.add_argument("--generation-repeats", type=int, default=GENERATION_REPEATS)
    parser.add_argument("--output-dir", default="outputs/ltx_image_modular")
    parser.add_argument("--show-metrics", action=argparse.BooleanOptionalAction, default=SHOW_METRICS)
    parser.add_argument("--save-metrics", action=argparse.BooleanOptionalAction, default=SAVE_METRICS)
    parser.add_argument("--show-denoise-steps", action=argparse.BooleanOptionalAction, default=SHOW_DENOISE_STEPS)
    parser.add_argument("--crisp-lora", action=argparse.BooleanOptionalAction, default=CRISP_LORA_ENABLED)
    parser.add_argument("--crisp-lora-path", default=CRISP_LORA_PATH)
    parser.add_argument("--crisp-lora-weight-name", default=CRISP_LORA_WEIGHT_NAME)
    parser.add_argument("--crisp-lora-adapter-name", default=CRISP_LORA_ADAPTER_NAME)
    parser.add_argument("--crisp-lora-scale", type=float, default=CRISP_LORA_SCALE)
    parser.add_argument("--soft-lora", action=argparse.BooleanOptionalAction, default=SOFT_LORA_ENABLED)
    parser.add_argument("--soft-lora-path", default=SOFT_LORA_PATH)
    parser.add_argument("--soft-lora-weight-name", default=SOFT_LORA_WEIGHT_NAME)
    parser.add_argument("--soft-lora-adapter-name", default=SOFT_LORA_ADAPTER_NAME)
    parser.add_argument("--soft-lora-scale", type=float, default=SOFT_LORA_SCALE)
    parser.add_argument(
        "--mm-strategy",
        choices=["auto", "no_offload", "model_offload", "group_offload", "block_pin"],
        default=DIFFUSERS_MM_STRATEGY,
        help="diffusers-mm strategy for the transformer.",
    )
    parser.add_argument(
        "--text-encoder-mm-strategy",
        choices=["auto", "no_offload", "model_offload", "group_offload", "block_pin"],
        default=TEXT_ENCODER_MM_STRATEGY,
        help="diffusers-mm strategy for the text encoder.",
    )
    parser.add_argument("--mm-use-stream", action=argparse.BooleanOptionalAction, default=GROUP_OFFLOAD_USE_STREAM)
    parser.add_argument("--mm-low-cpu-mem", action=argparse.BooleanOptionalAction, default=GROUP_OFFLOAD_LOW_CPU_MEM)
    parser.add_argument("--block-pin-count", type=int, default=BLOCK_PIN_COUNT)
    parser.add_argument("--block-pin-auto-evict", action=argparse.BooleanOptionalAction, default=BLOCK_PIN_AUTO_EVICT)
    parser.add_argument("--block-pin-spill-aware", action=argparse.BooleanOptionalAction, default=BLOCK_PIN_SPILL_AWARE)
    parser.add_argument("--block-pin-spill-margin-gb", type=float, default=BLOCK_PIN_SPILL_MARGIN_GB)
    parser.add_argument("--block-pin-workload-probe", action=argparse.BooleanOptionalAction, default=BLOCK_PIN_WORKLOAD_PROBE)
    parser.add_argument("--block-pin-call-workload", action=argparse.BooleanOptionalAction, default=BLOCK_PIN_CALL_WORKLOAD)
    parser.add_argument("--auto-block-pin-working-set-gb", type=float, default=AUTO_BLOCK_PIN_WORKING_SET_GB)
    parser.add_argument("--auto-block-pin-working-set-windows-gb", type=float, default=AUTO_BLOCK_PIN_WORKING_SET_WINDOWS_GB)
    parser.add_argument("--auto-block-pin-allocator-inflation", type=float, default=AUTO_BLOCK_PIN_ALLOCATOR_INFLATION)
    parser.add_argument(
        "--auto-block-pin-allocator-inflation-windows",
        type=float,
        default=AUTO_BLOCK_PIN_ALLOCATOR_INFLATION_WINDOWS,
    )
    parser.add_argument(
        "--auto-block-pin-allocator-pool-overhead-gb",
        type=float,
        default=AUTO_BLOCK_PIN_ALLOCATOR_POOL_OVERHEAD_GB,
    )
    parser.add_argument(
        "--auto-block-pin-allocator-pool-overhead-windows-gb",
        type=float,
        default=AUTO_BLOCK_PIN_ALLOCATOR_POOL_OVERHEAD_WINDOWS_GB,
    )
    parser.add_argument("--mm-vram-reserve-gb", type=float, default=MM_VRAM_RESERVE_GB)
    parser.add_argument("--mm-vram-reserve-windows-gb", type=float, default=MM_VRAM_RESERVE_WINDOWS_GB)
    parser.add_argument(
        "--mm-vram-reserve-windows-large-card-extra-gb",
        type=float,
        default=MM_VRAM_RESERVE_WINDOWS_LARGE_CARD_EXTRA_GB,
    )
    return parser.parse_args()


def build_run_slug(seed):
    pag_tag = f"pag{PAG_SCALE:g}_layers{'-'.join(map(str, PAG_APPLIED_LAYERS))}" if PAG_ENABLED else "nopag"
    lora_tags = []
    if SOFT_LORA_ENABLED:
        lora_tags.append(f"{SOFT_LORA_ADAPTER_NAME}{SOFT_LORA_SCALE:g}")
    if CRISP_LORA_ENABLED:
        lora_tags.append(f"{CRISP_LORA_ADAPTER_NAME}{CRISP_LORA_SCALE:g}")
    lora_tag = "lora_" + "-".join(lora_tags) if lora_tags else "nolora"
    return "_".join(
        [
            "ltx23_image",
            MODEL_TAG,
            "bf16",
            "text_encoder_original",
            pag_tag,
            lora_tag,
            f"{WIDTH}x{HEIGHT}",
            f"steps{NUM_INFERENCE_STEPS}",
            f"seed{seed}",
        ]
    )


def load_diffusers_mm():
    try:
        from diffusers_mm import ModelManager, block_pin_activation_scale
    except ImportError as exc:
        raise RuntimeError("Install diffusers-mm first, for example: uv pip install diffusers-mm") from exc
    return ModelManager, block_pin_activation_scale


def make_diffusers_mm_manager(strategy: str):
    ModelManager, _ = load_diffusers_mm()
    manager = ModelManager(
        strategy=strategy,
        group_offload_use_stream=GROUP_OFFLOAD_USE_STREAM,
        group_offload_low_cpu_mem=GROUP_OFFLOAD_LOW_CPU_MEM,
        block_pin_auto_evict=BLOCK_PIN_AUTO_EVICT,
        block_pin_spill_aware=BLOCK_PIN_SPILL_AWARE,
        block_pin_spill_margin_gb=BLOCK_PIN_SPILL_MARGIN_GB,
        block_pin_workload_probe=BLOCK_PIN_WORKLOAD_PROBE,
        block_pin_call_workload=BLOCK_PIN_CALL_WORKLOAD,
        auto_block_pin_working_set_gb=AUTO_BLOCK_PIN_WORKING_SET_GB,
        auto_block_pin_working_set_windows_gb=AUTO_BLOCK_PIN_WORKING_SET_WINDOWS_GB,
        auto_block_pin_allocator_inflation=AUTO_BLOCK_PIN_ALLOCATOR_INFLATION,
        auto_block_pin_allocator_inflation_windows=AUTO_BLOCK_PIN_ALLOCATOR_INFLATION_WINDOWS,
        auto_block_pin_allocator_pool_overhead_gb=AUTO_BLOCK_PIN_ALLOCATOR_POOL_OVERHEAD_GB,
        auto_block_pin_allocator_pool_overhead_windows_gb=AUTO_BLOCK_PIN_ALLOCATOR_POOL_OVERHEAD_WINDOWS_GB,
    )
    if MM_VRAM_RESERVE_GB is not None:
        manager.VRAM_RESERVE_GB = float(MM_VRAM_RESERVE_GB)
    if MM_VRAM_RESERVE_WINDOWS_GB is not None:
        manager.VRAM_RESERVE_WINDOWS_GB = float(MM_VRAM_RESERVE_WINDOWS_GB)
    if MM_VRAM_RESERVE_WINDOWS_LARGE_CARD_EXTRA_GB is not None:
        manager.VRAM_RESERVE_WINDOWS_LARGE_CARD_EXTRA_GB = float(MM_VRAM_RESERVE_WINDOWS_LARGE_CARD_EXTRA_GB)
    return manager


def apply_diffusers_mm(
    components: dict[str, torch.nn.Module],
    *,
    strategy: str,
    event_name: str,
    record_event,
    seq_len: int | None = None,
    batch: int = 1,
    activation_scale: float = 1.0,
) -> tuple[object, dict[str, torch.nn.Module]]:
    source = dict(components)
    manager = make_diffusers_mm_manager(strategy)
    if seq_len is not None:
        manager.set_block_pin_workload(seq_len, batch=batch, activation_scale=activation_scale)
    event_t0 = time.time()
    registered_components = manager.register_components(source)
    if BLOCK_PIN_COUNT is not None and "transformer" in source:
        manager.set_block_pin_count("transformer", BLOCK_PIN_COUNT)
    applied_strategy = manager.apply_offload_strategy(torch.device(DEVICE))
    record_event(
        event_name,
        time.time() - event_t0,
        requested_strategy=strategy,
        applied_strategy=applied_strategy,
        registered_components=registered_components,
        group_offload_use_stream=GROUP_OFFLOAD_USE_STREAM,
        group_offload_low_cpu_mem=GROUP_OFFLOAD_LOW_CPU_MEM,
        block_pin_count=BLOCK_PIN_COUNT,
    )
    print(f"  [diffusers-mm] {event_name}: requested={strategy} applied={applied_strategy}", flush=True)
    return manager, source



def cleanup_runtime_state(record_event, event_name: str):
    event_t0 = time.time()
    flush()
    record_event(event_name, time.time() - event_t0)


def load_lora_adapter(model, source: str, weight_name: str, adapter_name: str) -> str:
    state_dict, metadata = LTX2LoraLoaderMixin.lora_state_dict(
        source,
        weight_name=weight_name,
        return_lora_metadata=True,
    )
    model.load_lora_adapter(
        state_dict,
        prefix="transformer",
        adapter_name=adapter_name,
        metadata=metadata,
        low_cpu_mem_usage=LOW_CPU_MEM_USAGE,
    )
    return adapter_name


def maybe_load_lora(model, run_metrics, enabled, kind, path, weight_name, adapter_name, scale, record_event):
    if not enabled:
        return None
    event_t0 = time.time()
    actual_adapter_name = load_lora_adapter(model, path, weight_name, adapter_name)
    run_metrics["lora_adapters"].append(
        {
            "kind": kind,
            "path": path,
            "weight_name": weight_name,
            "adapter_name": actual_adapter_name,
            "scale": scale,
        }
    )
    record_event(
        f"load_{kind}_lora",
        time.time() - event_t0,
        path=path,
        weight_name=weight_name,
        adapter_name=actual_adapter_name,
        scale=scale,
    )
    return actual_adapter_name, scale


def activate_loras(model, adapters, record_event):
    if not adapters:
        return
    names, scales = zip(*adapters, strict=False)
    try:
        model.set_adapters(list(names), weights=list(scales))
    except TypeError:
        model.set_adapters(list(names), adapter_weights=list(scales))
    record_event("activate_loras", 0.0, adapters=dict(adapters))


def make_denoise_progress_callback(tracker):
    def callback(pipe, step_index, timestep, callback_kwargs):
        now = time.perf_counter()
        elapsed = now - callback.last_time
        callback.last_time = now
        callback.step_times.append(elapsed)
        if SHOW_DENOISE_STEPS:
            avg = (now - callback.start_time) / (step_index + 1)
            allocated = torch.cuda.memory_allocated(DEVICE) / (1024**3)
            reserved = torch.cuda.memory_reserved(DEVICE) / (1024**3)
            total_steps = len(callback.timesteps) if callback.timesteps is not None else NUM_INFERENCE_STEPS
            print(
                f"  [denoise] step {step_index + 1}/{total_steps} timestep={float(timestep):.4f} "
                f"elapsed={elapsed:.4f}s avg={avg:.4f}s/it "
                f"torch_alloc={allocated:.2f} GiB torch_reserved={reserved:.2f} GiB",
                flush=True,
            )
        return callback_kwargs

    callback.start_time = 0.0
    callback.last_time = 0.0
    callback.step_times = []
    callback.timesteps = None
    return callback


def main():
    global WIDTH, HEIGHT, SEED, NUM_INFERENCE_STEPS, GUIDANCE_SCALE, GUIDANCE_RESCALE
    global DECODE_TIMESTEP, DECODE_NOISE_SCALE, PAG_ENABLED, PAG_SCALE, PAG_APPLIED_LAYERS
    global GENERATION_REPEATS, SHOW_METRICS, SAVE_METRICS, SHOW_DENOISE_STEPS
    global CRISP_LORA_ENABLED, CRISP_LORA_PATH, CRISP_LORA_WEIGHT_NAME, CRISP_LORA_ADAPTER_NAME, CRISP_LORA_SCALE
    global SOFT_LORA_ENABLED, SOFT_LORA_PATH, SOFT_LORA_WEIGHT_NAME, SOFT_LORA_ADAPTER_NAME, SOFT_LORA_SCALE
    global DIFFUSERS_MM_STRATEGY, TEXT_ENCODER_MM_STRATEGY, GROUP_OFFLOAD_USE_STREAM, GROUP_OFFLOAD_LOW_CPU_MEM
    global BLOCK_PIN_COUNT, BLOCK_PIN_AUTO_EVICT, BLOCK_PIN_SPILL_AWARE, BLOCK_PIN_SPILL_MARGIN_GB
    global BLOCK_PIN_WORKLOAD_PROBE, BLOCK_PIN_CALL_WORKLOAD
    global AUTO_BLOCK_PIN_WORKING_SET_GB, AUTO_BLOCK_PIN_WORKING_SET_WINDOWS_GB
    global AUTO_BLOCK_PIN_ALLOCATOR_INFLATION, AUTO_BLOCK_PIN_ALLOCATOR_INFLATION_WINDOWS
    global AUTO_BLOCK_PIN_ALLOCATOR_POOL_OVERHEAD_GB, AUTO_BLOCK_PIN_ALLOCATOR_POOL_OVERHEAD_WINDOWS_GB
    global MM_VRAM_RESERVE_GB, MM_VRAM_RESERVE_WINDOWS_GB, MM_VRAM_RESERVE_WINDOWS_LARGE_CARD_EXTRA_GB
    global prompt, negative_prompt

    args = parse_args()
    WIDTH = args.width
    HEIGHT = args.height
    SEED = args.seed
    NUM_INFERENCE_STEPS = args.steps
    GUIDANCE_SCALE = args.guidance_scale
    GUIDANCE_RESCALE = args.guidance_rescale
    DECODE_TIMESTEP = args.decode_timestep
    DECODE_NOISE_SCALE = args.decode_noise_scale
    PAG_ENABLED = args.pag
    PAG_SCALE = args.pag_scale
    PAG_APPLIED_LAYERS = [int(item.strip()) for item in args.pag_layers.split(",") if item.strip()]
    GENERATION_REPEATS = args.generation_repeats
    SHOW_METRICS = args.show_metrics
    SAVE_METRICS = args.save_metrics
    SHOW_DENOISE_STEPS = args.show_denoise_steps
    CRISP_LORA_ENABLED = args.crisp_lora
    CRISP_LORA_PATH = args.crisp_lora_path
    CRISP_LORA_WEIGHT_NAME = args.crisp_lora_weight_name
    CRISP_LORA_ADAPTER_NAME = args.crisp_lora_adapter_name
    CRISP_LORA_SCALE = args.crisp_lora_scale
    SOFT_LORA_ENABLED = args.soft_lora
    SOFT_LORA_PATH = args.soft_lora_path
    SOFT_LORA_WEIGHT_NAME = args.soft_lora_weight_name
    SOFT_LORA_ADAPTER_NAME = args.soft_lora_adapter_name
    SOFT_LORA_SCALE = args.soft_lora_scale
    DIFFUSERS_MM_STRATEGY = args.mm_strategy
    TEXT_ENCODER_MM_STRATEGY = args.text_encoder_mm_strategy
    GROUP_OFFLOAD_USE_STREAM = args.mm_use_stream
    GROUP_OFFLOAD_LOW_CPU_MEM = args.mm_low_cpu_mem
    BLOCK_PIN_COUNT = args.block_pin_count
    BLOCK_PIN_AUTO_EVICT = args.block_pin_auto_evict
    BLOCK_PIN_SPILL_AWARE = args.block_pin_spill_aware
    BLOCK_PIN_SPILL_MARGIN_GB = args.block_pin_spill_margin_gb
    BLOCK_PIN_WORKLOAD_PROBE = args.block_pin_workload_probe
    BLOCK_PIN_CALL_WORKLOAD = args.block_pin_call_workload
    AUTO_BLOCK_PIN_WORKING_SET_GB = args.auto_block_pin_working_set_gb
    AUTO_BLOCK_PIN_WORKING_SET_WINDOWS_GB = args.auto_block_pin_working_set_windows_gb
    AUTO_BLOCK_PIN_ALLOCATOR_INFLATION = args.auto_block_pin_allocator_inflation
    AUTO_BLOCK_PIN_ALLOCATOR_INFLATION_WINDOWS = args.auto_block_pin_allocator_inflation_windows
    AUTO_BLOCK_PIN_ALLOCATOR_POOL_OVERHEAD_GB = args.auto_block_pin_allocator_pool_overhead_gb
    AUTO_BLOCK_PIN_ALLOCATOR_POOL_OVERHEAD_WINDOWS_GB = args.auto_block_pin_allocator_pool_overhead_windows_gb
    MM_VRAM_RESERVE_GB = args.mm_vram_reserve_gb
    MM_VRAM_RESERVE_WINDOWS_GB = args.mm_vram_reserve_windows_gb
    MM_VRAM_RESERVE_WINDOWS_LARGE_CARD_EXTRA_GB = args.mm_vram_reserve_windows_large_card_extra_gb
    prompt = args.prompt
    negative_prompt = args.negative_prompt

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this runner.")

    seed = SEED or torch.randint(0, 2**32, (1,)).item()
    if not SEED:
        print(f"  Using random seed: {seed}")
    generator = torch.Generator(device="cpu").manual_seed(seed)

    run_slug = build_run_slug(seed)
    output_dir = Path(args.output_dir)
    metrics_dir = output_dir / "metrics"
    run_metrics = {
        "run_slug": run_slug,
        "model_tag": MODEL_TAG,
        "model_path": MODEL_PATH,
        "text_encoder_kind": "original",
        "transformer_kind": "custom_modular",
        "width": WIDTH,
        "height": HEIGHT,
        "seed": seed,
        "num_inference_steps": NUM_INFERENCE_STEPS,
        "guidance_scale": GUIDANCE_SCALE,
        "guidance_rescale": GUIDANCE_RESCALE,
        "vae_decode_timestep": DECODE_TIMESTEP,
        "vae_decode_noise_scale": DECODE_NOISE_SCALE,
        "pag_enabled": PAG_ENABLED,
        "pag_scale": PAG_SCALE if PAG_ENABLED else 0.0,
        "pag_applied_layers": PAG_APPLIED_LAYERS if PAG_ENABLED else None,
        "lora_enabled": SOFT_LORA_ENABLED or CRISP_LORA_ENABLED,
        "lora_adapters": [],
        "dtype": str(DTYPE),
        "offload_backend": "diffusers_mm",
        "diffusers_mm_config": {
            "transformer_strategy": DIFFUSERS_MM_STRATEGY,
            "text_encoder_strategy": TEXT_ENCODER_MM_STRATEGY,
            "group_offload_use_stream": GROUP_OFFLOAD_USE_STREAM,
            "group_offload_low_cpu_mem": GROUP_OFFLOAD_LOW_CPU_MEM,
            "block_pin_count": BLOCK_PIN_COUNT,
            "block_pin_auto_evict": BLOCK_PIN_AUTO_EVICT,
            "block_pin_spill_aware": BLOCK_PIN_SPILL_AWARE,
            "block_pin_spill_margin_gb": BLOCK_PIN_SPILL_MARGIN_GB,
            "block_pin_workload_probe": BLOCK_PIN_WORKLOAD_PROBE,
            "block_pin_call_workload": BLOCK_PIN_CALL_WORKLOAD,
            "auto_block_pin_working_set_gb": AUTO_BLOCK_PIN_WORKING_SET_GB,
            "auto_block_pin_working_set_windows_gb": AUTO_BLOCK_PIN_WORKING_SET_WINDOWS_GB,
            "auto_block_pin_allocator_inflation": AUTO_BLOCK_PIN_ALLOCATOR_INFLATION,
            "auto_block_pin_allocator_inflation_windows": AUTO_BLOCK_PIN_ALLOCATOR_INFLATION_WINDOWS,
            "auto_block_pin_allocator_pool_overhead_gb": AUTO_BLOCK_PIN_ALLOCATOR_POOL_OVERHEAD_GB,
            "auto_block_pin_allocator_pool_overhead_windows_gb": AUTO_BLOCK_PIN_ALLOCATOR_POOL_OVERHEAD_WINDOWS_GB,
            "mm_vram_reserve_gb": MM_VRAM_RESERVE_GB,
            "mm_vram_reserve_windows_gb": MM_VRAM_RESERVE_WINDOWS_GB,
            "mm_vram_reserve_windows_large_card_extra_gb": MM_VRAM_RESERVE_WINDOWS_LARGE_CARD_EXTRA_GB,
        },
        "events": [],
        "steps": [],
    }

    tracker = RunTracker(DEVICE, run_metrics, interval=0.1, show_metrics=SHOW_METRICS)
    record_event = tracker.record_event
    step_start = tracker.step_start
    step_end = tracker.step_end
    denoise_progress_callback = make_denoise_progress_callback(tracker)

    print("Using diffusers-mm modular comparison runner", flush=True)
    print(
        f"  text_encoder_strategy={TEXT_ENCODER_MM_STRATEGY} transformer_strategy={DIFFUSERS_MM_STRATEGY} "
        f"model_path={MODEL_PATH}",
        flush=True,
    )
    print(f"  VRAM baseline: {torch.cuda.memory_allocated(DEVICE) / (1024**3):.2f} GB", flush=True)

    t0 = step_start("Pass 0: Encode prompts")

    event_t0 = time.time()
    text_encoder = Gemma3ForConditionalGeneration.from_pretrained(
        MODEL_PATH,
        subfolder="text_encoder",
        torch_dtype=DTYPE,
        low_cpu_mem_usage=LOW_CPU_MEM_USAGE,
    )
    record_event("load_text_encoder", time.time() - event_t0, source=MODEL_PATH)

    text_encoder_mm, text_encoder_source = apply_diffusers_mm(
        {"text_encoder": text_encoder},
        strategy=TEXT_ENCODER_MM_STRATEGY,
        event_name="setup_text_encoder_diffusers_mm",
        record_event=record_event,
    )

    event_t0 = time.time()
    tokenizer = GemmaTokenizerFast.from_pretrained(MODEL_PATH, subfolder="tokenizer")
    record_event("load_tokenizer", time.time() - event_t0, source=MODEL_PATH)

    event_t0 = time.time()
    prompt_pipe = LTX2ImageTextEncoderStep().init_pipeline()
    prompt_pipe.update_components(text_encoder=text_encoder, tokenizer=tokenizer)
    record_event("build_prompt_modular_pipeline", time.time() - event_t0, model_path=MODEL_PATH)

    event_t0 = time.time()
    with torch.inference_mode(), text_encoder_mm.device_scope(device=DEVICE, dtype=DTYPE):
        prompt_state = prompt_pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            guidance_scale=GUIDANCE_SCALE,
            output=[
                "prompt_embeds",
                "prompt_attention_mask",
                "negative_prompt_embeds",
                "negative_prompt_attention_mask",
                "batch_size",
                "dtype",
                "do_classifier_free_guidance",
            ],
        )
    record_event("encode_prompt_call", time.time() - event_t0, classifier_free_guidance=False)

    prompt_embeds = prompt_state["prompt_embeds"].to(OFFLOAD_DEVICE)
    prompt_attention_mask = prompt_state["prompt_attention_mask"].to(OFFLOAD_DEVICE)
    latent_batch_size = prompt_state["batch_size"]
    prompt_dtype = prompt_state["dtype"]
    do_classifier_free_guidance = prompt_state["do_classifier_free_guidance"]
    if SHOW_METRICS:
        print(f"  prompt_embeds shape: {prompt_embeds.shape}")

    text_encoder_mm.unregister_components(text_encoder_source)
    del prompt_state, prompt_pipe, text_encoder, tokenizer, text_encoder_mm, text_encoder_source
    cleanup_runtime_state(record_event, "cleanup_after_text_encoder")
    step_end("Pass 0: Encode prompts", t0)

    t0 = step_start(f"Pass 1: Generate at {WIDTH}x{HEIGHT}")

    event_t0 = time.time()
    connectors = LTX2ImageTextConnectors.from_pretrained(
        MODEL_PATH,
        subfolder="connectors",
        torch_dtype=DTYPE,
        low_cpu_mem_usage=LOW_CPU_MEM_USAGE,
    ).to(DEVICE)
    record_event("load_connectors_to_cuda", time.time() - event_t0, source=MODEL_PATH)

    event_t0 = time.time()
    connector_pipe = LTX2ImageConnectorStep().init_pipeline()
    connector_pipe.update_components(connectors=connectors)
    record_event("build_connector_modular_pipeline", time.time() - event_t0, model_path=MODEL_PATH)

    event_t0 = time.time()
    connector_state = connector_pipe(
        prompt_embeds=prompt_embeds.to(device=DEVICE, dtype=DTYPE),
        prompt_attention_mask=prompt_attention_mask.to(device=DEVICE),
        negative_prompt_embeds=None,
        negative_prompt_attention_mask=None,
        do_classifier_free_guidance=do_classifier_free_guidance,
        pag_scale=PAG_SCALE if PAG_ENABLED else 0.0,
        output=[
            "connector_prompt_embeds",
            "connector_attention_mask",
            "batch_size",
            "transformer_batch_multiplier",
            "do_perturbed_attention_guidance",
        ],
    )
    record_event("connector_modular_pipe_call", time.time() - event_t0)

    connector_prompt_embeds = connector_state["connector_prompt_embeds"].to(OFFLOAD_DEVICE)
    connector_attention_mask = connector_state["connector_attention_mask"].to(OFFLOAD_DEVICE)
    latent_batch_size = connector_state["batch_size"]
    transformer_batch_multiplier = connector_state["transformer_batch_multiplier"]
    do_perturbed_attention_guidance = connector_state["do_perturbed_attention_guidance"]

    del connector_state, connector_pipe, connectors, prompt_embeds, prompt_attention_mask
    cleanup_runtime_state(record_event, "cleanup_after_connectors")


    transformer_load_kwargs = {
        "subfolder": "transformer",
        "torch_dtype": prompt_dtype,
        "low_cpu_mem_usage": LOW_CPU_MEM_USAGE,
    }
    if LOW_CPU_MEM_USAGE:
        transformer_load_kwargs["device_map"] = "cpu"

    event_t0 = time.time()
    transformer = LTX2ImageTransformer2DModel.from_pretrained(MODEL_PATH, **transformer_load_kwargs)
    record_event(
        "load_transformer",
        time.time() - event_t0,
        source=MODEL_PATH,
        low_cpu_mem_usage=LOW_CPU_MEM_USAGE,
        device_map=transformer_load_kwargs.get("device_map"),
    )

    active_adapters = []
    for item in [
        (
            SOFT_LORA_ENABLED,
            "soft",
            SOFT_LORA_PATH,
            SOFT_LORA_WEIGHT_NAME,
            SOFT_LORA_ADAPTER_NAME,
            SOFT_LORA_SCALE,
        ),
        (
            CRISP_LORA_ENABLED,
            "crisp",
            CRISP_LORA_PATH,
            CRISP_LORA_WEIGHT_NAME,
            CRISP_LORA_ADAPTER_NAME,
            CRISP_LORA_SCALE,
        ),
    ]:
        adapter = maybe_load_lora(transformer, run_metrics, *item, record_event=record_event)
        if adapter is not None:
            active_adapters.append(adapter)
    activate_loras(transformer, active_adapters, record_event)

    latent_seq_len = max(1, (WIDTH // 32) * (HEIGHT // 32))
    transformer_forward_batch = max(1, latent_batch_size * transformer_batch_multiplier)
    lora_count = int(SOFT_LORA_ENABLED) + int(CRISP_LORA_ENABLED)
    transformer_mm, transformer_source = apply_diffusers_mm(
        {"transformer": transformer},
        strategy=DIFFUSERS_MM_STRATEGY,
        event_name="setup_transformer_diffusers_mm",
        record_event=record_event,
        seq_len=latent_seq_len,
        batch=transformer_forward_batch,
        activation_scale=load_diffusers_mm()[1](lora_count=lora_count),
    )

    event_t0 = time.time()
    scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(MODEL_PATH, subfolder="scheduler")
    record_event("load_scheduler", time.time() - event_t0, source=MODEL_PATH)

    event_t0 = time.time()
    prepare_pipe = LTX2ImagePrepareLatentsStep().init_pipeline()
    prepare_pipe.update_components(transformer=transformer, scheduler=scheduler)
    record_event("build_prepare_latents_modular_pipeline", time.time() - event_t0, model_path=MODEL_PATH)

    event_t0 = time.time()
    denoise_pipe = LTX2ImageDenoiseStep().init_pipeline()
    denoise_pipe.update_components(transformer=transformer, scheduler=scheduler)
    record_event("build_denoise_modular_pipeline", time.time() - event_t0, model_path=MODEL_PATH)

    image_latent = None
    latent_height = None
    latent_width = None
    latent_channels = None
    run_metrics["denoise_step_times_by_repeat"] = []

    for repeat_index in range(GENERATION_REPEATS):
        repeat_suffix = "" if GENERATION_REPEATS == 1 else f"_repeat_{repeat_index + 1}"
        if GENERATION_REPEATS > 1:
            print(f"  Warm generation repeat {repeat_index + 1}/{GENERATION_REPEATS}", flush=True)

        event_t0 = time.time()
        prepare_state = prepare_pipe(
            width=WIDTH,
            height=HEIGHT,
            num_inference_steps=NUM_INFERENCE_STEPS,
            batch_size=latent_batch_size,
            transformer_batch_multiplier=transformer_batch_multiplier,
            generator=generator,
            output=["latents", "timesteps", "latent_height", "latent_width", "in_channels", "video_rotary_emb"],
        )
        record_event(f"prepare_latents_modular_pipe_call{repeat_suffix}", time.time() - event_t0)

        denoise_progress_callback.timesteps = prepare_state["timesteps"]
        denoise_progress_callback.start_time = time.perf_counter()
        denoise_progress_callback.last_time = denoise_progress_callback.start_time
        denoise_progress_callback.step_times = []
        print(f"  Starting denoise loop{repeat_suffix}", flush=True)
        event_t0 = time.time()
        with transformer_mm.device_scope(device=DEVICE, dtype=DTYPE):
            denoise_state = denoise_pipe(
                latents=prepare_state["latents"],
                timesteps=prepare_state["timesteps"],
                connector_prompt_embeds=connector_prompt_embeds.to(device=DEVICE, dtype=DTYPE),
                connector_attention_mask=connector_attention_mask.to(device=DEVICE),
                latent_height=prepare_state["latent_height"],
                latent_width=prepare_state["latent_width"],
                video_rotary_emb=prepare_state["video_rotary_emb"],
            batch_size=latent_batch_size,
            transformer_batch_multiplier=transformer_batch_multiplier,
                do_classifier_free_guidance=False,
                do_perturbed_attention_guidance=do_perturbed_attention_guidance,
            guidance_scale=GUIDANCE_SCALE,
                guidance_rescale=GUIDANCE_RESCALE,
                pag_scale=PAG_SCALE if PAG_ENABLED else 0.0,
                pag_applied_layers=PAG_APPLIED_LAYERS if PAG_ENABLED else None,
                callback_on_step_end=denoise_progress_callback,
                callback_on_step_end_tensor_inputs=["latents"],
                output="latents",
            )
        record_event(
            f"denoise_modular_pipe_call{repeat_suffix}",
            time.time() - event_t0,
            repeat_index=repeat_index,
        )
        run_metrics["denoise_step_times_by_repeat"].append(list(denoise_progress_callback.step_times))
        run_metrics["denoise_step_times"] = denoise_progress_callback.step_times

        if image_latent is not None:
            del image_latent
        image_latent = denoise_state.to(OFFLOAD_DEVICE)
        latent_height = prepare_state["latent_height"]
        latent_width = prepare_state["latent_width"]
        latent_channels = prepare_state["in_channels"]
        if SHOW_METRICS:
            print(f"  Image latent: {image_latent.shape}")
        del prepare_state, denoise_state

    transformer_mm.unregister_components(transformer_source)
    del connector_prompt_embeds, connector_attention_mask
    del prepare_pipe, denoise_pipe, transformer, scheduler, transformer_mm, transformer_source
    cleanup_runtime_state(record_event, "cleanup_before_vae_decode")
    step_end(f"Pass 1: Generate at {WIDTH}x{HEIGHT}", t0)

    t0 = step_start("Pass 2: Decode VAE")

    event_t0 = time.time()
    vae = AutoencoderKLLTX2Video.from_pretrained(
        MODEL_PATH,
        subfolder="vae",
        torch_dtype=DTYPE,
        low_cpu_mem_usage=LOW_CPU_MEM_USAGE,
    ).to(DEVICE)
    record_event("load_vae_to_cuda", time.time() - event_t0, source=MODEL_PATH)

    event_t0 = time.time()
    decode_pipe = LTX2ImageDistilledBlocks().sub_blocks["decode"].init_pipeline()
    decode_pipe.update_components(vae=vae)
    image = decode_pipe(
        latents=image_latent.to(device=DEVICE, dtype=DTYPE),
        batch_size=latent_batch_size,
        latent_height=latent_height,
        latent_width=latent_width,
        in_channels=latent_channels,
        decode_timestep=DECODE_TIMESTEP,
        decode_noise_scale=DECODE_NOISE_SCALE,
        generator=generator,
        output_type="pil",
        output="images",
    )[0]
    record_event("vae_decode_modular_call", time.time() - event_t0)

    del decode_pipe, vae, image_latent
    cleanup_runtime_state(record_event, "cleanup_after_vae_decode")
    step_end("Pass 2: Decode VAE", t0)

    t0 = step_start("Save Image")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{run_slug}.png"
    image.save(output_path)
    print(f"  Image saved successfully to: {output_path}")
    step_end("Save Image", t0)

    total_time = tracker.total_elapsed()
    run_metrics["total_elapsed_sec"] = round(total_time, 4)
    run_metrics["global_peak_vram_gb"] = round(tracker.global_peak_vram, 4)
    run_metrics["global_peak_ram_gb"] = round(tracker.global_peak_ram, 4)

    metrics_path = metrics_dir / f"{run_slug}.json"
    if SAVE_METRICS:
        metrics_dir.mkdir(parents=True, exist_ok=True)
        metrics_path.write_text(json.dumps(run_metrics, indent=2), encoding="utf-8")

    print("\n" + "=" * 70)
    print(f"  TOTAL: {total_time:.1f}s | Peak VRAM: {tracker.global_peak_vram:.2f} GB | Peak RAM: {tracker.global_peak_ram:.2f} GB")
    print(f"  Output: {output_path}")
    if SAVE_METRICS:
        print(f"  Metrics JSON: {metrics_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
