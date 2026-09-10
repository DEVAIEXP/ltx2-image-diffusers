"""
Local low-VRAM parity runner for the experimental LTX 2.3 distilled modular T2I blocks.
"""

import argparse
import json
import os
import time
from pathlib import Path

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("HF_MODULES_CACHE", str((Path(__file__).parent / ".hf_modules").resolve()))

import torch
from diffusers import AutoencoderKLLTX2Video, FlowMatchEulerDiscreteScheduler
from diffusers_dynamic_offloader import (
    DynamicOffloadSettings,
    enable_offload,
    format_dynamic_offload_presets,
    from_pretrained_with_dynamic_offload,
    is_wsl_environment,
    maybe_purge_windows_standby_cache,
    remove_dynamic_offload,
)
from transformers import Gemma3ForConditionalGeneration, GemmaTokenizerFast

from custom_blocks.ltx2_image import LTX2ImageDistilledBlocks, LTX2ImageTextEncoderStep
from custom_blocks.ltx2_image.connectors_ltx2_image import LTX2ImageTextConnectors
from custom_blocks.ltx2_image.modular_blocks_ltx2_image import (
    LTX2ImageConnectorStep,
    LTX2ImageDenoiseStep,
    LTX2ImagePrepareLatentsStep,
)
from inference_utils import RunTracker, flush

RUNNING_ON_WSL = False
DEVICE = "cuda:0"
OFFLOAD_DEVICE = "cpu"
DTYPE = torch.bfloat16

MODEL_TAG = "distilled_modular"
MODEL_PATH = os.getenv("MODEL_PATH", r"elismasilva/ltx2.3-image-distilled-1.1")
TEXT_ENCODER_LOW_CPU_MEM_USAGE = True
MODEL_LOW_CPU_MEM_USAGE = True
DYNAMIC_OFFLOAD_SETTINGS = None
REQUESTED_DYNAMIC_OFFLOAD_PRESET = "auto"
DYNAMIC_OFFLOAD_PRESET = "auto"


def preset_env(name: str, default: str = "") -> str:
    if name in os.environ:
        return os.environ[name]
    value = DYNAMIC_OFFLOAD_SETTINGS.preset_value(name)
    if value != "":
        return value
    return default


def parse_bool_preset_env(name: str, default: str = "0") -> bool:
    return preset_env(name, default).strip().lower() in {"1", "true", "yes", "on"}


TEXT_ENCODER_GROUP_OFFLOAD = True
TEXT_ENCODER_DYNAMIC_OFFLOAD = False
TRANSFORMER_MEMORY_MANAGER = "off"
DYNAMIC_OFFLOAD_CONFIG = None
DYNAMIC_OFFLOAD_EXECUTION_MODE = "linear_runtime"
DYNAMIC_OFFLOAD_PIN_CPU_MEMORY = None
DYNAMIC_OFFLOAD_SHOW_PROFILE = False
DYNAMIC_OFFLOAD_EFFECTIVE_PIN_CPU_MEMORY = None
PRE_VAE_CLEANUP_REPEATS = 1
RESET_DYNAMIC_MEMORY_AFTER_RUN = False
METRICS_LEVEL = 1
SHOW_METRICS = True
SAVE_METRICS = False
SHOW_DENOISE_STEPS = True
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
FAKE_PROMPT_EMBEDS = False
GENERATION_REPEATS = 1
TRANSFORMER_PREPARE_REPEATS = 1

prompt = (
    "Fisheye close-up of a calico cat wearing a tiny flower crown, sniffing the camera lens in a sunny park, "
    "with bright colors, realistic fur detail, and playful viral-pet energy."
)
negative_prompt = ""


def parse_optional_float(value: str | None) -> float | None:
    if value in (None, ""):
        return None
    return float(value)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", default=MODEL_PATH)
    parser.add_argument("--preset", default="auto")
    parser.add_argument("--device", default=DEVICE)
    parser.add_argument("--prompt", default=prompt)
    parser.add_argument("--negative-prompt", default=negative_prompt)
    parser.add_argument("--width", type=int, default=WIDTH)
    parser.add_argument("--height", type=int, default=HEIGHT)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--steps", type=int, default=NUM_INFERENCE_STEPS)
    parser.add_argument("--guidance-scale", type=float, default=GUIDANCE_SCALE)
    parser.add_argument("--guidance-rescale", type=float, default=GUIDANCE_RESCALE)
    parser.add_argument("--decode-timestep", type=float, default=DECODE_TIMESTEP)
    parser.add_argument("--decode-noise-scale", type=parse_optional_float, default=DECODE_NOISE_SCALE)
    parser.add_argument("--pag", action=argparse.BooleanOptionalAction, default=PAG_ENABLED)
    parser.add_argument("--pag-scale", type=float, default=PAG_SCALE)
    parser.add_argument("--pag-layers", default=",".join(str(layer) for layer in PAG_APPLIED_LAYERS))
    parser.add_argument("--fake-prompt", action=argparse.BooleanOptionalAction, default=FAKE_PROMPT_EMBEDS)
    parser.add_argument("--generation-repeats", type=int, default=GENERATION_REPEATS)
    parser.add_argument("--transformer-prepare-repeats", type=int, default=TRANSFORMER_PREPARE_REPEATS)
    parser.add_argument("--pre-vae-cleanup-repeats", type=int, default=None)
    parser.add_argument("--output-dir", default="outputs/ltx_image_modular")
    parser.add_argument("--metrics-level", type=int, choices=(0, 1, 2), default=METRICS_LEVEL)
    parser.add_argument("--show-metrics", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--save-metrics", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--show-denoise-steps", action=argparse.BooleanOptionalAction, default=SHOW_DENOISE_STEPS)
    parser.add_argument("--low-cpu-mem-usage", action=argparse.BooleanOptionalAction, default=MODEL_LOW_CPU_MEM_USAGE)
    parser.add_argument("--text-encoder-low-cpu-mem-usage", action=argparse.BooleanOptionalAction, default=TEXT_ENCODER_LOW_CPU_MEM_USAGE)
    parser.add_argument("--text-encoder-group-offload", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--text-encoder-dynamic-offload", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--resident-module-budget-gb", type=float, default=None)
    parser.add_argument("--max-resident-module-budget-gb", type=float, default=None)
    parser.add_argument("--pin-weight-budget-gb", type=float, default=None)
    parser.add_argument("--reset-dynamic-memory-after-run", action=argparse.BooleanOptionalAction, default=RESET_DYNAMIC_MEMORY_AFTER_RUN)
    parser.add_argument("--print-presets", action="store_true")
    return parser.parse_args()


def build_run_slug(seed):
    pag_tag = f"pag{PAG_SCALE:g}_layers{'-'.join(map(str, PAG_APPLIED_LAYERS))}" if PAG_ENABLED else "nopag"
    return "_".join(
        [
            "ltx23_image",
            MODEL_TAG,
            "bf16",
            "text_encoder_original",
            pag_tag,
            f"{WIDTH}x{HEIGHT}",
            f"steps{NUM_INFERENCE_STEPS}",
            f"seed{seed}",
        ]
    )


def cleanup_runtime_state(record_event, event_name: str, *, repeats: int = 1, collect_cuda_ipc: bool = False) -> None:
    if repeats <= 0:
        return

    before_free_gb = before_total_gb = None
    after_free_gb = after_total_gb = None
    if torch.cuda.is_available():
        before_free, before_total = torch.cuda.mem_get_info(DEVICE)
        before_free_gb = before_free / 1024**3
        before_total_gb = before_total / 1024**3

    event_t0 = time.time()
    for _ in range(repeats):
        flush()
        if collect_cuda_ipc and torch.cuda.is_available() and hasattr(torch.cuda, "ipc_collect"):
            torch.cuda.ipc_collect()

    if torch.cuda.is_available():
        after_free, after_total = torch.cuda.mem_get_info(DEVICE)
        after_free_gb = after_free / 1024**3
        after_total_gb = after_total / 1024**3

    record_event(
        event_name,
        time.time() - event_t0,
        repeats=repeats,
        collect_cuda_ipc=collect_cuda_ipc,
        before_free_vram_gb=None if before_free_gb is None else round(before_free_gb, 4),
        before_total_vram_gb=None if before_total_gb is None else round(before_total_gb, 4),
        after_free_vram_gb=None if after_free_gb is None else round(after_free_gb, 4),
        after_total_vram_gb=None if after_total_gb is None else round(after_total_gb, 4),
    )


def cleanup_before_vae_decode(record_event) -> None:
    cleanup_runtime_state(
        record_event,
        "cleanup_before_vae_decode",
        repeats=PRE_VAE_CLEANUP_REPEATS,
        collect_cuda_ipc=True,
    )


def denoise_progress_callback(components, step_index, timestep, callback_kwargs):
    now = time.perf_counter()
    last_time = getattr(denoise_progress_callback, "last_time", now)
    start_time = getattr(denoise_progress_callback, "start_time", last_time)
    step_elapsed = now - last_time
    total_elapsed = now - start_time
    denoise_progress_callback.last_time = now
    denoise_progress_callback.step_times.append(step_elapsed)

    used_gb = torch.cuda.memory_allocated(DEVICE) / 1024**3
    reserved_gb = torch.cuda.memory_reserved(DEVICE) / 1024**3
    total_steps = len(denoise_progress_callback.timesteps)
    avg_elapsed = total_elapsed / (step_index + 1)
    if SHOW_DENOISE_STEPS:
        print(
            f"  [denoise] step {step_index + 1}/{total_steps} timestep={float(timestep):.4f} "
            f"elapsed={step_elapsed:.4f}s avg={avg_elapsed:.4f}s/it "
            f"torch_alloc={used_gb:.2f} GiB torch_reserved={reserved_gb:.2f} GiB",
            flush=True,
        )
    return callback_kwargs


def main():
    global RUNNING_ON_WSL, DEVICE, MODEL_PATH, TEXT_ENCODER_LOW_CPU_MEM_USAGE, MODEL_LOW_CPU_MEM_USAGE
    global DYNAMIC_OFFLOAD_SETTINGS, REQUESTED_DYNAMIC_OFFLOAD_PRESET, DYNAMIC_OFFLOAD_PRESET
    global TEXT_ENCODER_GROUP_OFFLOAD, TEXT_ENCODER_DYNAMIC_OFFLOAD, TRANSFORMER_MEMORY_MANAGER
    global DYNAMIC_OFFLOAD_CONFIG, DYNAMIC_OFFLOAD_EXECUTION_MODE, DYNAMIC_OFFLOAD_PIN_CPU_MEMORY
    global DYNAMIC_OFFLOAD_SHOW_PROFILE, DYNAMIC_OFFLOAD_EFFECTIVE_PIN_CPU_MEMORY
    global PRE_VAE_CLEANUP_REPEATS, RESET_DYNAMIC_MEMORY_AFTER_RUN
    global METRICS_LEVEL, SHOW_METRICS, SAVE_METRICS, SHOW_DENOISE_STEPS
    global WIDTH, HEIGHT, SEED, NUM_INFERENCE_STEPS, GUIDANCE_SCALE, GUIDANCE_RESCALE
    global DECODE_TIMESTEP, DECODE_NOISE_SCALE, PAG_ENABLED, PAG_SCALE, PAG_APPLIED_LAYERS
    global FAKE_PROMPT_EMBEDS, GENERATION_REPEATS, TRANSFORMER_PREPARE_REPEATS
    global prompt, negative_prompt

    args = parse_args()

    RUNNING_ON_WSL = is_wsl_environment()
    if args.print_presets:
        print(format_dynamic_offload_presets(default_preset=args.preset, running_on_wsl=RUNNING_ON_WSL))
        return

    DEVICE = args.device
    MODEL_PATH = args.model_path
    TEXT_ENCODER_LOW_CPU_MEM_USAGE = args.text_encoder_low_cpu_mem_usage
    MODEL_LOW_CPU_MEM_USAGE = args.low_cpu_mem_usage

    settings_environ = os.environ.copy()
    if args.resident_module_budget_gb is not None:
        settings_environ["DDO_RESIDENT_MODULE_BUDGET_GB"] = str(args.resident_module_budget_gb)
    if args.max_resident_module_budget_gb is not None:
        settings_environ["DDO_MAX_RESIDENT_MODULE_BUDGET_GB"] = str(args.max_resident_module_budget_gb)
    if args.pin_weight_budget_gb is not None:
        settings_environ["DDO_PIN_WEIGHT_BUDGET_GB"] = str(args.pin_weight_budget_gb)

    settings_kwargs = {
        "execution_device": DEVICE,
        "offload_device": OFFLOAD_DEVICE,
        "running_on_wsl": RUNNING_ON_WSL,
        "default_preset": args.preset,
        "environ": settings_environ,
    }

    DYNAMIC_OFFLOAD_SETTINGS = DynamicOffloadSettings.from_env(**settings_kwargs)
    REQUESTED_DYNAMIC_OFFLOAD_PRESET = DYNAMIC_OFFLOAD_SETTINGS.requested_preset
    DYNAMIC_OFFLOAD_PRESET = DYNAMIC_OFFLOAD_SETTINGS.effective_preset
    DYNAMIC_OFFLOAD_CONFIG = DYNAMIC_OFFLOAD_SETTINGS.config
    DYNAMIC_OFFLOAD_EXECUTION_MODE = DYNAMIC_OFFLOAD_CONFIG.execution_mode
    DYNAMIC_OFFLOAD_PIN_CPU_MEMORY = DYNAMIC_OFFLOAD_SETTINGS.requested_pin_cpu_memory
    DYNAMIC_OFFLOAD_SHOW_PROFILE = DYNAMIC_OFFLOAD_CONFIG.show_profile
    DYNAMIC_OFFLOAD_EFFECTIVE_PIN_CPU_MEMORY = DYNAMIC_OFFLOAD_SETTINGS.effective_pin_cpu_memory

    TEXT_ENCODER_GROUP_OFFLOAD = (
        args.text_encoder_group_offload
        if args.text_encoder_group_offload is not None
        else parse_bool_preset_env("DDO_RUNNER_TEXT_ENCODER_GROUP_OFFLOAD", "1")
    )
    TEXT_ENCODER_DYNAMIC_OFFLOAD = (
        args.text_encoder_dynamic_offload
        if args.text_encoder_dynamic_offload is not None
        else parse_bool_preset_env("DDO_RUNNER_TEXT_ENCODER_DYNAMIC_OFFLOAD")
    )
    TRANSFORMER_MEMORY_MANAGER = preset_env("DDO_RUNNER_TRANSFORMER_MEMORY_MANAGER", "off").lower()
    PRE_VAE_CLEANUP_REPEATS = (
        args.pre_vae_cleanup_repeats
        if args.pre_vae_cleanup_repeats is not None
        else int(preset_env("DDO_RUNNER_PRE_VAE_CLEANUP_REPEATS", "3" if RUNNING_ON_WSL else "1"))
    )
    RESET_DYNAMIC_MEMORY_AFTER_RUN = args.reset_dynamic_memory_after_run
    METRICS_LEVEL = args.metrics_level
    SHOW_METRICS = args.show_metrics if args.show_metrics is not None else METRICS_LEVEL >= 1
    SAVE_METRICS = args.save_metrics if args.save_metrics is not None else METRICS_LEVEL >= 2
    SHOW_DENOISE_STEPS = args.show_denoise_steps
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
    FAKE_PROMPT_EMBEDS = args.fake_prompt
    GENERATION_REPEATS = max(1, args.generation_repeats)
    TRANSFORMER_PREPARE_REPEATS = max(1, args.transformer_prepare_repeats)
    prompt = args.prompt
    negative_prompt = args.negative_prompt

    seed = SEED or torch.randint(0, 2**32, (1,)).item()
    if not SEED:
        print(f"Using random seed: {seed}")
    generator = torch.Generator(device="cpu").manual_seed(seed)

    run_slug = build_run_slug(seed)
    output_dir = Path(args.output_dir)
    metrics_dir = output_dir / "metrics"
    run_metrics = {
        "run_slug": run_slug,
        "model_tag": MODEL_TAG,
        "model_path": MODEL_PATH,
        "width": WIDTH,
        "height": HEIGHT,
        "seed": seed,
        "num_inference_steps": NUM_INFERENCE_STEPS,
        "generation_repeats": GENERATION_REPEATS,
        "transformer_prepare_repeats": TRANSFORMER_PREPARE_REPEATS,
        "guidance_scale": GUIDANCE_SCALE,
        "guidance_rescale": GUIDANCE_RESCALE,
        "vae_decode_timestep": DECODE_TIMESTEP,
        "vae_decode_noise_scale": DECODE_NOISE_SCALE,
        "pag_enabled": PAG_ENABLED,
        "pag_scale": PAG_SCALE if PAG_ENABLED else 0.0,
        "pag_applied_layers": PAG_APPLIED_LAYERS if PAG_ENABLED else None,
        "dtype": str(DTYPE),
        "text_encoder_low_cpu_mem_usage": TEXT_ENCODER_LOW_CPU_MEM_USAGE,
        "text_encoder_dynamic_offload": TEXT_ENCODER_DYNAMIC_OFFLOAD,
        "text_encoder_group_offload": TEXT_ENCODER_GROUP_OFFLOAD,
        "model_low_cpu_mem_usage": MODEL_LOW_CPU_MEM_USAGE,
        "running_on_wsl": RUNNING_ON_WSL,
        "reset_dynamic_memory_after_run": RESET_DYNAMIC_MEMORY_AFTER_RUN,
        "metrics_level": METRICS_LEVEL,
        "show_metrics": SHOW_METRICS,
        "save_metrics": SAVE_METRICS,
        "transformer_memory_manager": TRANSFORMER_MEMORY_MANAGER,
        **DYNAMIC_OFFLOAD_SETTINGS.as_metrics(),
        "pre_vae_cleanup_repeats": PRE_VAE_CLEANUP_REPEATS,
        "events": [],
        "steps": [],
    }

    if DYNAMIC_OFFLOAD_PRESET:
        if REQUESTED_DYNAMIC_OFFLOAD_PRESET == "auto":
            print(f"Using DDO preset: auto -> {DYNAMIC_OFFLOAD_PRESET}", flush=True)
        else:
            print(f"Using DDO preset: {DYNAMIC_OFFLOAD_PRESET}", flush=True)
    if DYNAMIC_OFFLOAD_PIN_CPU_MEMORY and not DYNAMIC_OFFLOAD_EFFECTIVE_PIN_CPU_MEMORY:
        print("  [dynamic-offload] disabling pinned CPU memory on WSL; set DDO_DISABLE_PIN_ON_WSL=0 to force it.", flush=True)
    text_encoder_route = (
        "dynamic_offload"
        if TEXT_ENCODER_DYNAMIC_OFFLOAD
        else "group_offload"
        if TEXT_ENCODER_GROUP_OFFLOAD
        else "cuda_to"
    )
    transformer_route = "dynamic_offload" if DYNAMIC_OFFLOAD_SETTINGS.enabled else "standard"
    print(
        "  [runner] "
        f"text_encoder_route={text_encoder_route} "
        f"text_encoder_group_offload={TEXT_ENCODER_GROUP_OFFLOAD} "
        f"text_encoder_dynamic_offload={TEXT_ENCODER_DYNAMIC_OFFLOAD} "
        f"transformer_route={transformer_route} "
        f"preset={DYNAMIC_OFFLOAD_PRESET or 'none'}",
        flush=True,
    )

    tracker = RunTracker(DEVICE, run_metrics, interval=0.1, show_metrics=SHOW_METRICS)
    record_event = tracker.record_event
    step_start = tracker.step_start
    step_end = tracker.step_end

    run_metrics["purge_windows_standby_before_run"] = maybe_purge_windows_standby_cache(
        DYNAMIC_OFFLOAD_SETTINGS,
        "before_run",
        record_event=record_event,
    )

    t0 = step_start("Pass 0: Encode prompts")
    if FAKE_PROMPT_EMBEDS:
        prompt_embeds = torch.zeros((1, 1024, 188160), dtype=DTYPE, device=OFFLOAD_DEVICE)
        prompt_attention_mask = torch.ones((1, 1024), dtype=torch.long, device=OFFLOAD_DEVICE)
        record_event("fake_prompt_embeds", 0.0, shape=list(prompt_embeds.shape))
    else:
        event_t0 = time.time()
        text_encoder = Gemma3ForConditionalGeneration.from_pretrained(
            MODEL_PATH,
            subfolder="text_encoder",
            torch_dtype=DTYPE,
            low_cpu_mem_usage=TEXT_ENCODER_LOW_CPU_MEM_USAGE,
        )
        record_event("load_text_encoder", time.time() - event_t0, source=MODEL_PATH)

        event_t0 = time.time()
        text_encoder_offload = enable_offload(
            text_encoder,
            settings=DYNAMIC_OFFLOAD_SETTINGS,
            component="text_encoder",
            execution_device=DEVICE,
            offload_device=OFFLOAD_DEVICE,
            low_cpu_mem_usage=TEXT_ENCODER_LOW_CPU_MEM_USAGE,
            record_event=record_event,
            dynamic_event_name="setup_text_encoder_dynamic_offload",
            group_event_name="setup_text_encoder_group_offload",
        )
        text_encoder_dynamic_offload_hook = text_encoder_offload.hook
        run_metrics["text_encoder_offload_route"] = text_encoder_offload.route

        event_t0 = time.time()
        tokenizer = GemmaTokenizerFast.from_pretrained(MODEL_PATH, subfolder="tokenizer")
        record_event("load_tokenizer", time.time() - event_t0, source=MODEL_PATH)

        event_t0 = time.time()
        prompt_pipe = LTX2ImageTextEncoderStep().init_pipeline()
        prompt_pipe.update_components(text_encoder=text_encoder, tokenizer=tokenizer)
        record_event("build_prompt_modular_pipeline", time.time() - event_t0, model_path=MODEL_PATH)

        event_t0 = time.time()
        with torch.inference_mode():
            prompt_state = prompt_pipe(
                prompt=prompt,
                negative_prompt=negative_prompt,
                guidance_scale=GUIDANCE_SCALE,
                output=["prompt_embeds", "prompt_attention_mask"],
            )
        record_event("encode_prompt_call", time.time() - event_t0, classifier_free_guidance=False)

        prompt_embeds = prompt_state["prompt_embeds"].to(OFFLOAD_DEVICE)
        prompt_attention_mask = prompt_state["prompt_attention_mask"].to(OFFLOAD_DEVICE)
        if text_encoder_dynamic_offload_hook is not None:
            run_metrics["text_encoder_dynamic_offload_runtime_summary"] = text_encoder_dynamic_offload_hook.state.as_dict()
            if DYNAMIC_OFFLOAD_SHOW_PROFILE:
                text_encoder_dynamic_offload_hook.print_profile_summary()
            remove_dynamic_offload(text_encoder)
            text_encoder_dynamic_offload_hook = None
        del text_encoder_offload
        del prompt_state
        del prompt_pipe, text_encoder, tokenizer
        cleanup_runtime_state(record_event, "cleanup_after_text_encoder")
        run_metrics["purge_windows_standby_after_text_encoder"] = maybe_purge_windows_standby_cache(
            DYNAMIC_OFFLOAD_SETTINGS,
            "after_text_encoder",
            record_event=record_event,
        )

    if SHOW_METRICS:
        print(f"  prompt_embeds shape: {prompt_embeds.shape}")
    step_end("Pass 0: Encode prompts", t0)

    t0 = step_start(f"Pass 1: Generate at {WIDTH}x{HEIGHT}")

    event_t0 = time.time()
    connectors = LTX2ImageTextConnectors.from_pretrained(
        MODEL_PATH,
        subfolder="connectors",
        torch_dtype=DTYPE,
        low_cpu_mem_usage=MODEL_LOW_CPU_MEM_USAGE,
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
        do_classifier_free_guidance=False,
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

    del prompt_embeds, prompt_attention_mask, connector_state
    del connector_pipe, connectors
    cleanup_runtime_state(record_event, "cleanup_after_connectors")

    dynamic_offload_enabled = DYNAMIC_OFFLOAD_SETTINGS.enabled
    dynamic_offload_config = DYNAMIC_OFFLOAD_CONFIG if dynamic_offload_enabled else None

    def transformer_prepare_event_name(name: str, repeat_index: int) -> str:
        if TRANSFORMER_PREPARE_REPEATS == 1:
            return name
        return f"{name}_prepare_{repeat_index + 1}"

    def load_and_prepare_transformer(repeat_index: int):
        transformer_load_kwargs = {
            "subfolder": "transformer",
            "torch_dtype": DTYPE,
            "low_cpu_mem_usage": MODEL_LOW_CPU_MEM_USAGE,
        }

        run_metrics[f"purge_windows_standby_before_transformer_{repeat_index + 1}"] = (
            maybe_purge_windows_standby_cache(
                DYNAMIC_OFFLOAD_SETTINGS,
                "before_transformer",
                record_event=record_event,
                event_name=transformer_prepare_event_name("purge_windows_standby_before_transformer", repeat_index),
            )
        )

        event_t0 = time.time()
        if MODEL_LOW_CPU_MEM_USAGE:
            transformer_load_kwargs["device_map"] = "cpu"
        transformer_load = from_pretrained_with_dynamic_offload(
            MODEL_PATH,
            dynamic_offload_config=dynamic_offload_config,
            apply_dynamic=False,
            **transformer_load_kwargs,
        )
        prepared_transformer = transformer_load.module
        prepared_dynamic_offload_hook = transformer_load.hook
        record_event(
            transformer_prepare_event_name("load_transformer", repeat_index),
            time.time() - event_t0,
            source=MODEL_PATH,
            low_cpu_mem_usage=MODEL_LOW_CPU_MEM_USAGE,
            device_map=transformer_load_kwargs.get("device_map"),
            loader="AutoModel",
            resolved_class=prepared_transformer.__class__.__name__,
        )

        if dynamic_offload_enabled and DYNAMIC_OFFLOAD_EXECUTION_MODE != "plan" and TRANSFORMER_MEMORY_MANAGER != "off":
            raise ValueError(
                "Dynamic offload execution currently requires "
                "DDO_RUNNER_TRANSFORMER_MEMORY_MANAGER='off'. "
                "Use execution_mode='plan' with the block manager."
            )

        event_t0 = time.time()
        if TRANSFORMER_MEMORY_MANAGER != "off":
            raise ValueError(
                "The transformer block manager is no longer used by this runner. "
                "Use DDO_PRESET or transformer group offload."
            )
        transformer_offload = enable_offload(
            prepared_transformer,
            settings=DYNAMIC_OFFLOAD_SETTINGS,
            config=dynamic_offload_config,
            component="transformer",
            execution_device=DEVICE,
            offload_device=OFFLOAD_DEVICE,
            low_cpu_mem_usage=MODEL_LOW_CPU_MEM_USAGE,
            record_event=record_event,
            dynamic_event_name=transformer_prepare_event_name("build_dynamic_offload_plan", repeat_index),
            group_event_name=transformer_prepare_event_name("setup_transformer_group_offload", repeat_index),
        )
        prepared_dynamic_offload_hook = transformer_offload.hook
        if transformer_offload.route == "dynamic_offload" and DYNAMIC_OFFLOAD_EXECUTION_MODE != "plan":
            record_event(
                transformer_prepare_event_name("skip_transformer_to_cuda", repeat_index),
                time.time() - event_t0,
                reason=f"dynamic_offload_{DYNAMIC_OFFLOAD_EXECUTION_MODE}",
            )
        return prepared_transformer, prepared_dynamic_offload_hook

    transformer = None
    dynamic_offload_hook = None
    for prepare_repeat_index in range(TRANSFORMER_PREPARE_REPEATS):
        if TRANSFORMER_PREPARE_REPEATS > 1:
            print(
                f"  Transformer prepare repeat {prepare_repeat_index + 1}/{TRANSFORMER_PREPARE_REPEATS}",
                flush=True,
            )
        transformer, dynamic_offload_hook = load_and_prepare_transformer(prepare_repeat_index)
        if prepare_repeat_index + 1 < TRANSFORMER_PREPARE_REPEATS:
            if dynamic_offload_enabled:
                remove_dynamic_offload(transformer)
            del transformer
            dynamic_offload_hook = None
            cleanup_runtime_state(record_event, transformer_prepare_event_name("cleanup_after_transformer_prepare", prepare_repeat_index))

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

    if dynamic_offload_enabled and DYNAMIC_OFFLOAD_EXECUTION_MODE != "plan":
        run_metrics["dynamic_offload_runtime_summary"] = dynamic_offload_hook.state.as_dict()
        if DYNAMIC_OFFLOAD_SHOW_PROFILE:
            dynamic_offload_hook.print_profile_summary()
    del connector_prompt_embeds, connector_attention_mask
    if dynamic_offload_enabled:
        if DYNAMIC_OFFLOAD_EXECUTION_MODE != "plan" and "dynamic_offload_runtime_summary" not in run_metrics:
            run_metrics["dynamic_offload_runtime_summary"] = dynamic_offload_hook.state.as_dict()
        remove_dynamic_offload(transformer)
        dynamic_offload_hook = None
    del prepare_pipe, denoise_pipe, transformer, scheduler
    cleanup_before_vae_decode(record_event)
    step_end(f"Pass 1: Generate at {WIDTH}x{HEIGHT}", t0)
    t0 = step_start("Pass 2: Decode VAE")

    event_t0 = time.time()
    vae = AutoencoderKLLTX2Video.from_pretrained(
        MODEL_PATH,
        subfolder="vae",
        torch_dtype=DTYPE,
        low_cpu_mem_usage=MODEL_LOW_CPU_MEM_USAGE,
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
    if RESET_DYNAMIC_MEMORY_AFTER_RUN:
        cleanup_runtime_state(record_event, "reset_dynamic_memory_after_run", collect_cuda_ipc=True)
        if SAVE_METRICS:
            metrics_path.write_text(json.dumps(run_metrics, indent=2), encoding="utf-8")
        print("  Dynamic memory state reset after run.")
    run_metrics["purge_windows_standby_after_run"] = maybe_purge_windows_standby_cache(
        DYNAMIC_OFFLOAD_SETTINGS,
        "after_run",
        record_event=record_event,
    )
    if SAVE_METRICS:
        metrics_path.write_text(json.dumps(run_metrics, indent=2), encoding="utf-8")
    print("=" * 70)


if __name__ == "__main__":
    main()
