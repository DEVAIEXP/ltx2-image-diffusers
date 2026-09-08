---
library_name: diffusers
pipeline_tag: text-to-image
base_model: elismasilva/ltx2.3-image-base
tags:
- text-to-image
- image-to-image
- modular-diffusers
- diffusion
- ltx
- ltx2-image
license: apache-2.0
---

# LTX 2 Image custom modular blocks

Custom [Modular Diffusers](https://huggingface.co/docs/diffusers/main/en/modular_diffusers/overview) blocks that
extend LTX 2 Image with **image-to-image**, plus a unified **`AutoBlocks`** that folds text-to-image and img2img into a
single pipeline. The workflow is chosen automatically from which inputs you pass.

```arduino
prompt             -> text2image
prompt + image     -> image2image        (optional strength)
```

## Loading & running

```python
import torch
from diffusers import ModularPipeline

pipe = ModularPipeline.from_pretrained(
    "elismasilva/ltx2.3_image_custom_blocks",
    trust_remote_code=True,
)
pipe.load_components(
    names=["text_encoder", "tokenizer", "connectors", "transformer", "vae", "scheduler"],
    pretrained_model_name_or_path="elismasilva/ltx2.3-image-base",
    torch_dtype=torch.bfloat16,
)
pipe.to("cuda")

image = pipe(
    prompt="a cinematic close-up portrait with crisp realistic detail",
    height=704,
    width=1280,
    num_inference_steps=8,
    output="images",
)[0]
image.save("ltx2_image_t2i.png")
```

### image-to-image

Pass an `image` and optional `strength` to the same pipe:

```python
import torch
from diffusers import ModularPipeline
from diffusers.utils import load_image

pipe = ModularPipeline.from_pretrained(
    "elismasilva/ltx2.3_image_custom_blocks",
    trust_remote_code=True,
)
pipe.load_components(
    names=["text_encoder", "tokenizer", "connectors", "transformer", "vae", "scheduler"],
    pretrained_model_name_or_path="elismasilva/ltx2.3-image-base",
    torch_dtype=torch.bfloat16,
)
pipe.to("cuda")

init_image = load_image("input.png")

result = pipe(
    prompt="preserve the scene while restoring crisp natural detail",
    image=init_image,
    height=704,
    width=1280,
    num_inference_steps=8,
    strength=0.20,
    output="images",
)[0]
result.save("ltx2_image_i2i.png")
```

## Blocks

The unified pipeline is composed of six modular blocks:

1. **`LTX2ImageTextEncoderStep`** encodes prompts into packed per-layer Gemma hidden states.
2. **`LTX2ImageConnectorStep`** adapts those hidden states with the image text connectors.
3. **`LTX2ImageVaeEncoderStep`** optionally encodes an input image for img2img.
4. **`LTX2ImagePrepareLatentsStep`** prepares one-frame image latents and FlowMatch timesteps.
5. **`LTX2ImageDenoiseStep`** runs the LTX 2 Image denoising loop.
6. **`LTX2ImageDecodeStep`** decodes latents into images or returns latent output.

## Components

The blocks expect the following components from `elismasilva/ltx2.3-image-base`:

- `text_encoder`
- `tokenizer`
- `connectors`
- `transformer`
- `vae`
- `scheduler`

## Notes

The pipeline supports prompt embeds, custom `sigmas` or `timesteps`, `guidance_scale`, `guidance_rescale`, PAG inputs
(`pag_scale` and `pag_applied_layers`), and VAE decode controls (`decode_timestep` and `decode_noise_scale`). For img2img,
`input_noise_sigma`, `input_sharpen`, `phase_cutoff`, `phase_transition_width`, and `phase_pad_factor` can be used when
you need more control over how the input image is converted into latents.
