from diffusers import ModularPipelineBlocks
from huggingface_hub import HfApi

local_dir = "./custom_blocks/ltx2_image"
repo_id = "elismasilva/ltx2.3_image_custom_blocks"
readme_path = f"{local_dir}/README.md"

with open(readme_path, encoding="utf-8") as f:
    readme = f.read()

blocks = ModularPipelineBlocks.from_pretrained(local_dir, trust_remote_code=True)
pipeline = blocks.init_pipeline()
pipeline.save_pretrained(local_dir, repo_id=repo_id, push_to_hub=True)

with open(readme_path, "w", encoding="utf-8", newline="\n") as f:
    f.write(readme)

HfApi().upload_file(
    path_or_fileobj=readme_path,
    path_in_repo="README.md",
    repo_id=repo_id,
)
