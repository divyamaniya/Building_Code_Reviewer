import os
from pathlib import Path
from huggingface_hub import snapshot_download
from huggingface_hub.utils import LocalEntryNotFoundError


# convert this into utils
# create function to load emebdding model, llm, vectordb

def download_hf_model(local_dir: Path, token: str):
    repo_id = '/'.join(Path(local_dir).parts[-2:])
    
    try:
        path = snapshot_download(
            repo_id = repo_id,
            local_dir = local_dir,
            local_files_only=True,          # the function will look for the model in your local directory. If it finds it, it returns the path immediately. If it's missing, it raises an error.
            token = token
        )
        print(f"Model found locally: {path}")
    except (LocalEntryNotFoundError, Exception):
        print(f"Model not found. Starting download for {repo_id}...")
        path = snapshot_download(
            repo_id= repo_id,
            local_dir=local_dir,
            local_files_only=False,
            token= token
        )

    return path