import os
from pathlib import Path
from omegaconf import OmegaConf, DictConfig


def get_hydra_fallback(args):
    """Fallback handler for hydra interpolations when running outside @hydra.main"""
    key = args[0] if args else ""
    if "runtime.cwd" in key or "cwd" in key:
        return os.getcwd()
    elif "runtime.output_dir" in key or "output_dir" in key:
        return os.getcwd()
    return ""

def load_config(config_path: str = "config/config.yaml") -> DictConfig:
    """
    Loads configuration using OmegaConf from the specified path.
    """
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found at: {path.absolute()}")

    # Register a custom resolver for ${hydra:...} if not already present
    if not OmegaConf.has_resolver("hydra"):
        OmegaConf.register_new_resolver("hydra", lambda *args, **kwargs: get_hydra_fallback(args))
    
    config = OmegaConf.load(path)
    prompts_path = path.parent / "prompts" / "templates.yaml"       # Manually loading prompt templates, because could not find P-003 and resolve prompt .yaml into config.yaml
    if prompts_path.exists():
        prompts_cfg = OmegaConf.load(prompts_path)
        # Merge prompts into the config structure under prompt_registry
        config = OmegaConf.merge(config, {"prompt_registry": prompts_cfg})
    return config