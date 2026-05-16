from omegaconf import OmegaConf
import argparse
from easydict import EasyDict as edict
import re
import os
import datetime
import torch
import torch.distributed as dist
import numpy as np
import random
import yaml
import wandb
import shutil
import copy
from pathlib import Path
import time
from torch.utils.tensorboard import SummaryWriter
from PIL import Image

#################Init Config  Begins#################

def process_overrides(overrides):
    """
    Handle space around "="
    """
    # First, join all items with spaces to create a single string
    combined = ' '.join(overrides)
    
    # Use regex to identify and fix patterns like 'param = value' to 'param=value'
    # This handles various spacing around the equals sign
    fixed_string = re.sub(r'(\S+)\s*=\s*(\S+)', r'\1=\2', combined)
    
    # Split the fixed string back into a list, preserving properly formatted args
    # We split on spaces that are not within a parameter=value pair
    processed = re.findall(r'[^\s=]+=\S+|\S+', fixed_string)
    
    return processed

def init_config():
    parser = argparse.ArgumentParser()

    parser.add_argument("--config", "-c", required=True)
    parser.add_argument("overrides", nargs="*")  # Capture all "key=value" args
    args = parser.parse_args()

    # 1. Load base config
    config = OmegaConf.load(args.config)

    # 2. Merge with CLI overrides directly
    cli_conf = OmegaConf.from_dotlist(args.overrides)
    config.merge_with(cli_conf)
    
    config.training['num_views'] = config.training.num_input_views + config.training.num_target_views
    # print(f'[Debug] num_input_views: {config.training.num_input_views}, num_target_views: {config.training.num_target_views}, num_views: {config.training.num_views}')

    config = OmegaConf.to_container(config, resolve=True)
    config = edict(config)
    return config

#################Init Config End#################



def init_distributed(seed=42):
    """
    Initialize distributed training environment and set random seeds for reproducibility.
    
    Args:
        seed (int): Random seed for PyTorch, NumPy, and Python's random module.
                   Default is 42.
    
    Returns:
        edict: Dictionary with attribute access containing:
            - local_rank: GPU rank within the current node
            - global_rank: Global rank of the process
            - world_size: Total number of processes
            - device: The CUDA device assigned to this process
            - is_main_process: Flag to identify the main process
            - seed: The random seed used for this process
    """
    global_rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])
    
    dist.init_process_group(
        backend="nccl",
        timeout=datetime.timedelta(seconds=3600)
    )
    
    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)
    
    # Set random seeds
    # Each process gets a different seed derived from the base seed
    process_seed = seed + global_rank
    torch.manual_seed(process_seed)
    torch.cuda.manual_seed(process_seed)
    torch.cuda.manual_seed_all(process_seed) 
    np.random.seed(process_seed)
    random.seed(process_seed)
    
    # Optional: For better performance
    torch.backends.cudnn.benchmark = True
    
    return edict({
        'local_rank': local_rank,
        'global_rank': global_rank,
        'world_size': world_size,
        'device': device,
        'is_main_process': global_rank == 0, 
        'seed': process_seed
    })




def local_backup_src_code(
    src_dir,
    dst_dir,
    max_size_MB=4.0,
    extension_to_backup=(".py", ".yaml", ".sh", ".bash", ".json"),
    exclude_dirs=("wandb", ".git", "checkpoints", "experiments"),
    verbose=True,
):
    """
    Backup source code files with size limit check.
    
    Args:
        src_dir: Source directory to backup
        dst_dir: Destination directory for backups
        max_size_MB: Maximum total size allowed for backup in MB
        extension_to_backup: File extensions to include in backup
        exclude_dirs: Directories to exclude from backup
        verbose: Whether to print progress information
    
    Returns:
        tuple: (num_files_backed_up, total_size_in_bytes)
    
    Raises:
        ValueError: If total size exceeds max_size_MB
    """
    start_time = time.time()
    src_path = Path(src_dir).resolve()
    dst_path = Path(dst_dir).resolve()
    
    # Convert to set for faster lookup
    extension_set = set(extension_to_backup)
    ignore_paths = {(src_path / d).resolve() for d in exclude_dirs}
    
    max_bytes = int(max_size_MB * 1024 * 1024)
    
    if not src_path.exists():
        raise FileNotFoundError(f"Source directory does not exist: {src_path}")
    
    files = []
    total_size = 0
    
    for dirpath, dirnames, filenames in os.walk(src_path):
        current_path = Path(dirpath).resolve()
        
        # Skip excluded directories
        if any(parent in ignore_paths for parent in current_path.parents) or current_path in ignore_paths:
            dirnames.clear()
            continue
        
        # Filter files by extension
        for filename in filenames:
            file_ext = os.path.splitext(filename)[1]
            if file_ext not in extension_set:
                continue
                
            src_file = current_path / filename
            rel_path = current_path.relative_to(src_path)
            dst_file = dst_path / rel_path / filename
            
            try:
                file_size = src_file.stat().st_size
                total_size += file_size
                files.append((src_file, dst_file, file_size))
            except (FileNotFoundError, PermissionError) as e:
                if verbose:
                    print(f"Warning: Could not access {src_file}: {e}")
    
    if total_size > max_bytes:
        if verbose:
            print(f"Size limit exceeded: {total_size / (1024*1024):.2f} MB > {max_size_MB} MB")
            print("Largest files:")
            for src_file, _, size in sorted(files, key=lambda x: x[2], reverse=True)[:5]:
                print(f"{src_file}: {size / 1024:.1f} KB")
        raise ValueError(f"Size limit exceeded: {total_size / (1024*1024):.2f} MB > {max_size_MB} MB")
    
    if verbose:
        print(f"Backing up {len(files)} files ({total_size / (1024*1024):.2f} MB)")
    
    dst_path.mkdir(parents=True, exist_ok=True)
    
    # Copy files
    successful_copies = 0
    for src_file, dst_file, _ in files:
        try:
            dst_file.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src_file, dst_file)
            successful_copies += 1
        except Exception as e:
            if verbose:
                print(f"Error copying {src_file} to {dst_file}: {e}")
    
    elapsed_time = time.time() - start_time
    if verbose:
        print(f"Backup completed: {successful_copies}/{len(files)} files copied in {elapsed_time:.2f} seconds")
    
    return successful_copies, total_size
    
def init_wandb_and_backup(config):
    # API key validation
    if config.training.wandb.mode in ['online']:
        # if there is already WANDB_API_KEY in the environment, skip the check
        if "WANDB_API_KEY" in os.environ:
            print("WandB API key already set in environment.")
        else:
            assert os.path.exists(
                config.training.wandb.api_key_path
            ), f"API key file does not exist: {config.training.wandb.api_key_path}"
            api_keys = edict(yaml.safe_load(open(config.training.wandb.api_key_path, "r")))
            assert api_keys.wandb is not None, "Wandb API key not found in api key file"
        # WandB setup and login
        os.environ["WANDB_API_KEY"] = api_keys.wandb

    # WandB initialization
    wandb.init(
        # entity=config.training.wandb.entity,
        project=config.training.wandb.project,
        name=config.training.wandb.exp_name,
        config=copy.deepcopy(config.training),
        mode=config.training.wandb.mode,
    )


class Logger:
    def __init__(self, log_config, save_config=None):
        """
        log_config: The training configuration need include:
            - wandb: dict, include:
                - project: str, the project name
                - exp_name: str, the experiment name
                - mode: str, 'online' or 'offline'
                - api_key_path: str, the path to the API key file
            - checkpoint_dir: str, the directory to save checkpoints
        save_config: The configuration to be saved in WandB or TensorBoard
        """
        self.config = log_config
        self.backend = log_config.wandb.mode
        init_success = False
        
        if self.backend == 'online':
            init_success = self._init_wandb(save_config)
            if init_success:
                print("[LOG] WandB online initialized!")

        if self.backend != 'online' or not init_success:
            self._init_tensorboard()
            self.backend = 'offline'
            print(f"[LOG] TensorBoard local initialized!, check at {self.config.checkpoint_dir}/{self.config.wandb.exp_name}")

    def _init_wandb(self, save_config):
        if "WANDB_API_KEY" not in os.environ:
            api_key_file = self.config.wandb.api_key_path
            # assert os.path.exists(api_key_file), f"API key file not found: {api_key_file}"
            if not os.path.exists(api_key_file):
                print(f"API key file does not exist: {api_key_file}")
                return False
            api_keys = edict(yaml.safe_load(open(api_key_file, "r")))
            # assert api_keys.wandb, "Wandb API key missing in key file"
            if api_keys.wandb is None:
                print("Wandb API key not found in api key file")
                return False
            os.environ["WANDB_API_KEY"] = api_keys.wandb

        wandb.init(
            project=self.config.wandb.project,
            name=self.config.wandb.exp_name,
            config=edict(save_config)
        )
        self.log_fn = wandb.log
        return True

    def _init_tensorboard(self):
        log_dir = os.path.join(self.config.checkpoint_dir, self.config.wandb.exp_name)
        os.makedirs(log_dir, exist_ok=True)
        self.writer = SummaryWriter(log_dir=log_dir)

        def tb_log(metrics: dict, step: int = None):
            """
            metrics: dict of tag->value
            step: global step index
            """
            for tag, value in metrics.items():
                self.writer.add_scalar(tag, value, global_step=step)
            self.writer.flush()
        self.log_fn = tb_log

    def log(self, metrics: dict, step: int = None):
        """
        Log metrics to WandB or TensorBoard.
            metrics: dict, e.g. {'loss': 0.23, 'acc': 0.91}
            step: int, global step index
        """
        if self.backend == 'online':
            # wandb.log 接口：wandb.log({ 'loss': 0.23 }, step=step)
            self.log_fn(metrics, step=step)
        else:
            # tensorboard: metrics 和 step 都由 tb_log 接收
            self.log_fn(metrics, step)

    def log_images(self, tag: str, images: list, step: int = None):
        """
        images: 
          - for TensorBoard: Sequence[np.ndarray] in CHW or HWC format
          - for W&B: either Sequence[np.ndarray] or Sequence[str] (filepaths)
        """
        if self.backend == 'online':
            wandb_images = []
            for img in images:
                wandb_images.append(wandb.Image(img))
            wandb.log({"Iteration Image": wandb_images}, step=step)
        else:
            for idx, img_path in enumerate(images):
                with Image.open(img_path) as pil_img:
                    pil_img.load() 
                    rgb_img = pil_img.convert("RGB")
                    arr = np.array(rgb_img)  # HWC, uint8
                    
                if arr.dtype == np.uint8:
                    arr = arr.astype(np.float32) / 255.0
                if arr.ndim == 3 and arr.shape[2] in (1,3):
                    arr = arr.transpose(2, 0, 1)

                self.writer.add_image(f"{tag}/{img_path.split('/')[-1]}", arr, global_step=step)
            self.writer.flush()

    def log_model(self, file_path: str, step: int = None):
        if self.backend == 'online':
            wandb.save(file_path)

    def finish(self):
        if self.backend != 'online':
            self.writer.close()
        else:
            wandb.finish()
