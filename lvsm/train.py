"""
# Created: 2025-05-22 16:40
# Author: Qingwen Zhang  (https://kin-zhang.github.io/)
#
#
# Description: Refactor the training script to:
# 1. ~use hydra for configuration management.~
# 2. easily switch between tensorboard and wandb for logging.
#
"""

from copy import deepcopy
import importlib
import os, sys
import time
import wandb
import torch
from rich import print
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
import torch.distributed as dist
from setup import init_distributed, Logger, init_config
from utils.metric_utils import visualize_intermediate_results
from utils.training_utils import create_optimizer, create_lr_scheduler, auto_resume_job, print_rank0, print_time
# import hydra
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader, Subset

def main():
    config = init_config()
    os.environ["OMP_NUM_THREADS"] = str(config.training.get("num_threads", 1))
    # Set up DDP for training/inference and Fix random seed
    ddp_info = init_distributed(seed=777)
    dist.barrier()
    # rename wandb exp_name:
    config.training.wandb.exp_name = f"{config.training.checkpoint_dir.split('/')[-1]}-{config.training.num_input_views}"
    config.training.checkpoint_dir = f"{config.training.checkpoint_dir}-{config.training.num_input_views}"
    # Set up wandb and backup source code
    if ddp_info.is_main_process:
        logger = Logger(config.training, save_config=config)
    dist.barrier()

    # Set up tf32
    torch.backends.cuda.matmul.allow_tf32 = config.training.use_tf32
    torch.backends.cudnn.allow_tf32 = config.training.use_tf32
    amp_dtype_mapping = {
        "fp16": torch.float16, 
        "bf16": torch.bfloat16, 
        "fp32": torch.float32, 
        'tf32': torch.float32
    }

    # Load dataset
    dataset_name = config.training.get("dataset_name", None)
    assert dataset_name is not None, "Dataset name must be specified."

    if isinstance(dataset_name, list) or OmegaConf.is_list(dataset_name):
        dataset = []
        for cnt, name in enumerate(dataset_name):
            module, class_name = name.rsplit(".", 1)
            Dataset = importlib.import_module(module).__dict__[class_name]
            percentage = config.training.get("dataset_percentage", 1.0)
            percentage = percentage[cnt] if isinstance(percentage, list) or OmegaConf.is_list(percentage) else percentage
            config_copy = deepcopy(config)
            config_copy.training.dataset_name = name
            config_copy.training.dataset_path = config.training.get("dataset_path", None)[cnt]
            Subset_dataset = Dataset(config_copy)
            num_samples = int(len(Subset_dataset) * percentage)
            if percentage < 1.0:
                indices = list(range(len(Subset_dataset)))
                selected_indices = indices[:num_samples]
                Subset_dataset = Subset(Subset_dataset, selected_indices)
            if ddp_info.is_main_process:
                print(f"--- [Log] Using {percentage*100:.1f}% of dataset {name}, total {num_samples} samples")
            dataset.append(Subset_dataset)

        dataset = torch.utils.data.ConcatDataset(dataset)
    else:
        module, class_name = dataset_name.rsplit(".", 1)
        Dataset = importlib.import_module(module).__dict__[class_name]
        dataset = Dataset(config)

    batch_size_per_gpu = config.training.batch_size_per_gpu
    if ddp_info.is_main_process:
        print('-'*30)
        print(f"Dataset loaded, the dataset length is: {len(dataset)}. #Input Images: {config.training.num_input_views}, #Target Images: {config.training.num_target_views}")
        if len(dataset) / batch_size_per_gpu < 1:
            print("WARNING: The dataset is too small for the batch size, please check your config file.")
            exit(0)
        print(f"Model name: {config.model.class_name}")
        print(f"Model ray_encoding: {config.model.get('ray_encoding', 'global')}, pos_enc: {config.model.transformer.get('pos_enc', 'none')}")
        print('-'*30)
    datasampler = DistributedSampler(dataset)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size_per_gpu,
        shuffle=False,
        num_workers=config.training.num_workers,
        persistent_workers=True,
        pin_memory=False,
        drop_last=True,
        prefetch_factor=config.training.prefetch_factor,
        sampler=datasampler,
    )
    if ddp_info.is_main_process:
        print("Starting training..., the dataloader length is: ", len(dataloader))
    dataloader_iter = iter(dataloader)

    total_train_steps = config.training.train_steps
    grad_accum_steps = config.training.grad_accum_steps
    total_param_update_steps = total_train_steps
    total_train_steps = total_train_steps * grad_accum_steps # real train steps when using gradient accumulation
    total_batch_size = batch_size_per_gpu * ddp_info.world_size * grad_accum_steps
    total_num_epochs = int(total_param_update_steps * total_batch_size / len(dataset))


    module, class_name = config.model.class_name.rsplit(".", 1)
    LVSM = importlib.import_module(module).__dict__[class_name]
    # num_target_views need changed accordingly to align with target_cam
    if "target_cam" in config.training and len(config.training.target_cam) > 0:
        # config.training.num_target_views = len(config.training.target_cam)
        if ddp_info.is_main_process:
            print(f"--- [Log] Using {config.training.num_target_views} target views based on target_cam (random select): {config.training.target_cam}")

    model = LVSM(config).to(ddp_info.device)
    model = DDP(model, device_ids=[ddp_info.local_rank])


    optimizer, optimized_param_dict, all_param_dict = create_optimizer(
        model,
        config.training.weight_decay,
        config.training.lr,
        (config.training.beta1, config.training.beta2),
    )
    optim_param_list = list(optimized_param_dict.values())


    scheduler_type = config.training.get("scheduler_type", "cosine")
    lr_scheduler = create_lr_scheduler(
        optimizer,
        total_param_update_steps,
        config.training.warmup,
        scheduler_type=scheduler_type,
    )


    if config.training.get("resume_ckpt", "") != "":
        ckpt_load_path = config.training.resume_ckpt
    else:
        ckpt_load_path = config.training.checkpoint_dir

    enable_grad_scaler = config.training.use_amp and (config.training.amp_dtype == "fp16" or config.training.amp_dtype == "bf16")
    scaler = torch.amp.GradScaler('cuda', enabled=enable_grad_scaler)
    print_rank0(f"Grad scaler enabled: {enable_grad_scaler}")
    reset_training_state = config.training.get("reset_training_state", False)
    optimizer, lr_scheduler, scaler, cur_train_step, cur_param_update_step = auto_resume_job(
        ckpt_load_path,
        model,
        optimizer,
        lr_scheduler,
        scaler,
        reset_training_state,
    )
    dist.barrier()

    start_train_step = cur_train_step
    model.train()

    # Set up curriculum-related variables
    num_fwdbwd_passes_per_epoch = max(1, int(len(dataset) / batch_size_per_gpu))
    if config.training.view_selector.get('use_curriculum', False):
        max_iter_epoch = config.training.get('max_iter_epoch', 100)  # use a small number for iter per epoch, more flexible for curriculum
        num_fwdbwd_passes_per_epoch = min(num_fwdbwd_passes_per_epoch, max_iter_epoch)
    else:
        num_fwdbwd_passes_per_epoch = num_fwdbwd_passes_per_epoch


    # Track when training started so we can estimate remaining time
    train_start_time = time.time()

    while cur_train_step <= total_train_steps:
        tic = time.time()
        cur_epoch = int(cur_train_step * (total_batch_size / grad_accum_steps) // num_fwdbwd_passes_per_epoch )
        dataset.current_iteration = cur_train_step
        try:
            data = next(dataloader_iter)
        except StopIteration:
            # print(f"Current Rank {ddp_info.local_rank} Ran out of data. Resetting dataloader epoch to {cur_epoch}; might take a while...")
            datasampler.set_epoch(cur_epoch)
            dataloader_iter = iter(dataloader)
            dataset.current_iteration = cur_train_step
            data = next(dataloader_iter)

        batch = {k: v.to(ddp_info.device) if type(v) == torch.Tensor else v for k, v in data.items()}


        with torch.autocast(
            enabled=config.training.use_amp,
            device_type="cuda",
            dtype=amp_dtype_mapping[config.training.amp_dtype],
        ):
            ret_dict = model(batch)

        update_grads = (cur_train_step + 1) % grad_accum_steps == 0 or cur_train_step == total_train_steps
        if update_grads:
            with model.no_sync(): # no sync grads for efficiency
                scaler.scale(ret_dict.loss_metrics.loss / grad_accum_steps).backward()
        else:
            scaler.scale(ret_dict.loss_metrics.loss / grad_accum_steps).backward()
        cur_train_step += 1

        export_inter_results = ((cur_train_step-1) == start_train_step) or (cur_train_step % config.training.checkpoint_every == 0)

        skip_optimizer_step = False
        # Skip optimizer step if loss is NaN or Inf
        if torch.isnan(ret_dict.loss_metrics.loss) or torch.isinf(ret_dict.loss_metrics.loss):
            print(f"NaN or Inf loss detected, skip this iteration")
            skip_optimizer_step = True
            ret_dict.loss_metrics.loss.data = torch.zeros_like(ret_dict.loss_metrics.loss)

        total_grad_norm = None
        # Check gradient norm and update optimizer if everything is fine
        if update_grads and (not skip_optimizer_step):
            # Unscales the gradients
            scaler.unscale_(optimizer) 
            # For all gradients, we safely change the NaN -> 0., inf -> 1e-6, -inf -> 1e-6.
            with torch.no_grad():
                for n, p in optimized_param_dict.items():
                    if p.requires_grad and (p.grad is not None):
                        p.grad.nan_to_num_(nan=0.0, posinf=1e-6, neginf=-1e-6)
        
            # visualize the grad norm of each layer of our transformer (FOR DEBUG)
            if ddp_info.is_main_process and config.training.get("log_grad_norm_details", False):
                grad_norms = {}  # Dictionary to store norms per layer
                for name, param in model.named_parameters():
                    if param.grad is not None:  # Some parameters might not have gradients
                        grad_norms[name] = param.grad.detach().norm().item()  # Detach for safety
                for layer_name, grad_norm in grad_norms.items():
                    logger.log({"grad_norm_details/" + layer_name: grad_norm}, step=cur_train_step)

            total_grad_norm = 0.0
            if config.training.grad_clip_norm > 0:
                total_grad_norm = torch.nn.utils.clip_grad_norm_(optim_param_list, max_norm=config.training.grad_clip_norm).item()

                # if total_grad_norm > config.training.grad_clip_norm * 2.0:
                #     print(f"WARNING: step {cur_train_step} grad norm too large {total_grad_norm} > {config.training.grad_clip_norm * 2.0}")

                allowed_gradnorm = config.training.grad_clip_norm * config.training.get("allowed_gradnorm_factor", 5)
                if total_grad_norm > allowed_gradnorm:
                    skip_optimizer_step = True
                    print(f"WARNING: step {cur_train_step} grad norm too large {total_grad_norm} > {allowed_gradnorm}, skipping optimizer step")

                # show grad norm in wandb if it's too large
                display_grad_norm = total_grad_norm > config.training.grad_clip_norm * 2.0 or total_grad_norm > allowed_gradnorm
                if display_grad_norm and ddp_info.is_main_process:
                    logger.log({"grad_norm": total_grad_norm}, step=cur_train_step)

            if not skip_optimizer_step:
                scaler.step(optimizer)
                cur_param_update_step += 1

            lr_scheduler.step()
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

    # Estimate remaining training time
        steps_done = (cur_train_step - start_train_step)
        time_passed = time.time() - train_start_time
        est_time_left = (time_passed / steps_done) * (total_train_steps - cur_train_step) if steps_done > 0 else 0

        # log and save checkpoint
        if ddp_info.is_main_process:
            loss_dict = {k: float(f"{v.item():.6f}") for k, v in ret_dict.loss_metrics.items()}
            # print in console
            if (cur_train_step % config.training.print_every == 0) or (cur_train_step < 100 + start_train_step):
                print_str = f"[Epoch {int(cur_epoch):>2d}] | Forwad step: {int(cur_train_step):>4d} (Param update step: {int(cur_param_update_step):>4d}) | LR: {optimizer.param_groups[0]['lr']:.6f}"
                print_str += f" | Iter Time: {print_time(time.time() - tic)} | Elapsed: {print_time(time_passed)} | ETA: {print_time(est_time_left)}\n"
                # Add loss values
                for k, v in loss_dict.items():
                    if k in ['lpips_loss', 'norm_lpips_loss']:
                        continue # skip lpips loss
                    print_str += f"{k}: {v:.6f} | "
                print(print_str)

            # log in wandb
            if (cur_train_step % config.training.wandb.log_every == 0) or (
                cur_train_step < 200 + start_train_step
            ):
                log_dict = {
                    "iter": cur_train_step, 
                    "forward_pass_step": cur_train_step,
                    "param_update_step": cur_param_update_step,
                    "lr": optimizer.param_groups[0]["lr"],
                    "iter_time (s)": time.time() - tic,
                    "grad_norm": total_grad_norm,
                    "epoch": cur_epoch,
                }
                for k, v in loss_dict.items():
                    if k in ['lpips_loss', 'norm_lpips_loss']:
                        continue # skip lpips loss
                    log_dict.update({"train/" + k: v})
                logger.log(
                    log_dict,
                    step=cur_train_step,
                )

            # save checkpoint
            if (cur_train_step % config.training.checkpoint_every == 0) or (cur_train_step == total_train_steps):
                if isinstance(model, DDP):
                    model_weights = model.module.state_dict()
                else:
                    model_weights = model.state_dict()
                checkpoint = {
                    "model": model_weights,
                    "optimizer": optimizer.state_dict(),
                    "lr_scheduler": lr_scheduler.state_dict(),
                    "fwdbwd_pass_step": cur_train_step,
                    "param_update_step": cur_param_update_step,
                    "scaler": scaler.state_dict(),
                }
                os.makedirs(config.training.checkpoint_dir, exist_ok=True)
                ckpt_path = os.path.join(config.training.checkpoint_dir, f"ckpt_{cur_train_step:016}.pt")
                torch.save(checkpoint, ckpt_path)
                # upload to wandb if online logger
                if cur_train_step == total_train_steps and logger.backend == 'online':
                    logger.log_model(ckpt_path, step=cur_train_step)
                    
                print(f"Saved checkpoint at step {cur_train_step} to {os.path.abspath(ckpt_path)}")
            
            # export intermediate visualization results
            if export_inter_results:
                vis_path = os.path.join(config.training.checkpoint_dir, f"iter_{cur_train_step:08d}")
                os.makedirs(vis_path, exist_ok=True)
                input_uid_based_filename, uid_based_filename = visualize_intermediate_results(vis_path, ret_dict)

                paths = [
                    os.path.join(vis_path, f"supervision_{uid_based_filename}.jpg"),
                    os.path.join(vis_path, f"input_{input_uid_based_filename}.jpg"),
                ]
                logger.log_images(f"iter_{cur_train_step:08d}", paths, step=cur_train_step)
                # logger.log_artifact(f"iter_{cur_train_step:08d}", vis_path, step=cur_train_step)

                torch.cuda.empty_cache()
                model.train()

                
        if export_inter_results:
            torch.cuda.empty_cache()
            dist.barrier()
    
    if ddp_info.is_main_process:
        logger.finish()
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
