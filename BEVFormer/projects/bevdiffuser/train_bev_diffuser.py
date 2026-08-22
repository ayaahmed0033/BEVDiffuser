# Copyright (c) 2025 Robert Bosch GmbH
# SPDX-License-Identifier: AGPL-3.0

'''
Following code is adapted from
https://github.com/huggingface/diffusers/blob/main/examples/text_to_image/train_text_to_image.py
'''

import argparse
import logging
import math
import os
import sys
import shutil
import warnings

import wandb
import accelerate
import datasets
import numpy as np
import torch

torch.backends.cudnn.enabled = False

import torch.nn.functional as F
import transformers
import diffusers

from tqdm.auto import tqdm
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration
from packaging import version
from diffusers import DDPMScheduler
from diffusers.optimization import get_scheduler

from mmcv import Config, DictAction
from mmcv.runner import get_dist_info
from mmdet3d.datasets import build_dataset
from mmdet.apis import set_random_seed

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + "/..")
from projects.mmdet3d_plugin.datasets.builder import build_dataloader

from layout_diffusion.layout_diffusion_unet import LayoutDiffusionUNetModel
from scheduler_utils import DDIMGuidedScheduler
from model_utils import get_bev_model, build_unet
from test_bev_diffuser import evaluate
from projects.bevdiffuser.noise_construction import NoiseConstructionModule



logger = get_logger(__name__, log_level="INFO")


def train():
    args = parse_args()

    bev_cfg = Config.fromfile(args.bev_config)
    if args.cfg_options is not None:
        bev_cfg.merge_from_dict(args.cfg_options)

    logging_dir = os.path.join(args.output_dir, args.logging_dir)
    accelerator_project_config = ProjectConfiguration(
        project_dir=args.output_dir,
        logging_dir=logging_dir
    )

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=None,
        log_with=args.report_to,
        project_config=accelerator_project_config,
    )

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)

    if args.resume_from_checkpoint:
        resume_ckpt_number = args.resume_from_checkpoint.split("-")[-1]
        args.output_dir = f"{args.output_dir}-resume-{resume_ckpt_number}"
        logger.info(f"change output dir to {args.output_dir}")

    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    if args.seed is not None:
        set_random_seed(args.seed, deterministic=args.deterministic)

    if accelerator.is_main_process and args.output_dir is not None:
        os.makedirs(args.output_dir, exist_ok=True)

    noise_scheduler = DDPMScheduler.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="scheduler"
    )

    DDIM_scheduler = DDIMGuidedScheduler.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="scheduler"
    )

    if args.prediction_type is not None:
        noise_scheduler.register_to_config(prediction_type=args.prediction_type)
        DDIM_scheduler.register_to_config(prediction_type=args.prediction_type)

    bev_model = get_bev_model(args)
    ncm = None
    if args.use_ncm:
        ncm = NoiseConstructionModule(
            num_train_timesteps=noise_scheduler.config.num_train_timesteps,
            translation_range=args.ncm_translation_range,
            scale_range=(args.ncm_scale_min, args.ncm_scale_max),
            rotation_range=args.ncm_rotation_range,
            p_identity=args.ncm_p_identity,
            p_translation=args.ncm_p_translation,
            p_scaling=args.ncm_p_scaling,
            p_rotation=args.ncm_p_rotation,
            noise_dist=args.ncm_noise_dist,
        )

    bev_model.requires_grad_(False)

    if args.task_loss_scale != 0:
        bev_model.module.pts_bbox_head.transformer.decoder.requires_grad_(True)
        bev_model.module.pts_bbox_head.transformer.reference_points.requires_grad_(True)
        bev_model.module.pts_bbox_head.cls_branches.requires_grad_(True)
        bev_model.module.pts_bbox_head.reg_branches.requires_grad_(True)

    def get_task_loss(x, **kwargs):
        x = x.permute(0, 2, 3, 1).contiguous()
        x = x.reshape(-1, bev_cfg.bev_h_ * bev_cfg.bev_w_, bev_cfg._dim_)
        losses = bev_model(return_loss=True, given_bev=x, **kwargs)
        loss, _ = bev_model.module._parse_losses(losses)
        return loss

    unet = build_unet(bev_cfg.unet)

    if args.pretrained_unet_checkpoint is not None and (
        os.path.isfile(args.pretrained_unet_checkpoint)
        or os.path.isdir(args.pretrained_unet_checkpoint)
    ):
        unet.from_pretrained(args.pretrained_unet_checkpoint, subfolder="unet")

        # original-style behavior:
        # train only downsample and upsample layers
        unet.requires_grad_(False)
        unet.downsample_blocks.requires_grad_(True)
        unet.upsample_blocks.requires_grad_(True)

    assert version.parse(accelerate.__version__) >= version.parse(
        "0.16.0"
    ), "accelerate 0.16.0 or above is required"

    def save_model_hook(models, weights, output_dir):
        for i, model in enumerate(models):
            model.save_pretrained(os.path.join(output_dir, "unet"))

            # make sure to pop weight so that corresponding model is not saved again
            weights.pop()

    def load_model_hook(models, input_dir):
        for i in range(len(models)):
            # pop models so that they are not loaded again
            model = models.pop()
            model.from_pretrained(os.path.join(input_dir, "unet"))

    accelerator.register_save_state_pre_hook(save_model_hook)
    accelerator.register_load_state_pre_hook(load_model_hook)

    if args.gradient_checkpointing:
        unet.enable_gradient_checkpointing()

    optimizer_cls = torch.optim.AdamW

    trained_params = list(unet.parameters())
    if args.task_loss_scale != 0:
        trained_params += list(bev_model.parameters())

    optimizer = optimizer_cls(
        trained_params,
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    with accelerator.main_process_first():
        train_dataset = build_dataset(
            bev_cfg.data.train,
            default_args={
                'pc_range': bev_cfg.point_cloud_range,
                'use_3d_bbox': bev_cfg.use_3d_bbox,
                'num_classes': bev_cfg.num_classes,
                'num_bboxes': bev_cfg.num_bboxes,
            }
        )

        bev_cfg.data.test.load_annos = True
        val_dataset = build_dataset(
            bev_cfg.data.test,
            default_args={
                'pc_range': bev_cfg.point_cloud_range,
                'use_3d_bbox': bev_cfg.use_3d_bbox,
                'num_classes': bev_cfg.num_classes,
                'num_bboxes': bev_cfg.num_bboxes,
            }
        )
        logger.info(f"train dataset: {len(train_dataset)} samples, val dataset: {len(val_dataset)} samples")

    # NOTE: kept relative to the upstream original (uses getattr + num_gpus=max(...,1))
    # because it is the change that made the dataloader build succeed on this
    # mmdetection3d/mmcv installation. Everything else in this file is reverted
    # to match the original train_bev_diffuser.py.
    train_dataloader = build_dataloader(
        train_dataset,
        samples_per_gpu=args.train_batch_size,
        workers_per_gpu=args.dataloader_num_workers,
        num_gpus=max(get_dist_info()[1], 1),
        dist=(args.launcher != 'none'),
        seed=args.seed,
        shuffler_sampler=getattr(bev_cfg.data, "shuffler_sampler", None),
        nonshuffler_sampler=getattr(bev_cfg.data, "nonshuffler_sampler", None),
    )

    val_dataloader = build_dataloader(
        val_dataset,
        samples_per_gpu=bev_cfg.data.samples_per_gpu,
        workers_per_gpu=bev_cfg.data.workers_per_gpu,
        dist=(args.launcher != 'none'),
        shuffle=False,
        nonshuffler_sampler=getattr(bev_cfg.data, "nonshuffler_sampler", None),
    )

    def get_condition(batch):
        cond = {}

        if 'layout_obj_classes' in batch:
            cond['obj_class'] = torch.stack(batch['layout_obj_classes'].data[0])

        if 'layout_obj_bboxes' in batch:
            cond['obj_bbox'] = torch.stack(batch['layout_obj_bboxes'].data[0])

        if 'layout_obj_is_valid' in batch:
            cond['is_valid_obj'] = torch.stack(batch['layout_obj_is_valid'].data[0])

        if 'layout_obj_names' in batch:
            cond['obj_name'] = torch.stack(batch['layout_obj_names'].data[0])

        if np.random.rand() < args.uncond_prob:
            if isinstance(unet.module, LayoutDiffusionUNetModel):

                if 'obj_class' in unet.module.layout_encoder.used_condition_types:
                    cond['obj_class'] = torch.ones_like(cond['obj_class']).fill_(
                        unet.module.layout_encoder.num_classes_for_layout_object - 1
                    )
                    cond['obj_class'][:, 0] = (
                        unet.module.layout_encoder.num_classes_for_layout_object - 2
                    )

                if 'obj_name' in unet.module.layout_encoder.used_condition_types:
                    cond['obj_name'] = torch.stack(batch['default_obj_names'].data[0])

                if 'obj_bbox' in unet.module.layout_encoder.used_condition_types:
                    cond['obj_bbox'] = torch.zeros_like(cond['obj_bbox'])

                    if unet.module.layout_encoder.use_3d_bbox:
                        cond['obj_bbox'][:, 0] = torch.FloatTensor(
                            [0, 0, 0, 1, 1, 1, 0, 0, 0]
                        )
                    else:
                        cond['obj_bbox'][:, 0] = torch.FloatTensor(
                            [0, 0, 1, 1]
                        )

                cond['is_valid_obj'] = torch.zeros_like(cond['is_valid_obj'])
                cond['is_valid_obj'][:, 0] = 1.0

        return cond

    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / args.gradient_accumulation_steps
    )

    args.num_train_epochs = math.ceil(
        args.max_train_steps / num_update_steps_per_epoch
    )

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=args.max_train_steps * accelerator.num_processes,
    )

    unet, optimizer, lr_scheduler = accelerator.prepare(
        unet,
        optimizer,
        lr_scheduler
    )

    weight_dtype = torch.float32
    bev_model.to(accelerator.device, dtype=weight_dtype)

    if accelerator.is_main_process:
        tracker_config = dict(vars(args))

        if args.resume_from_checkpoint:
            resume_ckpt_number = args.resume_from_checkpoint.split("-")[-1]
            args.tracker_run_name = f"{args.tracker_run_name}-resume-{resume_ckpt_number}"

        init_kwargs = {}

        if args.report_to == "wandb":
            wandb.init(
                project=args.tracker_project_name,
                name=args.tracker_run_name,
                id=args.tracker_run_name
            )
            init_kwargs = {
                "wandb": {
                    "name": args.tracker_run_name
                }
            }

        accelerator.init_trackers(
            project_name=args.tracker_project_name,
            config=tracker_config,
            init_kwargs=init_kwargs
        )

    total_batch_size = (
        args.train_batch_size
        * accelerator.num_processes
        * args.gradient_accumulation_steps
    )

    is_training_sd21 = (
        args.pretrained_model_name_or_path == "stabilityai/stable-diffusion-2-1"
    )

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Num update steps per epoch = {num_update_steps_per_epoch}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size = {total_batch_size}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    logger.info(f"  Is SD21: {is_training_sd21}")

    global_step = 0
    first_epoch = 0
    step_cnt = 0
    resume_step = 0

    if args.resume_from_checkpoint:
        resume_path = args.resume_from_checkpoint

        # NOTE: original upstream code calls `accelerator.logger.info(...)` here,
        # but `Accelerator` has no `.logger` attribute in current `accelerate`
        # versions -> AttributeError, crashes every resume. Fixed by calling the
        # module-level `logger` (the same one used everywhere else in this file)
        # instead. This is the one other change kept from the original.
        logger.info(f"Resuming from checkpoint {resume_path}")
        accelerator.load_state(resume_path)
        global_step = int(resume_path.split("-")[-1])

        resume_global_step = global_step * args.gradient_accumulation_steps
        first_epoch = global_step // num_update_steps_per_epoch
        resume_step = resume_global_step % (
            num_update_steps_per_epoch * args.gradient_accumulation_steps
        )

        step_cnt = global_step * args.gradient_accumulation_steps

    progress_bar = tqdm(
        range(global_step, args.max_train_steps),
        disable=not accelerator.is_local_main_process
    )
    progress_bar.set_description("Steps")

    for epoch in range(first_epoch, args.num_train_epochs):
        unet.train()
        train_loss = 0.0

        for step, batch in enumerate(train_dataloader):

            if args.resume_from_checkpoint and epoch == first_epoch and step < resume_step:
                if step % 10 == 0:
                    logger.info(f"skipping data {step} / {resume_step}")
                continue

            with accelerator.accumulate(unet):

                with torch.no_grad():
                    latents = bev_model(
                        return_loss=False,
                        only_bev=True,
                        **batch
                    ).detach()

                latents = latents.reshape(
                    -1,
                    bev_cfg.bev_h_,
                    bev_cfg.bev_w_,
                    bev_cfg._dim_
                )

                latents = latents.permute(0, 3, 1, 2).contiguous()

                if ncm is not None:
                    noise = ncm.sample_noise(latents)
                else:
                    noise = torch.randn_like(latents)

                bsz = latents.shape[0]
                max_timestep = noise_scheduler.config.num_train_timesteps
                timesteps = torch.randint(0, max_timestep, (bsz,), device=latents.device)
                timesteps = timesteps.long()

                ncm_params = None
                if ncm is not None:
                    noisy_latents, target, ncm_params = ncm.construct(
                        latents, noise, timesteps, noise_scheduler,
                        noise_scheduler.config.prediction_type)
                else:
                    noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
                    if noise_scheduler.config.prediction_type == "epsilon":
                        target = noise
                    elif noise_scheduler.config.prediction_type == "sample":
                        target = latents
                    elif noise_scheduler.config.prediction_type == "v_prediction":
                        target = noise_scheduler.get_velocity(latents, noise, timesteps)
                    else:
                        raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")

                cond = get_condition(batch)

                model_pred = unet(
                    noisy_latents,
                    timesteps,
                    **cond
                )[0]
                denoise_loss = F.mse_loss(
                        model_pred.float(),
                        target.float(),
                        reduction="mean"
                    )

               

                if args.task_loss_scale > 0 and noise_scheduler.config.prediction_type == "sample":
                    task_pred = ncm.invert(model_pred, ncm_params) if ncm_params is not None else model_pred
                    task_loss = get_task_loss(task_pred, **batch)
                else:
                    task_loss = 0

                total_loss = denoise_loss + args.task_loss_scale * task_loss

                lr = lr_scheduler.get_last_lr()[0]

                step_cnt += 1

                if accelerator.is_main_process and args.report_to == "wandb":
                    wandb.log(
                        {
                            "step/step_cnt": step_cnt,
                            "step/epoch": epoch,
                            "lr/learning_rate": lr,
                            "train/denoise_loss": denoise_loss,
                            "train/task_loss": task_loss,
                            "train/total_loss": total_loss,
                        },
                        step=step_cnt
                    )

                loss = total_loss

                avg_loss = accelerator.gather(
                    loss.repeat(args.train_batch_size)
                ).mean()

                train_loss += avg_loss.item() / args.gradient_accumulation_steps

                accelerator.backward(loss)

                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(
                        unet.parameters(),
                        args.max_grad_norm
                    )

                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1

                accelerator.log(
                    {"train_loss": train_loss},
                    step=global_step
                )

                train_loss = 0.0

                if global_step % args.checkpointing_steps == 0:
                    save_path = os.path.join(
                        args.output_dir,
                        f"checkpoint-{global_step}"
                    )

                    if accelerator.is_main_process:

                        if args.checkpoints_total_limit is not None:
                            checkpoints = os.listdir(args.output_dir)
                            checkpoints = [
                                d for d in checkpoints
                                if d.startswith("checkpoint")
                            ]
                            checkpoints = sorted(
                                checkpoints,
                                key=lambda x: int(x.split("-")[1])
                            )

                            if len(checkpoints) >= args.checkpoints_total_limit:
                                num_to_remove = (
                                    len(checkpoints)
                                    - args.checkpoints_total_limit
                                    + 1
                                )

                                removing_checkpoints = checkpoints[0:num_to_remove]

                                logger.info(
                                    f"{len(checkpoints)} checkpoints already exist, "
                                    f"removing {len(removing_checkpoints)} checkpoints"
                                )

                                logger.info(
                                    f"removing checkpoints: {', '.join(removing_checkpoints)}"
                                )

                                for removing_checkpoint in removing_checkpoints:
                                    removing_checkpoint = os.path.join(
                                        args.output_dir,
                                        removing_checkpoint
                                    )
                                    shutil.rmtree(removing_checkpoint)

                        accelerator.save_state(save_path)

                       

                        logger.info(f"Saved state to {save_path}")

                    unet.eval()

                    with torch.no_grad():
                        eval_path = os.path.join(save_path, 'val')

                        eval_results = evaluate(
                            unet=unet.module,
                            bev_model=bev_model,
                            noise_scheduler=DDIM_scheduler,
                            dataset=val_dataset,
                            dataloader=val_dataloader,
                            bev_cfg=bev_cfg,
                            eval='bbox',
                            save_path=eval_path,
                            noise_timesteps=5,
                            denoise_timesteps=5,
                            num_inference_steps=5,
                            use_classifier_guidence=False
                        )

                    # if accelerator.is_main_process and args.report_to == "wandb":
                    #     for metric, score in eval_results.items():
                    #         wandb.log(
                    #             {f"val/{metric}": score},
                    #             step=step_cnt
                    #         )

                    unet.train()

            progress_bar.set_postfix(
                step_loss=loss.detach().item(),
                lr=lr_scheduler.get_last_lr()[0],
                epoch=epoch
            )

            if global_step >= args.max_train_steps:
                break

    accelerator.wait_for_everyone()
    accelerator.end_training()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Training script for BEVDiffuser"
    )

    parser.add_argument(
        '--bev_config',
        default="/home/aya/BEVDiffuser/BEVFormer/projects/configs/bevdiffuser/layout_tiny.py",
        help='config file path'
    )

    parser.add_argument(
        '--bev_checkpoint',
        default=None,
        help='checkpoint file'
    )

    parser.add_argument(
        '--pretrained_unet_checkpoint',
        default=None,
        help='checkpoint file'
    )

    group_gpus = parser.add_mutually_exclusive_group()

    group_gpus.add_argument(
        '--gpus',
        type=int
    )

    group_gpus.add_argument(
        '--gpu-ids',
        type=int,
        nargs='+'
    )

    parser.add_argument(
        '--seed',
        type=int,
        default=0
    )

    parser.add_argument(
        '--deterministic',
        action='store_true'
    )

    parser.add_argument(
        '--options',
        nargs='+',
        action=DictAction
    )

    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction
    )

    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='pytorch',
        help='job launcher'
    )

    parser.add_argument(
        '--local_rank',
        type=int,
        default=0
    )

    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default="/home/aya/BEVDiffuser/BEVFormer/hf_models/stable-diffusion-2-1",
    )

    parser.add_argument(
        "--uncond_prob",
        default=0.2,
        type=float
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default='results/test'
    )

    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None
    )

    parser.add_argument(
        "--train_batch_size",
        type=int,
        default=1
    )

    parser.add_argument(
        "--num_train_epochs",
        type=int,
        default=3
    )

    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=500,
        help="Total number of training steps to perform. If provided, overrides num_train_epochs.",
    )

    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1
    )

    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true"
    )

    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4
    )

    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant"
    )

    parser.add_argument(
        "--lr_warmup_steps",
        type=int,
        default=500
    )

    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=2
    )

    parser.add_argument(
        "--adam_beta1",
        type=float,
        default=0.9
    )

    parser.add_argument(
        "--adam_beta2",
        type=float,
        default=0.999
    )

    parser.add_argument(
        "--adam_weight_decay",
        type=float,
        default=1e-2
    )

    parser.add_argument(
        "--adam_epsilon",
        type=float,
        default=1e-08
    )

    parser.add_argument(
        "--max_grad_norm",
        default=1.0,
        type=float
    )

    parser.add_argument(
        "--prediction_type",
        type=str,
        default=None
    )

    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs"
    )

    parser.add_argument(
        "--report_to",
        type=str,
        default=None,
        choices=[None, "wandb"]
    )

    parser.add_argument(
        "--checkpointing_steps",
        type=int,
        default=250
    )

    parser.add_argument(
        "--checkpoints_total_limit",
        type=int,
        default=10
    )

    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None
    )

    parser.add_argument(
        "--tracker_project_name",
        type=str,
        default=None
    )

    parser.add_argument(
        "--tracker_run_name",
        type=str,
        default=None
    )

    parser.add_argument(
        "--task_loss_scale",
        type=float,
        default=0.0
    )
    parser.add_argument("--use_ncm", action="store_true")
    parser.add_argument("--ncm_translation_range", type=float, default=2.0)
    parser.add_argument("--ncm_scale_min", type=float, default=0.5)
    parser.add_argument("--ncm_scale_max", type=float, default=1.5)
    parser.add_argument("--ncm_rotation_range", type=float, default=1.5708)
    parser.add_argument("--ncm_p_identity", type=float, default=0.25)
    parser.add_argument("--ncm_p_translation", type=float, default=0.25)
    parser.add_argument("--ncm_p_scaling", type=float, default=0.25)
    parser.add_argument("--ncm_p_rotation", type=float, default=0.25)
    parser.add_argument("--ncm_noise_dist", type=str, default="gaussian",
                        choices=["gaussian", "laplace", "uniform"])

    

    

    args = parser.parse_args()

    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    if args.options and args.cfg_options:
        raise ValueError(
            '--options and --cfg-options cannot both be specified'
        )

    if args.options:
        warnings.warn(
            '--options is deprecated in favor of --cfg-options'
        )
        args.cfg_options = args.options

    return args


if __name__ == "__main__":
    train()