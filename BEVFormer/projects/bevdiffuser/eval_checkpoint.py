import os
import torch
from mmcv import Config
from mmdet3d.datasets import build_dataset
from scheduler_utils import DDIMGuidedScheduler
from model_utils import get_bev_model, build_unet
from test_bev_diffuser import evaluate
from projects.mmdet3d_plugin.datasets.builder import build_dataloader

class Args:
    bev_config = "/home/aya/BEVDiffuser/BEVFormer/projects/configs/bevdiffuser/layout_tiny.py"
    bev_checkpoint = "/home/aya/BEVDiffuser/BEVFormer/results/option1_run/checkpoint-2462/bev_model.pth"
    cfg_options = None
    launcher = "none"

args = Args()
bev_cfg = Config.fromfile(args.bev_config)

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

val_dataloader = build_dataloader(
    val_dataset,
    samples_per_gpu=bev_cfg.data.samples_per_gpu,
    workers_per_gpu=bev_cfg.data.workers_per_gpu,
    dist=False,
    shuffle=False,
    nonshuffler_sampler=getattr(bev_cfg.data, "nonshuffler_sampler", None),
)

bev_model = get_bev_model(args)
unet = build_unet(bev_cfg.unet)
unet.from_pretrained("/home/aya/BEVDiffuser/BEVFormer/results/option1_run/checkpoint-2462/unet", subfolder=None)
unet = unet.cuda().eval()

noise_scheduler = DDIMGuidedScheduler.from_pretrained(
    "/home/aya/BEVDiffuser/BEVFormer/hf_models/stable-diffusion-2-1",
    subfolder="scheduler"
)

save_path = "/home/aya/BEVDiffuser/BEVFormer/results/option1_run/checkpoint-2462/val"
os.makedirs(save_path, exist_ok=True)

with torch.no_grad():
    eval_results = evaluate(
        unet=unet,
        bev_model=bev_model,
        noise_scheduler=noise_scheduler,
        dataset=val_dataset,
        dataloader=val_dataloader,
        bev_cfg=bev_cfg,
        eval='bbox',
        save_path=save_path,
        noise_timesteps=5,
        denoise_timesteps=5,
        num_inference_steps=5,
        use_classifier_guidence=False,
    )

print(eval_results)