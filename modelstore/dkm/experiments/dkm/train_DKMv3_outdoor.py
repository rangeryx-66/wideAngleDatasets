import os
import torch
from argparse import ArgumentParser

from torch import nn
from torch.utils.data import ConcatDataset

import json
import wandb

from dkm.benchmarks import Megadepth1500Benchmark
from dkm.datasets.megadepth import MegadepthBuilder
from dkm.losses import DepthRegressionLoss
from dkm.train.train import train_k_steps
from dkm.checkpointing.checkpoint import CheckPoint
from dkm.models.dkm import *
from dkm.models.encoders import *


def get_model(pretrained_backbone=True, resolution="low", **kwargs):
    gp_dim = 256
    dfn_dim = 384
    feat_dim = 256
    coordinate_decoder = DFN(
        internal_dim=dfn_dim,
        feat_input_modules=nn.ModuleDict(
            {
                "32": nn.Conv2d(512, feat_dim, 1, 1),
                "16": nn.Conv2d(512, feat_dim, 1, 1),
            }
        ),
        pred_input_modules=nn.ModuleDict(
            {
                "32": nn.Identity(),
                "16": nn.Identity(),
            }
        ),
        rrb_d_dict=nn.ModuleDict(
            {
                "32": RRB(gp_dim + feat_dim, dfn_dim),
                "16": RRB(gp_dim + feat_dim, dfn_dim),
            }
        ),
        cab_dict=nn.ModuleDict(
            {
                "32": CAB(2 * dfn_dim, dfn_dim),
                "16": CAB(2 * dfn_dim, dfn_dim),
            }
        ),
        rrb_u_dict=nn.ModuleDict(
            {
                "32": RRB(dfn_dim, dfn_dim),
                "16": RRB(dfn_dim, dfn_dim),
            }
        ),
        terminal_module=nn.ModuleDict(
            {
                "32": nn.Conv2d(dfn_dim, 3, 1, 1, 0),
                "16": nn.Conv2d(dfn_dim, 3, 1, 1, 0),
            }
        ),
    )
    dw = True
    hidden_blocks = 8
    kernel_size = 5
    displacement_emb = "linear"
    local_corr = True
    conv_refiner = nn.ModuleDict(
        {
            "16": ConvRefiner(
                2 * 512 + 128 + (2 * 7 + 1) ** 2,
                2 * 512 + 128 + (2 * 7 + 1) ** 2,
                3,
                kernel_size=kernel_size,
                dw=dw,
                hidden_blocks=hidden_blocks,
                displacement_emb=displacement_emb,
                displacement_emb_dim=128,
                local_corr_radius=7,
                corr_in_other=True,
            ),
            "8": ConvRefiner(
                2 * 512 + 64 + (2 * 3 + 1) ** 2,
                2 * 512 + 64 + (2 * 3 + 1) ** 2,
                3,
                kernel_size=kernel_size,
                dw=dw,
                hidden_blocks=hidden_blocks,
                displacement_emb=displacement_emb,
                displacement_emb_dim=64,
                local_corr_radius=3,
                corr_in_other=True,
            ),
            "4": ConvRefiner(
                2 * 256 + 32 + (2 * 2 + 1) ** 2,
                2 * 256 + 32 + (2 * 2 + 1) ** 2,
                3,
                kernel_size=kernel_size,
                dw=dw,
                hidden_blocks=hidden_blocks,
                displacement_emb=displacement_emb,
                displacement_emb_dim=32,
                local_corr_radius=2,
                corr_in_other=True,
            ),
            "2": ConvRefiner(
                2 * 64 + 16,
                128 + 16,
                3,
                kernel_size=kernel_size,
                dw=dw,
                hidden_blocks=hidden_blocks,
                displacement_emb=displacement_emb,
                displacement_emb_dim=16,
            ),
            "1": ConvRefiner(
                2 * 3 + 6,
                24,
                3,
                kernel_size=kernel_size,
                dw=dw,
                hidden_blocks=hidden_blocks,
                displacement_emb=displacement_emb,
                displacement_emb_dim=6,
            ),
        }
    )
    kernel_temperature = 0.2
    learn_temperature = False
    no_cov = True
    kernel = CosKernel
    only_attention = False
    basis = "fourier"
    gp32 = GP(
        kernel,
        T=kernel_temperature,
        learn_temperature=learn_temperature,
        only_attention=only_attention,
        gp_dim=gp_dim,
        basis=basis,
        no_cov=no_cov,
    )
    gp16 = GP(
        kernel,
        T=kernel_temperature,
        learn_temperature=learn_temperature,
        only_attention=only_attention,
        gp_dim=gp_dim,
        basis=basis,
        no_cov=no_cov,
    )
    gps = nn.ModuleDict({"32": gp32, "16": gp16})
    proj = nn.ModuleDict(
        {"16": nn.Conv2d(1024, 512, 1, 1), "32": nn.Conv2d(2048, 512, 1, 1)}
    )
    decoder = Decoder(coordinate_decoder, gps, proj, conv_refiner, detach=True)
    if resolution == "low":
        h, w = 384, 512
    elif resolution == "high":
        h, w = 480, 640
    elif resolution == "higher":
        h, w = 540, 720
    elif resolution == "highest":
        h, w = 600, 800
    elif resolution == "highester":
        h, w = 660, 880
        
    encoder = ResNet50(pretrained=pretrained_backbone, high_res = False, freeze_bn=False)
    matcher = RegressionMatcher(encoder, decoder, h=h, w=w, alpha=1, beta=0,**kwargs).cuda()
    return matcher

def train(args):
    gpus, wandb_log, wandb_entity = args.gpus, not args.dont_log_wandb, args.wandb_entity 
    experiment_name = os.path.splitext(os.path.basename(__file__))[0]
    wandb_mode = "online" if wandb_log else "disabled"
    wandb.init(project="dkm", entity=wandb_entity, name=experiment_name, reinit=False, mode = wandb_mode)
    checkpoint_dir = "workspace/checkpoints/"
    h, w = 540, 720
    # model = get_model(pretrained_backbone=True, resolution="higher")
    # 替换原来的模型初始化
    model = get_model(pretrained_backbone=False, resolution="higher")  # 关闭ImageNet预训练
    # 加载自定义预训练权重
    pretrained_dict = torch.load("/root/autodl-tmp/DKM/workplace/checkpoints/DKMv3_outdoor.pth")
    model.load_state_dict(pretrained_dict)

    wandb.watch(model)
    # Num steps
    n0 = 0
    batch_size = gpus * 4
    N = (32 * 25000) // batch_size  # 250k steps of batch size 32
    # checkpoint every
    k = 1000

    # Data
    mega = MegadepthBuilder(data_root="data/megadepth")
    megadepth_train1 = mega.build_scenes(
        split="train_loftr", min_overlap=0.01, ht=h, wt=w, shake_t=32
    )
    megadepth_train2 = mega.build_scenes(
        split="train_loftr", min_overlap=0.05, ht=h, wt=w, shake_t=32
    )
    megadepth_train = ConcatDataset(megadepth_train1 + megadepth_train2)
    mega_ws = mega.weight_scenes(megadepth_train, alpha=0.75)
    # Loss and optimizer
    depth_loss = DepthRegressionLoss(ce_weight=0.01)

    parameters = [
        {"params": model.encoder.parameters(), "lr": gpus * 1e-6},
        {"params": model.decoder.parameters(), "lr": gpus * 5e-5},
    ]
    optimizer = torch.optim.AdamW(parameters, weight_decay=0.01)
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[(N) // 2, (4 * N) // 5], gamma=0.2
    )
    mega_pose_benchmark = Megadepth1500Benchmark("data/megadepth")
    checkpointer = CheckPoint(checkpoint_dir, experiment_name)
    checkpoint_name = checkpoint_dir + experiment_name + "_latest.pth"
    states = {}
    # Load checkpoint (if existing)
    if os.path.exists(checkpoint_name):
        states = torch.load(checkpoint_name)
        if "model" in states:
            model.load_state_dict(states["model"])
        if "n" in states:
            n0 = states["n"] if states["n"] else 0
        if "optimizer" in states:
            optimizer.load_state_dict(states["optimizer"])
        if "lr_scheduler" in states:
            lr_scheduler.load_state_dict(states["lr_scheduler"])
    if states:
        print(f"Loaded states {list(states.keys())}, at step {n0}")
    dp_model = nn.DataParallel(model)
    for n in range(n0, N, k):
        mega_sampler = torch.utils.data.WeightedRandomSampler(
            mega_ws, num_samples=batch_size * k, replacement=False
        )
        mega_dataloader = iter(
            torch.utils.data.DataLoader(
                megadepth_train,
                batch_size=batch_size,
                sampler=mega_sampler,
                num_workers=gpus * 8,
            )
        )
        train_k_steps(
            n, k, mega_dataloader, dp_model, depth_loss, optimizer, lr_scheduler
        )
        checkpointer(model, optimizer, lr_scheduler, n)
        wandb.log(mega_pose_benchmark.benchmark(model))


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--gpus", default=1, type=int)
    parser.add_argument("--dont_log_wandb", action='store_true')
    parser.add_argument("--wandb_entity", type=str)
    args, _ = parser.parse_known_args()
    train(args)