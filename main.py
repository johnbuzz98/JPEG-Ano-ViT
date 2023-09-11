import argparse
import logging
import os

import torch
import torch.nn as nn
from accelerate import Accelerator, DistributedDataParallelKwargs
from accelerate.logging import get_logger

from data import get_dataloader
from log import setup_default_logging
from models import Decoder, ViT
from scheduler import CosineAnnealingWarmupRestarts
from train import training
from utils import configs, torch_seed

_logger = logging.getLogger('train')

def run(cfg):
    # setting seed and device
    setup_default_logging()
    torch_seed(cfg.SEED)
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(log_with="wandb",kwargs_handlers=[ddp_kwargs])
    device = accelerator.device
    cfg.RANK = str(device)
#    _logger.info("Device: {}".format(cfg.RANK))
    print(accelerator.log_with)
    # savedir
    cfg.EXP_NAME = cfg.EXP_NAME + f"-{cfg.DATASET.TARGET}"
    savedir = os.path.join(cfg.RESULT.SAVEDIR, cfg.EXP_NAME)
    os.makedirs(savedir, exist_ok=True)
    config_ = {"model_arch": cfg.MODEL.ARCH, "Image_domain": cfg.MODEL.DOMAIN, "Image_format": cfg.MODEL.IMAGE_FORMAT,
                "patch_size": cfg.MODEL.PATCHSIZE, "embed_size": cfg.MODEL.EMBEDSIZE,
                "learning_rate": cfg.OPTIMIZER.LR, "batch_size": cfg.TRAIN.BATCHSIZE,}

    accelerator.init_trackers(project_name=cfg.EXP_NAME,
                            config = config_, 
                            init_kwargs={"wandb":
                                        {'entity': 'woojun_lee'}
                                        }
    )
    accelerator.trackers = []
    from accelerate.tracking import (
        LOGGER_TYPE_TO_CLASS,
        GeneralTracker,
        filter_trackers,
    )

    init_kwargs={"wandb": {'entity': 'woojun_lee'}}
    for tracker in accelerator.log_with:
        if issubclass(type(tracker), GeneralTracker):
            # Custom trackers are already initialized
            print('1')
            accelerator.trackers.append(tracker)
        else:
            tracker_init = LOGGER_TYPE_TO_CLASS[str(tracker)]
            print(tracker_init)
            if getattr(tracker_init, "requires_logging_directory"):
                # We can skip this check since it was done in `__init__`
                print('2')
                accelerator.trackers.append(
                    tracker_init(cfg.EXP_NAME, '/workspace', **init_kwargs.get(str(tracker), {}))
                )
            else:
                print('3')
                print(tracker_init(cfg.EXP_NAME, **init_kwargs.get(str(tracker), {})))
                accelerator.trackers.append(tracker_init(cfg.EXP_NAME, **init_kwargs.get(str(tracker), {})))
    if config_ is not None:
        for tracker in accelerator.trackers:
            tracker.store_init_configuration(config_)
    print(accelerator.trackers)
    # build dataloader
    trainloader, testloader = get_dataloader(cfg)

    # build ViT Encoder
    vit_encoder = ViT(
        patch_size = cfg.MODEL.PATCHSIZE,
        emb_size = cfg.MODEL.EMBEDSIZE,
        depth = cfg.MODEL.DEPTH,
        drop_p = cfg.TRAIN.DROP,
        device = cfg.RANK,
        dtype = torch.float32,
        num_heads=cfg.MODEL.HEADS,
        head_size=cfg.MODEL.HEADSIZE,
        pixel_space = cfg.MODEL.DOMAIN,
        use_subblock = True
                )
    #load pretrained weights
    vit_encoder_state_dict = torch.load(cfg.MODEL.WEIGHT, map_location=device)
    vit_encoder.load_state_dict(vit_encoder_state_dict, strict=False)

    # build Decoder
    decoder = Decoder(
        emb_size = cfg.MODEL.EMBEDSIZE,
        image_size = cfg.MODEL.IMG_SIZE,
        patch_size=cfg.MODEL.PATCHSIZE,
        domain = cfg.MODEL.DOMAIN,
    )
    
    # wrap model
    model = nn.Sequential(vit_encoder, decoder)

    # Set training
    criterion = nn.MSELoss(reduction='none')
    optimizer = torch.optim.AdamW(model.parameters(),
                                  lr=cfg.OPTIMIZER.LR,
                                  betas = (0.5, 0.999),)
    
    # accelerator setting
    trainloader, testloader, model, optimizer = accelerator.prepare(
        trainloader, testloader, model, optimizer
    )
    

    cfg.TRAIN.NUM_TRAINING_STEPS = len(trainloader) * cfg.TRAIN.EPOCHS

    if cfg.SCHEDULER.USE_SCHEDULER:
        scheduler = CosineAnnealingWarmupRestarts(
            optimizer,
            first_cycle_steps=cfg.TRAIN.NUM_TRAINING_STEPS,
            max_lr=cfg.OPTIMIZER.LR,
            min_lr=cfg.SCHEDULER.MIN_LR,
            warmup_steps=int(cfg.TRAIN.NUM_TRAINING_STEPS * cfg.SCHEDULER.WARMUP_RATIO),
        )
    else:
        scheduler = None
    """if cfg.TRAIN.USE_WANDB:
        config_ = {"model_arch": cfg.MODEL.ARCH, "Image_domain": cfg.MODEL.DOMAIN, "Image_format": cfg.MODEL.IMAGE_FORMAT,
                "patch_size": cfg.MODEL.PATCHSIZE, "embed_size": cfg.MODEL.EMBEDSIZE,
                "learning_rate": cfg.OPTIMIZER.LR, "batch_size": cfg.TRAIN.BATCHSIZE,}
"""
    print("Accelerator prepared")
    print("trainloader length: ", len(trainloader))
    print("testloader length: ", len(testloader))
    print("model arch: ", cfg.MODEL.ARCH)
    print("Mvtec target: ", cfg.DATASET.TARGET)
    print("Mvtec domain: ", cfg.MODEL.DOMAIN)
    print("Mvtec image format: ", cfg.MODEL.IMAGE_FORMAT)
    # Fitting model
    training(cfg,
        model=model,
        trainloader=trainloader,
        validloader=testloader,
        criterion=criterion,    
        optimizer=optimizer,
        scheduler=scheduler,
        log_interval=cfg.LOG.LOG_INTERVAL,
        eval_interval=cfg.LOG.EVAL_INTERVAL,
        savedir=savedir,
        device=device,
        use_wandb=cfg.TRAIN.USE_WANDB,
        accelerator = accelerator,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="JPEG-AnoViT")

    parser.add_argument('--target', type=str, default='bottle', help='Target class')
    # model config
    parser.add_argument('--model_arch', type=str, default='vits', help='Model architecture (vitti, vits, vitb, vitl, swinv2)')
    parser.add_argument('--no_subblock', action='store_true', help='If set, disable subblock conversion')
    parser.add_argument("--embed_type", type=int, default=1, help='Embedding layer type. (1: grouped, 2: separate, 3: concatenate). Default 1')
    parser.add_argument("--domain", type=str, default="dct", help="(DCT/RGB) Choose domain type")
    parser.add_argument("--img_format", type=str, default="jpeg", help="(JPEG/PNG) Choose image format")

    parser.add_argument("--configs", type=str, default=None, help="exp config file")
    parser.add_argument("--device", type=str, default="cuda", help="gpu id")
    parser.add_argument("--img_size", type=int, default=224, help="image size")
    parser.add_argument("--patch_size", type=int, default=16, help="patch size")

    # training config
    parser.add_argument("--seed", type=int, default=42, help="random seed")
    parser.add_argument("--epochs", type=int, default=500, help="number of training epochs")
    parser.add_argument("--batch", type=int, default=32, help="batch size")
    parser.add_argument("--lr", type=float, default=0.001, help="learning rate")
    parser.add_argument("--wd", type=float, default=0.05, help="weight decay")
    parser.add_argument("--drop", type=float, default=0.0, help="dropout rate")
    parser.add_argument("--warmup_steps", type=int, default=100, help="warmup steps")
    parser.add_argument("--ops_list", type=str, default="", help="augmentation operations")
    parser.add_argument("--num_ops", type=int, default=-1, help="number of augmentation operations")
    parser.add_argument("--ops_magnitude", type=float, default=-1, help="magnitude of augmentation operations")
    parser.add_argument("--amp", type=int, default=-1, help="use mixed precision training or not")
    parser.add_argument("--ampdtype", type=str, default="", help="amp dtype")
    parser.add_argument("--use_msrsync", action="store_true", help="use msr_sync_batchnorm")
    args = parser.parse_args()

    # load cfg
    cfg = configs.generate_config(
        modelarch = args.model_arch.lower(),
        domain = args.domain,
        image_format = args.img_format,
        target = args.target,
        modelver=args.embed_type,
        subblock=True if not args.no_subblock else False,
        epochs=None if args.epochs < 0 else args.epochs, # need to add
        batchsize=None if args.batch < 0 else args.batch, # need to change order
        lr=None if args.lr < 0 else args.lr,
        wd=None if args.wd < 0 else args.wd,
        drop=None if args.drop < 0 else args.drop,
        warmup_steps=None if args.warmup_steps < 0 else args.warmup_steps, # need to add
        auglist=None if args.ops_list == '' else args.ops_list.split(","),
        num_ops=None if args.num_ops < 0 else args.num_ops, # need to add
        ops_magnitude=None if args.ops_magnitude < 0 else args.ops_magnitude, # need to add
        seed=None if args.seed < 0 else args.seed, # need to add
        amp=None if args.amp < 0 else args.amp,
        ampdtype=None if args.ampdtype == '' else args.ampdtype,
        use_msrsync=args.use_msrsync,
    )

    run(cfg)
