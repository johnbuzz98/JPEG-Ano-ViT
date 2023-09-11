import torch

from .dataset import MvtecAd


def get_dataloader(cfg):

    train_dataset = MvtecAd(
        datadir = cfg.DATASET.DATADIR,
        target = cfg.DATASET.TARGET,
        is_train=True,
        resize=cfg.MODEL.IMG_SIZE,
        image_domain=cfg.MODEL.DOMAIN,
        image_format=cfg.MODEL.IMAGE_FORMAT,
    )
    valid_dataset = MvtecAd(
        datadir = cfg.DATASET.DATADIR,
        target = cfg.DATASET.TARGET,
        is_train=False,
        resize=cfg.MODEL.IMG_SIZE,
        image_domain=cfg.MODEL.DOMAIN,
        image_format=cfg.MODEL.IMAGE_FORMAT,
    )
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=cfg.TRAIN.BATCHSIZE,
    )
    val_loader = torch.utils.data.DataLoader(
        valid_dataset, batch_size=cfg.TRAIN.BATCHSIZE,
    )
    
    return train_loader, val_loader
