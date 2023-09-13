import json
import logging
import os
import time
from copy import deepcopy

import numpy as np
import torch
import torch.nn.functional as F
from accelerate.logging import get_logger
from scipy.ndimage import gaussian_filter
from sklearn.metrics import roc_auc_score

_logger = logging.getLogger('train')
class AverageMeter:
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count



def training(cfg, model, trainloader, validloader, criterion, optimizer, scheduler, 
             log_interval: int = 1, eval_interval: int = 1, savedir: str = None, use_wandb: bool = False, 
             accelerator= None, device: str ='cpu') -> dict:   

    batch_time_m = AverageMeter()
    data_time_m = AverageMeter()
    losses_m = AverageMeter()
    
    # set train mode
    model.train()

    # set optimizer
    optimizer.zero_grad()

    # training
    best_score = 0
    step = 0
    train_mode = True
    cfg.TRAIN.NUM_TRAINIG_STEPS = len(trainloader) * cfg.TRAIN.EPOCHS
    num_training_steps = cfg.TRAIN.NUM_TRAINIG_STEPS
    print ("Training for {} steps".format(num_training_steps))
    while train_mode:
        end = time.time()
        for idx, (inputs, masks, targets) in enumerate(trainloader):
            # batch
            #inputs, masks, targets = inputs.to(device), masks.to(device), targets.to(device)

            data_time_m.update(time.time() - end)
            
            # predict
            outputs = model(inputs)

            loss_y = criterion(inputs[0], outputs[0])
            loss_C = criterion(inputs[1], outputs[1]) # TODO 굳이 Flatten? ->  굳이 reduction none in training
            flattened_loss_y = loss_y.view(loss_y.size(0), -1)  # shape: [32, 1*28*28*8*8]
            flattened_loss_C = loss_C.view(loss_C.size(0), -1)  # shape: [32, 2*14*14*8*8]
            # Concatenate (이어붙이기) along the second dimension
            loss = torch.cat((flattened_loss_y, flattened_loss_C), dim=1).mean()
            accelerator.backward(loss)
            
            # update weight
            optimizer.step()
            optimizer.zero_grad()

            # log loss
            losses_m.update(loss.item())
            batch_time_m.update(time.time() - end)

            # wandb
            if use_wandb:
                accelerator.log({"lr": optimizer.param_groups[0]["lr"], "train_loss": losses_m.val,}, step=step,)
            
            if (step+1) % log_interval == 0 or step == 0: 
                _logger.info('TRAIN [{:>4d}/{}] '
                            'Loss: {loss.val:>6.4f} ({loss.avg:>6.4f}) '
                            'LR: {lr:.3e} '
                            'Time: {batch_time.val:.3f}s, {rate:>7.2f}/s ({batch_time.avg:.3f}s, {rate_avg:>7.2f}/s) '
                            'Data: {data_time.val:.3f} ({data_time.avg:.3f})'.format(
                            step+1, num_training_steps, 
                            loss       = losses_m, 
                            lr         = optimizer.param_groups[0]['lr'],
                            batch_time = batch_time_m,
                            rate       = inputs[0].size(0) / batch_time_m.val,
                            rate_avg   = inputs[0].size(0) / batch_time_m.avg,
                            data_time  = data_time_m))


            if ((step+1) % eval_interval == 0 and step != 0) or (step+1) == num_training_steps: 
                eval_metrics = evaluate(
                    cfg          = cfg,
                    model        = model, 
                    dataloader   = validloader, 
                    criterion   = criterion,
                )
                model.train()

                eval_log = dict([(f'eval_{k}', v) for k, v in eval_metrics.items()])

                # wandb
                if use_wandb:
                    accelerator.log(eval_log, step=step)

                # checkpoint
                if best_score < np.mean(list(eval_metrics.values())):
                    # save best score
                    state = {'best_step':step}
                    state.update(eval_log)
                    json.dump(state, open(os.path.join(savedir, 'best_score.json'),'w'), indent='\t')

                    # save best model
                    torch.save(model.state_dict(), os.path.join(savedir, f'best_model.pt'))
                    
                    _logger.info('Best Score {0:.3%} to {1:.3%}'.format(best_score, np.mean(list(eval_metrics.values()))))

                    best_score = np.mean(list(eval_metrics.values()))

            # scheduler
            if scheduler:
                scheduler.step()

            end = time.time()

            step += 1

            if step == num_training_steps:
                train_mode = False
                accelerator.end_training()
                break

    # print best score and step
    _logger.info('Best Metric: {0:.3%} (step {1:})'.format(best_score, state['best_step']))

    # save latest model
    torch.save(model.state_dict(), os.path.join(savedir, f'latest_model.pt'))

    # save latest score
    state = {'latest_step':step}
    state.update(eval_log)
    json.dump(state, open(os.path.join(savedir, 'latest_score.json'),'w'), indent='\t')

    

        
def evaluate(cfg, model, dataloader, criterion):

    # for image-level auroc
    total_loss_img = []
    total_targets = []
    # for pixel-level auroc
    total_masks = []
    total_loss_pixel = []

    model.eval()
    with torch.no_grad():
        for idx, (inputs, masks, targets) in enumerate(dataloader):
            # get masks
            total_masks.append(masks.cpu().numpy())

            # get targets
            total_targets.extend(targets.cpu().tolist())
            # predict
            outputs = model(inputs)
            
            # loss
            loss_y = criterion(inputs[0], outputs[0])
            loss_C = criterion(inputs[1], outputs[1])
            b, c, h, w, kh, kw = loss_C.size()
            upscaled_C = F.interpolate(loss_C.view(b, c*kh*kw , h, w), scale_factor=2, mode='bilinear', align_corners=True)
            upscaled_C = upscaled_C.view(b, c, h*2, w*2, kh, kw)

            loss = torch.cat((loss_y, upscaled_C), dim=1)
            # loss image
            total_loss_img.extend(loss.flatten(start_dim=1).max(dim=1)[0].cpu().tolist())

            # loss pixel with gaussian filter
            loss = loss.mean(dim=1)
            #여기서 에바 #reshape loss
            b, h, w, kh, kw = loss.size()
            loss = loss.view(b, h*kh, w*kw)
            #여기까지 에바
            loss_pixel = np.zeros_like(loss.cpu())

            for i, loss_b in enumerate(loss):
                loss_pixel[i] = gaussian_filter(deepcopy(loss_b.cpu()), sigma=6)
            total_loss_pixel.append(loss_pixel)
            
   # image-level auroc
    auroc_img = roc_auc_score(total_targets, total_loss_img)

    # pixel-level auroc
    total_loss_pixel = np.vstack(total_loss_pixel).reshape(-1)
    total_masks = np.vstack(total_masks).reshape(-1)
    auroc_pixel = roc_auc_score(total_masks, total_loss_pixel)

    _logger.info('TEST: AUROC-image: %.3f%% | AUROC-pixel: %.3f%%' % (auroc_img, auroc_pixel))


    return {"auroc_img": auroc_img, "auroc_pixel": auroc_pixel}
