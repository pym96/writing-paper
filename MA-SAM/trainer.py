import argparse
import logging
import os
import random
import sys
import time
import math
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch.nn.modules.loss import CrossEntropyLoss
from torch.utils.data import DataLoader
import torch.nn.functional as F
from tqdm import tqdm
from utils import DiceLoss
from torchvision import transforms
from icecream import ic
from datetime import datetime

from metrics import *
from models import *

def calc_loss(outputs, low_res_label_batch, ce_loss, dice_loss, dice_weight:float=0.8):
    low_res_logits = outputs['low_res_logits']
    loss_ce = ce_loss(low_res_logits, low_res_label_batch[:].long())
    loss_dice = dice_loss(low_res_logits, low_res_label_batch, softmax=True)
    loss = (1 - dice_weight) * loss_ce + dice_weight * loss_dice
    return loss, loss_ce, loss_dice


def trainer_run(args, model, snapshot_path, multimask_output, low_res):
    from datasets.dataset import dataset_reader, RandomGenerator
    
    output_filename = datetime.now().strftime("%Y%m%d-%H%M%S")
    
    if not os.path.exists('./training_log'):
        os.mkdir('./training_log')
    logging.basicConfig(filename= './training_log/' + args.output.split('/')[-1] + '_log.txt', level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    base_lr = args.base_lr
    num_classes = args.num_classes
    batch_size = args.batch_size * args.n_gpu
    
    db_train = dataset_reader(base_dir=args.root_path, split="train", num_classes=args.num_classes, 
                                transform=transforms.Compose([RandomGenerator(output_size=[args.img_size, args.img_size], low_res=[low_res, low_res])]))
    print("The length of train set is: {}".format(len(db_train)))

    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)

    trainloader = DataLoader(db_train, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True,
                             worker_init_fn=worker_init_fn)

    if args.n_gpu > 1:
        model = nn.DataParallel(model)
    model.train()
    ce_loss = CrossEntropyLoss(ignore_index=-100)
    dice_loss = DiceLoss(num_classes + 1)
    if args.warmup:
        b_lr = base_lr / args.warmup_period
    else:
        b_lr = base_lr
    if args.AdamW:
        optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=b_lr, betas=(0.9, 0.999), weight_decay=0.1)
    else:
        optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=b_lr, momentum=0.9, weight_decay=0.0001) 
    if args.use_amp:
        scaler = torch.cuda.amp.GradScaler(enabled=args.use_amp)
    writer = SummaryWriter(snapshot_path + '/log')
    iter_num = 0
    max_epoch = args.max_epochs
    stop_epoch = args.stop_epoch
    max_iterations = args.max_epochs * len(trainloader)
    logging.info("{} iterations per epoch. {} max iterations ".format(len(trainloader), max_iterations))
    
    iterator = tqdm(range(max_epoch), ncols=70)
    
    inpainter = Inpainter(args.sigma, args.kernel_size, args.reps, args.scale_factor).to(args.device)
    boundary = Boundary().to(args.device)
    
    for epoch_num in iterator:
        for i_batch, sampled_batch in enumerate(trainloader):
            image_batch, label_batch = sampled_batch['image'], sampled_batch['label'] 
            image_batch = image_batch.unsqueeze(2)
            image_batch = torch.cat((image_batch, image_batch, image_batch), dim=2)
  
            hw_size = image_batch.shape[-1]
            label_batch = label_batch.contiguous().view(-1, hw_size, hw_size)

            low_res_label_batch = sampled_batch['low_res_label']
            image_batch, label_batch = image_batch.cuda(), label_batch.cuda()
            
            # TODO: Use iem to process image_batch, and then take it as trained data, I think this is reasonable
            pre_mask = torch.nn.Parameter(torch.zeros(len(image_batch), 1, args.size, args.size).to(args.device))
            init_start, init_end = args.size//5, args.size - args.size//5
            pre_mask.data[:,:,init_start:init_end,init_start:init_end].fill_(1.0)
        
            for i in range(args.iters):
                foreground = image_batch * pre_mask
                background = image_batch * (1-pre_mask)

                pred_foreground = inpainter(background, (1-pre_mask))
                pred_background = inpainter(foreground, pre_mask)

                # inpainting error is equiv to negative coeff. of constraint between foreground and background
                inp_error = neg_coeff_constraint(image_batch, pre_mask, pred_foreground, pred_background)
                # diversity term is the total deviation of foreground and background pixels
                mask_diversity = diversity(image_batch, pre_mask, foreground, background)

                # regularized IEM objective (to be maximized) is the inpainting error minus diversity regularizer
                total_loss = inp_error - args.lmbda * mask_diversity
                total_loss.sum().backward()

                with torch.no_grad():
                    grad = pre_mask.grad.data
                    
                    # we only update mask pixels that are in the boundary AND have non-zero gradient
                    update_bool = boundary(pre_mask) * (grad != 0)
                    # pixels with positive gradients are set to 1 and with negative gradients are set to 0
                    pre_mask.data[update_bool] = (grad[update_bool] > 0).float()
                    grad.zero_()
                    
                    # smoothing procedure: we set a pixel to 1 if there are 4 or more 1-valued pixels in its 3x3 neighborhood
                    pre_mask.data = (F.avg_pool2d(pre_mask, 3, 1, 1, divisor_override=1) >= 4).float()
                
                
            low_res_label_batch = low_res_label_batch.cuda()
            
            if args.use_amp:
                with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=args.use_amp):
                    outputs = model(pre_mask, multimask_output, args.img_size)
                    loss, loss_ce, loss_dice = calc_loss(outputs, label_batch, ce_loss, dice_loss, args.dice_param)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
            
            if args.warmup and iter_num < args.warmup_period:
                lr_ = base_lr * ((iter_num + 1) / args.warmup_period)
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr_
            else:
                if args.warmup:
                    shift_iter = iter_num - args.warmup_period
                    assert shift_iter >= 0, f'Shift iter is {shift_iter}, smaller than zero'
                else:
                    shift_iter = iter_num
                lr_ = base_lr * (1.0 - shift_iter / max_iterations) ** args.lr_exp
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr_

            iter_num = iter_num + 1
            writer.add_scalar('info/lr', lr_, iter_num)
            writer.add_scalar('info/total_loss', loss, iter_num)
            writer.add_scalar('info/loss_ce', loss_ce, iter_num)
            writer.add_scalar('info/loss_dice', loss_dice, iter_num)

            logging.info('iteration %d : loss : %f, loss_ce: %f, loss_dice: %f' % (iter_num, loss.item(), loss_ce.item(), loss_dice.item()))

        save_interval = 20 
        if (epoch_num + 1) % save_interval == 0:
            save_mode_path = os.path.join(snapshot_path, 'epoch_' + str(epoch_num) + '.pth')
            try:
                model.save_parameters(save_mode_path)
            except:
                model.module.save_parameters(save_mode_path)
            logging.info("save model to {}".format(save_mode_path))

        if epoch_num >= max_epoch - 1 or epoch_num >= stop_epoch - 1:
            save_mode_path = os.path.join(snapshot_path, 'epoch_' + str(epoch_num) + '.pth')
            try:
                model.save_parameters(save_mode_path)
            except:
                model.module.save_parameters(save_mode_path)
            logging.info("save model to {}".format(save_mode_path))
            iterator.close()
            break

    writer.close()
    return "Training Finished!"
