#!/usr/bin/env python
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import argparse
import builtins
import os
import shutil
import copy

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim

import torchvision.models as models
from data.dataset import load_train, load_val
from research_tools.store import ExperimentLogWriter
import utils

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--dataset', choices=['imagenet', 'tiny-imagenet'], default='tiny-imagenet',
                    help='Which dataset to evaluate on.')
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet50')
parser.add_argument('-j', '--workers', default=32, type=int, metavar='N',
                    help='number of data loading workers (default: 32)')
parser.add_argument('--epochs', default=100, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch_size', default=256, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', '--learning-rate', default=30., type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--schedule', default=[60, 80], nargs='*', type=int,
                    help='learning rate schedule (when to drop lr by a ratio)')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--wd', '--weight_decay', default=0., type=float,
                    metavar='W', help='weight decay (default: 0.)',
                    dest='weight_decay')
parser.add_argument('-p', '--print_freq', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--eval_first', action='store_true',
                    help='If this is true, we eval -1.pth.')

parser.add_argument('--num_per_class', type=int, default=int(1e10),
                    help='Number of images per class for getting a subset of Imagenet')
parser.add_argument('--data_aug', type=str, help='Choice of augmentation to use for training linear classifier.')
parser.add_argument('--checkpoint_every', type=int, default=5, help='How often to evaluate linear classification (every how many epochs).')
parser.add_argument('--specific_ckpts', nargs='*', help='filenames of specific checkpoints to evaluate')

best_acc1 = 0

def load_model(state_dict):
    for key in list(state_dict.keys()):
        if key.startswith('online_encoder.net'):
            state_dict[key[len('online_encoder.net.'):]] = state_dict[key]
        del state_dict[key]
    model = models.resnet50(num_classes=200)
    model.load_state_dict(state_dict, strict=True)
    return model

def main():
    args = parser.parse_args()
    logger = ExperimentLogWriter('/tiger/u/kshen6/byol-pytorch', None)
    for fname in sorted(os.listdir('.')):
        if fname not in args.specific_ckpts: continue
        eval_ckpt(fname, args, logger)

def eval_ckpt(fname, args, logger):
    model = load_model(torch.load(fname)).cuda()
    dict_id = fname.split('.')[0]
    logger.create_data_dict(
        ['epoch', 'train_acc', 'val_acc','train_loss', 'val_loss', 'train5', 'val5'],
        dict_id=dict_id)

    # freeze all layers but the last fc
    for name, param in model.named_parameters():
        if name not in ['fc.weight', 'fc.bias']:
            param.requires_grad = False
    # init the fc layer
    model.fc.weight.data.normal_(mean=0.0, std=0.01)
    model.fc.bias.data.zero_()

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda()

    # optimize only the linear classifier
    parameters = list(filter(lambda p: p.requires_grad, model.parameters()))
    assert len(parameters) == 2  # fc.weight, fc.bias
    optimizer = torch.optim.SGD(parameters, args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    cudnn.benchmark = True

    # Data loading code
    train_sampler, train_loader = load_train(args.dataset, args.num_per_class, False,
                                             args.batch_size, args.workers, data_aug=args.data_aug)
    val_loader = load_val(args.dataset, args.batch_size, args.workers)

    best_acc1 = 0

    for epoch in range(args.start_epoch, args.epochs):
        adjust_learning_rate(optimizer, epoch, args)

        # train for one epoch
        top1, top5, losses = train(train_loader, model, criterion, optimizer, epoch, args)

        # always test after 1 epoch of linear evaluation
        if epoch == 0 or (epoch + 1) % args.checkpoint_every == 0:
            # evaluate on validation set
            acc1, acc5, val_losses = validate(val_loader, model, criterion, args)

            # remember best acc@1 and save checkpoint
            is_best = acc1 > best_acc1
            best_acc1 = max(acc1, best_acc1)

            logger.update_data_dict(
                {
                'epoch' : epoch + 1,
                'train_acc' : top1, 
                'val_acc' : acc1,
                'train_loss' : losses,
                'val_loss' : val_losses,
                'train5' : top5,
                'val5' : acc5
                }, dict_id=dict_id)
            logger.save_data_dict(dict_id=dict_id)
            save_checkpoint({
                'epoch': epoch + 1,
                'arch': args.arch,
                'state_dict': model.state_dict(),
                'best_acc1': best_acc1,
                'optimizer' : optimizer.state_dict(),
            }, filename='linevalckpt-' + dict_id + '.tar')

def train(train_loader, model, criterion, optimizer, epoch, args):
    losses = utils.AverageMeter('Loss', ':.4e')
    top1 = utils.AverageMeter('Acc@1', ':6.2f')
    top5 = utils.AverageMeter('Acc@5', ':6.2f')
    progress = utils.ProgressMeter(
        len(train_loader),
        [losses, top1, top5],
        prefix="Epoch: [{}]".format(epoch))

    """
    Switch to eval mode:
    Under the protocol of linear classification on frozen features/models,
    it is not legitimate to change any part of the pre-trained model.
    BatchNorm in train mode may revise running mean/std (even if it receives
    no gradient), which are part of the model parameters too.
    """
    model.eval()

    for i, (images, target) in enumerate(train_loader):
        images = images.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)

        # compute output
        output = model(images)
        loss = criterion(output, target)

        # measure accuracy and record loss
        acc1, acc5 = utils.accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), images.size(0))
        top1.update(acc1[0], images.size(0))
        top5.update(acc5[0], images.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i + 1) % args.print_freq == 0:
            progress.display(i)

    return top1.avg, top5.avg, losses.avg

def validate(val_loader, model, criterion, args):
    losses = utils.AverageMeter('Loss', ':.4e')
    top1 = utils.AverageMeter('Acc@1', ':6.2f')
    top5 = utils.AverageMeter('Acc@5', ':6.2f')
    progress = utils.ProgressMeter(
        len(val_loader),
        [losses, top1, top5],
        prefix='Test: ')

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        for i, (images, target) in enumerate(val_loader):
            images = images.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)

            # compute output
            output = model(images)
            loss = criterion(output, target)

            # measure accuracy and record loss
            acc1, acc5 = utils.accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1[0], images.size(0))
            top5.update(acc5[0], images.size(0))

            if i % args.print_freq == 0:
                progress.display(i)

        # TODO: this should also be done with the ProgressMeter
        # is the above todo done??
        print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
              .format(top1=top1, top5=top5))

    return top1.avg, top5.avg, losses.avg

def save_checkpoint(state, filename):
    torch.save(state, filename)

def adjust_learning_rate(optimizer, epoch, args):
    """Decay the learning rate based on schedule"""
    lr = args.lr
    for milestone in args.schedule:
        lr *= 0.1 if epoch >= milestone else 1.
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

if __name__ == '__main__':
    main()
