from network.resnet import resnet18
from network.unet import UNet
from dataloader import testData
from loss import SoftDiceLoss

import os
import time

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable

from collections import OrderedDict

import numpy as np
from sklearn.metrics import accuracy_score

def classification(args):
    param = OrderedDict()

    dataloader = testData(args)
    model = resnet18()

    if args.gpu:
        assert (torch.cuda.is_available())
        param['gpu_ids'] = [args.gpu_ids]
        torch.cuda.set_device(device=param['gpu_ids'][0])

        model.cuda()

    if args.resume:
        network.load_state_dict(
            torch.load(os.path.join(args.save_path, '/resnet_18/%s_resnet.pth' % args.resume_num),
                map_location={'cuda:0' : 'cuda:0'}))

    acc_total = 0
    count = 0
    for i, batch_sample in enumerate(dataloader):
        input, class_label, mask, name = batch_sample['input_image'], batch_sample['class_label'], \
                                   batch_sample['mask'], batch_sample['image_name']
        if args.gpu:
            input, class_label = Variable(input).cuda(args.gpu_ids), Variable(class_label).cuda(args.gpu_ids).squeeze(1)

        class_pred = model(input).numpy()
        pred = np.argmax(class_pred,axis=1)
        acc = accuracy_score(pred, class_label)

        acc_total += acc

        count += 1

    print(f'accuracy: {acc_total/count }')

def segmentation(args):
    param = OrderedDict()

    dataloader = testData(args)
    model = UNet()

    if args.gpu:
        assert (torch.cuda.is_available())
        param['gpu_ids'] = [args.gpu_ids]
        torch.cuda.set_device(device=param['gpu_ids'][0])

        model.cuda()

    if args.resume:
        network.load_state_dict(
            torch.load(os.path.join(args.save_path, '/unet/%s_unet.pth' % args.resume_num),
                       map_location={'cuda:0': 'cuda:0'}))

    acc_total = 0
    count = 0
    for i, batch_sample in enumerate(dataloader):
        input, class_label, mask, name = batch_sample['input_image'], batch_sample['class_label'], \
                                         batch_sample['mask'], batch_sample['image_name']
        if args.gpu:
            input, mask = Variable(input).cuda(args.gpu_ids), Variable(mask).cuda(args.gpu_ids)

        mask_pred = model(input)

        acc = accuracy_score(mask_pred, mask)

        acc_total += acc

        count += 1

    print(f'accuracy: {acc_total / count}')

def test(args):
    if args.task == 'classification':
        classification(args)
    elif args.task == 'segmentation':
        segmentation(args)
    else:
        raise ValueError(f'The {args.task} is NOT EXIST. Please check out the task option.')




