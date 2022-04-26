from network.resnet import resnet18
from network.unet import UNet
from dataloader import trainData
from loss import SoftDiceLoss

import os
import time

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable

from collections import OrderedDict

def classification(args):
    param = OrderedDict()

    dataloader = trainData(args)
    model = resnet18()
    crossloss = nn.CrossEntropyLoss()
    if not os.path.exists(os.path.join(args.save_path,'/checkpoint')):
        os.mkdir(os.path.join(args.save_path,'/checkpoint'))

    checkpoint_file = open(os.path.join(args.save_path, '/checkpoint/resnet_checkpoint.txt'), 'a+')

    if args.gpu:
        assert (torch.cuda.is_available())
        param['gpu_ids'] = [args.gpu_ids]
        torch.cuda.set_device(device=param['gpu_ids'][0])

        model.cuda()

    if args.resume:
        network.load_state_dict(
            torch.load(os.path.join(args.save_path, '/resnet_18/%s_resnet.pth' % args.resume_num),
                map_location={'cuda:0' : 'cuda:0'}))



    model.train()
    optimizer = optim.Adam(model.parameters(), lr=1e-2, betas=(0.9, 0.999))

    start_num = args.resume_num+1 if args.resume else 0
    for epoch in range(start_num, args.epoch):
        start_time = time.time()
        loss_total = 0
        count = 0
        for i, batch_sample in enumerate(dataloader):
            input, class_label, mask, name = batch_sample['input_image'], batch_sample['class_label'], \
                                       batch_sample['mask'], batch_sample['image_name']
            if args.gpu:
                input, class_label = Variable(input).cuda(args.gpu_ids), Variable(class_label).cuda(args.gpu_ids).squeeze(1)

            class_pred = model(input)

            optimizer.zero_grad()

            loss = crossloss(class_pred,class_label)
            loss_total += loss.item()
            loss.backward()
            optimizer.step()

            count += 1

        end_time = time.time()
        print(f'epoch: {epoch}, loss: {loss_total/count }, consuming time: {end_time-start_time}')

        print(f'epoch: {epoch}, loss: {loss_total / count}, consuming time: {end_time - start_time}',
              file=checkpoint_file)

        torch.save(model.state_dict(),
                   os.path.join(args.save_path, '/resnet_18/','%s_resnet.pth' % epoch))

def segmentation(args):
    param = OrderedDict()

    dataloader = trainData(args)

    model = UNet(n_channels=3,n_classes=1)

    if not os.path.exists(os.path.join(args.save_path, 'checkpoint')):
        os.mkdir(os.path.join(args.save_path, 'checkpoint'))
    checkpoint_file = open(os.path.join(args.save_path, 'checkpoint/unet_checkpoint.txt'), 'a+')

    if args.gpu:
        assert (torch.cuda.is_available())
        param['gpu_ids'] = [args.gpu_ids]
        torch.cuda.set_device(device=param['gpu_ids'][0])

        model.cuda()

    if args.resume:
        network.load_state_dict(
            torch.load(os.path.join(args.save_path, '/unet/%s_unet.pth' % args.resume_num),
                map_location = {'cuda:0': 'cuda:0'}))



    model.train()
    diceloss = SoftDiceLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-2, betas=(0.9, 0.999))

    start_num = args.resume_num + 1 if args.resume else 0
    for epoch in range(start_num, args.epoch):
        start_time = time.time()
        loss_total = 0
        count = 0
        for i, batch_sample in enumerate(dataloader):
            input, class_label, mask, name = batch_sample['input_image'], batch_sample['class_label'], \
                                             batch_sample['mask'], batch_sample['image_name']
            if args.gpu:
                input, mask = Variable(input).cuda(args.gpu_ids), Variable(mask).cuda(args.gpu_ids)

            mask_pred = model(input)

            optimizer.zero_grad()
            loss = diceloss(mask_pred,mask)
            loss_total += loss.item()
            loss.backward()
            optimizer.step()

            count += 1

        end_time = time.time()
        print(f'epoch: {epoch}, loss: {loss_total / count}, consuming time: {end_time - start_time}')

        print(f'epoch: {epoch}, loss: {loss_total / count}, consuming time: {end_time - start_time}',
              file=checkpoint_file)

        torch.save(model.state_dict(),
                   os.path.join(args.save_path, '/unet/%s_unet.pth' % epoch))

def train(args):
    if args.task == 'classification':
        classification(args)
    elif args.task == 'segmentation':
        segmentation(args)
    else:
        raise ValueError(f'The {args.task} is NOT EXIST. Please check out the task option.')




