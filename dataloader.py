import os.path
import random

import torchvision.transforms as transforms
from torch.utils.data import DataLoader

import torch
import numpy as np
import pandas as pd
from PIL import Image

import matplotlib.pyplot as plt

import argparse

def make_dataset(paths):

    data = pd.read_csv(paths)
    index = list(data.columns)

    image_name = np.array(data[index[0]][:-1])
    melanoma = data[index[1]][:-1]
    seborrheic_keratosis = data[index[2]][:-1]
    benign_nevi = data[index[3]][:-1]

    label = []
    label.append(melanoma)
    label.append(seborrheic_keratosis)
    label.append(benign_nevi)
    label = np.array(label).transpose(1,0)
    label = np.argmax(label,axis=1)

    label = label[:,np.newaxis]

    print("Label Information...")
    print(f"class label: {index[1]} {index[2]} {index[3]}")
    print(f"The number of {index[1]}: {np.sum(melanoma, axis=0)}")
    print(f"The number of {index[2]}: {np.sum(seborrheic_keratosis, axis=0)}")
    print(f"The number of {index[3]}: {np.sum(benign_nevi, axis=0)}")


    # for i in range(len(image_name)):
    #     if melanoma[i]+seborrheic_keratosis[i]+benign_nevi[i] != 1:
    #         raise ValueError(f"The image label are not unique, Please check out the {i}th row of the corresponding file!")
    #     else:
    #         if melanoma[i]==1:
    #             label.append(index[1])
    #         elif seborrheic_keratosis[i]==1:
    #             label.append(index[2])
    #         elif benign_nevi[i]==1:
    #             label.append(index[3])
    #         else:
    #             raise ValueError(f"The image label are not unique, Please check out the {i}th row of the corresponding file!")


    return image_name,label


class Dataset():
    def __init__(self, args):
        super(Dataset, self).__init__()
        self.dir = args.dir

        self.image_path = os.path.join(args.dir,args.image_path)
        self.mask_path = os.path.join(args.dir,args.mask_path)
        self.ground_truth = os.path.join(args.dir, args.ground_truth)

        self.image_name, self.image_label = make_dataset(self.ground_truth)

        self.transform = torch.from_numpy

    def __getitem__(self, item):

        image_name = self.image_name[item]
        image_path = os.path.join(self.image_path,image_name+'.jpg')
        mask_path = os.path.join(self.mask_path,image_name+'_segmentation.png')
        class_label = self.image_label[item]

        input_image = np.array(Image.open(image_path))/255
        mask = np.array(Image.open(mask_path))/255
        mask = mask[np.newaxis,:]



        input_image = self.transform(input_image).type(torch.FloatTensor).permute(2,0,1)
        mask = self.transform(mask).type(torch.FloatTensor)
        class_label = self.transform(class_label)

        return {'input_image':input_image,'class_label':class_label,'mask':mask,'image_name':image_name}

    def __len__(self):
        return len(self.image_name)

def trainData(args):
    trainDataset = Dataset(args)
    return DataLoader(dataset=trainDataset,batch_size=args.batch_size,shuffle=True,drop_last=True,num_workers=0)

def testData(args):
    testDataset = Dataset(args)
    return DataLoader(dataset=testDataset,batch_size=args.batch_size,shuffle=True,drop_last=True,num_workers=0)


def build_Dataset(args):
    dataset = Dataset(args)
    return DataLoader(dataset=dataset,batch_size=args.batch_size,shuffle=True,drop_last=True,num_workers=0)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument('--dir', type=str, default='.\ISIC2017_preprocessed')
    parser.add_argument('--image_path', type=str, default='ISIC-2017_Test_v2_Data_Resized')
    parser.add_argument('--mask_path', type=str, default='ISIC-2017_Test_v2_Part1_GroundTruth_Resized')
    parser.add_argument('--ground_truth', type=str, default='ISIC-2017_Test_v2_Part3_GroundTruth.csv')
    parser.add_argument('--batch_size', type=int, default=1)
    args = parser.parse_args()

    dataloader = build_Dataset(args)

    print("load data......")
    for batch_sample in dataloader:
        input, label, mask, name = batch_sample['input_image'], batch_sample['class_label'], \
                                   batch_sample['mask'], batch_sample['image_name']
        print("image_name:", name)
        print("input_shape:", input.shape)
        print("mask_shape:", mask.shape)
        print("label_shape:",label.shape)
        print("label:",label)


        fig = plt.figure()
        ax1 = fig.add_subplot(1,2,1)
        plt.imshow(input[0].permute(1,2,0), cmap='gray')

        ax2 = fig.add_subplot(1, 2, 2)
        plt.imshow(mask[0,0,:,:],cmap='gray')

        plt.show()




