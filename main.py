import os
import argparse
import torch

from train import train

def main(args):
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True

    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)
        print('Create path : {}'.format(args.save_path))
    if args.operation=='train':
        train(args)
    elif args.operation=='test':
        print('')
    else:
        raise ValueError(f"The {args.operation} operation is NOT EXIST!")

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument('--dir', type=str, default='./ISIC2017_preprocessed')
    parser.add_argument('--image_path', type=str, default='ISIC-2017_Test_v2_Data_Resized')
    parser.add_argument('--mask_path', type=str, default='ISIC-2017_Test_v2_Part1_GroundTruth_Resized')
    parser.add_argument('--ground_truth', type=str, default='ISIC-2017_Test_v2_Part3_GroundTruth.csv')
    parser.add_argument('--batch_size', type=int, default=1)

    parser.add_argument('--save_path', type=str, default=r'D:/大学课程/大三下课程/人工智能实践/ISIC/result')

    parser.add_argument('--operation', type=str, default='train')
    parser.add_argument('--task', type=str, default='classification')

    parser.add_argument('--gpu', type=bool, default=True)
    parser.add_argument('--gpu_ids', type=int, default=0)

    parser.add_argument('--resume', type=bool, default=False)
    parser.add_argument('--resume_num', type=int, default=0)

    parser.add_argument('--epoch',type=int, default=100)

    args = parser.parse_args()

    main(args)
