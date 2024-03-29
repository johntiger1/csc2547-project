import torch
from torchvision import datasets, transforms
import torch.utils.data.sampler  as sampler
import torch.utils.data as data

import numpy as np
import argparse
import random
import os

from custom_datasets import *
import model
import vgg
from solver import Solver
from utils import *
import arguments
from tqdm import tqdm

def cifar_transformer():
    return transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ])

def main(args):
    if args.dataset == 'cifar10':

        all_indices = set(np.arange(10000))
        initial_indices = random.sample(all_indices, 2000)
        test_sampler = data.sampler.SubsetRandomSampler(initial_indices)


        test_dataloader = data.DataLoader(
                datasets.CIFAR10(args.data_path, download=True, transform=cifar_transformer(), train=False),
            batch_size=args.batch_size, drop_last=False, sampler=test_sampler )

        '''
        The length is still (orig_length)
        But the times it will iterate through is only #random indices now
        '''
        # print(len(test_dataloader.dataset))
        # total = 0
        # for i, batch in enumerate(tqdm(test_dataloader)):
        #     print("good")
        #     total += len(batch[0])
        # print("total was {}".format(total))
        # construct a new dataset, from these iterated ones


        train_dataset = CIFAR10(args.data_path)

        args.num_images = 5000 #a type of curriculum learning could be useful here!
        args.budget = 250
        args.initial_budget = 2000
        args.num_classes = 10
    elif args.dataset == 'cifar100':
        test_dataloader = data.DataLoader(
                datasets.CIFAR100(args.data_path, download=True, transform=cifar_transformer(), train=False),
             batch_size=args.batch_size, drop_last=False)

        train_dataset = CIFAR100(args.data_path)

        args.num_images = 50000
        args.budget = 2500
        args.initial_budget = 5000
        args.num_classes = 100

    elif args.dataset == 'imagenet':
        test_dataloader = data.DataLoader(
                datasets.ImageFolder(args.data_path, transform=imagenet_transformer()),
            drop_last=False, batch_size=args.batch_size)

        train_dataset = ImageNet(args.data_path)

        args.num_images = 1281167
        args.budget = 64060
        args.initial_budget = 128120
        args.num_classes = 1000
    else:
        raise NotImplementedError

    all_indices = set(np.arange(args.num_images))
    initial_indices = random.sample(all_indices, args.initial_budget)
    sampler = data.sampler.SubsetRandomSampler(initial_indices)

    # dataset with labels available
    querry_dataloader = data.DataLoader(train_dataset, sampler=sampler, 
            batch_size=args.batch_size, drop_last=True)

    # print("in main")
    # print(len(querry_dataloader.dataset))
            
    args.cuda = args.cuda and torch.cuda.is_available()
    solver = Solver(args, test_dataloader)

    splits = [0.4,0.45,0.5] #splits actually has no effect on anything (just for formatting)!

    current_indices = list(initial_indices)

    accuracies = []

    # let's say we give it just 10% of the initial dataset, and say the rest was "unlabelled". We will see how it performs

    for split in tqdm(splits):
        # need to retrain all the models on the new images
        # re initialize and retrain the models
        # task_model = vgg.vgg16_bn(num_classes=args.num_classes)
        task_model = model.SimpleTaskModel(args.latent_dim*2, args.num_classes) # 2 for the different params (variance, mean); you are trying to predict
        vae = model.VAE(args.latent_dim)
        discriminator = model.Discriminator(args.latent_dim)

        # OK, so they do actually have a strong separation: we DO keep track of which indices have been sampled so far
        unlabeled_indices = np.setdiff1d(list(all_indices), current_indices)
        unlabeled_sampler = data.sampler.SubsetRandomSampler(unlabeled_indices)
        unlabeled_dataloader = data.DataLoader(train_dataset,
                sampler=unlabeled_sampler, batch_size=args.batch_size, drop_last=False)

        # train the models on the current data
        acc, vae, discriminator = solver.train(querry_dataloader,
                                               task_model,
                                               vae,
                                               discriminator,
                                               unlabeled_dataloader)


        print('Final accuracy with {}% of data is: {:.2f}'.format(int(split*100), acc))
        accuracies.append(acc)

        sampled_indices = solver.sample_for_labeling(vae, discriminator, unlabeled_dataloader)
        current_indices = list(current_indices) + list(sampled_indices)
        sampler = data.sampler.SubsetRandomSampler(current_indices)
        querry_dataloader = data.DataLoader(train_dataset, sampler=sampler,
                batch_size=args.batch_size, drop_last=True)

    torch.save(accuracies, os.path.join(args.out_path, args.log_name))

if __name__ == '__main__':
    import datetime
    from datetime import date

    print("running")
    print(datetime.datetime.now())
    # print(datetime.date.today())

    with open("{}_myfile.txt".format(str(datetime.datetime.now())), "a") as file:
        file.write("starting {}\n".format( datetime.datetime.now()))
        args = arguments.get_args()
        main(args)
        file.write("Finished it all! {}".format( datetime.datetime.now()))




