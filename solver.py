import torch
import torch.nn as nn
import torch.optim as optim

import os
import numpy as np
from sklearn.metrics import accuracy_score

import sampler




class Solver:
    def __init__(self, args, test_dataloader):
        self.args = args
        self.test_dataloader = test_dataloader
        # print(len(self.test_dataloader))
        # exit(len(self.test_dataloader))
        self.bce_loss = nn.BCELoss()
        self.mse_loss = nn.MSELoss()
        self.ce_loss = nn.CrossEntropyLoss()

        self.sampler = sampler.AdversarySampler(self.args.budget)


    def read_data(self, dataloader, labels=True):
        if labels:
            while True:
                for img, label, _ in dataloader:
                    yield img, label
        else:
            while True:
                for img, _, _ in dataloader: #this INFINITELY yields images, interestingly! (oh it just looks around, when it sees everyting!)
                    yield img

    '''
    Now, we will simply run a new task model at the very end, that takes the EMBEDDINGS as inputs. 
    And check the performance.
    We can also: train this jointly. But that can be investigated later
    '''
    def train(self, querry_dataloader, task_model, vae, discriminator, unlabeled_dataloader):
        from tqdm import tqdm
        
        labeled_data = self.read_data(querry_dataloader)

        aa = 0
        while True:
            labeled_imgs, labels = next(labeled_data)
            aa +=1
            print(aa)

        unlabeled_data = self.read_data(unlabeled_dataloader, labels=False)

        optim_vae = optim.Adam(vae.parameters(), lr=5e-4)
        optim_task_model = optim.Adam(task_model.parameters(), lr=5e-4)
        optim_discriminator = optim.Adam(discriminator.parameters(), lr=5e-4)

        vae.train()
        discriminator.train()
        task_model.train()

        if self.args.cuda:
            vae = vae.cuda()
            discriminator = discriminator.cuda()
            task_model = task_model.cuda()
        
        change_lr_iter = self.args.train_iterations // 25

        for iter_count in tqdm(range(self.args.train_iterations)):
            if iter_count is not 0 and iter_count % change_lr_iter == 0:
                for param in optim_vae.param_groups:
                    param['lr'] = param['lr'] * 0.9
    
                for param in optim_task_model.param_groups:
                    param['lr'] = param['lr'] * 0.9 

                for param in optim_discriminator.param_groups:
                    param['lr'] = param['lr'] * 0.9 

            labeled_imgs, labels = next(labeled_data)
            unlabeled_imgs = next(unlabeled_data)

            if self.args.cuda:
                labeled_imgs = labeled_imgs.cuda()
                unlabeled_imgs = unlabeled_imgs.cuda()
                labels = labels.cuda()

            # task_model step
            # preds = task_model(labeled_imgs)
            # task_loss = self.ce_loss(preds, labels)
            # optim_task_model.zero_grad()
            # task_loss.backward()
            # optim_task_model.step()

            # VAE step
            for count in (range(self.args.num_vae_steps)):

                recon, z, mu, logvar = vae(labeled_imgs)
                unsup_loss = self.vae_loss(labeled_imgs, recon, mu, logvar, self.args.beta)
                unlab_recon, unlab_z, unlab_mu, unlab_logvar = vae(unlabeled_imgs)
                transductive_loss = self.vae_loss(unlabeled_imgs, 
                        unlab_recon, unlab_mu, unlab_logvar, self.args.beta)
            
                labeled_preds = discriminator(mu)
                unlabeled_preds = discriminator(unlab_mu)
                
                lab_real_preds = torch.ones(labeled_imgs.size(0))
                unlab_real_preds = torch.ones(unlabeled_imgs.size(0))
                    
                if self.args.cuda:
                    lab_real_preds = lab_real_preds.cuda()
                    unlab_real_preds = unlab_real_preds.cuda()

                dsc_loss = self.bce_loss(labeled_preds, lab_real_preds) + \
                        self.bce_loss(unlabeled_preds, unlab_real_preds)
                total_vae_loss = unsup_loss + transductive_loss + self.args.adversary_param * dsc_loss
                optim_vae.zero_grad()
                total_vae_loss.backward()
                optim_vae.step()

                # sample new batch if needed to train the adversarial network
                if count < (self.args.num_vae_steps - 1):
                    labeled_imgs, _ = next(labeled_data)
                    unlabeled_imgs = next(unlabeled_data)

                    if self.args.cuda:
                        labeled_imgs = labeled_imgs.cuda()
                        unlabeled_imgs = unlabeled_imgs.cuda()
                        labels = labels.cuda()

            # Discriminator step
            for count in (range(self.args.num_adv_steps)):
                with torch.no_grad():
                    _, _, mu, _ = vae(labeled_imgs)
                    _, _, unlab_mu, _ = vae(unlabeled_imgs)
                
                labeled_preds = discriminator(mu)
                unlabeled_preds = discriminator(unlab_mu)
                
                lab_real_preds = torch.ones(labeled_imgs.size(0))
                unlab_fake_preds = torch.zeros(unlabeled_imgs.size(0))

                if self.args.cuda:
                    lab_real_preds = lab_real_preds.cuda()
                    unlab_fake_preds = unlab_fake_preds.cuda()
                
                dsc_loss = self.bce_loss(labeled_preds, lab_real_preds) + \
                        self.bce_loss(unlabeled_preds, unlab_fake_preds)

                optim_discriminator.zero_grad()
                dsc_loss.backward()
                optim_discriminator.step()

                # sample new batch if needed to train the adversarial network
                if count < (self.args.num_adv_steps - 1):
                    labeled_imgs, _ = next(labeled_data)
                    unlabeled_imgs = next(unlabeled_data)

                    if self.args.cuda:
                        labeled_imgs = labeled_imgs.cuda()
                        unlabeled_imgs = unlabeled_imgs.cuda()
                        labels = labels.cuda()



            if iter_count % 1000 == 0:
                print('Current training iteration: {}'.format(iter_count))
                # print('Current task model loss: {:.4f}'.format(task_loss.item()))
                print('Current vae model loss: {:.4f}'.format(total_vae_loss.item()))
                print('Current discriminator model loss: {:.4f}'.format(dsc_loss.item()))

        # We need to generate the embeddings, and the dataset to do iteration over

        labeled_data = self.read_data(querry_dataloader)

        # for i, labeled_data_batch in enumerate(labeled_data ): # just need to encode these guys
        #     labeled_imgs, labels =  labeled_data_batch
        #     recon, z, mu, logvar = vae(labeled_imgs)
            # we can easily just do the inference here, and then keep doing this in turn


        # train the task model on the embeddings (of the labelled data)
        # also need to run for several epochs.

        # print(len(querry_dataloader))
        # print(len(labeled_data))

        NUM_EPOCHS = 1
        from tqdm import tqdm

        total_task_loss = 0
        total_examples = 0
        # for iter_count in tqdm(range(self.args.train_iterations)):
        for labeled_data in tqdm(querry_dataloader):
            # labeled_imgs, labels =  next(labeled_data)
            labeled_imgs, labels =  labeled_data

            if self.args.cuda:
                labeled_imgs = labeled_imgs.cuda()
                labels = labels.cuda()



            recon, z, mu, logvar = vae(labeled_imgs)

            # now, we just need to train a classifier on these datapoints; also need to associate the labels then
            # compute loss

            X = torch.cat((mu, logvar),1) #assuming batch size first, ambient space dimension second
            y = labels
            total_examples += len(X)

            preds = task_model(X)
            task_loss = self.ce_loss(preds, labels)
            total_task_loss += task_loss.item()
            optim_task_model.zero_grad()
            task_loss.backward()
            optim_task_model.step()

            if iter_count %100:
                print("Loss on iter_count {} is {}".format(100*iter_count, total_task_loss/len(total_examples )))

        final_accuracy = self.test_via_embedding(task_model, vae)



        return final_accuracy, vae, discriminator


    def sample_for_labeling(self, vae, discriminator, unlabeled_dataloader):
        querry_indices = self.sampler.sample(vae, 
                                             discriminator, 
                                             unlabeled_dataloader, 
                                             self.args.cuda)

        return querry_indices


    def test_via_embedding(self, task_model, vae):
        task_model.eval()
        vae.eval()
        total, correct = 0, 0
        for imgs, labels in self.test_dataloader:
            if self.args.cuda:
                imgs = imgs.cuda()

            with torch.no_grad():
                print("calling the test func")
                print(imgs.shape)
                recon, z, mu, logvar = vae(imgs)
                X = torch.cat((mu, logvar), 1)  # assuming batch size first, ambient space dimension second
                y = labels
                preds = task_model(X)

            preds = torch.argmax(preds, dim=1).cpu().numpy()
            correct += accuracy_score(labels, preds, normalize=False)
            total += imgs.size(0)
            print(total)
        return correct / total * 100


    def test(self, task_model):
        task_model.eval()
        total, correct = 0, 0
        for imgs, labels in self.test_dataloader:
            if self.args.cuda:
                imgs = imgs.cuda()

            with torch.no_grad():
                print("calling the test func")
                print(imgs.shape)
                preds = task_model(imgs)

            preds = torch.argmax(preds, dim=1).cpu().numpy()
            correct += accuracy_score(labels, preds, normalize=False)
            total += imgs.size(0)
        return correct / total * 100


    def vae_loss(self, x, recon, mu, logvar, beta):
        MSE = self.mse_loss(recon, x)
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        KLD = KLD * beta
        return MSE + KLD
