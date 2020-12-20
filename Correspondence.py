from LineMOD import *
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data.sampler import SubsetRandomSampler
from Correspondance_Network import UNet
import torch.optim as optim
import re
from scipy.spatial.transform import Rotation as R
from torchvision import transforms, utils
import time

def train_correspondence_block(root_dir, classes, numEpoch=20, batchSize = 3, validationSplit = 0.2):

    trainData = LineMODDataset(root_dir, classes=classes,
                                transform=transforms.Compose([transforms.ToTensor(),
                                transforms.ColorJitter(brightness=0, contrast=0, saturation=0, hue=0)]))

    
    num_workers = 0

    nTrain = len(trainData)
    ind = list(range(nTrain))
    np.random.shuffle(ind)
    split = int(validationSplit * nTrain)


    # prepare data loaders (combine dataset and sampler)
    trainer      = torch.utils.data.DataLoader(trainData, 
                                               batch_size=batchSize,
                                               sampler=SubsetRandomSampler(ind[split:]), 
                                               num_workers=num_workers)
    
    validator    = torch.utils.data.DataLoader(trainData, 
                                               batch_size=batchSize,
                                               sampler=SubsetRandomSampler(ind[:split]), 
                                               num_workers=num_workers)

    # architecture for correspondence block - 13 objects + backgound = 14 channels for ID masks
    corrNetwork          = UNet(n_channels=3, 
                                out_channels_id=14, 
                                out_channels_uv=256, 
                                bilinear=True)
    
    corrNetwork.cuda()

    # Loss functions for U V and ID
    # Corss entropy
    # Categorical
    # NLL
    # L2
    
    [lossU, lossV, lossID] = [nn.CrossEntropyLoss()]*3
    
    
    # RMSprop  XX
    # ADAM     VV
    
    optimizer = optim.Adam(corrNetwork.parameters(), 
                           lr=3e-4, 
                           weight_decay=3e-5)

    # Initialize the minimum loss to infinity
    minValidLoss = np.Inf
    
    corrTrainLoss, corrValidLoss = [], []
    
    def computer(sampler):
        Loss = 0.0
        for _, image, idmask, umask, vmask in sampler:
            image, idmask, umask, vmask = image.cuda(), idmask.cuda(), umask.cuda(), vmask.cuda()
            optimizer.zero_grad()
            idPred, uPred, vPred = corrNetwork(image)
            loss = lossID(idPred, idmask) + lossU(uPred, umask) + lossV(vPred, vmask)
            loss.backward()
            optimizer.step()
            Loss += loss.item()
        return Loss
    
    for epoch in range(0, numEpoch):
        start = time.time()
        torch.cuda.empty_cache()
        # keep track of training and validation loss
        trainLoss, validLoss = 0.0, 0.0

        print("==================== Epoch ", epoch+1, " ====================")
     
        corrNetwork.train()
        trainLoss = computer(trainer)
        corrNetwork.eval()
        validLoss = computer(validator)
        
        trainLoss /= len(trainer.sampler)
        validLoss /= len(validator.sampler)
        
        corrTrainLoss.append(trainLoss)
        corrValidLoss.append(validLoss)
        

        print('Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f}'.format(
            epoch, trainLoss, validLoss))

        if validLoss <= minValidLoss:
            print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(minValidLoss,validLoss))
            torch.save(corrNetwork.state_dict(),'correspondence_block.pt')
            minValidLoss = validLoss
            
        print("Time for epoch: {:.6f}".format(time.time() - start))

    print("==================== History ====================")
    print("correspondance train loss history", corrTrainLoss)
    print("correspondance valid loss history", corrValidLoss)
    plt.style.use('ggplot')
    plt.plot(corrTrainLoss, label = "Training Loss")
    plt.plot(corrValidLoss, label = "Validation Loss")
    plt.legend()
    plt.show()