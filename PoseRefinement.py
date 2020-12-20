import os
import numpy as np
import re
import cv2
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from Helper import *
from LineMOD import *
from scipy.spatial.transform import Rotation as R
from Correspondance_Network import UNet
from torchvision import transforms, utils
import matplotlib.image as mpimg
from torchvision import models
from torch.utils.data.sampler import SubsetRandomSampler
import torch.optim as optim
from torchsummary import summary
################################ Change Superclass ##################################
class PoseRefinerDataset(Dataset):

    """
    Args:
        root_dir (str): path to the dataset directory
        classes (dict): dictionary containing classes as key  
        transform : Transforms for input image
            """

    def __init__(self, root_dir, classes=None, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.classes = classes
        self.list_all_images = load_obj(root_dir + "all_images_adr")
        self.training_images_idx = load_obj(root_dir + "train_images_indices")

    def __len__(self):
        return len(self.training_images_idx)

    def __getitem__(self, i):
        imgAddress = self.list_all_images[self.training_images_idx[i]]
        label = os.path.split(os.path.split(os.path.dirname(imgAddress))[0])[1]
        regex = re.compile(r'\d+')
        ind = regex.findall(os.path.split(imgAddress)[1])[0]
        realImage = cv2.imread(self.root_dir + label +
                           '/pose_refinement/real/color' + str(ind) + ".png")
        renderedImage = cv2.imread(
            self.root_dir + label + '/pose_refinement/rendered/color' + str(ind) + ".png", cv2.IMREAD_GRAYSCALE)
        renderedImage = cv2.cvtColor(renderedImage.astype('uint8'), cv2.COLOR_GRAY2RGB)
        truePose = get_rot_tra(self.root_dir + label + '/data/rot' + str(ind) + ".rot",
                                self.root_dir + label + '/data/tra' + str(ind) + ".tra")
        predictedPoseAddress = self.root_dir + label + \
            '/predicted_pose/info_' + str(ind) + ".txt"
        predictedPose = np.loadtxt(predictedPoseAddress)
        if self.transform:
            realImage = self.transform(realImage)
            renderedImage = self.transform(renderedImage)
        return label, realImage, renderedImage, truePose, predictedPose

########################################################################################################

class Pose_Refiner(nn.Module):

    def __init__(self):
        super(Pose_Refiner, self).__init__()
        self.featureExtractorRealImage = nn.Sequential(*list(models.resnet18(pretrained=True,
                                                                              progress=True).children())[:9])   # extracts features of real images; equivalent to head E11 in paper.
        self.featureExtractorRenderedImage = nn.Sequential(*list(models.resnet18(pretrained=True,
                                                                                 progress=True).children())[:9])
        self.xyHead1 = nn.Linear(512, 253)
        self.xyHead2 = nn.Linear(256, 2)
        self.zHead = nn.Sequential(nn.Linear(512, 256),
                                   nn.ReLU(),
                                   nn.Linear(256, 1))
        self.rotationHead1 = nn.Linear(512, 252)
        self.rotationHead2 = nn.Linear(256, 4)

        self.relu = nn.ReLU()

    def _initialize_weights(self):
        # weight initialization with zeros in the begining
        nn.init.constant_(self.xyHead1.weight, 0.)
        nn.init.constant_(self.xyHead1.bias, 0.)

        weights = torch.zeros((2, 256))
        weights[0, 253] = torch.tensor(1.)
        weights[1, 254] = torch.tensor(1.)
        self.xyHead2.weight = nn.Parameter(weights)
        nn.init.constant_(self.xyHead2.bias, 0.)

        nn.init.constant_(self.zHead.weight, 0.)
        nn.init.constant_(self.zHead.bias, 0.)

        nn.init.constant_(self.rotationHead1.weight, 0.)
        nn.init.constant_(self.rotationHead1.bias, 0.)

        randWeights = torch.zeros((4, 256))
        randWeights[0, 252] = torch.tensor(1.)
        randWeights[1, 253] = torch.tensor(1.)
        randWeights[2, 254] = torch.tensor(1.)
        randWeights[3, 255] = torch.tensor(1.)
        self.rotationHead2.weight = nn.Parameter(randWeights)
        nn.init.constant_(self.rotationHead2.bias, 0.)

    def forward(self, image, rendered, pred_pose, bs=1):
        # extracting the feature vector f
        featureRGBImage = self.featureExtractorRealImage(image)
        featureRenderedImage = self.featureExtractorRenderedImage(rendered)
        featureRGBImage = featureRGBImage.view(bs, -1)
        featureRGBImage = self.relu(featureRGBImage)
        featureRenderedImage = featureRenderedImage.view(bs, -1)
        featureRenderedImage = self.relu(featureRenderedImage)
        featureDifference = featureRGBImage - featureRenderedImage
        
        # Z refinement head
        z = self.zHead(featureDifference)

        # XY refinement head
        refinementHeadXY1 = self.xyHead1(featureDifference)
        refinementHeadXY1 = self.relu(refinementHeadXY1)
        xPred = np.reshape(pred_pose[:, 0, 3], (bs, -1)).float().cuda()
        yPred = np.reshape(pred_pose[:, 1, 3], (bs, -1)).float().cuda()
        refinementHeadXY1 = torch.cat((refinementHeadXY1, xPred), 1)
        refinementHeadXY1 = torch.cat((refinementHeadXY1, yPred), 1)
        refinementHeadXY1 = torch.cat((refinementHeadXY1, z), 1)
        xy = self.xyHead2(refinementHeadXY1.cuda())

        # Rotation head
        refinementRoatation1 = self.rotationHead1(featureDifference)
        refinementRoatation1 = self.relu(refinementRoatation1)
        r = R.from_matrix(pred_pose[:, 0:3, 0:3])
        r = r.as_quat()
        r = np.reshape(r, (bs, -1))
        refinementRoatation1 = torch.cat(
            (refinementRoatation1, torch.from_numpy(r).float().cuda()), 1)
        rot = self.rotationHead2(refinementRoatation1)

        return xy, z, rot
    
############################################################################################################

def create_refinement_inputs(root_dir, classes, intrinsic_matrix):
    correspondenceBlock = UNet(
        n_channels=3, out_channels_id=14, out_channels_uv=256, bilinear=True)
    correspondenceBlock.cuda()
    correspondenceBlock.load_state_dict(torch.load(
        'correspondence_block.pt', map_location=torch.device('cpu')))

    trainData = LineMODDataset(root_dir,
                                classes=classes,
                                transform=transforms.Compose([transforms.ToTensor()]))

    upsampled = nn.Upsample(size=[240, 320], mode='bilinear',align_corners=False)

    regex = re.compile(r'\d+')
    count = 0
    
    for i in range(len(trainData)):
        if i % 1000 == 0:
            print(str(i) + "/" + str(len(trainData)) + " finished!")
        imgAddress, image, _, _, _ = trainData[i]

        label = os.path.split(os.path.split(os.path.dirname(imgAddress))[0])[1]
        ind = regex.findall(os.path.split(imgAddress)[1])[0]
        renderedAddress = root_dir + label + \
            "/pose_refinement/rendered/color" + str(ind) + ".png"
        addressImg = root_dir + label + \
            "/pose_refinement/real/color" + str(ind) + ".png"
        # find the object in the image using the idmask
        image = image.view(1, image.shape[0], image.shape[1], image.shape[2])
        idMaskPredicted, _, _ = correspondenceBlock(image.cuda())
        idMask = torch.argmax(idMaskPredicted, dim=1).squeeze().cpu()
        coordinates2d = (idMask == classes[label]).nonzero(as_tuple=True)
        if coordinates2d[0].nelement() != 0:
            coordinates2d = torch.cat((coordinates2d[0].view(
                coordinates2d[0].shape[0], 1), coordinates2d[1].view(coordinates2d[1].shape[0], 1)), 1)
            min_x = coordinates2d[:, 0].min()
            max_x = coordinates2d[:, 0].max()
            min_y = coordinates2d[:, 1].min()
            max_y = coordinates2d[:, 1].max()
            image = image.squeeze().transpose(1, 2).transpose(0, 2)
            obj_img = image[min_x:max_x+1, min_y:max_y+1, :]
            # saving in the correct format using upsampling
            obj_img = obj_img.transpose(0, 1).transpose(0, 2).unsqueeze(dim=0)
            obj_img = upsampled(obj_img)
            obj_img = obj_img.squeeze().transpose(0, 2).transpose(0, 1)
            mpimg.imsave(addressImg, obj_img.squeeze().numpy())

            # create rendering for an object
            croppedRenderedImage = create_rendering(
                root_dir, intrinsic_matrix, label, ind)
            renderedImg = torch.from_numpy(croppedRenderedImage)
            renderedImg = renderedImg.unsqueeze(dim=0)
            renderedImg = renderedImg.transpose(1, 3).transpose(2, 3)
            renderedImg = upsampled(renderedImg)
            renderedImg = renderedImg.squeeze().transpose(0, 2).transpose(0, 1)
            mpimg.imsave(renderedAddress, renderedImg.numpy())

        else:  # object not present in idmask prediction
            count += 1
            mpimg.imsave(renderedAddress, np.zeros((240, 320)))
            mpimg.imsave(addressImg, np.zeros((240, 320)))
    print("Number of outliers: ", count)
    
########################################################################################################

def train_pose_refinement(root_dir, classes, epochs=5):
    
    trainData = PoseRefinerDataset(root_dir, classes=classes,
                                    transform=transforms.Compose([
                                        transforms.ToPILImage(mode=None),
                                        transforms.Resize(size=(224, 224)),
                                        transforms.ToTensor(),
                                        transforms.Normalize([0.485, 0.456, 0.406], [
                                                             0.229, 0.224, 0.225]),
                                        transforms.ColorJitter(
                                            brightness=0, contrast=0, saturation=0, hue=0)
                                    ]))

    poseRefiner = Pose_Refiner()
    
    poseRefiner.cuda()
   
    # freeze resnet
    # pose_refiner.feature_extractor[0].weight.requires_grad = False

    batchSize = 1
    numWorkers = 0
    validSize = 0.2
    # obtain training indices that will be used for validation
    numTrain = len(trainData)
    indices = list(range(numTrain))
    np.random.shuffle(indices)
    split = int(np.floor(validSize * numTrain))
    train_ind, valid_ind = indices[split:], indices[:split]

    # define samplers for obtaining training and validation batches
    trainSampler = SubsetRandomSampler(train_ind)
    validSampler = SubsetRandomSampler(valid_ind)

    # prepare data loaders (combine dataset and sampler)
    trainLoader = torch.utils.data.DataLoader(trainData, batch_size=batchSize,
                                               sampler=trainSampler, num_workers=numWorkers)
    validLoader = torch.utils.data.DataLoader(trainData, batch_size=batchSize,
                                               sampler=validSampler, num_workers=numWorkers)

    optimizer = optim.Adam(poseRefiner.parameters(),
                           lr=1.5e-4, weight_decay=3e-5)

    # number of epochs to train the model
    n_epochs = epochs

    minimumValidationLoss = np.Inf  # track change in validation loss
    for epoch in range(1, n_epochs+1):
        torch.cuda.empty_cache()
        print("----- Epoch Number: ", epoch, "--------")

        # keep track of training and validation loss
        train_loss = 0.0
        valid_loss = 0.0

        ###################
        # train the model #
        ###################
        poseRefiner.train()
        for label, image, rendered, true_pose, pred_pose in trainLoader:
            # move tensors to GPU
            #summary(poseRefiner, [image.size(), rendered.size(), pred_pose.size()])
            image, rendered = image.cuda(), rendered.cuda()
            # clear the gradients of all optimized variables
            optimizer.zero_grad()
            # forward pass: compute predicted outputs by passing inputs to the model
            xy, z, rot = poseRefiner(image, rendered, pred_pose, batchSize)
            # convert rot quarternion to rotational matrix
            rot[torch.isnan(rot)] = 1  # take care of NaN and inf values
            rot[rot == float("Inf")] = 1
            xy[torch.isnan(xy)] == 0
            z[torch.isnan(z)] == 0

            rot = torch.tensor(
                (R.from_quat(rot.detach().cpu().numpy())).as_matrix())
            # update predicted pose
            pred_pose[:, 0:3, 0:3] = rot
            pred_pose[:, 0, 3] = xy[:, 0]
            pred_pose[:, 1, 3] = xy[:, 1]
            pred_pose[:, 2, 3] = z.squeeze()
            # fetch point cloud data
            pt_cld = fetch_ptcld_data(root_dir, label, batchSize)
            # calculate the batch loss
            loss = Matching_loss(pt_cld, true_pose, pred_pose, batchSize)
            # backward pass: compute gradient of the loss with respect to model parameters
            loss.backward()
            # perform a single optimization step (parameter update)
            optimizer.step()
            # update training loss
            train_loss += loss.item()

        ######################
        # validate the model #
        ######################
        poseRefiner.eval()
        for label, image, rendered, true_pose, pred_pose in validLoader:
            # move tensors to GPU
            image, rendered = image.cuda(), rendered.cuda()
            # forward pass: compute predicted outputs by passing inputs to the model
            xy, z, rot = poseRefiner(image, rendered, pred_pose, batchSize)
            rot[torch.isnan(rot)] = 1  # take care of NaN and inf values
            rot[rot == float("Inf")] = 1            
            xy[torch.isnan(xy)] == 0
            z[torch.isnan(z)] == 0
            # convert R quarternion to rotational matrix
            rot = torch.tensor(
                (R.from_quat(rot.detach().cpu().numpy())).as_matrix())
            # update predicted pose
            pred_pose[:, 0:3, 0:3] = rot
            pred_pose[:, 0, 3] = xy[:, 0]
            pred_pose[:, 1, 3] = xy[:, 1]
            pred_pose[:, 2, 3] = z.squeeze()
            # fetch point cloud data
            pt_cld = fetch_ptcld_data(root_dir, label, batchSize)
            # calculate the batch loss
            loss = Matching_loss(pt_cld, true_pose, pred_pose, batchSize)
            # update average validation loss
            valid_loss += loss.item()

        # calculate average losses
        train_loss = train_loss/len(trainLoader.sampler)
        valid_loss = valid_loss/len(validLoader.sampler)

        # print training/validation statistics
        print('Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f}'.format(
            epoch, train_loss, valid_loss))

        # save model if validation loss has decreased
        if valid_loss <= minimumValidationLoss:
            print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(
                minimumValidationLoss, valid_loss))
            torch.save(poseRefiner.state_dict(), 'pose_refiner.pt')
            minimumValidationLoss = valid_loss

########################################################################################################
def fetch_ptcld_data(root_dir, label, bs):
    # detch pt cld data for batchsize
    pointCloudData = []
    for i in range(bs):
        obj_dir = root_dir + label[i] + "/object.xyz"
        pointCloud = np.loadtxt(obj_dir, skiprows=1, usecols=(0, 1, 2))
        index = np.random.choice(pointCloud.shape[0], 3000, replace=False)
        pointCloudData.append(pointCloud[index, :])
    pointCloudData = np.stack(pointCloudData, axis=0)
    return pointCloudData

########################################################################################################
# no. of points is always 3000
def Matching_loss(pt_cld_rand, true_pose, pred_pose, bs, training=True):

    total_loss = torch.tensor([0.])
    total_loss.requires_grad = True
    for i in range(0, bs):
        pt_cld = pt_cld_rand[i, :, :].squeeze()
        TP = true_pose[i, :, :].squeeze()
        PP = pred_pose[i, :, :].squeeze()
        target = torch.tensor(pt_cld) @ TP[0:3, 0:3] + torch.cat(
            (TP[0, 3].view(-1, 1), TP[1, 3].view(-1, 1), TP[2, 3].view(-1, 1)), 1)
        output = torch.tensor(pt_cld) @ PP[0:3, 0:3] + torch.cat(
            (PP[0, 3].view(-1, 1), PP[1, 3].view(-1, 1), PP[2, 3].view(-1, 1)), 1)
        loss = (torch.abs(output - target).sum())/3000
        if loss < 100:
            total_loss = total_loss + loss
        else:  # so that loss isn't NaN
            total_loss = total_loss + torch.tensor([100.])

    return total_loss  