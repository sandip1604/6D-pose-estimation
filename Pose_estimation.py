from LineMOD import *
from Correspondance_Network import UNet
import torch
import re
import os
import cv2
import numpy as np
from Correspondance_Network import UNet
from Helper import *
from scipy.spatial.transform import Rotation as R
from torchvision import transforms, utils

def initial_pose_estimation(root_dir, classes, intrinsicCameraMatrix):

    
    training    = LineMODDataset(root_dir, 
                                classes=classes,
                                transform=transforms.Compose([transforms.ToTensor()]))

    corrNetwork = UNet(n_channels=3, 
                       out_channels_id=14, 
                       out_channels_uv=256, 
                       bilinear=True)
    corrNetwork.cuda()
    corrNetwork.load_state_dict(torch.load('correspondence_block.pt', 
                                           map_location=torch.device('cpu')))

    regex = re.compile(r'\d+')
    exception = 0
    
    for i in range(len(training)):

        imgAddress, img, _, _, _ = training[i]
        
        label = os.path.split(os.path.split(os.path.dirname(imgAddress))[0])[1]        
        ind   = regex.findall(os.path.split(imgAddress)[1])[0]
        img   = img.view(1, img.shape[0], img.shape[1], img.shape[2])
        
        idPred, uPred, vPred = corrNetwork(img.cuda())
        
        prediction = lambda x: torch.argmax(x, dim=1).squeeze().cpu()
        
        temp, upred, vpred  = prediction(idPred), prediction(uPred), prediction(vPred)
        
        coord2d = (temp == classes[label]).nonzero(as_tuple=True)

        path    = root_dir + label + "/predicted_pose/" + "info_" + str(ind) + ".txt"

        coord2d = torch.cat((coord2d[0].view(coord2d[0].shape[0], 1), 
                             coord2d[1].view(coord2d[1].shape[0], 1)), 1)
        

        keys    = torch.cat((upred[coord2d[:, 0], coord2d[:, 1]].view(-1, 1), 
                             vpred[coord2d[:, 0], coord2d[:, 1]].view(-1, 1)), 1)
        keys    = tuple(keys.numpy())
        dictionary = load_obj(root_dir + label + "/UV-XYZ_mapping")
        
        align2D, align3D = [], []

        for count, (u, v) in enumerate(keys):
            if (u, v) in dictionary:
                align2D.append(np.array(coord2d[count]))
                align3D.append(dictionary[(u, v)])

        if len(align2D) >= 4 or len(align3D) >= 4:
            _, rotVector, transVector, inliers = cv2.solvePnPRansac(np.array(align3D, dtype=np.float32),
                                                          np.array(align2D, dtype=np.float32), 
                                                          intrinsicCameraMatrix, 
                                                          distCoeffs=None,
                                                          iterationsCount=150, 
                                                          reprojectionError=1.0, 
                                                          flags=cv2.SOLVEPNP_P3P)
            rot, _ = cv2.Rodrigues(rotVector, jacobian=None)
            rot[np.isnan(rot)] = 1
            transVector[np.isnan(transVector)] = 1
            
            transVector = np.where(-100 < transVector, transVector, np.array([-100.]))
            transVector = np.where(transVector < 100, transVector, np.array([100.]))
            rot_trans   = np.append(rot, transVector, axis=1)

            np.savetxt(path, rot_trans)
        else:
            exception += 1
            rot_trans = np.ones((3, 4))
            rot_trans[:, 3] = 0
            np.savetxt(path, rot_trans)
        if i % 1000 == 0:
            print(str(i) + "/" + str(len(training)) + " Done")
            
    print("PnP Failed on: ", exception)