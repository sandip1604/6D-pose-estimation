import copy
import time
import argparse
import torch.nn as nn
from torchvision import transforms, utils
from Correspondance_Network import UNet
import re
import os
from Helper import *
import cv2
import torch
from PoseRefinement import *
from scipy.spatial.transform import Rotation as R
import matplotlib.image as mpimg

def test(numSample, intrinsicCameraMatrix, classes):
    parser = argparse.ArgumentParser(
        description='Script to create the Ground Truth masks')
    parser.add_argument("--root_dir", default="LineMOD_Dataset/",
                        help="path to dataset directory")
    args, unknown = parser.parse_known_args()

    root_dir = args.root_dir

    [classScore, classInst] = [dict.fromkeys(classes, 0)]*2 


    transform = transforms.Compose([transforms.ToPILImage(mode=None),
                                    transforms.Resize(size=(224, 224)),
                                    transforms.ToTensor(),
                                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    corrNet   = UNet(n_channels=3, 
                     out_channels_id=14,
                     out_channels_uv=256, 
                     bilinear=True)

    corrNet.load_state_dict(torch.load('correspondence_block.pt', 
                                       map_location=torch.device('cpu')))
    poseRefineNet = Pose_Refiner()

    poseRefineNet.load_state_dict(torch.load('pose_refiner.pt', 
                                             map_location=torch.device('cpu')))

    corrNet.cuda()
    poseRefineNet.cuda()
    poseRefineNet.eval()
    corrNet.eval()

    imageList = load_obj(root_dir + "all_images_adr")
    imageTestId = load_obj(root_dir + "test_images_indices")

    regex = re.compile(r'\d+')
    upsampled = nn.Upsample(size=[240, 320], mode='bilinear', align_corners=False)
    totalScore, totalFloatScore = 0, 0

    try:
        os.mkdir("Evaluated_Images")
    except FileExistsError:
        pass
    start = time.time()

    

    for i in range(numSample):

        imageAddress = imageList[imageTestId[i]]
        label        = os.path.split(os.path.split(os.path.dirname(imageAddress))[0])[1]
        idx          = regex.findall(os.path.split(imageAddress)[1])[0]

        gtPose       = get_rot_tra(root_dir + label + "/data/rot" + str(idx) + ".rot", 
                                   root_dir + label + "/data/tra" + str(idx) + ".tra")

        imageTest    = cv2.imread(imageAddress)
        vizImage     = copy.deepcopy(imageTest)
        imageTest    = cv2.resize(imageTest, 
                                  (imageTest.shape[1]//2, imageTest.shape[0]//2), 
                                  interpolation=cv2.INTER_AREA)

        imageTest    = torch.from_numpy(imageTest).type(torch.double).transpose(1, 2).transpose(0, 1)

        if len(imageTest.shape) != 4:
            imageView = lambda image: image.view(1, image.shape[0], image.shape[1], image.shape[2])
            imageTest = imageView(imageTest)

        # pass through correspondence block
        idMask, uMask, vMask = corrNet(imageTest.float().cuda())

        # convert the masks to 240,320 shape
        Pred    = lambda mask: torch.argmax(mask, dim=1).squeeze().cpu()
        idPred, uPred, vPred = Pred(idMask), Pred(uMask), Pred(vMask)
        
        coord2d = (idPred == classes[label]).nonzero(as_tuple=True)
        if coord2d[0].nelement() != 0:  # label is detected in the image
            coord2d = torch.cat((coord2d[0].view(coord2d[0].shape[0], 1), 
                                 coord2d[1].view(coord2d[1].shape[0], 1)), 1)
            uVal = uPred[coord2d[:, 0], coord2d[:, 1]]
            vVal = vPred[coord2d[:, 0], coord2d[:, 1]]
            Keys = torch.cat((uVal.view(-1, 1), vVal.view(-1, 1)), 1)
            Keys = tuple(Keys.numpy())
            dct  = load_obj(root_dir + label + "/UV-XYZ_mapping")
            mapping2d, mapping3d = [], []

            for count, (u, v) in enumerate(Keys):
                if (u, v) in dct:
                    mapping2d.append(np.array(coord2d[count]))
                    mapping3d.append(dct[(u, v)])

            if len(mapping2d) >= 6 or len(mapping3d) >= 6:
                _, rotVec, transVec, inliers = cv2.solvePnPRansac(np.array(mapping3d, dtype=np.float32),
                                                              np.array(mapping2d, dtype=np.float32), 
                                                              intrinsicCameraMatrix, 
                                                              distCoeffs=None,
                                                              iterationsCount=150, 
                                                              reprojectionError=1.0, 
                                                              flags=cv2.SOLVEPNP_P3P)
                rot, _ = cv2.Rodrigues(rotVec, jacobian=None)
                posePred = np.append(rot, transVec, axis=1)

            else:  # save an empty file
                posePred = np.zeros((3, 4))
#################################################################################################################

            img         = imageTest.squeeze().transpose(1, 2).transpose(0, 2)
            classImage  = img[coord2d[:, 0].min():coord2d[:, 0].max()+1, 
                          coord2d[:, 1].min():coord2d[:, 1].max()+1, :]
            ##########################################
            classImage  = upsampled(classImage.transpose(0, 1).transpose(0, 2).unsqueeze(dim=0))
            classImage  = classImage.squeeze().transpose(0, 2).transpose(0, 1)
            classImage  = transform(torch.as_tensor(classImage, dtype=torch.float32))
            ##########################################
            renderImage = create_rendering_eval(root_dir, intrinsicCameraMatrix, label, posePred)
            renderImage = torch.from_numpy(renderImage).unsqueeze(dim=0)
            renderImage = upsampled(renderImage.transpose(1, 3).transpose(2, 3)).squeeze()
            renderImage = transform(torch.as_tensor(renderImage, dtype=torch.float32))
            ##########################################
            if len(renderImage.shape) != 4:
                renderImage = imageView(renderImage)
            ##########################################
            if len(classImage.shape) != 4:
                classImage  = imageView(classImage)
            ##########################################
            posePred   = (torch.from_numpy(posePred)).unsqueeze(0)

            xy, z, rot = poseRefineNet(classImage.float().cuda(),
                                       renderImage.float().cuda(), 
                                       posePred)                      
            rot[torch.isnan(rot)]    = 1
            rot[rot == float("Inf")] = 1
            xy[torch.isnan(xy)]      = 0
            z[torch.isnan(z)]        = 0
            rot      = (R.from_quat(rot.detach().cpu().numpy())).as_matrix()
            posePred = posePred.squeeze().numpy()
            xy       = xy.squeeze()
            posePred[0:3, 0:3] = rot
            posePred[0, 3]     = xy[0]
            posePred[1, 3]     = xy[1]
            posePred[2, 3]     = z

            diameter = np.loadtxt(root_dir + label + "/distance.txt")
            ptCloud  = np.loadtxt(root_dir + label + "/object.xyz", 
                                  skiprows=1, 
                                  usecols=(0, 1, 2))
            vizTest = create_bounding_box(vizImage, posePred, ptCloud, intrinsicCameraMatrix)
            vizTest = create_bounding_box(vizTest, gtPose, ptCloud, intrinsicCameraMatrix, color = (0,255,0))

            score      = ADD_score(ptCloud, gtPose, posePred, diameter)
            floatScore = ADD_score_norm(ptCloud, gtPose, posePred, diameter)
            
            path = "Evaluated_Images/" + str(floatScore) + "_score_" + str(i) +'.png'
            cv2.imwrite(path, vizTest)
            totalScore += score
            totalFloatScore += (1 - floatScore)
            classScore[label] += score
        else:
            classScore[label] += 0

        classInst[label] += 1

    print("Average Evaluation Time:", (time.time() - start)/numSample)
    print("ADD Score for all testing images is: ",totalScore/numSample)
    print("ADD Score for all testing images is: ",totalFloatScore/numSample)
    return classScore, classInst