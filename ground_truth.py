from Helper import *
import os
import re
import cv2
import numpy as np
import random
import matplotlib.image as mpimg
from scipy.spatial.transform import Rotation as R


def create_GT_masks(background_dir, root_dir, classes, intrinsicCameraMatrix):
    """
    Helper function to create the Ground Truth ID,U and V masks
        Args:
        root_dir (str): path to the root directory of the dataset
        background_dir(str): path t
        intrinsicCameraMatrix (array): matrix containing camera intrinsics
        classes (dict) : dictionary containing classes and their ids
        Saves the masks to their respective directories
    """
    imageList = load_obj(root_dir + "all_images_adr")
    trainImagesInd = load_obj(root_dir + "train_images_indices")
    for i in range(len(trainImagesInd)):
        imageAddress = imageList[trainImagesInd[i]]
        label = os.path.split(os.path.split(os.path.dirname(imageAddress))[0])[1]
        regex = re.compile(r'\d+')
        ind = regex.findall(os.path.split(imageAddress)[1])[0]

        if i % 1000 == 0:
            print(str(i) + "/" + str(len(trainImagesInd)) + " finished!")

        image = cv2.imread(imageAddress)
        [IDMask, UMask, VMask] = [np.zeros((image.shape[0], image.shape[1]))]*3
 
        ID_mask_file = root_dir + label + \
            "/ground_truth/IDmasks/color" + str(ind) + ".png"
        U_mask_file = root_dir + label + \
            "/ground_truth/Umasks/color" + str(ind) + ".png"
        V_mask_file = root_dir + label + \
            "/ground_truth/Vmasks/color" + str(ind) + ".png"

        tranAddress = root_dir + label + "/data/tra" + str(ind) + ".tra"
        rotAddress = root_dir + label + "/data/rot" + str(ind) + ".rot"
        netTransformation = get_rot_tra(rotAddress, tranAddress)

        # Read point Point Cloud Data
        point_cloud_file = root_dir + label + "/object.xyz"
        pt_cld_data = np.loadtxt(point_cloud_file, skiprows=1, usecols=(0, 1, 2))
        ones = np.ones((pt_cld_data.shape[0], 1))
        homogenousCoord = np.append(pt_cld_data[:, :3], ones, axis=1)

        # Perspective Projection to obtain 2D coordinates for masks
        homogenous2D = intrinsicCameraMatrix @ (netTransformation @ homogenousCoord.T)
        coord2D      = ((np.floor(homogenous2D[:2, :] / homogenous2D[2, :])).T).astype(int)
        x2d, y2d     = np.clip(coord2D[:, 0], 0, 639), np.clip(coord2D[:, 1], 0, 479)

        IDMask[y2d, x2d] = classes[label]

        if i % 100 != 0:  # change background for every 99/100 images
            background_img_adr = background_dir + random.choice(os.listdir(background_dir))
            background_img     = cv2.imread(background_img_adr)
            background_img     = cv2.resize(background_img, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_AREA)
            background_img[y2d, x2d, :] = image[y2d, x2d, :]
            background_adr     = root_dir + label + "/changed_background/color" + str(ind) + ".png"
            mpimg.imsave(background_adr, background_img)

        # Generate Ground Truth UV Maps
        centre = np.mean(pt_cld_data, axis=0)
        length = np.sqrt((centre[0]-pt_cld_data[:, 0])**2 + (centre[1] - pt_cld_data[:, 1])**2 + (centre[2]-pt_cld_data[:, 2])**2)
        unit_vector = [(pt_cld_data[:, 0]-centre[0])/length, (pt_cld_data[:,1]-centre[1])/length, (pt_cld_data[:, 2]-centre[2])/length]
        U = 0.5 + (np.arctan2(unit_vector[2], unit_vector[0])/(2*np.pi))
        V = 0.5 - (np.arcsin(unit_vector[1])/np.pi)
        UMask[y2d, x2d] = U
        VMask[y2d, x2d] = V

        # Saving ID, U and V masks after using the fill holes function
        IDMask, UMask, VMask = fill_holes(IDMask, UMask, VMask)
        cv2.imwrite(ID_mask_file, IDMask)
        mpimg.imsave(U_mask_file, UMask, cmap='gray')
        mpimg.imsave(V_mask_file, VMask, cmap='gray')
        
#         try:
#             os.mkdir("UV_XYZ")
#         except FileExistsError:
#             pass
       
#         if i%100 == 0:
#             print("U:", np.shape(U_mask))
#             print("V:", np.shape(V_mask))
            
#             cv2.imwrite(os.path.join("UV_XYZ" , "U_" + str(i) +".jpg"), U_mask)
#             cv2.imwrite(os.path.join("UV_XYZ" , "V_" + str(i) +".jpg"), V_mask)
