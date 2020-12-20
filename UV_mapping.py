import os
import numpy as np
from Helper import save_obj
from scipy.spatial.transform import Rotation as R

def create_UV_XYZ_dictionary(root_dir):

    classes = ['ape', 'phone', 'cam', 'duck','can', 'cat', 'driller','iron', 'eggbox', 'glue', 'holepuncher', 'benchviseblue', 'lamp']

    for label in classes:
        ptCloud = np.loadtxt(root_dir + label + "/object.xyz", skiprows=1, usecols=(0, 1, 2))
        # calculate u and v coordinates from the xyz point cloud file
        centre = np.mean(ptCloud, axis=0)
        length = np.sqrt((centre[0]-ptCloud[:, 0])**2 + (centre[1] - ptCloud[:, 1])**2 + (centre[2]-ptCloud[:, 2])**2)
        unit_vector = [(ptCloud[:, 0]-centre[0])/length, (ptCloud[:,1]-centre[1])/length, (ptCloud[:, 2]-centre[2])/length]

        uCoord = ((0.5 + (np.arctan2(unit_vector[2], unit_vector[0])/(2*np.pi))) * 255).astype(int)
        vCoord = ((0.5 - (np.arcsin(unit_vector[1])/np.pi)) * 255).astype(int)

        # Create a dictionary of the UV Mappnig
        dictionary = {}
        
        for u, v, xyz in zip(uCoord, vCoord, ptCloud):
            key = (u, v)
            if key not in dictionary:
                dictionary[key] = xyz
        
        # save it in pkl file      
        save_obj(dictionary, root_dir + label + "/UV-XYZ_mapping")