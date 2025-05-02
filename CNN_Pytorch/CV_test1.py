import os
import BratsName
import pre_processing
# import load_training_data
import Network_ThreePathways
# import Simple_UNet
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
import time
import datetime
import nibabel as nib
from scipy.ndimage import uniform_filter
from scipy.ndimage import gaussian_filter

import Testing_function

# from torchsummary import summary
# from thop import profile
# from torchstat import stat


branch_name_all = ['SEG_2CV1']
num_experiments = len(branch_name_all)

device = torch.device('cuda')

roi_training_path = '/scratch/sym/BrainTumorSegmentation/Code/Coarse_Fine_Seg/SEG_2C/TestTrainingSet_ROI_2D_ROI/'
roi_test_path = '/scratch/sym/BrainTumorSegmentation/Code/Coarse_Fine_Seg/SEG_2C/Test_ROI_2D_ROI/'


# DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu") # 让torch判断是否使用GPU，建议使用GPU环境，因为会快很多

data_path_id = BratsName.Brats_Path_ID()
training_path = data_path_id.path_training_cases
validation_path = data_path_id.path_validation_cases
brats_training_ID = data_path_id.training_cases
brats_validation_ID = data_path_id.validation_cases

num_input_channels = 4

num_patch_per_patient = 155
# percent_interest_slice = 28741/155/(335+76)    #

total_patient = len(brats_training_ID)

for i in range(num_experiments):
    branch_name = branch_name_all[i]
    ################################################################
    checkpoint_folder = './CheckPoint_' + branch_name + '/'
    checkpoint_path_et = os.path.join(checkpoint_folder, 'model_et49')

    cnn_et = Network_ThreePathways.TC_Pathway()
    cnn_et.load_state_dict(torch.load(checkpoint_path_et, map_location=device))

    ###################################################################
    # test
    ####################################################################
    cnn_et.eval()


    # print((int(training_images.num_total_image/total_patient), num_input_channels, L_3D, H_3D, W_3D))

    # input_r = torch.randn(int(training_images.num_total_image/total_patient), num_input_channels, L_3D, H_3D, W_3D)
    # macs, params = profile(cnn, inputs=(input_r, ))
    # print('macs: ', macs, ', params: ', params)

    # summary(cnn, (num_input_channels, L_3D, H_3D, W_3D))

    # stat(cnn, (int(training_images.num_total_image/total_patient), num_input_channels, L_3D, H_3D, W_3D))  # int(training_images.num_total_image/total_patient), num_input_channels


    Testing_function.TestBRATS(cnn_et, roi_training_path, roi_test_path, branch_name, device)
