import os
import BratsName
import pre_processing
import load_training_data
# import Network_BRATS
# import Simple_UNet
import Network_Old
import Testing_function
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

# from torchsummary import summary
# from thop import profile
# from torchstat import stat

branch_name = 'SEG_III1'

device = torch.device('cuda')

batch_size = 100
num_epoch = 50
plot_step = 50

# DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu") # 让torch判断是否使用GPU，建议使用GPU环境，因为会快很多

data_path_id = BratsName.Brats_Path_ID()
training_path = data_path_id.path_training_cases
validation_path = data_path_id.path_validation_cases
brats_training_ID = data_path_id.training_cases
brats_validation_ID = data_path_id.validation_cases
roi_training_path = data_path_id.path_roi_training
roi_test_path = data_path_id.path_roi_validation
# training
#################################################################
print('Start training: ')

total_patient = len(brats_training_ID)

############################################
print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
training_start_time = datetime.datetime.now()
#####################################################

training_images = pre_processing.BratsTrainingSet(training_path, roi_training_path, brats_training_ID)
print(training_images.num_total_image)

train_data = load_training_data.Brats_dataset_batch(image_x=training_images.image_x, image_seg=training_images.image_seg,
                                             patch_map=training_images.patch_map, repeat=1)
train_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True, num_workers=16)

net = Network_Old.Net_Old_version()
net = net.to(device)
net.train()

optimizer = torch.optim.Adam(net.parameters(), lr=0.01)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=int(training_images.num_total_image/batch_size * num_epoch)+100, eta_min=0.000001)

# loss_function = nn.BCELoss().to(device)
loss_function = nn.CrossEntropyLoss().to(device)

checkpoint_folder = './CheckPoint_' + branch_name + '/'
if not os.path.exists(checkpoint_folder):
    os.makedirs(checkpoint_folder)
checkpoint_path = os.path.join(checkpoint_folder, 'model')

for epoch in range(num_epoch):
    start = time.time()
    for step, (batch_image, batch_label) in enumerate(train_loader):
        # batch_image = Variable(batch_image, requires_grad=False)
        # batch_label = Variable(batch_label, requires_grad=False)
        batch_image = batch_image.to(device)
        batch_label = batch_label.type(torch.LongTensor)
        batch_label = batch_label.to(device)

        output_map = net(batch_image)
        loss_val = loss_function(output_map, batch_label)

        lr_present = optimizer.param_groups[0]['lr']

        if step % plot_step == 0:
            print('Epoch: ', epoch, ', Step: ', step, ', LR: ', lr_present,
                  ', Loss: %.4f' % loss_val.item())

        optimizer.zero_grad()
        loss_val.backward()
        optimizer.step()
        scheduler.step()

    if num_epoch - epoch <= 5:
        torch.save(net.state_dict(), checkpoint_path + str(epoch) + '.pkl')

    duration = time.time() - start
    print('Training duation: %.4f' % duration)


print('training finished')
print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
training_end_time = datetime.datetime.now()
print('training time(s):', (training_end_time - training_start_time).seconds)

###################################################################
# test
####################################################################
cnn_test = Network_Old.Net_Old_version()
cnn_test.load_state_dict(torch.load(checkpoint_path + str(num_epoch-1) + '.pkl', map_location=device))

cnn_test.eval()

print('Start testing: ')

Testing_function.TestBRATS(cnn_test, roi_training_path, roi_test_path, branch_name, device)
