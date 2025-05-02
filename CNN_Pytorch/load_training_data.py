import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import random


class Brats_dataset_batch(Dataset):
    def __init__(self, image_x, image_seg, patch_map, repeat=1):

        ###############
        self.img_H_2D = 168
        self.img_W_2D = 200
        self.img_H_2D_s = int((240 - self.img_H_2D) / 2)  # 36
        self.img_H_2D_e = int(240 - (240 - self.img_H_2D) / 2)  # 204
        self.img_W_2D_s = 26
        self.img_W_2D_e = 226

        self.img_H_3D = 168
        self.img_W_3D = 200
        self.img_H_3D_s = int((240 - self.img_H_3D) / 2)  # 34
        self.img_H_3D_e = int(240 - (240 - self.img_H_3D) / 2)  # 206
        self.img_W_3D_s = 26
        self.img_W_3D_e = 226

        '''
        self.img_H_3D = 172
        self.img_W_3D = 204
        self.img_H_3D_s = int((240 - self.img_H_3D) / 2)  # 34
        self.img_H_3D_e = int(240 - (240 - self.img_H_3D) / 2)  # 206
        self.img_W_3D_s = 24
        self.img_W_3D_e = 228
        '''

        # self.img_L_3D = 5

        self.img_num_slice = 155

        ####################

        self.patch_map = patch_map
        self.image_x = image_x
        self.image_seg = image_seg

        self.len = np.shape(self.patch_map)[0]
        self.repeat = repeat

    def __getitem__(self, i):
        index = i % self.len
        # print("i={},index={}".format(i, index))
        id_case = self.patch_map[index, 0]
        id_slice = self.patch_map[index, 1]

        temp_image = np.zeros((4, self.img_H_3D, self.img_W_3D), dtype=np.float32)
        temp_label = np.zeros((self.img_H_2D, self.img_W_2D), dtype=np.float32)

        '''
        for j in range(4):
            if id_slice - 3 < 0:
                # temp_image[(7-id_slice-4):7, :, :] = self.image_x[id_case, 0, 0:(id_slice+4), :, :]
                temp_image[(7*j+7-id_slice-4):(7*j+7), :, :] = self.image_x[id_case, j, 0:(id_slice+4), :, :]
            elif id_slice + 4 > 155:
                # temp_image[0:(155+3-id_slice), :, :] = self.image_x[id_case, 0, (id_slice-3):155, :, :]
                temp_image[(0+7*j):(7*j+155+3-id_slice), :, :] = self.image_x[id_case, j, (id_slice-3):155, :, :]
            else:
                # temp_image[0:7, :, :] = self.image_x[id_case, 0, (id_slice-3):(id_slice+4), :, :]
                temp_image[(7*j+0):(7*j+7), :, :] = self.image_x[id_case, j, (id_slice - 3):(id_slice + 4), :, :]
        '''

        temp_image[:, :, :] = self.image_x[id_case, :, id_slice, :, :]
        temp_label[:, :] = self.image_seg[id_case, id_slice, :, :]

        temp_image, temp_label = self.aug_sample(temp_image, temp_label)

        temp_label = self.process_ground_truth(temp_label)

        # temp_image = torch.from_numpy(temp_image)

        # temp_image = torch.from_numpy(temp_image.copy()),
        # temp_label = torch.from_numpy(temp_label.copy())

        return temp_image.copy(), temp_label.copy()


    def __len__(self):
        if self.repeat == None:
            data_len = 10000000
        else:
            data_len = self.len * self.repeat
        return data_len


    def aug_sample(self, temp_image, temp_label):
        '''
        flip_length = random.choice([0, 1])
        if flip_length == 1:
            temp_image = np.flip(temp_image, axis=1)
            '''

        flip_high = random.choice([0, 1])
        if flip_high == 1:
            temp_image = np.flip(temp_image, axis=1)
            temp_label = np.flip(temp_label, axis=0)

        flip_width = random.choice([0, 1])
        if flip_width == 1:
            temp_image = np.flip(temp_image, axis=2)
            temp_label = np.flip(temp_label, axis=1)

        return temp_image, temp_label


    def process_ground_truth(self, img_label):
        # size_img = np.shape(img_label)

        label_out = np.copy(img_label)
        label_out[label_out == 4] = 3

        return label_out


