import os
import nibabel as nib
import numpy as np
# import traceback
# import tensorflow as tf
from scipy.ndimage import uniform_filter


def count_pixel(arr, target):
    mask = (arr == target)
    arr_new = arr[mask]
    return arr_new.size


def normalization(data):
    _range = np.max(data) - np.min(data)
    if _range == 0:
        return_data = data - np.min(data)
    else:
        return_data = (data - np.min(data)) / _range
    return return_data


def normalization_signal(data):
    mask_data = data.copy()
    mask_data[mask_data != 0] = 1

    _range = np.max(data[data != 0]) - np.min(data[data != 0])
    if _range == 0:
        return_data = data - np.min(data[data != 0])
    else:
        return_data = (data - np.min(data[data != 0])) / _range
    return return_data * mask_data


def standardization(data):
    mu = np.mean(data)
    sigma = np.std(data)
    if sigma == 0:
        return_data = (data - mu)
    else:
        return_data = (data - mu) / sigma
    return return_data


def standardize_signal(data):
    mean_data = np.mean(data[data != 0])
    std_data = np.std(data[data != 0])

    if std_data == 0:
        standard_data = data - mean_data
    else:
        standard_data = (data - mean_data) / std_data

    return standard_data


def threshold_mask_v2(img_flair, img_t2, alpha):
    size_img = np.shape(img_flair)
    temp_flair = img_flair.copy()
    temp_t2 = img_t2.copy()

    mask_flair = img_flair.copy()
    mask_t2 = img_t2.copy()
    mask_flair[mask_flair != 0] = 1
    mask_t2[mask_t2 != 0] = 1
    mask_flair_t2 = mask_flair + mask_t2
    mask_flair_t2[mask_flair_t2 != 0] = 1

    temp_flair = uniform_filter(temp_flair, size=5, mode='mirror')
    temp_flair = temp_flair * mask_flair
    temp_t2 = uniform_filter(temp_t2, size=5, mode='mirror')
    temp_t2 = temp_t2 * mask_t2

    temp_flair = normalization_signal(temp_flair)
    temp_t2 = normalization_signal(temp_t2)

    flair_mean = np.mean(temp_flair[temp_flair != 0])
    flair_median = np.median(temp_flair[temp_flair != 0])
    flair_stdev = np.std(temp_flair[temp_flair != 0])
    flair_threshold = np.minimum(flair_mean, flair_median) + alpha * np.exp(-0.5 / flair_stdev)

    t2_mean = np.mean(temp_t2[temp_t2 != 0])
    t2_median = np.median(temp_t2[temp_t2 != 0])
    t2_stdev = np.std(temp_t2[temp_t2 != 0])
    t2_threshold = np.minimum(t2_mean, t2_median) + alpha * np.exp(-0.5 / t2_stdev)

    temp_flair[temp_flair <= flair_threshold] = 0
    temp_t2[temp_t2 <= t2_threshold] = 0

    intersect_flair_t2 = temp_flair * temp_t2
    intersect_flair_t2 = uniform_filter(intersect_flair_t2, size=5, mode='mirror')
    intersect_flair_t2 = intersect_flair_t2 * mask_flair_t2

    return standardize_signal(intersect_flair_t2)


class BratsTrainingSet(object):
    def __init__(self, training_location, roi_location, patient_ID):
        self.brats_name = patient_ID

        ###############
        self.img_H_2D = 168
        self.img_W_2D = 200
        self.img_H_2D_s = int((240 - self.img_H_2D) / 2)  # 36
        self.img_H_2D_e = int(240 - (240 - self.img_H_2D) / 2)  # 204
        self.img_W_2D_s = 26
        self.img_W_2D_e = 226

        self.img_H_3D = 168
        self.img_W_3D = 200
        self.img_H_3D_s = int((240 - self.img_H_3D) / 2)  # 36
        self.img_H_3D_e = int(240 - (240 - self.img_H_3D) / 2)  # 204
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

        self.img_num_slice = 155

        ####################

        self.num_floder = len(self.brats_name)
        self.num_total_image = 0

        self.image_x = np.zeros((self.num_floder, 4, self.img_num_slice, self.img_H_3D, self.img_W_3D), dtype=np.float32)
        self.image_seg = np.zeros((self.num_floder, self.img_num_slice, self.img_H_2D, self.img_W_2D), dtype=np.float32)

        self.patch_map = np.zeros((self.num_floder * self.img_num_slice, 2), dtype=np.int32)

        print('Start loading samples:')

        # read image
        for i in range(self.num_floder):
            path_flair = os.path.join(
                training_location + self.brats_name[i] + '/' + self.brats_name[i][
                                                           4:len(self.brats_name[i])] + '_flair.nii.gz')
            path_t2 = os.path.join(
                training_location + self.brats_name[i] + '/' + self.brats_name[i][
                                                               4:len(self.brats_name[i])] + '_t2.nii.gz')
            path_t1 = os.path.join(
                training_location + self.brats_name[i] + '/' + self.brats_name[i][4:len(self.brats_name[i])] + '_t1.nii.gz')
            path_t1ce = os.path.join(
                training_location + self.brats_name[i] + '/' + self.brats_name[i][
                                                           4:len(self.brats_name[i])] + '_t1ce.nii.gz')
            path_seg = os.path.join(
                training_location + self.brats_name[i] + '/' + self.brats_name[i][
                                                           4:len(self.brats_name[i])] + '_seg.nii.gz')
            # path_roi = os.path.join(roi_location + self.brats_name[i][4:len(self.brats_name[i])] + '.nii.gz')

            if i % 10 == 0 or i + 1 == self.num_floder:
                print('Read and process %.4f %%' % (100 * (i+1)/self.num_floder))

            seg_temp = nib.load(path_seg)
            seg_temp_arr = seg_temp.get_fdata()
            seg_temp_arr_sq = np.squeeze(seg_temp_arr)
            seg_temp_arr_sq = seg_temp_arr_sq[self.img_H_2D_s:self.img_H_2D_e, self.img_W_2D_s:self.img_W_2D_e, :]

            '''
            roi_temp = nib.load(path_roi)
            roi_temp_arr = roi_temp.get_fdata()
            roi_temp_arr_sq = np.squeeze(roi_temp_arr)
            roi_temp_arr_sq[roi_temp_arr_sq != 0] = 1
            # roi_temp_arr_sq = roi_temp_arr_sq[self.img_H_3D_s:self.img_H_3D_e, self.img_W_3D_s:self.img_W_3D_e, :]

            roi_temp_arr_sq = uniform_filter(roi_temp_arr_sq, size=[15, 15, 9], mode='constant', cval=0.0)
            roi_temp_arr_sq[roi_temp_arr_sq > 20 / (15 * 15 * 9)] = 1
            roi_temp_arr_sq[roi_temp_arr_sq <= 20 / (15 * 15 * 9)] = 0
            
            interest_slice = np.zeros(self.img_num_slice, dtype=np.int16)
            for j in range(self.img_num_slice):
                interest_slice[j] = np.max(roi_temp_arr_sq[:, :, j])
            '''

            flair_temp = nib.load(path_flair)
            flair_temp_arr = flair_temp.get_fdata()
            flair_temp_arr_sq = np.squeeze(flair_temp_arr)
            # flair_temp_arr_sq = flair_temp_arr_sq * roi_temp_arr_sq
            flair_temp_arr_sq = flair_temp_arr_sq[self.img_H_3D_s:self.img_H_3D_e, self.img_W_3D_s:self.img_W_3D_e, :]

            t2_temp = nib.load(path_t2)
            t2_temp_arr = t2_temp.get_fdata()
            t2_temp_arr_sq = np.squeeze(t2_temp_arr)
            # t2_temp_arr_sq = t2_temp_arr_sq * roi_temp_arr_sq
            t2_temp_arr_sq = t2_temp_arr_sq[self.img_H_3D_s:self.img_H_3D_e, self.img_W_3D_s:self.img_W_3D_e, :]

            t1_temp = nib.load(path_t1)
            t1_temp_arr = t1_temp.get_fdata()
            t1_temp_arr_sq = np.squeeze(t1_temp_arr)
            # t1_temp_arr_sq = t1_temp_arr_sq * roi_temp_arr_sq
            t1_temp_arr_sq = t1_temp_arr_sq[self.img_H_3D_s:self.img_H_3D_e, self.img_W_3D_s:self.img_W_3D_e, :]

            t1ce_temp = nib.load(path_t1ce)
            t1ce_temp_arr = t1ce_temp.get_fdata()
            t1ce_temp_arr_sq = np.squeeze(t1ce_temp_arr)
            # t1ce_temp_arr_sq = t1ce_temp_arr_sq * roi_temp_arr_sq
            t1ce_temp_arr_sq = t1ce_temp_arr_sq[self.img_H_3D_s:self.img_H_3D_e, self.img_W_3D_s:self.img_W_3D_e, :]

            background_flair = np.zeros(155, dtype=np.float32)
            for j in range(155):
                background_flair[j] = count_pixel(flair_temp_arr_sq[:, :, j], 0)
                background_flair[j] = background_flair[j] / self.img_H_3D / self.img_W_3D

            interest_slice = np.zeros(self.img_num_slice, dtype=np.int16)
            for j in range(155):
                if j <= 77 and background_flair[j] <= 0.90:
                    interest_slice[j] = 1
                if j > 77 and background_flair[j] < 1:
                    interest_slice[j] = 1

            for j in range(self.img_num_slice):
                if interest_slice[j] == 1:
                    self.patch_map[self.num_total_image, 0] = i
                    self.patch_map[self.num_total_image, 1] = j
                    self.num_total_image += 1

            flair_temp_arr_sq_norm = standardize_signal(flair_temp_arr_sq)
            t2_temp_arr_sq_norm = standardize_signal(t2_temp_arr_sq)
            t1_temp_arr_sq_norm = standardize_signal(t1_temp_arr_sq)
            t1ce_temp_arr_sq_norm = standardize_signal(t1ce_temp_arr_sq)
            # flair_t2_complementary = threshold_mask_v2(flair_temp_arr_sq, t2_temp_arr_sq, 0.1)

            for j in range(self.img_num_slice):
                self.image_x[i, 0, j, :, :] = flair_temp_arr_sq_norm[:, :, j]
                self.image_x[i, 1, j, :, :] = t2_temp_arr_sq_norm[:, :, j]
                self.image_x[i, 2, j, :, :] = t1_temp_arr_sq_norm[:, :, j]
                self.image_x[i, 3, j, :, :] = t1ce_temp_arr_sq_norm[:, :, j]
                self.image_seg[i, j, :, :] = seg_temp_arr_sq[:, :, j]

            ##### calculate mask
            # self.image_x[i, 0, :, :, :] = standardize_signal(t1_temp_arr_sq)
            # self.image_x[i, 1, :, :, :] = standardize_signal(t1ce_temp_arr_sq)
            # self.image_x[i, 2, :, :, :] = standardize_signal(flair_temp_arr_sq)
            # self.image_x[i, 3, :, :, :] = standardize_signal(t2_temp_arr_sq)

            # self.image_seg[i, :, :, :] = seg_temp_arr_sq

        self.patch_map = self.patch_map[0:self.num_total_image, :]

        ##########################################################################################
        # shuffle

        self.image_order = np.arange(self.num_total_image)
        np.random.seed(0)
        np.random.shuffle(self.image_order)

        patch_map_temp = np.copy(self.patch_map)

        for j in range(self.num_total_image):
            self.patch_map[j, :] = patch_map_temp[self.image_order[j], :]
        ##########################################################################################


