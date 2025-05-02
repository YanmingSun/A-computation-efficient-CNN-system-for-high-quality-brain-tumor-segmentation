import os
import BratsName
import torch
import numpy as np
# import matplotlib.pyplot as plt
import time
import datetime
import nibabel as nib
from scipy.ndimage import uniform_filter
from scipy.ndimage import gaussian_filter


data_path_id = BratsName.Brats_Path_ID()
training_path = data_path_id.path_training_cases
validation_path = data_path_id.path_validation_cases
brats_training_ID = data_path_id.training_cases
brats_validation_ID = data_path_id.validation_cases

#################################################################

H_2D = 168
W_2D = 200
H_2D_s = int((240 - H_2D) / 2)  # 36
H_2D_e = int(240 - (240 - H_2D) / 2)  # 204
W_2D_s = 26
W_2D_e = 226

H_3D = 168
W_3D = 200
H_3D_s = int((240 - H_3D) / 2)  # 34
H_3D_e = int(240 - (240 - H_3D) / 2)  # 206
W_3D_s = 26
W_3D_e = 226

'''
H_3D = 172
W_3D = 204
H_3D_s = int((240 - H_3D) / 2)  # 34
H_3D_e = int(240 - (240 - H_3D) / 2)  # 206
W_3D_s = 24
W_3D_e = 228
'''

# L_3D = 5

# num_input_channels = 1

num_patch_per_patient = 155
# percent_interest_slice = 28741/155/(335+76)    #

total_patient = len(brats_training_ID)


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


def FusionPredictImage_v5(input_wt, input_net, input_et, patch_num=155):
    input_wt = np.squeeze(input_wt)
    input_net = np.squeeze(input_net)
    input_et = np.squeeze(input_et)

    size_img = np.shape(input_wt)

    eu_d0 = np.power(input_wt, 2) + np.power(input_net, 2) + np.power(input_et, 2)
    eu_d1 = np.power((input_wt - 1), 2) + np.power((input_net - 1), 2) + np.power(input_et, 2)
    eu_d2 = np.power((input_wt - 1), 2) + np.power(input_net, 2) + np.power(input_et, 2)
    eu_d4 = np.power((input_wt - 1), 2) + np.power(input_net, 2) + np.power((input_et - 1), 2)

    eu_d0 = np.reshape(eu_d0, (size_img[0], size_img[1], size_img[2], 1))
    eu_d1 = np.reshape(eu_d1, (size_img[0], size_img[1], size_img[2], 1))
    eu_d2 = np.reshape(eu_d2, (size_img[0], size_img[1], size_img[2], 1))
    eu_d4 = np.reshape(eu_d4, (size_img[0], size_img[1], size_img[2], 1))

    eu_d_matrix = np.concatenate((eu_d0, eu_d1, eu_d2, eu_d4), axis=3)

    label_all = np.argmin(eu_d_matrix, axis=3)
    label_all[label_all == 3] = 4

    # generate images
    image_predict_ALL = np.zeros((240, 240, 155), dtype=np.float32)

    for i in range(patch_num):
        image_predict_ALL[H_2D_s:H_2D_e, W_2D_s:W_2D_e, i] = label_all[i, :, :]

    return image_predict_ALL


def gaussian_lowpass3D(input_map, sigma, kernel_size):
    return gaussian_filter(input_map, sigma=sigma, mode='mirror', truncate=((kernel_size - 1) / 2 - 0.5) / sigma)
    # kernel size = 2 * int(truncate * sd + 0.5) + 1


def refine_binary_map_postv4(input_map, sigma, kernel_size, threshold_l, threshold_h):
    output_map = input_map.copy()
    map_f = gaussian_lowpass3D(input_map, sigma, kernel_size)
    map_f_h = map_f.copy()
    map_f_h[map_f_h >= threshold_h] = 1
    map_f_h[map_f_h < threshold_h] = 0
    map_f_l = map_f.copy()
    map_f_l[map_f_l <= threshold_l] = 0
    map_f_l[map_f_l > threshold_l] = 1
    map_f_l = 1 - map_f_l

    output_map = output_map + map_f_h
    output_map[output_map > 1] = 1
    output_map = output_map - map_f_l
    output_map[output_map < 0] = 0

    return output_map


def post_processing_v5(input_map, sigma=5, kernel_size=13, threshold_wt_l=0.2, threshold_wt_h=0.7,
                       threshold_tc_l=0.2, threshold_tc_h=0.5, threshold_et_l=0.1, threshold_et_h=0.8):
    wt_data = input_map.copy()
    et_data = input_map.copy()
    tc_data = input_map.copy()

    wt_data[wt_data != 0] = 1

    et_data[et_data < 4] = 0
    et_data[et_data == 4] = 1

    tc_data[tc_data == 2] = 0
    tc_data[tc_data > 0] = 1

    if np.sum(wt_data) < 2000:
        threshold_et_l = 0
        threshold_tc_l = 0
        threshold_wt_l = 0

    et_output = refine_binary_map_postv4(et_data, sigma, kernel_size, threshold_et_l, threshold_et_h)
    tc_output = refine_binary_map_postv4(tc_data, sigma, kernel_size, threshold_tc_l, threshold_tc_h)
    wt_output = refine_binary_map_postv4(wt_data, sigma, kernel_size, threshold_wt_l, threshold_wt_h)

    output_data = 4 * et_output + 1 * tc_output
    output_data[output_data == 5] = 4

    output_data = output_data + 2 * wt_output
    output_data[output_data == 3] = 1
    output_data[output_data == 6] = 4

    return output_data


####################
# test post
#################

def brats_test_post(cnn, input_dir_path, roi_dir_path, patient_ID, branch_name, device):
    path_flair = os.path.join(input_dir_path + patient_ID + '/' + patient_ID + '_flair.nii.gz')
    path_t2 = os.path.join(input_dir_path + patient_ID + '/' + patient_ID + '_t2.nii.gz')
    path_t1 = os.path.join(input_dir_path + patient_ID + '/' + patient_ID + '_t1.nii.gz')
    path_t1ce = os.path.join(input_dir_path + patient_ID + '/' + patient_ID + '_t1ce.nii.gz')
    # path_roi = os.path.join(roi_dir_path + patient_ID + '.nii.gz')

    '''
    roi_temp = nib.load(path_roi)
    roi_temp_arr = roi_temp.get_fdata()
    roi_temp_arr_sq = np.squeeze(roi_temp_arr)
    roi_temp_arr_sq[roi_temp_arr_sq != 0] = 1

    roi_temp_arr_sq = uniform_filter(roi_temp_arr_sq, size=[15, 15, 9], mode='constant', cval=0.0)
    roi_temp_arr_sq[roi_temp_arr_sq > 20 / (15 * 15 * 9)] = 1
    roi_temp_arr_sq[roi_temp_arr_sq <= 20 / (15 * 15 * 9)] = 0

    interest_slice = np.zeros(num_patch_per_patient, dtype=np.int16)
    for j in range(num_patch_per_patient):
        interest_slice[j] = np.max(roi_temp_arr_sq[:, :, j])
    '''

    flair_temp = nib.load(path_flair)
    flair_temp_arr = flair_temp.get_fdata()
    flair_temp_arr_sq = np.squeeze(flair_temp_arr)
    # flair_temp_arr_sq = flair_temp_arr_sq * roi_temp_arr_sq
    flair_temp_arr_sq = flair_temp_arr_sq[H_3D_s:H_3D_e, W_3D_s:W_3D_e, :]

    t2_temp = nib.load(path_t2)
    t2_temp_arr = t2_temp.get_fdata()
    t2_temp_arr_sq = np.squeeze(t2_temp_arr)
    # t2_temp_arr_sq = t2_temp_arr_sq * roi_temp_arr_sq
    t2_temp_arr_sq = t2_temp_arr_sq[H_3D_s:H_3D_e, W_3D_s:W_3D_e, :]

    t1_temp = nib.load(path_t1)
    t1_temp_arr = t1_temp.get_fdata()
    t1_temp_arr_sq = np.squeeze(t1_temp_arr)
    # t1_temp_arr_sq = t1_temp_arr_sq * roi_temp_arr_sq
    t1_temp_arr_sq = t1_temp_arr_sq[H_3D_s:H_3D_e, W_3D_s:W_3D_e, :]

    t1ce_temp = nib.load(path_t1ce)
    t1ce_temp_arr = t1ce_temp.get_fdata()
    t1ce_temp_arr_sq = np.squeeze(t1ce_temp_arr)
    # t1ce_temp_arr_sq = t1ce_temp_arr_sq * roi_temp_arr_sq
    t1ce_temp_arr_sq = t1ce_temp_arr_sq[H_3D_s:H_3D_e, W_3D_s:W_3D_e, :]

    affine_array = flair_temp.affine

    ######  Normalization and Standardization  ########
    x_test = np.zeros((4, num_patch_per_patient, H_3D, W_3D), dtype=np.float32)

    flair_temp_arr_sq_norm = standardize_signal(flair_temp_arr_sq)
    t2_temp_arr_sq_norm = standardize_signal(t2_temp_arr_sq)
    t1_temp_arr_sq_norm = standardize_signal(t1_temp_arr_sq)
    t1ce_temp_arr_sq_norm = standardize_signal(t1ce_temp_arr_sq)
    # flair_t2_complementary = threshold_mask_v2(flair_temp_arr_sq, t2_temp_arr_sq, 0.1)
    for j in range(num_patch_per_patient):
        x_test[0, j, :, :] = flair_temp_arr_sq_norm[:, :, j]
        x_test[1, j, :, :] = t2_temp_arr_sq_norm[:, :, j]
        x_test[2, j, :, :] = t1_temp_arr_sq_norm[:, :, j]
        x_test[3, j, :, :] = t1ce_temp_arr_sq_norm[:, :, j]

    x_test_batch = np.zeros((1, 4, H_3D, W_3D), dtype=np.float32)
    t_predict_image = np.zeros((240, 240, 155), dtype=np.float32)

    background_flair = np.zeros(155, dtype=np.float32)
    for j in range(155):
        background_flair[j] = count_pixel(flair_temp_arr_sq[:, :, j], 0)
        background_flair[j] = background_flair[j] / H_3D / W_3D

    interest_slice = np.zeros(155, dtype=np.int16)
    for j in range(155):
        if j <= 77 and background_flair[j] <= 0.90:
            interest_slice[j] = 1
        if j > 77 and background_flair[j] < 1:
            interest_slice[j] = 1

    for j in range(155):
        if interest_slice[j] == 1:
            x_test_batch[0, :, :, :] = x_test[:, j, :, :]

            '''
            x_test_batch = x_test_batch * 0
            
            for k in range(4):
                if j - 3 < 0:
                    # x_test_batch[0, (7-j-4):7, :, :] = x_test[0, 0:(j+4), :, :]
                    x_test_batch[0, (7*k+7 - j - 4):(7*k+7), :, :] = x_test[k, 0:(j + 4), :, :]
                elif j + 4 > 155:
                    # x_test_batch[0, 0:(155 + 3 - j), :, :] = x_test[0, (j - 3):155, :, :]
                    x_test_batch[0, (7*k+0):(7*k+155 + 3 - j), :, :] = x_test[k, (j - 3):155, :, :]
                else:
                    # x_test_batch[0, 0:7, :, :] = x_test[0, (j - 3):(j + 4), :, :]
                    x_test_batch[0, (0+7*k):(7+7*k), :, :] = x_test[k, (j - 3):(j + 4), :, :]
            '''

            x_test_batch_tensor = torch.from_numpy(x_test_batch)
            x_test_batch_tensor.to(device)

            temp_pred = cnn(x_test_batch_tensor)
            temp_pred = temp_pred.detach().numpy()
            temp_pred = np.squeeze(temp_pred)
            temp_pred = np.argmax(temp_pred, axis=0)

            t_predict_image[H_2D_s:H_2D_e, W_2D_s:W_2D_e, j] = temp_pred

    t_predict_image[t_predict_image == 3] = 4
    image_predict_all = t_predict_image

    image_post_all = post_processing_v5(image_predict_all, sigma=5, kernel_size=13, threshold_wt_l=0.2,
                                        threshold_wt_h=0.7,
                                        threshold_tc_l=0.2, threshold_tc_h=0.5, threshold_et_l=0.1, threshold_et_h=0.8)

    '''
    output_path_roi = './Test_' + branch_name + '_ROI/'
    if not os.path.exists(output_path_roi):
        os.makedirs(output_path_roi)
    '''

    output_path_all = './Test_' + branch_name + '_ALL/'
    if not os.path.exists(output_path_all):
        os.makedirs(output_path_all)

    output_path_post_all = './Test_' + branch_name + '_ALL_Post/'
    if not os.path.exists(output_path_post_all):
        os.makedirs(output_path_post_all)

    # image_predict_roi = image_predict_roi.astype(np.int16)
    image_predict_all = image_predict_all.astype(np.int16)
    image_post_all = image_post_all.astype(np.int16)

    # new_image_predict_roi = nib.Nifti1Image(image_predict_roi, affine_array)
    new_image_predict_all = nib.Nifti1Image(image_predict_all, affine_array)
    new_image_post_all = nib.Nifti1Image(image_post_all, affine_array)

    # nib.save(new_image_predict_roi, os.path.join(output_path_roi + patient_ID + '.nii.gz'))
    nib.save(new_image_predict_all, os.path.join(output_path_all + patient_ID + '.nii.gz'))
    nib.save(new_image_post_all, os.path.join(output_path_post_all + patient_ID + '.nii.gz'))

    return np.sum(interest_slice)


###############################################################
# test training-set post
##############################################################

def calculate_score_v5(predict_image, seg_image):
    predict_image = predict_image.astype(np.float32)
    seg_image = seg_image.astype(np.float32)

    shape_image = np.shape(seg_image)

    ###############
    predict_ET = np.copy(predict_image)
    predict_ET[predict_ET < 4] = 0
    predict_ET[predict_ET == 4] = 1
    seg_ET = np.copy(seg_image)
    seg_ET[seg_ET < 4] = 0
    seg_ET[seg_ET == 4] = 1

    predict_WT = np.copy(predict_image)
    predict_WT[predict_WT > 0] = 1
    seg_WT = np.copy(seg_image)
    seg_WT[seg_WT > 0] = 1

    predict_TC = np.copy(predict_image)
    predict_TC[predict_TC == 2] = 0
    predict_TC[predict_TC > 0] = 1
    seg_TC = np.copy(seg_image)
    seg_TC[seg_TC == 2] = 0
    seg_TC[seg_TC > 0] = 1

    score = np.zeros((1, 6), dtype=np.float32)

    ep = 0.0000001
    ################
    if np.sum(seg_ET) == 0 and np.sum(predict_ET) == 0:
        score[0, 0] = 1.0
        score[0, 3] = 1.0
    elif np.sum(seg_ET) == 0 and np.sum(predict_ET) > 0:
        score[0, 0] = 0.0
        score[0, 3] = np.nan
    else:
        score[0, 0] = 2 * np.sum(seg_ET * predict_ET) / (np.sum(seg_ET) + np.sum(predict_ET) + ep)
        score[0, 3] = np.sum(seg_ET * predict_ET) / (np.sum(seg_ET) + ep)

    ###########
    if np.sum(seg_WT) == 0 and np.sum(predict_WT) == 0:
        score[0, 1] = 1.0
        score[0, 4] = 1.0
    elif np.sum(seg_WT) == 0 and np.sum(predict_WT) > 0:
        score[0, 1] = 0.0
        score[0, 4] = np.nan
    else:
        score[0, 1] = 2 * np.sum(seg_WT * predict_WT) / (np.sum(seg_WT) + np.sum(predict_WT) + ep)
        score[0, 4] = np.sum(seg_WT * predict_WT) / (np.sum(seg_WT) + ep)

    #################
    if np.sum(seg_TC) == 0 and np.sum(predict_TC) == 0:
        score[0, 2] = 1.0
        score[0, 5] = 1.0
    elif np.sum(seg_TC) == 0 and np.sum(predict_TC) > 0:
        score[0, 2] = 0.0
        score[0, 5] = np.nan
    else:
        score[0, 2] = 2 * np.sum(seg_TC * predict_TC) / (np.sum(seg_TC) + np.sum(predict_TC) + ep)
        score[0, 5] = np.sum(seg_TC * predict_TC) / (np.sum(seg_TC) + ep)

    return score


def brats_test_trainingset_post(cnn, input_dir_path, roi_dir_path, patient_path, branch_name, device):
    path_flair = os.path.join(input_dir_path + patient_path + '/' + patient_path[4:len(patient_path)] + '_flair.nii.gz')
    path_t2 = os.path.join(input_dir_path + patient_path + '/' + patient_path[4:len(patient_path)] + '_t2.nii.gz')
    path_t1 = os.path.join(input_dir_path + patient_path + '/' + patient_path[4:len(patient_path)] + '_t1.nii.gz')
    path_t1ce = os.path.join(input_dir_path + patient_path + '/' + patient_path[4:len(patient_path)] + '_t1ce.nii.gz')
    path_seg = os.path.join(input_dir_path + patient_path + '/' + patient_path[4:len(patient_path)] + '_seg.nii.gz')

    # path_roi = os.path.join(roi_dir_path + patient_path[4:len(patient_path)] + '.nii.gz')

    seg_temp = nib.load(path_seg)
    seg_temp_arr = seg_temp.get_fdata()
    seg_temp_arr_sq = np.squeeze(seg_temp_arr)

    '''
    roi_temp = nib.load(path_roi)
    roi_temp_arr = roi_temp.get_fdata()
    roi_temp_arr_sq = np.squeeze(roi_temp_arr)
    roi_temp_arr_sq[roi_temp_arr_sq != 0] = 1

    roi_temp_arr_sq = uniform_filter(roi_temp_arr_sq, size=[15, 15, 9], mode='constant', cval=0.0)
    roi_temp_arr_sq[roi_temp_arr_sq > 20 / (15 * 15 * 9)] = 1
    roi_temp_arr_sq[roi_temp_arr_sq <= 20 / (15 * 15 * 9)] = 0

    interest_slice = np.zeros(num_patch_per_patient, dtype=np.int16)
    for j in range(num_patch_per_patient):
        interest_slice[j] = np.max(roi_temp_arr_sq[:, :, j])
    '''

    flair_temp = nib.load(path_flair)
    flair_temp_arr = flair_temp.get_fdata()
    flair_temp_arr_sq = np.squeeze(flair_temp_arr)
    # flair_temp_arr_sq = flair_temp_arr_sq * roi_temp_arr_sq
    flair_temp_arr_sq = flair_temp_arr_sq[H_3D_s:H_3D_e, W_3D_s:W_3D_e, :]

    t2_temp = nib.load(path_t2)
    t2_temp_arr = t2_temp.get_fdata()
    t2_temp_arr_sq = np.squeeze(t2_temp_arr)
    # t2_temp_arr_sq = t2_temp_arr_sq * roi_temp_arr_sq
    t2_temp_arr_sq = t2_temp_arr_sq[H_3D_s:H_3D_e, W_3D_s:W_3D_e, :]

    t1_temp = nib.load(path_t1)
    t1_temp_arr = t1_temp.get_fdata()
    t1_temp_arr_sq = np.squeeze(t1_temp_arr)
    # t1_temp_arr_sq = t1_temp_arr_sq * roi_temp_arr_sq
    t1_temp_arr_sq = t1_temp_arr_sq[H_3D_s:H_3D_e, W_3D_s:W_3D_e, :]

    t1ce_temp = nib.load(path_t1ce)
    t1ce_temp_arr = t1ce_temp.get_fdata()
    t1ce_temp_arr_sq = np.squeeze(t1ce_temp_arr)
    # t1ce_temp_arr_sq = t1ce_temp_arr_sq * roi_temp_arr_sq
    t1ce_temp_arr_sq = t1ce_temp_arr_sq[H_3D_s:H_3D_e, W_3D_s:W_3D_e, :]

    affine_array = flair_temp.affine

    ######  Normalization and Standardization  ########
    x_test = np.zeros((4, num_patch_per_patient, H_3D, W_3D), dtype=np.float32)

    flair_temp_arr_sq_norm = standardize_signal(flair_temp_arr_sq)
    t2_temp_arr_sq_norm = standardize_signal(t2_temp_arr_sq)
    t1_temp_arr_sq_norm = standardize_signal(t1_temp_arr_sq)
    t1ce_temp_arr_sq_norm = standardize_signal(t1ce_temp_arr_sq)
    # flair_t2_complementary = threshold_mask_v2(flair_temp_arr_sq, t2_temp_arr_sq, 0.1)
    for j in range(num_patch_per_patient):
        x_test[0, j, :, :] = flair_temp_arr_sq_norm[:, :, j]
        x_test[1, j, :, :] = t2_temp_arr_sq_norm[:, :, j]
        x_test[2, j, :, :] = t1_temp_arr_sq_norm[:, :, j]
        x_test[3, j, :, :] = t1ce_temp_arr_sq_norm[:, :, j]

    x_test_batch = np.zeros((1, 4, H_3D, W_3D), dtype=np.float32)
    t_predict_image = np.zeros((240, 240, 155), dtype=np.float32)

    background_flair = np.zeros(155, dtype=np.float32)
    for j in range(155):
        background_flair[j] = count_pixel(flair_temp_arr_sq[:, :, j], 0)
        background_flair[j] = background_flair[j] / H_3D / W_3D

    interest_slice = np.zeros(155, dtype=np.int16)
    for j in range(155):
        if j <= 77 and background_flair[j] <= 0.90:
            interest_slice[j] = 1
        if j > 77 and background_flair[j] < 1:
            interest_slice[j] = 1

    for j in range(155):
        if interest_slice[j] == 1:
            x_test_batch[0, :, :, :] = x_test[:, j, :, :]

            '''
            x_test_batch = x_test_batch * 0

            for k in range(4):
                if j - 3 < 0:
                    # x_test_batch[0, (7-j-4):7, :, :] = x_test[0, 0:(j+4), :, :]
                    x_test_batch[0, (7*k+7 - j - 4):(7*k+7), :, :] = x_test[k, 0:(j + 4), :, :]
                elif j + 4 > 155:
                    # x_test_batch[0, 0:(155 + 3 - j), :, :] = x_test[0, (j - 3):155, :, :]
                    x_test_batch[0, (7*k+0):(7*k+155 + 3 - j), :, :] = x_test[k, (j - 3):155, :, :]
                else:
                    # x_test_batch[0, 0:7, :, :] = x_test[0, (j - 3):(j + 4), :, :]
                    x_test_batch[0, (0+7*k):(7+7*k), :, :] = x_test[k, (j - 3):(j + 4), :, :]
            '''

            x_test_batch_tensor = torch.from_numpy(x_test_batch)
            x_test_batch_tensor.to(device)

            temp_pred = cnn(x_test_batch_tensor)
            temp_pred = temp_pred.detach().numpy()
            temp_pred = np.squeeze(temp_pred)
            temp_pred = np.argmax(temp_pred, axis=0)

            t_predict_image[H_2D_s:H_2D_e, W_2D_s:W_2D_e, j] = temp_pred

    t_predict_image[t_predict_image == 3] = 4
    image_predict_all = t_predict_image

    image_post_all = post_processing_v5(image_predict_all, sigma=5, kernel_size=13, threshold_wt_l=0.2,
                                        threshold_wt_h=0.7,
                                        threshold_tc_l=0.2, threshold_tc_h=0.5, threshold_et_l=0.1, threshold_et_h=0.8)

    output_path_tts_all = './TestTrainingSet_' + branch_name + '_ALL/'
    if not os.path.exists(output_path_tts_all):
        os.makedirs(output_path_tts_all)

    output_path_tts_post_all = './TestTrainingSet_' + branch_name + '_ALL_Post/'
    if not os.path.exists(output_path_tts_post_all):
        os.makedirs(output_path_tts_post_all)

    # image_predict_roi = image_predict_roi.astype(np.int16)
    image_predict_all = image_predict_all.astype(np.int16)
    image_post_all = image_post_all.astype(np.int16)

    # new_image_predict_roi = nib.Nifti1Image(image_predict_roi, affine_array)
    new_image_predict_all = nib.Nifti1Image(image_predict_all, affine_array)
    new_image_post_all = nib.Nifti1Image(image_post_all, affine_array)

    # nib.save(new_image_predict_roi, os.path.join(output_path_tts_roi + patient_path[4:len(patient_path)] + '.nii.gz'))
    nib.save(new_image_predict_all, os.path.join(output_path_tts_all + patient_path[4:len(patient_path)] + '.nii.gz'))
    nib.save(new_image_post_all, os.path.join(output_path_tts_post_all + patient_path[4:len(patient_path)] + '.nii.gz'))

    # score_roi = calculate_score(image_predict_roi, seg_temp_arr_sq)
    score_all = calculate_score_v5(image_predict_all, seg_temp_arr_sq)
    score_post_all = calculate_score_v5(image_post_all, seg_temp_arr_sq)

    print(patient_path, '%.4f, %.4f, %.4f, %.4f, %.4f, %.4f' % (
        score_all[0, 0], score_all[0, 1], score_all[0, 2], score_all[0, 3], score_all[0, 4], score_all[0, 5]))

    # return score_roi, score_all, score_post_all
    return score_all, score_post_all, np.sum(interest_slice)


class TestBRATS(object):
    def __init__(self, cnn, roi_training_path, roi_test_path, branch_name, device):
        self.cnn = cnn
        self.roi_training_path = roi_training_path
        self.roi_test_path = roi_test_path
        self.branch_name = branch_name
        #######################################################################################################
        # test
        #########################################################################################################
        start = time.time()

        num_validation_patient = len(brats_validation_ID)
        num_interest_slice_validation = np.zeros(num_validation_patient, dtype=np.int16)

        print('patient ID,  number of interest slices')

        for j in range(num_validation_patient):
            num_interest_slice_validation[j] = brats_test_post(self.cnn, validation_path, self.roi_test_path, brats_validation_ID[j], self.branch_name, device)
            print(brats_validation_ID[j], '  ', num_interest_slice_validation[j])

        print('mean of number of interest slice: ', np.mean(num_interest_slice_validation))

        duration = time.time() - start
        print('test duation: %.4f' % duration)


        #######################################################################################################
        # test
        #########################################################################################################
        start = time.time()

        num_training_cases = np.min([total_patient, 335])

        score_all = np.zeros((num_training_cases, 6), dtype=np.float32)  # Dice_ET, Dice_WT, Dice_TC
        score_post_all = np.zeros((num_training_cases, 6), dtype=np.float32)  # Dice_ET, Dice_WT, Dice_TC
        num_interest_slice_train = np.zeros(num_training_cases, dtype=np.float32)

        print('Scores (Test trainingset):')
        print('patient_ID, Dice_ET, Dice_WT, Dice_TC, Sensitivity_ET, Sensitivity_WT, Sensitivity_TC')

        for j in range(num_training_cases):
            score_all[j:j + 1, :], score_post_all[j:j + 1, :], num_interest_slice_train[
                j] = brats_test_trainingset_post(self.cnn, training_path, self.roi_training_path, brats_training_ID[j], self.branch_name, device)

        np.savetxt(('training_' + branch_name + '_Score_ALL.txt'), score_all, fmt='%f')
        np.savetxt(('training_' + branch_name + '_Score_post_ALL.txt'), score_post_all, fmt='%f')

        mean_score_all_training = np.nanmean(score_all, axis=0)
        mean_score_post_all_training = np.nanmean(score_post_all, axis=0)

        print('Mean scores (trainingset):')
        print('ALL')
        print('Dice_ET, Dice_WT, Dice_TC, Sensitivity_ET, Sensitivity_WT, Sensitivity_TC')
        print('%.4f, %.4f, %.4f, %.4f, %.4f, %.4f' % (
            mean_score_all_training[0], mean_score_all_training[1], mean_score_all_training[2], mean_score_all_training[3], mean_score_all_training[4],
            mean_score_all_training[5]))

        print('ALL_post')
        print('Dice_ET, Dice_WT, Dice_TC, Sensitivity_ET, Sensitivity_WT, Sensitivity_TC')
        print('%.4f, %.4f, %.4f, %.4f, %.4f, %.4f' % (
            mean_score_post_all_training[0], mean_score_post_all_training[1], mean_score_post_all_training[2], mean_score_post_all_training[3],
            mean_score_post_all_training[4], mean_score_post_all_training[5]))

        print('number of interest slice:')
        for j in range(num_training_cases):
            print(brats_training_ID[j], ': ', num_interest_slice_train[j])

        print('mean number of interest slice: ', np.mean(num_interest_slice_train))

        duration = time.time() - start
        print('test training set duation: %.4f' % duration)

