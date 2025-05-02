import os
import nibabel as nib
# import skimage.io as io
import numpy as np
import random
from skimage.measure import compare_ssim

def count_pixel(arr, target):
    mask = (arr == target)
    arr_new = arr[mask]
    return arr_new.size


def count_contour_ssim(image):
    if image.max() > 0:
        v_start = 0
        v_end = 0
        h_start = 0
        h_end = 0
        shape_image = np.shape(image)
        for k in range(shape_image[0]):
            if image[k, :].max() > 0:
                v_start = k
                break
        for k in range(shape_image[0]):
            if image[int(shape_image[0] - k - 1), :].max() > 0:
                v_end = int(shape_image[0] - k - 1)
                break
        for k in range(shape_image[1]):
            if image[:, k].max() > 0:
                h_start = k
                break
        for k in range(shape_image[1]):
            if image[:, int(shape_image[1] - k - 1)].max() > 0:
                h_end = int(shape_image[1] - k - 1)
                break
        v_start = min([shape_image[0] - 1 - v_end + 1, v_start - 0 + 1]) - 1
        v_end = shape_image[0] - 1 - v_start

        image_s_shape = [int(v_end - v_start + 1), int(h_end - h_start + 1)]

        if image_s_shape[0] > 15 and image_s_shape[1] > 15:
            #            image_s = np.zeros((image_s_shape[0], image_s_shape[1]), dtype=np.float)
            image_s_up1 = np.zeros((int(image_s_shape[0] / 2), image_s_shape[1]), dtype=np.float)
            image_s_down1 = np.zeros((int(image_s_shape[0] / 2), image_s_shape[1]), dtype=np.float)
            image_s_up2 = np.zeros((int(image_s_shape[0] / 2), image_s_shape[1]), dtype=np.float)
            image_s_down2 = np.zeros((int(image_s_shape[0] / 2), image_s_shape[1]), dtype=np.float)

            #            image_s[:, :] = image[v_start:v_end+1, h_start:h_end+1]
            image_s_up1[:, :] = image[v_start:int(shape_image[0] / 2), h_start:h_end + 1]
            image_s_up1[:, :] = FlipImage_UpDown(image_s_up1[:, :])
            image_s_down1[:, :] = image[int(shape_image[0] / 2):v_end + 1, h_start:h_end + 1]

            image_s_up2[:, :] = image_s_up1[:, :]
            image_s_down2[:, :] = image_s_down1[:, :]

            image_s_up1[image_s_up1 != 0] = 255
            image_s_down1[image_s_down1 != 0] = 255

            #            print(np.shape(image_s_up1))
            #            print(np.shape(image_s_down1))
            contour_ssim = compare_ssim(image_s_up1, image_s_down1)

            signal_ssim = compare_ssim(image_s_up2, image_s_down2)

            return contour_ssim, signal_ssim
        else:
            return 1, 1
    else:
        return 1, 1




def extract_patch(input_data, image_high, image_width, image_high_s, image_high_e, image_width_s, image_width_e):
    shape_data = np.shape(input_data)
    image_output = np.zeros((image_high, image_width, shape_data[2]), dtype=np.float)

    for j in range(shape_data[2]):
        image_output[:, :, j] = input_data[image_high_s:image_high_e, image_width_s:image_width_e, j]

    return image_output

def Generate_RandomImageOrder(num_floder):
    num_floder = int(num_floder)
    num_totel_image = int(num_floder*155)

    data = np.zeros((num_totel_image, 4), dtype=np.int)

    data[:, 0] = range(num_totel_image)

    for i in range(num_floder):
        for ii in range(155):
            data[i * 155 + ii, 1] = i
            data[i * 155 + ii, 2] = ii

    for i in range(num_totel_image):
        if data[i, 2] >= 0 and data[i, 2] <= 30:
            data[i, 3] = 0
        elif data[i, 2] >= 31 and data[i, 2] <= 61:
            data[i, 3] = 1
        elif data[i, 2] >= 62 and data[i, 2] <= 92:
            data[i, 3] = 2
        elif data[i, 2] >= 93 and data[i, 2] <= 123:
            data[i, 3] = 3
        else:
            data[i, 3] = 4

    order_class0 = []
    order_class1 = []
    order_class2 = []
    order_class3 = []
    order_class4 = []

    for i in range(num_totel_image):
        if data[i, 3] == 0:
            order_class0.append(data[i, 0])
        elif data[i, 3] == 1:
            order_class1.append(data[i, 0])
        elif data[i, 3] == 2:
            order_class2.append(data[i, 0])
        elif data[i, 3] == 3:
            order_class3.append(data[i, 0])
        else:
            order_class4.append(data[i, 0])

    random.shuffle(order_class0)
    random.shuffle(order_class1)
    random.shuffle(order_class2)
    random.shuffle(order_class3)
    random.shuffle(order_class4)

    output_order = np.zeros(num_totel_image, dtype=np.int)

    NumberImage_PerClass = int(num_totel_image / 5)

    for i in range(NumberImage_PerClass):
        output_order[i * 5] = order_class0[i]
        output_order[i * 5 + 1] = order_class1[i]
        output_order[i * 5 + 2] = order_class2[i]
        output_order[i * 5 + 3] = order_class3[i]
        output_order[i * 5 + 4] = order_class4[i]

#    np.savetxt(txt_name, output_order, fmt='%d')

    return output_order


def FlipImage_UpDown(image):
    shape_image = np.shape(image)
    temp_image = np.zeros((shape_image[0], shape_image[1]), dtype=np.float)
    for i in range(shape_image[0]):
        temp_image[i, :] = image[shape_image[0]-1-i, :]
    return temp_image

def FlipImage_LeftRight(image):
    shape_image = np.shape(image)
    temp_image = np.zeros((shape_image[0], shape_image[1]), dtype=np.float)
    for i in range(shape_image[1]):
        temp_image[:, i] = image[:, shape_image[1]-1-i]
    return temp_image

def normalization(data):
    _range = np.max(data) - np.min(data)
    if _range == 0:
        return_data = data - np.min(data)
    else:
        return_data = (data - np.min(data)) / _range
    return return_data


def standardization(data):
    mu = np.mean(data)
    sigma = np.std(data)
    if sigma == 0:
        return_data = (data - mu)
    else:
        return_data = (data - mu) / sigma
    return return_data


class BratsTrainingSet:
    def __init__(self):
        self.brats_name = [
            'LGG/Brats18_TCIA10_103_1',
            'LGG/Brats18_TCIA12_470_1',
            'HGG/Brats18_TCIA02_171_1',
            'LGG/Brats18_TCIA13_624_1',
            'HGG/Brats18_CBICA_ASU_1',
            'HGG/Brats18_2013_23_1',
            'HGG/Brats18_TCIA02_368_1',
            'HGG/Brats18_TCIA08_319_1',
            'HGG/Brats18_TCIA02_331_1',
            'HGG/Brats18_2013_20_1',
            'LGG/Brats18_2013_0_1',
            'LGG/Brats18_TCIA10_387_1',
            'LGG/Brats18_TCIA13_653_1',
            'HGG/Brats18_TCIA01_412_1',
            'HGG/Brats18_CBICA_ABE_1',
            'HGG/Brats18_2013_22_1',
            'HGG/Brats18_TCIA02_283_1',
            'HGG/Brats18_CBICA_AVJ_1',
            'HGG/Brats18_CBICA_ABY_1',
            'LGG/Brats18_2013_8_1',
            'LGG/Brats18_2013_29_1',
            'HGG/Brats18_CBICA_AUR_1',
            'HGG/Brats18_2013_5_1',
            'LGG/Brats18_TCIA10_351_1',
            'HGG/Brats18_TCIA06_165_1',
            'LGG/Brats18_TCIA10_310_1',
            'HGG/Brats18_TCIA01_190_1',
            'HGG/Brats18_CBICA_ATP_1',
            'HGG/Brats18_CBICA_ATB_1',
            'HGG/Brats18_TCIA05_444_1',
            'HGG/Brats18_TCIA04_328_1',
            'HGG/Brats18_TCIA01_235_1',
            'HGG/Brats18_TCIA02_300_1',
            'HGG/Brats18_CBICA_ASY_1',
            'HGG/Brats18_TCIA04_149_1',
            'HGG/Brats18_CBICA_AOO_1',
            'LGG/Brats18_TCIA13_621_1',
            'LGG/Brats18_TCIA13_623_1',
            'HGG/Brats18_TCIA02_471_1',
            'LGG/Brats18_TCIA10_625_1',
            'LGG/Brats18_TCIA10_330_1',
            'HGG/Brats18_TCIA02_430_1',
            'HGG/Brats18_CBICA_ASN_1',
            'LGG/Brats18_TCIA10_282_1',
            'HGG/Brats18_TCIA01_180_1',
            'HGG/Brats18_TCIA03_338_1',
            'LGG/Brats18_TCIA09_451_1',
            'HGG/Brats18_CBICA_ANZ_1',
            'HGG/Brats18_TCIA08_278_1',
            'HGG/Brats18_TCIA06_409_1',
            'HGG/Brats18_TCIA03_498_1',
            'HGG/Brats18_TCIA03_199_1',
            'HGG/Brats18_TCIA03_133_1',
            'HGG/Brats18_TCIA08_162_1',
            'HGG/Brats18_CBICA_ABO_1',
            'HGG/Brats18_2013_3_1',
            'HGG/Brats18_TCIA01_390_1',
            'HGG/Brats18_TCIA06_603_1',
            'HGG/Brats18_TCIA02_374_1',
            'HGG/Brats18_TCIA01_411_1',
            'HGG/Brats18_2013_19_1',
            'HGG/Brats18_2013_12_1',
            'LGG/Brats18_2013_9_1',
            'HGG/Brats18_TCIA06_247_1',
            'HGG/Brats18_TCIA08_234_1',
            'HGG/Brats18_CBICA_AAG_1',
            'LGG/Brats18_TCIA10_639_1',
            'HGG/Brats18_CBICA_ASV_1',
            'LGG/Brats18_TCIA09_177_1',
            'HGG/Brats18_TCIA02_322_1',
            'LGG/Brats18_TCIA09_428_1',
            'HGG/Brats18_CBICA_AYW_1',
            'LGG/Brats18_TCIA10_393_1',
            'LGG/Brats18_TCIA10_632_1',
            'HGG/Brats18_2013_18_1',
            'HGG/Brats18_CBICA_ATF_1',
            'HGG/Brats18_TCIA01_499_1',
            'HGG/Brats18_CBICA_BFP_1',
            'HGG/Brats18_CBICA_ARF_1',
            'LGG/Brats18_TCIA10_628_1',
            'HGG/Brats18_CBICA_AOH_1',
            'HGG/Brats18_CBICA_APY_1',
            'HGG/Brats18_TCIA02_608_1',
            'HGG/Brats18_CBICA_AOD_1',
            'HGG/Brats18_TCIA02_473_1',
            'HGG/Brats18_2013_17_1',
            'HGG/Brats18_CBICA_AQP_1',
            'HGG/Brats18_TCIA02_208_1',
            'HGG/Brats18_TCIA02_607_1',
            'HGG/Brats18_TCIA06_372_1',
            'HGG/Brats18_TCIA02_118_1',
            'HGG/Brats18_TCIA01_425_1',
            'LGG/Brats18_TCIA10_449_1',
            'HGG/Brats18_CBICA_ASK_1',
            'HGG/Brats18_TCIA01_186_1',
            'HGG/Brats18_CBICA_AQY_1',
            'HGG/Brats18_CBICA_ANP_1',
            'LGG/Brats18_TCIA13_645_1',
            'HGG/Brats18_TCIA02_309_1',
            'HGG/Brats18_2013_4_1',
            'HGG/Brats18_TCIA03_121_1',
            'LGG/Brats18_TCIA10_644_1',
            'HGG/Brats18_TCIA08_469_1',
            'HGG/Brats18_CBICA_ABM_1',
            'LGG/Brats18_TCIA10_307_1',
            'HGG/Brats18_TCIA02_377_1',
            'HGG/Brats18_TCIA05_396_1',
            'HGG/Brats18_CBICA_ANI_1',
            'LGG/Brats18_TCIA10_410_1',
            'HGG/Brats18_CBICA_AXL_1',
            'HGG/Brats18_TCIA01_203_1',
            'HGG/Brats18_TCIA03_375_1',
            'LGG/Brats18_TCIA10_202_1',
            'HGG/Brats18_CBICA_AQJ_1',
            'LGG/Brats18_TCIA13_630_1',
            'HGG/Brats18_CBICA_AQD_1',
            'LGG/Brats18_TCIA13_615_1',
            'HGG/Brats18_TCIA01_131_1',
            'HGG/Brats18_TCIA01_448_1',
            'LGG/Brats18_TCIA13_618_1',
            'HGG/Brats18_TCIA08_242_1',
            'LGG/Brats18_2013_6_1',
            'HGG/Brats18_CBICA_AXJ_1',
            'HGG/Brats18_TCIA04_192_1',
            'LGG/Brats18_TCIA10_420_1',
            'LGG/Brats18_TCIA10_637_1',
            'HGG/Brats18_CBICA_AQV_1',
            'LGG/Brats18_2013_24_1',
            'HGG/Brats18_CBICA_ASO_1',
            'LGG/Brats18_TCIA12_298_1',
            'LGG/Brats18_TCIA12_249_1',
            'LGG/Brats18_TCIA10_299_1',
            'LGG/Brats18_2013_28_1',
            'HGG/Brats18_CBICA_ALU_1',
            'HGG/Brats18_CBICA_AVG_1',
            'HGG/Brats18_TCIA05_478_1',
            'HGG/Brats18_CBICA_AXO_1',
            'HGG/Brats18_TCIA01_201_1',
            'HGG/Brats18_CBICA_ASA_1',
            'LGG/Brats18_TCIA12_101_1',
            'HGG/Brats18_2013_7_1',
            'LGG/Brats18_TCIA13_634_1',
            'HGG/Brats18_TCIA02_198_1',
            'HGG/Brats18_TCIA02_394_1',
            'HGG/Brats18_2013_14_1',
            'HGG/Brats18_CBICA_AQR_1',
            'HGG/Brats18_TCIA06_211_1',
            'HGG/Brats18_TCIA02_606_1',
            'HGG/Brats18_CBICA_AME_1',
            'HGG/Brats18_TCIA05_277_1',
            'LGG/Brats18_TCIA10_629_1',
            'HGG/Brats18_CBICA_APZ_1',
            'LGG/Brats18_TCIA10_442_1',
            'HGG/Brats18_TCIA03_474_1',
            'HGG/Brats18_TCIA01_221_1',
            'HGG/Brats18_TCIA02_314_1',
            'HGG/Brats18_CBICA_AOZ_1',
            'HGG/Brats18_CBICA_ANG_1',
            'HGG/Brats18_TCIA02_135_1',
            'LGG/Brats18_TCIA10_346_1',
            'HGG/Brats18_CBICA_ASG_1',
            'HGG/Brats18_CBICA_BHK_1',
            'HGG/Brats18_CBICA_AWH_1',
            'HGG/Brats18_CBICA_ATX_1',
            'HGG/Brats18_TCIA02_605_1',
            'LGG/Brats18_TCIA13_654_1',
            'HGG/Brats18_CBICA_BHM_1',
            'HGG/Brats18_CBICA_AOP_1',
            'HGG/Brats18_2013_13_1',
            'HGG/Brats18_TCIA06_184_1',
            'HGG/Brats18_TCIA02_226_1',
            'LGG/Brats18_TCIA10_152_1',
            'HGG/Brats18_CBICA_AQN_1',
            'HGG/Brats18_TCIA02_179_1',
            'HGG/Brats18_CBICA_ALX_1',
            'HGG/Brats18_CBICA_ATD_1',
            'LGG/Brats18_TCIA12_466_1',
            'HGG/Brats18_CBICA_AAP_1',
            'LGG/Brats18_TCIA09_255_1',
            'HGG/Brats18_2013_21_1',
            'HGG/Brats18_CBICA_ARW_1',
            'LGG/Brats18_TCIA10_175_1',
            'HGG/Brats18_2013_2_1',
            'HGG/Brats18_TCIA02_455_1',
            'HGG/Brats18_CBICA_ALN_1',
            'HGG/Brats18_2013_27_1',
            'HGG/Brats18_TCIA08_167_1',
            'HGG/Brats18_TCIA08_205_1',
            'HGG/Brats18_TCIA08_406_1',
            'HGG/Brats18_TCIA03_296_1',
            'HGG/Brats18_CBICA_AAB_1',
            'HGG/Brats18_CBICA_AYI_1',
            'HGG/Brats18_TCIA04_437_1',
            'HGG/Brats18_TCIA02_370_1',
            'HGG/Brats18_TCIA03_265_1',
            'LGG/Brats18_TCIA10_325_1',
            'HGG/Brats18_TCIA02_168_1',
            'HGG/Brats18_CBICA_AQU_1',
            'LGG/Brats18_TCIA10_261_1',
            'HGG/Brats18_2013_10_1',
            'LGG/Brats18_TCIA12_480_1',
            'HGG/Brats18_2013_11_1',
            'HGG/Brats18_TCIA08_113_1',
            'LGG/Brats18_TCIA10_490_1',
            'LGG/Brats18_TCIA09_141_1',
            'HGG/Brats18_CBICA_AWI_1',
            'HGG/Brats18_CBICA_ABN_1',
            'HGG/Brats18_CBICA_AAL_1',
            'HGG/Brats18_CBICA_APR_1',
            'HGG/Brats18_CBICA_ARZ_1',
            'HGG/Brats18_TCIA01_401_1',
            'HGG/Brats18_CBICA_AYU_1',
            'HGG/Brats18_TCIA02_151_1',
            'HGG/Brats18_CBICA_AQT_1',
            'LGG/Brats18_2013_15_1',
            'HGG/Brats18_CBICA_AUQ_1',
            'HGG/Brats18_TCIA01_150_1',
            'HGG/Brats18_CBICA_AWG_1',
            'HGG/Brats18_TCIA01_429_1',
            'HGG/Brats18_CBICA_AUN_1',
            'LGG/Brats18_TCIA09_462_1',
            'LGG/Brats18_TCIA09_312_1',
            'HGG/Brats18_TCIA02_117_1',
            'HGG/Brats18_TCIA01_335_1',
            'HGG/Brats18_TCIA02_222_1',
            'LGG/Brats18_TCIA13_650_1',
            'HGG/Brats18_CBICA_ABB_1',
            'HGG/Brats18_TCIA04_361_1',
            'LGG/Brats18_TCIA10_640_1',
            'HGG/Brats18_CBICA_AQG_1',
            'HGG/Brats18_TCIA08_436_1',
            'LGG/Brats18_2013_16_1',
            'HGG/Brats18_CBICA_ASH_1',
            'LGG/Brats18_TCIA13_633_1',
            'HGG/Brats18_2013_25_1',
            'HGG/Brats18_CBICA_BFB_1',
            'HGG/Brats18_CBICA_AZD_1',
            'HGG/Brats18_TCIA02_321_1',
            'HGG/Brats18_2013_26_1',
            'HGG/Brats18_CBICA_AQO_1',
            'HGG/Brats18_TCIA02_491_1',
            'HGG/Brats18_TCIA06_332_1',
            'LGG/Brats18_TCIA10_413_1',
            'HGG/Brats18_TCIA03_257_1',
            'HGG/Brats18_TCIA03_138_1',
            'HGG/Brats18_CBICA_AZH_1',
            'HGG/Brats18_TCIA02_290_1',
            'HGG/Brats18_CBICA_ASE_1',
            'HGG/Brats18_TCIA08_218_1',
            'LGG/Brats18_TCIA09_620_1',
            'HGG/Brats18_TCIA08_280_1',
            'LGG/Brats18_TCIA09_254_1',
            'HGG/Brats18_TCIA04_343_1',
            'LGG/Brats18_TCIA09_402_1',
            'HGG/Brats18_CBICA_AXM_1',
            'HGG/Brats18_CBICA_ATV_1',
            'HGG/Brats18_CBICA_AXW_1',
            'HGG/Brats18_TCIA03_419_1',
            'HGG/Brats18_TCIA04_479_1',
            'HGG/Brats18_TCIA01_147_1',
            'HGG/Brats18_CBICA_AMH_1',
            'HGG/Brats18_CBICA_AQZ_1',
            'HGG/Brats18_TCIA04_111_1',
            'HGG/Brats18_TCIA02_274_1',
            'HGG/Brats18_CBICA_ASW_1',
            'HGG/Brats18_CBICA_AYA_1',
            'HGG/Brats18_TCIA08_105_1',
            'LGG/Brats18_TCIA13_642_1',
            'LGG/Brats18_2013_1_1',
            'LGG/Brats18_TCIA10_408_1',
            'LGG/Brats18_TCIA10_276_1',
            'LGG/Brats18_TCIA10_109_1',
            'HGG/Brats18_TCIA01_231_1',
            'LGG/Brats18_TCIA10_266_1',
            'HGG/Brats18_CBICA_AXQ_1',
            'HGG/Brats18_CBICA_AVV_1',
            'LGG/Brats18_TCIA10_241_1',
            'HGG/Brats18_CBICA_AXN_1',
            'HGG/Brats18_CBICA_AQQ_1',
            'HGG/Brats18_CBICA_BHB_1',
            'HGG/Brats18_TCIA01_460_1',
            'HGG/Brats18_TCIA01_378_1',
            'HGG/Brats18_CBICA_AQA_1',
            'LGG/Brats18_TCIA09_493_1',
            'LGG/Brats18_TCIA10_130_1'
        ]

        ############### 所有切割按照python下标的标准
        self.image_high = 168
        self.image_width = 200
        self.image_high_s = int((240 - self.image_high) / 2)  # 36
        self.image_high_e = int(240 - (240 - self.image_high) / 2)  # 204
        self.image_width_s = 26
        self.image_width_e = 226
        ####################

        self.num_floder = len(self.brats_name)
        self.num_totel_image = 0
        self.epoch_completed = 0
        self.index_in_epoch = 0  # point to the start-index of next batch

        ###  CUT
        self.image_flair_all = np.zeros((self.num_floder*155, self.image_high, self.image_width), dtype=np.float)
        self.image_t1_all = np.zeros((self.num_floder*155, self.image_high, self.image_width), dtype=np.float)
        self.image_t1ce_all = np.zeros((self.num_floder*155, self.image_high, self.image_width), dtype=np.float)
        self.image_t2_all = np.zeros((self.num_floder*155, self.image_high, self.image_width), dtype=np.float)
        self.image_seg_all = np.zeros((self.num_floder*155, self.image_high, self.image_width), dtype=np.float)

        self.dir_path = '/project/XXXXXXX/XXX/MyBrats/Brats2018/MICCAI_BraTS_2018_Data_Training/'
    
        # read image
        for i in range(self.num_floder):
            path_flair = os.path.join(
                self.dir_path + self.brats_name[i] + '/' + self.brats_name[i][4:len(self.brats_name[i])] + '_flair.nii.gz')
            path_t1 = os.path.join(
                self.dir_path + self.brats_name[i] + '/' + self.brats_name[i][4:len(self.brats_name[i])] + '_t1.nii.gz')
            path_t1ce = os.path.join(
                self.dir_path + self.brats_name[i] + '/' + self.brats_name[i][4:len(self.brats_name[i])] + '_t1ce.nii.gz')
            path_t2 = os.path.join(
                self.dir_path + self.brats_name[i] + '/' + self.brats_name[i][4:len(self.brats_name[i])] + '_t2.nii.gz')
            path_seg = os.path.join(
                self.dir_path + self.brats_name[i] + '/' + self.brats_name[i][4:len(self.brats_name[i])] + '_seg.nii.gz')

            if i == round(self.num_floder/4):
                print('rate of progress 25%')
            if i == round(self.num_floder/2):
                print('rate of progress 50%')
            if i == round(self.num_floder/4*3):
                print('rate of progress 75%')
            if i == self.num_floder-1:
                print('rate of progress 100%')

            flair_temp = nib.load(path_flair)
            flair_temp_arr = flair_temp.get_fdata()
            flair_temp_arr_sq = np.squeeze(flair_temp_arr)
            flair_temp_arr_sq = extract_patch(flair_temp_arr_sq, self.image_high, self.image_width, self.image_high_s,
                                              self.image_high_e, self.image_width_s, self.image_width_e)

            t1_temp = nib.load(path_t1)
            t1_temp_arr = t1_temp.get_fdata()
            t1_temp_arr_sq = np.squeeze(t1_temp_arr)
            t1_temp_arr_sq = extract_patch(t1_temp_arr_sq, self.image_high, self.image_width, self.image_high_s,
                                              self.image_high_e, self.image_width_s, self.image_width_e)

            t1ce_temp = nib.load(path_t1ce)
            t1ce_temp_arr = t1ce_temp.get_fdata()
            t1ce_temp_arr_sq = np.squeeze(t1ce_temp_arr)
            t1ce_temp_arr_sq = extract_patch(t1ce_temp_arr_sq, self.image_high, self.image_width, self.image_high_s,
                                              self.image_high_e, self.image_width_s, self.image_width_e)

            t2_temp = nib.load(path_t2)
            t2_temp_arr = t2_temp.get_fdata()
            t2_temp_arr_sq = np.squeeze(t2_temp_arr)
            t2_temp_arr_sq = extract_patch(t2_temp_arr_sq, self.image_high, self.image_width, self.image_high_s,
                                              self.image_high_e, self.image_width_s, self.image_width_e)

            seg_temp = nib.load(path_seg)
            seg_temp_arr = seg_temp.get_fdata()
            seg_temp_arr_sq = np.squeeze(seg_temp_arr)
            seg_temp_arr_sq = extract_patch(seg_temp_arr_sq, self.image_high, self.image_width, self.image_high_s,
                                              self.image_high_e, self.image_width_s, self.image_width_e)

            ###################### calculate cutting condition

            background_flair = np.zeros(155, dtype=np.float)
            for k in range(155):
                background_flair[k] = count_pixel(flair_temp_arr_sq[:, :, k], 0)
                background_flair[k] = background_flair[k] / self.image_high / self.image_width

            contour_ssim_flair = np.zeros(155, dtype=np.float)
            signal_ssim_flair = np.zeros(155, dtype=np.float)

            for k in range(155):
                contour_ssim_flair[k], signal_ssim_flair[k] = count_contour_ssim(flair_temp_arr_sq[:, :, k])

            ######################

            ######  Normalization and Standardization  ########

            flair_temp_arr_sq = normalization(flair_temp_arr_sq)
            flair_temp_arr_sq = standardization(flair_temp_arr_sq)

            t1_temp_arr_sq = normalization(t1_temp_arr_sq)
            t1_temp_arr_sq = standardization(t1_temp_arr_sq)

            t1ce_temp_arr_sq = normalization(t1ce_temp_arr_sq)
            t1ce_temp_arr_sq = standardization(t1ce_temp_arr_sq)

            t2_temp_arr_sq = normalization(t2_temp_arr_sq)
            t2_temp_arr_sq = standardization(t2_temp_arr_sq)

            ###################################################
            ###  CUT

            for j in range(155):
                if j <= 77 and background_flair[j] <= 0.90 and contour_ssim_flair[j] >= 0.35 and signal_ssim_flair[j] <= 0.58:
                    self.image_flair_all[self.num_totel_image, :, :] = flair_temp_arr_sq[:, :, j]
                    self.image_t1_all[self.num_totel_image, :, :] = t1_temp_arr_sq[:, :, j]
                    self.image_t1ce_all[self.num_totel_image, :, :] = t1ce_temp_arr_sq[:, :, j]
                    self.image_t2_all[self.num_totel_image, :, :] = t2_temp_arr_sq[:, :, j]
                    self.image_seg_all[self.num_totel_image, :, :] = seg_temp_arr_sq[:, :, j]

                    self.num_totel_image += 1

                if j > 77 and background_flair[j] < 1 and signal_ssim_flair[j] <= 0.58:
                    self.image_flair_all[self.num_totel_image, :, :] = flair_temp_arr_sq[:, :, j]
                    self.image_t1_all[self.num_totel_image, :, :] = t1_temp_arr_sq[:, :, j]
                    self.image_t1ce_all[self.num_totel_image, :, :] = t1ce_temp_arr_sq[:, :, j]
                    self.image_t2_all[self.num_totel_image, :, :] = t2_temp_arr_sq[:, :, j]
                    self.image_seg_all[self.num_totel_image, :, :] = seg_temp_arr_sq[:, :, j]

                    self.num_totel_image += 1

        self.image_flair = self.image_flair_all[0:self.num_totel_image, :, :]
        self.image_t1 = self.image_t1_all[0:self.num_totel_image, :, :]
        self.image_t1ce = self.image_t1ce_all[0:self.num_totel_image, :, :]
        self.image_t2 = self.image_t2_all[0:self.num_totel_image, :, :]
        self.image_seg = self.image_seg_all[0:self.num_totel_image, :, :]

        self.image_order_0 = np.arange(self.num_totel_image)
        np.random.shuffle(self.image_order_0)
        self.image_order_1 = np.arange(self.num_totel_image)
        np.random.shuffle(self.image_order_1)
        self.image_order_2 = np.arange(self.num_totel_image)
        np.random.shuffle(self.image_order_2)

    def next_batch(self, batch_size, flip_b):
        r_image_flair = np.zeros((batch_size, self.image_high, self.image_width, 1), dtype=np.float)
        r_image_t1 = np.zeros((batch_size, self.image_high, self.image_width, 1), dtype=np.float)
        r_image_t1ce = np.zeros((batch_size, self.image_high, self.image_width, 1), dtype=np.float)
        r_image_t2 = np.zeros((batch_size, self.image_high, self.image_width, 1), dtype=np.float)
        r_image_seg = np.zeros((batch_size, self.image_high, self.image_width, 1), dtype=np.float)

        batch_order = np.zeros(batch_size, dtype=np.int)

        if self.num_totel_image - self.index_in_epoch >= batch_size:
            start = self.index_in_epoch
            end = self.index_in_epoch + batch_size

            if self.num_totel_image - self.index_in_epoch > batch_size:
                self.index_in_epoch = end
            else:
                self.index_in_epoch = 0
                self.epoch_completed += 1

            if flip_b == 0:
                batch_order = self.image_order_0[start:end]
            elif flip_b == 1:
                batch_order = self.image_order_1[start:end]
            else:
                batch_order = self.image_order_2[start:end]

        else:
            num_remnant_image = self.num_totel_image - self.index_in_epoch

            if flip_b == 0:
                batch_order[0:num_remnant_image] = self.image_order_0[self.index_in_epoch:self.num_totel_image]
                batch_order[num_remnant_image:batch_size] = self.image_order_0[0:(batch_size - num_remnant_image)]
            elif flip_b == 1:
                batch_order[0:num_remnant_image] = self.image_order_1[self.index_in_epoch:self.num_totel_image]
                batch_order[num_remnant_image:batch_size] = self.image_order_1[0:(batch_size - num_remnant_image)]
            else:
                batch_order[0:num_remnant_image] = self.image_order_2[self.index_in_epoch:self.num_totel_image]
                batch_order[num_remnant_image:batch_size] = self.image_order_2[0:(batch_size - num_remnant_image)]

            self.epoch_completed += 1
            self.index_in_epoch = batch_size - num_remnant_image

        for i in range(batch_size):
            if flip_b == 0:
                r_image_flair[i, :, :, 0] = self.image_flair[batch_order[i], :, :]
                r_image_t1[i, :, :, 0] = self.image_t1[batch_order[i], :, :]
                r_image_t1ce[i, :, :, 0] = self.image_t1ce[batch_order[i], :, :]
                r_image_t2[i, :, :, 0] = self.image_t2[batch_order[i], :, :]
                r_image_seg[i, :, :, 0] = self.image_seg[batch_order[i], :, :]
            elif flip_b == 1:
                r_image_flair[i, :, :, 0] = FlipImage_UpDown(self.image_flair[batch_order[i], :, :])
                r_image_t1[i, :, :, 0] = FlipImage_UpDown(self.image_t1[batch_order[i], :, :])
                r_image_t1ce[i, :, :, 0] = FlipImage_UpDown(self.image_t1ce[batch_order[i], :, :])
                r_image_t2[i, :, :, 0] = FlipImage_UpDown(self.image_t2[batch_order[i], :, :])
                r_image_seg[i, :, :, 0] = FlipImage_UpDown(self.image_seg[batch_order[i], :, :])
            else:
                r_image_flair[i, :, :, 0] = FlipImage_LeftRight(self.image_flair[batch_order[i], :, :])
                r_image_t1[i, :, :, 0] = FlipImage_LeftRight(self.image_t1[batch_order[i], :, :])
                r_image_t1ce[i, :, :, 0] = FlipImage_LeftRight(self.image_t1ce[batch_order[i], :, :])
                r_image_t2[i, :, :, 0] = FlipImage_LeftRight(self.image_t2[batch_order[i], :, :])
                r_image_seg[i, :, :, 0] = FlipImage_LeftRight(self.image_seg[batch_order[i], :, :])

        return r_image_flair, r_image_t1, r_image_t1ce, r_image_t2, r_image_seg
