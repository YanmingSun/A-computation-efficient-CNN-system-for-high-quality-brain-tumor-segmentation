import os
import nibabel as nib
import skimage.io as io
import numpy as np
import tensorflow as tf
import LoadTrainingSet_PreprocessingFour_SmallSizeImageThree_CutTrain

import matplotlib.pyplot as plt  # plt 用于显示图片
import matplotlib.image as mpimg  # mpimg 用于读取图片

from sklearn.preprocessing import MinMaxScaler
from PIL import Image
from pylab import *

from functools import reduce

import math
import random
import time
import datetime

from skimage.measure import compare_ssim

branch_name = 'FullReluDepthwiseConvolution1'

input_path = '/project/XXXXXXX/XXX/MyBrats/Brats2018/MICCAI_BraTS_2018_Data_Validation/'

print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
training_start_time = datetime.datetime.now()

print('Start loading dataset: ')
training_set = LoadTrainingSet_PreprocessingFour_SmallSizeImageThree_CutTrain.BratsTrainingSet()  # load data from hard disk to RAM

np.savetxt(('training_' + branch_name + '_imageorder0.txt'), training_set.image_order_0, fmt='%d')
np.savetxt(('training_' + branch_name + '_imageorder1.txt'), training_set.image_order_1, fmt='%d')
np.savetxt(('training_' + branch_name + '_imageorder2.txt'), training_set.image_order_2, fmt='%d')

###################################################################
# training
#################################################################
print('Start training: ')
sess = tf.Session()

img_H = training_set.image_high  # 168
img_W = training_set.image_width  # 200
img_H_s = training_set.image_high_s
img_H_e = training_set.image_high_e
img_W_s = training_set.image_width_s
img_W_e = training_set.image_width_e

flair_in = tf.placeholder("float", shape=[None, img_H, img_W, 1])
t1_in = tf.placeholder("float", shape=[None, img_H, img_W, 1])
t1ce_in = tf.placeholder("float", shape=[None, img_H, img_W, 1])
t2_in = tf.placeholder("float", shape=[None, img_H, img_W, 1])
seg_in = tf.placeholder("float", shape=[None, img_H, img_W, 1])
# test flag for batch norm
tst = tf.placeholder(tf.bool)
iter = tf.placeholder(tf.int32)


def generalized_dice_loss(pred, true, p=1, q=1, eps=1E-6):
    """pred and true are tensors of shape (b, w_0, w_1, ..., c) where
             b   ... batch size
             w_k ... width of input in k-th dimension
             c   ... number of segments/classes
       Furthermore, boths tensors have exclusively values in [0, 1].
       more than already good ones. The remaining parameters are as follows:
             p   ... power of inverse weigthing (p=2 default, p=0 uniform)
             q   ... power of inverse loss weighting (q=1 default, q=0 none)
             eps ... regularization term if empty classes occur"""

    assert (p >= 0)
    assert (q >= 0)
    assert (eps >= 0)
    assert (pred.get_shape()[1:] == true.get_shape()[1:])

    m = "the values in your last layer must be strictly in [0, 1]"
    with tf.control_dependencies([]):

        shape_pred = pred.get_shape()
        shape_true = true.get_shape()
        prod_pred = reduce(lambda x, y: x * y, shape_pred[1:-1], tf.Dimension(1))
        prod_true = reduce(lambda x, y: x * y, shape_true[1:-1], tf.Dimension(1))

        # reshape to shape (b, W, c) where W is product of w_k
        pred = tf.reshape(pred, [-1, prod_pred, shape_pred[-1]])
        true = tf.reshape(true, [-1, prod_true, shape_true[-1]])

        # no class reweighting at all
        if p == 0:
            # unweighted intersection and union
            inter = tf.reduce_mean(pred * true, axis=[1, 2])
            union = tf.reduce_mean(pred + true, axis=[1, 2])
        else:
            # inverse L_p weighting for class cardinalities
            weights = tf.abs(tf.reduce_sum(true, axis=[1])) ** p + eps
            weights = tf.expand_dims(tf.reduce_sum(weights, axis=[-1]), -1) \
                      / weights

            # weighted intersection and union
            inter = tf.reduce_mean(weights * tf.reduce_mean(pred * true, axis=[1]),
                                   axis=[-1])
            union = tf.reduce_mean(weights * tf.reduce_mean(pred + true, axis=[1]),
                                   axis=[-1])

        # the traditional dice formula
        loss = 1.0 - 2.0 * (inter + eps) / (union + eps)

        # no reweighting of the batch
        if q == 0:
            return tf.reduce_mean(loss)

        # inverse L_q weighting for loss scores
        weights = tf.abs(loss) ** q + eps
        weights = tf.reduce_sum(weights) / weights

        return tf.reduce_mean(loss * weights) / tf.reduce_mean(weights)


def batchnorm(Ylogits, is_test, iteration, offset, convolutional=False):
    exp_moving_avg = tf.train.ExponentialMovingAverage(0.999,
                                                       iteration)  # adding the iteration prevents from averaging across non-existing iterations
    bnepsilon = 1e-5
    if convolutional:
        mean, variance = tf.nn.moments(Ylogits, [0, 1, 2])
    else:
        mean, variance = tf.nn.moments(Ylogits, [0])
    update_moving_averages = exp_moving_avg.apply([mean, variance])
    m = tf.cond(is_test, lambda: exp_moving_avg.average(mean), lambda: mean)
    v = tf.cond(is_test, lambda: exp_moving_avg.average(variance), lambda: variance)
    Ybn = tf.nn.batch_normalization(Ylogits, m, v, offset, None, bnepsilon)
    return Ybn, update_moving_averages


def no_batchnorm(Ylogits, is_test, iteration, offset, convolutional=False):
    return Ylogits, tf.no_op()


def instance_norm(x):
    mean, variance = tf.nn.moments(x, axes=[1,2], keep_dims=True)
    epsilon = 1e-5
    inv = tf.rsqrt(variance + epsilon)
    normalized = (x-mean)*inv
    return normalized


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def depthwise_conv2d(x, W):
    return tf.nn.depthwise_conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


'''
def conv2d_transpose(x, w):
#    shape_x = x.get_shape().as_list()
    shape_w = w.get_shape().as_list()
    inputs_shape = tf.shape(x)
    outputs_shape = [inputs_shape[0], inputs_shape[1], inputs_shape[2], shape_w[2]]
#    shape_y = [x[0], shape_x[1] * 2, shape_x[2] * 2, shape_w[2]]
    r_Y = tf.nn.conv2d_transpose(x, w, output_shape=outputs_shape, strides=[1, 2, 2, 1], padding="SAME")
    return r_Y
'''


def full_relu(x):
    x1 = tf.nn.relu(x)
    x2 = tf.nn.relu(-x)
    return tf.concat([x1, x2], 3)


input_image_concat = tf.concat([flair_in, t1_in, t1ce_in, t2_in], 3)


### layer1_1
W_conv1_1 = weight_variable([3, 3, 4, 2])
b_conv1_1 = bias_variable([8])

h_conv1_1 = depthwise_conv2d(input_image_concat, W_conv1_1)

h_bn1_1 = instance_norm(h_conv1_1) + b_conv1_1

h_relu1_1 = full_relu(h_bn1_1)
h_pool1_1 = max_pool_2x2(h_relu1_1)

### layer1_2

W_conv1_2 = weight_variable([3, 3, 4, 8])
b_conv1_2 = bias_variable([8])

h_conv1_2 = conv2d(input_image_concat, W_conv1_2)
h_bn1_2, update_ema1_2 = batchnorm(h_conv1_2, tst, iter, b_conv1_2, convolutional=True)
h_relu1_2 = full_relu(h_bn1_2)
h_pool1_2 = max_pool_2x2(h_relu1_2)

h_relu1 = tf.concat([h_relu1_1, h_relu1_2], 3)
h_pool1 = tf.concat([h_pool1_1, h_pool1_2], 3)

# convolution + BN + relu + pool
### layer2_1
W_conv2_1 = weight_variable([3, 3, 16, 1])
b_conv2_1 = bias_variable([16])

h_conv2_1 = depthwise_conv2d(h_pool1_1, W_conv2_1)

h_bn2_1 = instance_norm(h_conv2_1) + b_conv2_1

h_relu2_1 = full_relu(h_bn2_1)
h_pool2_1 = max_pool_2x2(h_relu2_1)

# convolution + BN + relu + pool
### layer2_2
W_conv2_2 = weight_variable([3, 3, 16, 8])
b_conv2_2 = bias_variable([8])

h_conv2_2 = conv2d(h_pool1_2, W_conv2_2)
h_bn2_2, update_ema2_2 = batchnorm(h_conv2_2, tst, iter, b_conv2_2, convolutional=True)
h_relu2_2 = full_relu(h_bn2_2)
h_pool2_2 = max_pool_2x2(h_relu2_2)

h_relu2 = tf.concat([h_relu2_1, h_relu2_2], 3)
h_pool2 = tf.concat([h_pool2_1, h_pool2_2], 3)

# convolution + BN + relu + pool
W_conv3 = weight_variable([3, 3, 48, 16])
b_conv3 = bias_variable([16])

h_conv3 = conv2d(h_pool2, W_conv3)
h_bn3, update_ema3 = batchnorm(h_conv3, tst, iter, b_conv3, convolutional=True)
h_relu3 = full_relu(h_bn3)
h_pool3 = max_pool_2x2(h_relu3)

#############################
W_conv4 = weight_variable([3, 3, 32, 16])
b_conv4 = bias_variable([16])

h_conv4 = conv2d(h_pool3, W_conv4)
h_bn4, update_ema4 = batchnorm(h_conv4, tst, iter, b_conv4, convolutional=True)
h_relu4 = tf.nn.relu(h_bn4)
###################################


# upsample + conv + BN + relu
W_conv5 = weight_variable([3, 3, 16, 16])
b_conv5 = bias_variable([16])

h_upsample5 = tf.image.resize_images(h_relu4, [int(img_H / 4), int(img_W / 4)], 0)
W_conv5_1 = weight_variable([1, 1, 32, 16])
b_conv5_1 = bias_variable([16])
h_conv5_1 = conv2d(h_relu3, W_conv5_1) + b_conv5_1
h_upsample5 = h_upsample5 + h_conv5_1

h_conv5 = conv2d(h_upsample5, W_conv5)
h_bn5, update_ema5 = batchnorm(h_conv5, tst, iter, b_conv5, convolutional=True)
h_relu5 = tf.nn.relu(h_bn5)

# upsample + conv + BN + relu
W_conv6 = weight_variable([3, 3, 16, 16])
b_conv6 = bias_variable([16])

h_upsample6 = tf.image.resize_images(h_relu5, [int(img_H / 2), int(img_W / 2)], 0)
W_conv6_1 = weight_variable([1, 1, 48, 16])
b_conv6_1 = bias_variable([16])
h_conv6_1 = conv2d(h_relu2, W_conv6_1) + b_conv6_1
h_upsample6 = h_upsample6 + h_conv6_1

h_conv6 = conv2d(h_upsample6, W_conv6)
h_bn6, update_ema6 = batchnorm(h_conv6, tst, iter, b_conv6, convolutional=True)
h_relu6 = tf.nn.relu(h_bn6)

# upsample + conv + BN + relu
W_conv7 = weight_variable([3, 3, 16, 4])
b_conv7 = bias_variable([4])

h_upsample7 = tf.image.resize_images(h_relu6, [int(img_H), int(img_W)], 0)
W_conv7_1 = weight_variable([1, 1, 32, 16])
b_conv7_1 = bias_variable([16])
h_conv7_1 = conv2d(h_relu1, W_conv7_1) + b_conv7_1
h_upsample7 = h_upsample7 + h_conv7_1

h_conv7 = conv2d(h_upsample7, W_conv7) + b_conv7
# h_bn6, update_ema6 = batchnorm(h_conv6, tst, iter, b_conv6, convolutional=True)
# h_relu6 = tf.nn.relu(h_bn6)

#####################################################
# preprocess the ground true

# shape_seg_in = seg_in.get_shape().as_list()
# shape_seg_in[0] = -1
temp_zero = seg_in * 0.0
temp_one = temp_zero + 1.0

# 10 times error penalty for label 1, 2 and 4
seg_in_label_0 = tf.where(tf.equal(seg_in, temp_zero), temp_one, temp_zero)
seg_in_label_1 = tf.where(tf.equal(seg_in, temp_one), temp_one, temp_zero)
seg_in_label_2 = tf.where(tf.equal(seg_in, 2.0 * temp_one), temp_one, temp_zero)
seg_in_label_4 = tf.where(tf.equal(seg_in, 4.0 * temp_one), temp_one, temp_zero)
seg_in_label = tf.concat([seg_in_label_0, seg_in_label_1, seg_in_label_2, seg_in_label_4], 3)

##########################################

accuracy_list = []  # 保存准确率序列
cross_entropy_list = []
learning_rate_list = []
step_list = []
flip_b = -1  # flip training-set images: 0 no flip, 1 Up-Down flip, 2 Left-Right flip
n = 22101  # 训练次数
s = 50  # plot输出步长
batch_size = 100

#########################################
y_softmax = tf.nn.softmax(h_conv7, axis=3)
# cross_entropy = -tf.reduce_sum(seg_in_label * tf.log(y_softmax))
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=h_conv7, labels=seg_in_label)
cross_entropy = tf.reduce_mean(cross_entropy) * 100000
# dice_loss = generalized_dice_loss(y_softmax, seg_in_label, p=1, q=1, eps=1E-6)

# the learning rate is: # 0.00001 + 0.03 * (1/e)^(step/1000)), i.e. exponential decay from 0.03->0.0001
# lr = 0.000005 + tf.train.exponential_decay(0.0001, iter, 1000, 1 / math.e)

lr = tf.train.cosine_decay(learning_rate=0.01, global_step=iter, decay_steps=19900, alpha=0.0001)
train_step = tf.train.AdamOptimizer(lr).minimize(cross_entropy)
predict_image = tf.argmax(y_softmax, 3)  # if label equal 3, change to 4.
correct_prediction = tf.equal(predict_image, tf.argmax(seg_in_label, 3))
accuracy_pixel = tf.reduce_mean(tf.cast(correct_prediction, "float"))

update_ema = tf.group(update_ema1_2, update_ema2_2,
                      update_ema3, update_ema4, update_ema5, update_ema6)

#####################################################
######################################################

sess.run(tf.global_variables_initializer())

for i in range(n):
    if i % 500 == 0:
        flip_b = flip_b + 1
        if flip_b >= 3:
            flip_b = 0

    r_image_flair, r_image_t1, r_image_t1ce, r_image_t2, r_image_seg = training_set.next_batch(batch_size, flip_b)

    if i % s == 0:
        training_accuracy = sess.run(accuracy_pixel, feed_dict={flair_in: r_image_flair, t1_in: r_image_t1,
                                                                t1ce_in: r_image_t1ce, t2_in: r_image_t2,
                                                                seg_in: r_image_seg, tst: False, iter: i})
        training_cross_entropy = sess.run(cross_entropy, feed_dict={flair_in: r_image_flair, t1_in: r_image_t1,
                                                                    t1ce_in: r_image_t1ce, t2_in: r_image_t2,
                                                                    seg_in: r_image_seg, tst: False, iter: i})
        learning_rate = sess.run(lr, feed_dict={flair_in: r_image_flair, t1_in: r_image_t1,
                                                t1ce_in: r_image_t1ce, t2_in: r_image_t2,
                                                seg_in: r_image_seg, tst: False, iter: i})
        step_list.append(i)
        accuracy_list.append(training_accuracy)
        cross_entropy_list.append(training_cross_entropy)
        learning_rate_list.append(learning_rate)

        print("step %d, training accuracy %g, cross entropy %g, learning rate %g" % (
            i, training_accuracy, training_cross_entropy, learning_rate))
    sess.run(train_step, feed_dict={flair_in: r_image_flair, t1_in: r_image_t1,
                                    t1ce_in: r_image_t1ce, t2_in: r_image_t2,
                                    seg_in: r_image_seg, tst: False, iter: i})
    sess.run(update_ema, feed_dict={flair_in: r_image_flair, t1_in: r_image_t1,
                                    t1ce_in: r_image_t1ce, t2_in: r_image_t2,
                                    seg_in: r_image_seg, tst: False, iter: i})

print_content = [step_list, accuracy_list, cross_entropy_list, learning_rate_list]
print_content = list(map(list, zip(*print_content)))
np.savetxt(('training_' + branch_name + '_printcontent.txt'), print_content, fmt='%f')

print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
training_end_time = datetime.datetime.now()
print('training time(s):', (training_end_time - training_start_time).seconds)

#########################################################################
# validation and submittion
###########################################################################
print('Start validation: ')

brats_validation_ID = ['Brats18_CBICA_AAM_1',
                       'Brats18_CBICA_ABT_1',
                       'Brats18_CBICA_ALA_1',
                       'Brats18_CBICA_ALT_1',
                       'Brats18_CBICA_ALV_1',
                       'Brats18_CBICA_ALZ_1',
                       'Brats18_CBICA_AMF_1',
                       'Brats18_CBICA_AMU_1',
                       'Brats18_CBICA_ANK_1',
                       'Brats18_CBICA_APM_1',
                       'Brats18_CBICA_AQE_1',
                       'Brats18_CBICA_ARR_1',
                       'Brats18_CBICA_ATW_1',
                       'Brats18_CBICA_AUC_1',
                       'Brats18_CBICA_AUE_1',
                       'Brats18_CBICA_AZA_1',
                       'Brats18_CBICA_BHF_1',
                       'Brats18_CBICA_BHN_1',
                       'Brats18_CBICA_BKY_1',
                       'Brats18_CBICA_BLI_1',
                       'Brats18_CBICA_BLK_1',
                       'Brats18_MDA_907_1',
                       'Brats18_MDA_922_1',
                       'Brats18_MDA_1012_1',
                       'Brats18_MDA_1015_1',
                       'Brats18_MDA_1081_1',
                       'Brats18_TCIA02_230_1',
                       'Brats18_TCIA02_400_1',
                       'Brats18_TCIA03_216_1',
                       'Brats18_TCIA03_288_1',
                       'Brats18_TCIA03_313_1',
                       'Brats18_TCIA03_604_1',
                       'Brats18_TCIA04_212_1',
                       'Brats18_TCIA04_253_1',
                       'Brats18_TCIA07_600_1',
                       'Brats18_TCIA07_601_1',
                       'Brats18_TCIA07_602_1',
                       'Brats18_TCIA09_248_1',
                       'Brats18_TCIA10_195_1',
                       'Brats18_TCIA10_311_1',
                       'Brats18_TCIA10_609_1',
                       'Brats18_TCIA11_612_1',
                       'Brats18_TCIA12_613_1',
                       'Brats18_TCIA13_610_1',
                       'Brats18_TCIA13_611_1',
                       'Brats18_TCIA13_617_1',
                       'Brats18_TCIA13_636_1',
                       'Brats18_TCIA13_638_1',
                       'Brats18_TCIA13_646_1',
                       'Brats18_TCIA13_652_1',
                       'Brats18_UAB_3446_1',
                       'Brats18_UAB_3448_1',
                       'Brats18_UAB_3449_1',
                       'Brats18_UAB_3454_1',
                       'Brats18_UAB_3455_1',
                       'Brats18_UAB_3456_1',
                       'Brats18_UAB_3490_1',
                       'Brats18_UAB_3498_1',
                       'Brats18_UAB_3499_1',
                       'Brats18_WashU_S036_1',
                       'Brats18_WashU_S037_1',
                       'Brats18_WashU_S041_1',
                       'Brats18_WashU_W033_1',
                       'Brats18_WashU_W038_1',
                       'Brats18_WashU_W047_1',
                       'Brats18_WashU_W053_1',
                       ]


def extract_patch(input_data, image_high, image_width, image_high_s, image_high_e, image_width_s, image_width_e):
    shape_data = np.shape(input_data)
    image_output = np.zeros((image_high, image_width, shape_data[2]), dtype=np.float)

    for j in range(shape_data[2]):
        image_output[:, :, j] = input_data[image_high_s:image_high_e, image_width_s:image_width_e, j]

    return image_output


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


def brats_validation(input_dir_path, output_dir_path, patient_ID):
    image_flair = np.zeros((155, img_H, img_W, 1), dtype=np.float)
    image_t1 = np.zeros((155, img_H, img_W, 1), dtype=np.float)
    image_t1ce = np.zeros((155, img_H, img_W, 1), dtype=np.float)
    image_t2 = np.zeros((155, img_H, img_W, 1), dtype=np.float)

    image_predict = np.zeros((240, 240, 155), dtype=np.float)

    path_flair = os.path.join(input_dir_path + patient_ID + '/' + patient_ID + '_flair.nii.gz')
    path_t1 = os.path.join(input_dir_path + patient_ID + '/' + patient_ID + '_t1.nii.gz')
    path_t1ce = os.path.join(input_dir_path + patient_ID + '/' + patient_ID + '_t1ce.nii.gz')
    path_t2 = os.path.join(input_dir_path + patient_ID + '/' + patient_ID + '_t2.nii.gz')

    flair_temp = nib.load(path_flair)
    flair_temp_arr = flair_temp.get_fdata()
    flair_temp_arr_sq = np.squeeze(flair_temp_arr)
    flair_temp_arr_sq = extract_patch(flair_temp_arr_sq, img_H, img_W, img_H_s, img_H_e, img_W_s, img_W_e)

    t1_temp = nib.load(path_t1)
    t1_temp_arr = t1_temp.get_fdata()
    t1_temp_arr_sq = np.squeeze(t1_temp_arr)
    t1_temp_arr_sq = extract_patch(t1_temp_arr_sq, img_H, img_W, img_H_s, img_H_e, img_W_s, img_W_e)

    t1ce_temp = nib.load(path_t1ce)
    t1ce_temp_arr = t1ce_temp.get_fdata()
    t1ce_temp_arr_sq = np.squeeze(t1ce_temp_arr)
    t1ce_temp_arr_sq = extract_patch(t1ce_temp_arr_sq, img_H, img_W, img_H_s, img_H_e, img_W_s, img_W_e)

    t2_temp = nib.load(path_t2)
    t2_temp_arr = t2_temp.get_fdata()
    t2_temp_arr_sq = np.squeeze(t2_temp_arr)
    t2_temp_arr_sq = extract_patch(t2_temp_arr_sq, img_H, img_W, img_H_s, img_H_e, img_W_s, img_W_e)

    affine_array = flair_temp.affine

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

    for i in range(155):
        image_flair[i, :, :, 0] = flair_temp_arr_sq[:, :, i]
        image_t1[i, :, :, 0] = t1_temp_arr_sq[:, :, i]
        image_t1ce[i, :, :, 0] = t1ce_temp_arr_sq[:, :, i]
        image_t2[i, :, :, 0] = t2_temp_arr_sq[:, :, i]

    batch_flair = np.zeros((1, img_H, img_W, 1), dtype=np.float)
    batch_t1 = np.zeros((1, img_H, img_W, 1), dtype=np.float)
    batch_t1ce = np.zeros((1, img_H, img_W, 1), dtype=np.float)
    batch_t2 = np.zeros((1, img_H, img_W, 1), dtype=np.float)

    for i in range(155):
        batch_flair[0, :, :, 0] = image_flair[i, :, :, 0]
        batch_t1[0, :, :, 0] = image_t1[i, :, :, 0]
        batch_t1ce[0, :, :, 0] = image_t1ce[i, :, :, 0]
        batch_t2[0, :, :, 0] = image_t2[i, :, :, 0]

        t_predict_image = sess.run(predict_image, feed_dict={flair_in: batch_flair, t1_in: batch_t1,
                                                             t1ce_in: batch_t1ce, t2_in: batch_t2,
                                                             tst: True})
        t_predict_image_p = np.squeeze(t_predict_image)
        t_predict_image_p[t_predict_image_p == 3] = 4
        image_predict[img_H_s:img_H_e, img_W_s:img_W_e, i] = t_predict_image_p[:, :]

    image_predict = image_predict.astype(int16)
    new_image_predict = nib.Nifti1Image(image_predict, affine_array)
    nib.save(new_image_predict, os.path.join(output_dir_path + patient_ID + '.nii.gz'))


num_validation_patient = len(brats_validation_ID)


output_path = './Validation_' + branch_name + '_submit/'
if not os.path.exists(output_path):
    os.makedirs(output_path)

for j in range(num_validation_patient):
    brats_validation(input_path, output_path, brats_validation_ID[j])

    if j == round(num_validation_patient / 4):
        print('rate of progress 25%')
    if j == round(num_validation_patient / 2):
        print('rate of progress 50%')
    if j == round(num_validation_patient / 4 * 3):
        print('rate of progress 75%')
    if j == num_validation_patient - 1:
        print('rate of progress 100%')

print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
validation_end_time = datetime.datetime.now()
print('validation time(s):', (validation_end_time - training_end_time).seconds)


######################################################
# validation (cut image based on contour , ssim and background)
######################################################


def FlipImage_UpDown(image):
    shape_image = np.shape(image)
    temp_image = np.zeros((shape_image[0], shape_image[1]), dtype=np.float)
    for i in range(shape_image[0]):
        temp_image[i, :] = image[shape_image[0] - 1 - i, :]
    return temp_image


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


def brats_validation_cut(input_dir_path, output_dir_path, patient_ID):
    image_flair = np.zeros((155, img_H, img_W, 1), dtype=np.float)
    image_t1 = np.zeros((155, img_H, img_W, 1), dtype=np.float)
    image_t1ce = np.zeros((155, img_H, img_W, 1), dtype=np.float)
    image_t2 = np.zeros((155, img_H, img_W, 1), dtype=np.float)

    image_predict = np.zeros((240, 240, 155), dtype=np.float)

    path_flair = os.path.join(input_dir_path + patient_ID + '/' + patient_ID + '_flair.nii.gz')
    path_t1 = os.path.join(input_dir_path + patient_ID + '/' + patient_ID + '_t1.nii.gz')
    path_t1ce = os.path.join(input_dir_path + patient_ID + '/' + patient_ID + '_t1ce.nii.gz')
    path_t2 = os.path.join(input_dir_path + patient_ID + '/' + patient_ID + '_t2.nii.gz')

    flair_temp = nib.load(path_flair)
    flair_temp_arr = flair_temp.get_fdata()
    flair_temp_arr_sq = np.squeeze(flair_temp_arr)
    flair_temp_arr_sq = extract_patch(flair_temp_arr_sq, img_H, img_W, img_H_s, img_H_e, img_W_s, img_W_e)

    t1_temp = nib.load(path_t1)
    t1_temp_arr = t1_temp.get_fdata()
    t1_temp_arr_sq = np.squeeze(t1_temp_arr)
    t1_temp_arr_sq = extract_patch(t1_temp_arr_sq, img_H, img_W, img_H_s, img_H_e, img_W_s, img_W_e)

    t1ce_temp = nib.load(path_t1ce)
    t1ce_temp_arr = t1ce_temp.get_fdata()
    t1ce_temp_arr_sq = np.squeeze(t1ce_temp_arr)
    t1ce_temp_arr_sq = extract_patch(t1ce_temp_arr_sq, img_H, img_W, img_H_s, img_H_e, img_W_s, img_W_e)

    t2_temp = nib.load(path_t2)
    t2_temp_arr = t2_temp.get_fdata()
    t2_temp_arr_sq = np.squeeze(t2_temp_arr)
    t2_temp_arr_sq = extract_patch(t2_temp_arr_sq, img_H, img_W, img_H_s, img_H_e, img_W_s, img_W_e)

    affine_array = flair_temp.affine

    ###################### calculate cutting condition

    background_flair = np.zeros(155, dtype=np.float)
    for k in range(155):
        background_flair[k] = count_pixel(flair_temp_arr_sq[:, :, k], 0)
        background_flair[k] = background_flair[k] / img_H / img_W

    contour_ssim_flair = np.zeros(155, dtype=np.float)
    signal_ssim_flair = np.zeros(155, dtype=np.float)

    for i in range(155):
        contour_ssim_flair[i], signal_ssim_flair[i] = count_contour_ssim(flair_temp_arr_sq[:, :, i])

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

    for i in range(155):
        image_flair[i, :, :, 0] = flair_temp_arr_sq[:, :, i]
        image_t1[i, :, :, 0] = t1_temp_arr_sq[:, :, i]
        image_t1ce[i, :, :, 0] = t1ce_temp_arr_sq[:, :, i]
        image_t2[i, :, :, 0] = t2_temp_arr_sq[:, :, i]

    batch_flair = np.zeros((1, img_H, img_W, 1), dtype=np.float)
    batch_t1 = np.zeros((1, img_H, img_W, 1), dtype=np.float)
    batch_t1ce = np.zeros((1, img_H, img_W, 1), dtype=np.float)
    batch_t2 = np.zeros((1, img_H, img_W, 1), dtype=np.float)

    number_slide = 0

    for i in range(155):
        if i <= 77 and background_flair[i] <= 0.90 and contour_ssim_flair[i] >= 0.35 and signal_ssim_flair[
            i] <= 0.58:
            batch_flair[0, :, :, 0] = image_flair[i, :, :, 0]
            batch_t1[0, :, :, 0] = image_t1[i, :, :, 0]
            batch_t1ce[0, :, :, 0] = image_t1ce[i, :, :, 0]
            batch_t2[0, :, :, 0] = image_t2[i, :, :, 0]

            t_predict_image = sess.run(predict_image, feed_dict={flair_in: batch_flair, t1_in: batch_t1,
                                                                 t1ce_in: batch_t1ce, t2_in: batch_t2,
                                                                 tst: True})
            t_predict_image_p = np.squeeze(t_predict_image)
            t_predict_image_p[t_predict_image_p == 3] = 4
            image_predict[img_H_s:img_H_e, img_W_s:img_W_e, i] = t_predict_image_p[:, :]

            number_slide = number_slide + 1

        if i > 77 and background_flair[i] < 1 and signal_ssim_flair[i] <= 0.58:
            batch_flair[0, :, :, 0] = image_flair[i, :, :, 0]
            batch_t1[0, :, :, 0] = image_t1[i, :, :, 0]
            batch_t1ce[0, :, :, 0] = image_t1ce[i, :, :, 0]
            batch_t2[0, :, :, 0] = image_t2[i, :, :, 0]

            t_predict_image = sess.run(predict_image, feed_dict={flair_in: batch_flair, t1_in: batch_t1,
                                                                 t1ce_in: batch_t1ce, t2_in: batch_t2,
                                                                 tst: True})
            t_predict_image_p = np.squeeze(t_predict_image)
            t_predict_image_p[t_predict_image_p == 3] = 4
            image_predict[img_H_s:img_H_e, img_W_s:img_W_e, i] = t_predict_image_p[:, :]

            number_slide = number_slide + 1

    image_predict = image_predict.astype(int16)
    new_image_predict = nib.Nifti1Image(image_predict, affine_array)
    nib.save(new_image_predict, os.path.join(output_dir_path + patient_ID + '.nii.gz'))

    return number_slide


# num_validation_patient = len(brats_validation_ID)


output_path_cut = './Validation_' + branch_name + '_submit_Cut/'
if not os.path.exists(output_path_cut):
    os.makedirs(output_path_cut)

slide_number = np.zeros(num_validation_patient, dtype=np.int)

print('start validation (cut)')
print('Patient_ID', 'number_slides')

for j in range(num_validation_patient):
    slide_number[j] = brats_validation_cut(input_path, output_path_cut, brats_validation_ID[j])
    print(brats_validation_ID[j], slide_number[j])

print('Average number of slides: ', np.mean(slide_number))

print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
validation_cut_end_time = datetime.datetime.now()
print('validation_time_cut(s):', (validation_cut_end_time - validation_end_time).seconds)


######################################################
# validation (cut image based on contour , ssim and background; Post-processing)
######################################################
def post_processing(input_data, num_WT, num_ET, num_TC):
    shape_data = np.shape(input_data)

    pred_max = np.zeros(shape_data[2], dtype=np.float)
    output_data = np.zeros((shape_data[0], shape_data[1], shape_data[2]), dtype=np.float)

    for i in range(shape_data[2]):
        pred_max[i] = input_data[:, :, i].max()

    for i in range(shape_data[2]):
        num_index_WT = 0
        num_index_ET = 0
        num_index_TC = 0
        if i >= num_WT - 1 and i <= shape_data[2] - num_WT and pred_max[i] > 0:
            for j in range(num_WT):
                if pred_max[i - j] > 0:
                    num_index_WT = num_index_WT + 1
                else:
                    break
            for j in range(num_WT):
                if pred_max[i + j] > 0:
                    num_index_WT = num_index_WT + 1
                else:
                    break

            if num_index_WT - 1 >= num_WT:  # 上面两个循环把当前点算了两次
                output_data[:, :, i] = input_data[:, :, i]
            else:
                output_data[:, :, i] = 0

        if i >= num_ET - 1 and i <= shape_data[2] - num_ET and pred_max[i] == 4:
            for j in range(num_ET):
                if pred_max[i - j] == 4:
                    num_index_ET = num_index_ET + 1
                else:
                    break
            for j in range(num_ET):
                if pred_max[i + j] == 4:
                    num_index_ET = num_index_ET + 1
                else:
                    break
            if num_index_ET - 1 < num_ET:
                output_data[:, :, i][output_data[:, :, i] == 4] = 1

        if i >= num_TC - 1 and i <= shape_data[2] - num_TC and pred_max[i] == 1:
            for j in range(num_TC):
                if pred_max[i - j] == 1:
                    num_index_TC = num_index_TC + 1
                else:
                    break
            for j in range(num_TC):
                if pred_max[i + j] == 1:
                    num_index_TC = num_index_TC + 1
                else:
                    break
            if num_index_TC - 1 < num_TC:
                output_data[:, :, i][output_data[:, :, i] == 1] = 2

    return output_data


def brats_validation_cut_post(input_dir_path, output_dir_path, patient_ID):
    image_flair = np.zeros((155, img_H, img_W, 1), dtype=np.float)
    image_t1 = np.zeros((155, img_H, img_W, 1), dtype=np.float)
    image_t1ce = np.zeros((155, img_H, img_W, 1), dtype=np.float)
    image_t2 = np.zeros((155, img_H, img_W, 1), dtype=np.float)

    image_predict = np.zeros((240, 240, 155), dtype=np.float)

    path_flair = os.path.join(input_dir_path + patient_ID + '/' + patient_ID + '_flair.nii.gz')
    path_t1 = os.path.join(input_dir_path + patient_ID + '/' + patient_ID + '_t1.nii.gz')
    path_t1ce = os.path.join(input_dir_path + patient_ID + '/' + patient_ID + '_t1ce.nii.gz')
    path_t2 = os.path.join(input_dir_path + patient_ID + '/' + patient_ID + '_t2.nii.gz')

    flair_temp = nib.load(path_flair)
    flair_temp_arr = flair_temp.get_fdata()
    flair_temp_arr_sq = np.squeeze(flair_temp_arr)
    flair_temp_arr_sq = extract_patch(flair_temp_arr_sq, img_H, img_W, img_H_s, img_H_e, img_W_s, img_W_e)

    t1_temp = nib.load(path_t1)
    t1_temp_arr = t1_temp.get_fdata()
    t1_temp_arr_sq = np.squeeze(t1_temp_arr)
    t1_temp_arr_sq = extract_patch(t1_temp_arr_sq, img_H, img_W, img_H_s, img_H_e, img_W_s, img_W_e)

    t1ce_temp = nib.load(path_t1ce)
    t1ce_temp_arr = t1ce_temp.get_fdata()
    t1ce_temp_arr_sq = np.squeeze(t1ce_temp_arr)
    t1ce_temp_arr_sq = extract_patch(t1ce_temp_arr_sq, img_H, img_W, img_H_s, img_H_e, img_W_s, img_W_e)

    t2_temp = nib.load(path_t2)
    t2_temp_arr = t2_temp.get_fdata()
    t2_temp_arr_sq = np.squeeze(t2_temp_arr)
    t2_temp_arr_sq = extract_patch(t2_temp_arr_sq, img_H, img_W, img_H_s, img_H_e, img_W_s, img_W_e)

    affine_array = flair_temp.affine

    ###################### calculate cutting condition

    background_flair = np.zeros(155, dtype=np.float)
    for k in range(155):
        background_flair[k] = count_pixel(flair_temp_arr_sq[:, :, k], 0)
        background_flair[k] = background_flair[k] / img_H / img_W

    contour_ssim_flair = np.zeros(155, dtype=np.float)
    signal_ssim_flair = np.zeros(155, dtype=np.float)

    for i in range(155):
        contour_ssim_flair[i], signal_ssim_flair[i] = count_contour_ssim(flair_temp_arr_sq[:, :, i])

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

    for i in range(155):
        image_flair[i, :, :, 0] = flair_temp_arr_sq[:, :, i]
        image_t1[i, :, :, 0] = t1_temp_arr_sq[:, :, i]
        image_t1ce[i, :, :, 0] = t1ce_temp_arr_sq[:, :, i]
        image_t2[i, :, :, 0] = t2_temp_arr_sq[:, :, i]

    batch_flair = np.zeros((1, img_H, img_W, 1), dtype=np.float)
    batch_t1 = np.zeros((1, img_H, img_W, 1), dtype=np.float)
    batch_t1ce = np.zeros((1, img_H, img_W, 1), dtype=np.float)
    batch_t2 = np.zeros((1, img_H, img_W, 1), dtype=np.float)

    number_slide = 0

    for i in range(155):
        if i <= 77 and background_flair[i] <= 0.90 and contour_ssim_flair[i] >= 0.35 and signal_ssim_flair[
            i] <= 0.58:
            batch_flair[0, :, :, 0] = image_flair[i, :, :, 0]
            batch_t1[0, :, :, 0] = image_t1[i, :, :, 0]
            batch_t1ce[0, :, :, 0] = image_t1ce[i, :, :, 0]
            batch_t2[0, :, :, 0] = image_t2[i, :, :, 0]

            t_predict_image = sess.run(predict_image, feed_dict={flair_in: batch_flair, t1_in: batch_t1,
                                                                 t1ce_in: batch_t1ce, t2_in: batch_t2,
                                                                 tst: True})
            t_predict_image_p = np.squeeze(t_predict_image)
            t_predict_image_p[t_predict_image_p == 3] = 4
            image_predict[img_H_s:img_H_e, img_W_s:img_W_e, i] = t_predict_image_p[:, :]

            number_slide = number_slide + 1

        if i > 77 and background_flair[i] < 1 and signal_ssim_flair[i] <= 0.58:
            batch_flair[0, :, :, 0] = image_flair[i, :, :, 0]
            batch_t1[0, :, :, 0] = image_t1[i, :, :, 0]
            batch_t1ce[0, :, :, 0] = image_t1ce[i, :, :, 0]
            batch_t2[0, :, :, 0] = image_t2[i, :, :, 0]

            t_predict_image = sess.run(predict_image, feed_dict={flair_in: batch_flair, t1_in: batch_t1,
                                                                 t1ce_in: batch_t1ce, t2_in: batch_t2,
                                                                 tst: True})
            t_predict_image_p = np.squeeze(t_predict_image)
            t_predict_image_p[t_predict_image_p == 3] = 4
            image_predict[img_H_s:img_H_e, img_W_s:img_W_e, i] = t_predict_image_p[:, :]

            number_slide = number_slide + 1

    image_post = post_processing(image_predict, 7, 6, 1)

    image_post = image_post.astype(int16)
    new_image_post = nib.Nifti1Image(image_post, affine_array)
    nib.save(new_image_post, os.path.join(output_dir_path + patient_ID + '.nii.gz'))


# num_validation_patient = len(brats_validation_ID)


output_path_cut_post = './Validation_' + branch_name + '_submit_CutPost/'
if not os.path.exists(output_path_cut_post):
    os.makedirs(output_path_cut_post)

print('start validation (cut and post)')

for j in range(num_validation_patient):
    brats_validation_cut_post(input_path, output_path_cut_post, brats_validation_ID[j])

    if j == round(num_validation_patient / 4):
        print('rate of progress 25%')
    if j == round(num_validation_patient / 2):
        print('rate of progress 50%')
    if j == round(num_validation_patient / 4 * 3):
        print('rate of progress 75%')
    if j == num_validation_patient - 1:
        print('rate of progress 100%')

print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
validation_cutandpost_end_time = datetime.datetime.now()
print('validation_time_cut_post(s):', (validation_cutandpost_end_time - validation_cut_end_time).seconds)


###############################################################
# test training-set
##############################################################

def calculate_dice(predict_image, seg_image, patient_ID):
    predict_image = predict_image.astype(np.float)
    seg_image = seg_image.astype(np.float)

    shape_image = np.shape(seg_image)

    predict_image = predict_image.reshape((shape_image[0], shape_image[1] * shape_image[2]))
    seg_image = seg_image.reshape((shape_image[0], shape_image[1] * shape_image[2]))

    #################
    predict_ET = np.zeros((shape_image[0], shape_image[1] * shape_image[2]), dtype=np.float)
    seg_ET = np.zeros((shape_image[0], shape_image[1] * shape_image[2]), dtype=np.float)

    predict_WT = np.zeros((shape_image[0], shape_image[1] * shape_image[2]), dtype=np.float)
    seg_WT = np.zeros((shape_image[0], shape_image[1] * shape_image[2]), dtype=np.float)

    predict_TC = np.zeros((shape_image[0], shape_image[1] * shape_image[2]), dtype=np.float)
    seg_TC = np.zeros((shape_image[0], shape_image[1] * shape_image[2]), dtype=np.float)

    ###############
    predict_ET[:, :] = predict_image[:, :]
    predict_ET[predict_ET == 1] = 0
    predict_ET[predict_ET == 2] = 0
    predict_ET[predict_ET == 4] = 1
    seg_ET[:, :] = seg_image[:, :]
    seg_ET[seg_ET == 1] = 0
    seg_ET[seg_ET == 2] = 0
    seg_ET[seg_ET == 4] = 1

    predict_WT[:, :] = predict_image[:, :]
    predict_WT[predict_WT == 1] = 1
    predict_WT[predict_WT == 2] = 1
    predict_WT[predict_WT == 4] = 1
    seg_WT[:, :] = seg_image[:, :]
    seg_WT[seg_WT == 1] = 1
    seg_WT[seg_WT == 2] = 1
    seg_WT[seg_WT == 4] = 1

    predict_TC[:, :] = predict_image[:, :]
    predict_TC[predict_TC == 1] = 1
    predict_TC[predict_TC == 2] = 0
    predict_TC[predict_TC == 4] = 1
    seg_TC[:, :] = seg_image[:, :]
    seg_TC[seg_TC == 1] = 1
    seg_TC[seg_TC == 2] = 0
    seg_TC[seg_TC == 4] = 1

    ep = 0.0000001
    if np.sum(seg_ET) == 0:
        dice_ET = 1
    else:
        dice_ET = 2 * np.sum(seg_ET * predict_ET) / (np.sum(seg_ET) + np.sum(predict_ET) + ep)
    dice_WT = 2 * np.sum(seg_WT * predict_WT) / (np.sum(seg_WT) + np.sum(predict_WT) + ep)
    dice_TC = 2 * np.sum(seg_TC * predict_TC) / (np.sum(seg_TC) + np.sum(predict_TC) + ep)

    print(patient_ID, dice_ET, dice_WT, dice_TC)
    return dice_ET, dice_WT, dice_TC


def brats_test_trainingset(input_dir_path, output_dir_path, patient_path):
    image_flair = np.zeros((155, img_H, img_W, 1), dtype=np.float)
    image_t1 = np.zeros((155, img_H, img_W, 1), dtype=np.float)
    image_t1ce = np.zeros((155, img_H, img_W, 1), dtype=np.float)
    image_t2 = np.zeros((155, img_H, img_W, 1), dtype=np.float)

    image_predict = np.zeros((240, 240, 155), dtype=np.float)
    image_seg = np.zeros((240, 240, 155), dtype=np.float)

    zero_pixel_percent = np.zeros(155, dtype=np.float)

    path_flair = os.path.join(input_dir_path + patient_path + '/' + patient_path[4:len(patient_path)] + '_flair.nii.gz')
    path_t1 = os.path.join(input_dir_path + patient_path + '/' + patient_path[4:len(patient_path)] + '_t1.nii.gz')
    path_t1ce = os.path.join(input_dir_path + patient_path + '/' + patient_path[4:len(patient_path)] + '_t1ce.nii.gz')
    path_t2 = os.path.join(input_dir_path + patient_path + '/' + patient_path[4:len(patient_path)] + '_t2.nii.gz')
    path_seg = os.path.join(input_dir_path + patient_path + '/' + patient_path[4:len(patient_path)] + '_seg.nii.gz')

    flair_temp = nib.load(path_flair)
    flair_temp_arr = flair_temp.get_fdata()
    flair_temp_arr_sq = np.squeeze(flair_temp_arr)
    flair_temp_arr_sq = extract_patch(flair_temp_arr_sq, img_H, img_W, img_H_s, img_H_e, img_W_s, img_W_e)

    t1_temp = nib.load(path_t1)
    t1_temp_arr = t1_temp.get_fdata()
    t1_temp_arr_sq = np.squeeze(t1_temp_arr)
    t1_temp_arr_sq = extract_patch(t1_temp_arr_sq, img_H, img_W, img_H_s, img_H_e, img_W_s, img_W_e)

    t1ce_temp = nib.load(path_t1ce)
    t1ce_temp_arr = t1ce_temp.get_fdata()
    t1ce_temp_arr_sq = np.squeeze(t1ce_temp_arr)
    t1ce_temp_arr_sq = extract_patch(t1ce_temp_arr_sq, img_H, img_W, img_H_s, img_H_e, img_W_s, img_W_e)

    t2_temp = nib.load(path_t2)
    t2_temp_arr = t2_temp.get_fdata()
    t2_temp_arr_sq = np.squeeze(t2_temp_arr)
    t2_temp_arr_sq = extract_patch(t2_temp_arr_sq, img_H, img_W, img_H_s, img_H_e, img_W_s, img_W_e)

    seg_temp = nib.load(path_seg)
    seg_temp_arr = seg_temp.get_fdata()
    seg_temp_arr_sq = np.squeeze(seg_temp_arr)

    affine_array = flair_temp.affine

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

    for i in range(155):
        image_flair[i, :, :, 0] = flair_temp_arr_sq[:, :, i]
        image_t1[i, :, :, 0] = t1_temp_arr_sq[:, :, i]
        image_t1ce[i, :, :, 0] = t1ce_temp_arr_sq[:, :, i]
        image_t2[i, :, :, 0] = t2_temp_arr_sq[:, :, i]

        image_seg[:, :, i] = seg_temp_arr_sq[:, :, i]

    batch_flair = np.zeros((1, img_H, img_W, 1), dtype=np.float)
    batch_t1 = np.zeros((1, img_H, img_W, 1), dtype=np.float)
    batch_t1ce = np.zeros((1, img_H, img_W, 1), dtype=np.float)
    batch_t2 = np.zeros((1, img_H, img_W, 1), dtype=np.float)

    for i in range(155):
        batch_flair[0, :, :, 0] = image_flair[i, :, :, 0]
        batch_t1[0, :, :, 0] = image_t1[i, :, :, 0]
        batch_t1ce[0, :, :, 0] = image_t1ce[i, :, :, 0]
        batch_t2[0, :, :, 0] = image_t2[i, :, :, 0]

        t_predict_image = sess.run(predict_image, feed_dict={flair_in: batch_flair, t1_in: batch_t1,
                                                             t1ce_in: batch_t1ce, t2_in: batch_t2,
                                                             tst: True})
        t_predict_image_p = np.squeeze(t_predict_image)
        t_predict_image_p[t_predict_image_p == 3] = 4
        image_predict[img_H_s:img_H_e, img_W_s:img_W_e, i] = t_predict_image_p[:, :]

    image_predict = image_predict.astype(int16)
    new_image_predict = nib.Nifti1Image(image_predict, affine_array)
    nib.save(new_image_predict, os.path.join(output_dir_path + patient_path[4:len(patient_path)] + '.nii.gz'))

    dice_ET, dice_WT, dice_TC = calculate_dice(image_predict, image_seg, patient_path)
    return dice_ET, dice_WT, dice_TC


output_path_tts = './TTS_' + branch_name + '_submit/'
if not os.path.exists(output_path_tts):
    os.makedirs(output_path_tts)

dice_trainingset = np.zeros((training_set.num_floder, 3), dtype=np.float)  # Dice_ET, Dice_WT, Dice_TC

print('dice:')
print('patient_ID, Dice_ET, Dice_WT, Dice_TC')

for j in range(training_set.num_floder):
    dice_trainingset[j, 0], dice_trainingset[j, 1], dice_trainingset[j, 2] = brats_test_trainingset(
        training_set.dir_path, output_path_tts, training_set.brats_name[j])

np.savetxt(('training_' + branch_name + '_DiceOfTrainingSet.txt'), dice_trainingset, fmt='%f')
print('mean dice:')
print(np.mean(dice_trainingset[:, 0]), np.mean(dice_trainingset[:, 1]), np.mean(dice_trainingset[:, 2]))


######################################
# test training-set, cut
#######################################

def brats_test_trainingset_cut(input_dir_path, output_dir_path, patient_path):
    image_flair = np.zeros((155, img_H, img_W, 1), dtype=np.float)
    image_t1 = np.zeros((155, img_H, img_W, 1), dtype=np.float)
    image_t1ce = np.zeros((155, img_H, img_W, 1), dtype=np.float)
    image_t2 = np.zeros((155, img_H, img_W, 1), dtype=np.float)

    image_predict = np.zeros((240, 240, 155), dtype=np.float)
    image_seg = np.zeros((240, 240, 155), dtype=np.float)

    zero_pixel_percent = np.zeros(155, dtype=np.float)

    path_flair = os.path.join(input_dir_path + patient_path + '/' + patient_path[4:len(patient_path)] + '_flair.nii.gz')
    path_t1 = os.path.join(input_dir_path + patient_path + '/' + patient_path[4:len(patient_path)] + '_t1.nii.gz')
    path_t1ce = os.path.join(input_dir_path + patient_path + '/' + patient_path[4:len(patient_path)] + '_t1ce.nii.gz')
    path_t2 = os.path.join(input_dir_path + patient_path + '/' + patient_path[4:len(patient_path)] + '_t2.nii.gz')
    path_seg = os.path.join(input_dir_path + patient_path + '/' + patient_path[4:len(patient_path)] + '_seg.nii.gz')

    flair_temp = nib.load(path_flair)
    flair_temp_arr = flair_temp.get_fdata()
    flair_temp_arr_sq = np.squeeze(flair_temp_arr)
    flair_temp_arr_sq = extract_patch(flair_temp_arr_sq, img_H, img_W, img_H_s, img_H_e, img_W_s, img_W_e)

    t1_temp = nib.load(path_t1)
    t1_temp_arr = t1_temp.get_fdata()
    t1_temp_arr_sq = np.squeeze(t1_temp_arr)
    t1_temp_arr_sq = extract_patch(t1_temp_arr_sq, img_H, img_W, img_H_s, img_H_e, img_W_s, img_W_e)

    t1ce_temp = nib.load(path_t1ce)
    t1ce_temp_arr = t1ce_temp.get_fdata()
    t1ce_temp_arr_sq = np.squeeze(t1ce_temp_arr)
    t1ce_temp_arr_sq = extract_patch(t1ce_temp_arr_sq, img_H, img_W, img_H_s, img_H_e, img_W_s, img_W_e)

    t2_temp = nib.load(path_t2)
    t2_temp_arr = t2_temp.get_fdata()
    t2_temp_arr_sq = np.squeeze(t2_temp_arr)
    t2_temp_arr_sq = extract_patch(t2_temp_arr_sq, img_H, img_W, img_H_s, img_H_e, img_W_s, img_W_e)

    seg_temp = nib.load(path_seg)
    seg_temp_arr = seg_temp.get_fdata()
    seg_temp_arr_sq = np.squeeze(seg_temp_arr)

    affine_array = flair_temp.affine

    ###################### calculate cutting condition

    background_flair = np.zeros(155, dtype=np.float)
    for k in range(155):
        background_flair[k] = count_pixel(flair_temp_arr_sq[:, :, k], 0)
        background_flair[k] = background_flair[k] / img_H / img_W

    contour_ssim_flair = np.zeros(155, dtype=np.float)
    signal_ssim_flair = np.zeros(155, dtype=np.float)

    for i in range(155):
        contour_ssim_flair[i], signal_ssim_flair[i] = count_contour_ssim(flair_temp_arr_sq[:, :, i])

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

    for i in range(155):
        image_flair[i, :, :, 0] = flair_temp_arr_sq[:, :, i]
        image_t1[i, :, :, 0] = t1_temp_arr_sq[:, :, i]
        image_t1ce[i, :, :, 0] = t1ce_temp_arr_sq[:, :, i]
        image_t2[i, :, :, 0] = t2_temp_arr_sq[:, :, i]

        image_seg[:, :, i] = seg_temp_arr_sq[:, :, i]

    batch_flair = np.zeros((1, img_H, img_W, 1), dtype=np.float)
    batch_t1 = np.zeros((1, img_H, img_W, 1), dtype=np.float)
    batch_t1ce = np.zeros((1, img_H, img_W, 1), dtype=np.float)
    batch_t2 = np.zeros((1, img_H, img_W, 1), dtype=np.float)

    number_slide = 0

    for i in range(155):
        if i <= 77 and background_flair[i] <= 0.90 and contour_ssim_flair[i] >= 0.35 and signal_ssim_flair[
            i] <= 0.58:
            batch_flair[0, :, :, 0] = image_flair[i, :, :, 0]
            batch_t1[0, :, :, 0] = image_t1[i, :, :, 0]
            batch_t1ce[0, :, :, 0] = image_t1ce[i, :, :, 0]
            batch_t2[0, :, :, 0] = image_t2[i, :, :, 0]

            t_predict_image = sess.run(predict_image, feed_dict={flair_in: batch_flair, t1_in: batch_t1,
                                                                 t1ce_in: batch_t1ce, t2_in: batch_t2,
                                                                 tst: True})
            t_predict_image_p = np.squeeze(t_predict_image)
            t_predict_image_p[t_predict_image_p == 3] = 4
            image_predict[img_H_s:img_H_e, img_W_s:img_W_e, i] = t_predict_image_p[:, :]

            number_slide = number_slide + 1

        if i > 77 and background_flair[i] < 1 and signal_ssim_flair[i] <= 0.58:
            batch_flair[0, :, :, 0] = image_flair[i, :, :, 0]
            batch_t1[0, :, :, 0] = image_t1[i, :, :, 0]
            batch_t1ce[0, :, :, 0] = image_t1ce[i, :, :, 0]
            batch_t2[0, :, :, 0] = image_t2[i, :, :, 0]

            t_predict_image = sess.run(predict_image, feed_dict={flair_in: batch_flair, t1_in: batch_t1,
                                                                 t1ce_in: batch_t1ce, t2_in: batch_t2,
                                                                 tst: True})
            t_predict_image_p = np.squeeze(t_predict_image)
            t_predict_image_p[t_predict_image_p == 3] = 4
            image_predict[img_H_s:img_H_e, img_W_s:img_W_e, i] = t_predict_image_p[:, :]

            number_slide = number_slide + 1

    image_predict = image_predict.astype(int16)
    new_image_predict = nib.Nifti1Image(image_predict, affine_array)
    nib.save(new_image_predict, os.path.join(output_dir_path + patient_path[4:len(patient_path)] + '.nii.gz'))

    dice_ET, dice_WT, dice_TC = calculate_dice(image_predict, image_seg, patient_path)
    return dice_ET, dice_WT, dice_TC, number_slide


output_path_tts_cut = './TTS_' + branch_name + '_submit_Cut/'
if not os.path.exists(output_path_tts_cut):
    os.makedirs(output_path_tts_cut)

dice_trainingset_cut = np.zeros((training_set.num_floder, 3), dtype=np.float)  # Dice_ET, Dice_WT, Dice_TC

print('dice (cut):')
print('patient_ID, Dice_ET, Dice_WT, Dice_TC')

slide_number_tts = np.zeros(training_set.num_floder, dtype=np.int)

for j in range(training_set.num_floder):
    dice_trainingset_cut[j, 0], dice_trainingset_cut[j, 1], dice_trainingset_cut[j, 2], slide_number_tts[
        j] = brats_test_trainingset_cut(
        training_set.dir_path, output_path_tts_cut, training_set.brats_name[j])

np.savetxt(('training_' + branch_name + '_DiceOfTrainingSet_cut.txt'), dice_trainingset_cut, fmt='%f')
print('mean dice (cut):')
print(np.mean(dice_trainingset_cut[:, 0]), np.mean(dice_trainingset_cut[:, 1]), np.mean(dice_trainingset_cut[:, 2]))

for j in range(training_set.num_floder):
    print(training_set.brats_name[j], slide_number_tts[j])
print('Average number of slides: ', np.mean(slide_number_tts))


######################################
# test training-set, cut, post-processing
#######################################

def brats_test_trainingset_cut_post(input_dir_path, output_dir_path, patient_path):
    image_flair = np.zeros((155, img_H, img_W, 1), dtype=np.float)
    image_t1 = np.zeros((155, img_H, img_W, 1), dtype=np.float)
    image_t1ce = np.zeros((155, img_H, img_W, 1), dtype=np.float)
    image_t2 = np.zeros((155, img_H, img_W, 1), dtype=np.float)

    image_predict = np.zeros((240, 240, 155), dtype=np.float)
    image_seg = np.zeros((240, 240, 155), dtype=np.float)

    zero_pixel_percent = np.zeros(155, dtype=np.float)

    path_flair = os.path.join(input_dir_path + patient_path + '/' + patient_path[4:len(patient_path)] + '_flair.nii.gz')
    path_t1 = os.path.join(input_dir_path + patient_path + '/' + patient_path[4:len(patient_path)] + '_t1.nii.gz')
    path_t1ce = os.path.join(input_dir_path + patient_path + '/' + patient_path[4:len(patient_path)] + '_t1ce.nii.gz')
    path_t2 = os.path.join(input_dir_path + patient_path + '/' + patient_path[4:len(patient_path)] + '_t2.nii.gz')
    path_seg = os.path.join(input_dir_path + patient_path + '/' + patient_path[4:len(patient_path)] + '_seg.nii.gz')

    flair_temp = nib.load(path_flair)
    flair_temp_arr = flair_temp.get_fdata()
    flair_temp_arr_sq = np.squeeze(flair_temp_arr)
    flair_temp_arr_sq = extract_patch(flair_temp_arr_sq, img_H, img_W, img_H_s, img_H_e, img_W_s, img_W_e)

    t1_temp = nib.load(path_t1)
    t1_temp_arr = t1_temp.get_fdata()
    t1_temp_arr_sq = np.squeeze(t1_temp_arr)
    t1_temp_arr_sq = extract_patch(t1_temp_arr_sq, img_H, img_W, img_H_s, img_H_e, img_W_s, img_W_e)

    t1ce_temp = nib.load(path_t1ce)
    t1ce_temp_arr = t1ce_temp.get_fdata()
    t1ce_temp_arr_sq = np.squeeze(t1ce_temp_arr)
    t1ce_temp_arr_sq = extract_patch(t1ce_temp_arr_sq, img_H, img_W, img_H_s, img_H_e, img_W_s, img_W_e)

    t2_temp = nib.load(path_t2)
    t2_temp_arr = t2_temp.get_fdata()
    t2_temp_arr_sq = np.squeeze(t2_temp_arr)
    t2_temp_arr_sq = extract_patch(t2_temp_arr_sq, img_H, img_W, img_H_s, img_H_e, img_W_s, img_W_e)

    seg_temp = nib.load(path_seg)
    seg_temp_arr = seg_temp.get_fdata()
    seg_temp_arr_sq = np.squeeze(seg_temp_arr)

    affine_array = flair_temp.affine

    ###################### calculate cutting condition

    background_flair = np.zeros(155, dtype=np.float)
    for k in range(155):
        background_flair[k] = count_pixel(flair_temp_arr_sq[:, :, k], 0)
        background_flair[k] = background_flair[k] / img_H / img_W

    contour_ssim_flair = np.zeros(155, dtype=np.float)
    signal_ssim_flair = np.zeros(155, dtype=np.float)

    for i in range(155):
        contour_ssim_flair[i], signal_ssim_flair[i] = count_contour_ssim(flair_temp_arr_sq[:, :, i])

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

    for i in range(155):
        image_flair[i, :, :, 0] = flair_temp_arr_sq[:, :, i]
        image_t1[i, :, :, 0] = t1_temp_arr_sq[:, :, i]
        image_t1ce[i, :, :, 0] = t1ce_temp_arr_sq[:, :, i]
        image_t2[i, :, :, 0] = t2_temp_arr_sq[:, :, i]

        image_seg[:, :, i] = seg_temp_arr_sq[:, :, i]

    batch_flair = np.zeros((1, img_H, img_W, 1), dtype=np.float)
    batch_t1 = np.zeros((1, img_H, img_W, 1), dtype=np.float)
    batch_t1ce = np.zeros((1, img_H, img_W, 1), dtype=np.float)
    batch_t2 = np.zeros((1, img_H, img_W, 1), dtype=np.float)

    number_slide = 0

    for i in range(155):
        if i <= 77 and background_flair[i] <= 0.90 and contour_ssim_flair[i] >= 0.35 and signal_ssim_flair[
            i] <= 0.58:
            batch_flair[0, :, :, 0] = image_flair[i, :, :, 0]
            batch_t1[0, :, :, 0] = image_t1[i, :, :, 0]
            batch_t1ce[0, :, :, 0] = image_t1ce[i, :, :, 0]
            batch_t2[0, :, :, 0] = image_t2[i, :, :, 0]

            t_predict_image = sess.run(predict_image, feed_dict={flair_in: batch_flair, t1_in: batch_t1,
                                                                 t1ce_in: batch_t1ce, t2_in: batch_t2,
                                                                 tst: True})
            t_predict_image_p = np.squeeze(t_predict_image)
            t_predict_image_p[t_predict_image_p == 3] = 4
            image_predict[img_H_s:img_H_e, img_W_s:img_W_e, i] = t_predict_image_p[:, :]

            number_slide = number_slide + 1

        if i > 77 and background_flair[i] < 1 and signal_ssim_flair[i] <= 0.58:
            batch_flair[0, :, :, 0] = image_flair[i, :, :, 0]
            batch_t1[0, :, :, 0] = image_t1[i, :, :, 0]
            batch_t1ce[0, :, :, 0] = image_t1ce[i, :, :, 0]
            batch_t2[0, :, :, 0] = image_t2[i, :, :, 0]

            t_predict_image = sess.run(predict_image, feed_dict={flair_in: batch_flair, t1_in: batch_t1,
                                                                 t1ce_in: batch_t1ce, t2_in: batch_t2,
                                                                 tst: True})
            t_predict_image_p = np.squeeze(t_predict_image)
            t_predict_image_p[t_predict_image_p == 3] = 4
            image_predict[img_H_s:img_H_e, img_W_s:img_W_e, i] = t_predict_image_p[:, :]

            number_slide = number_slide + 1

    image_post = post_processing(image_predict, 7, 6, 1)

    image_post = image_post.astype(int16)
    new_image_post = nib.Nifti1Image(image_post, affine_array)
    nib.save(new_image_post, os.path.join(output_dir_path + patient_path[4:len(patient_path)] + '.nii.gz'))

    dice_ET, dice_WT, dice_TC = calculate_dice(image_post, image_seg, patient_path)
    return dice_ET, dice_WT, dice_TC, number_slide


output_path_tts_cut_post = './TTS_' + branch_name + '_submit_CutPost/'
if not os.path.exists(output_path_tts_cut_post):
    os.makedirs(output_path_tts_cut_post)

dice_trainingset_cut_post = np.zeros((training_set.num_floder, 3), dtype=np.float)  # Dice_ET, Dice_WT, Dice_TC

print('dice (cut and post-processing):')
print('patient_ID, Dice_ET, Dice_WT, Dice_TC')

slide_number_tts_post = np.zeros(training_set.num_floder, dtype=np.int)

for j in range(training_set.num_floder):
    dice_trainingset_cut_post[j, 0], dice_trainingset_cut_post[j, 1], dice_trainingset_cut_post[j, 2], \
    slide_number_tts_post[
        j] = brats_test_trainingset_cut_post(
        training_set.dir_path, output_path_tts_cut_post, training_set.brats_name[j])

np.savetxt(('training_' + branch_name + '_DiceOfTrainingSet_cutpost.txt'), dice_trainingset_cut_post, fmt='%f')
print('mean dice (cut and post-processing):')
print(np.mean(dice_trainingset_cut_post[:, 0]), np.mean(dice_trainingset_cut_post[:, 1]),
      np.mean(dice_trainingset_cut_post[:, 2]))

for j in range(training_set.num_floder):
    print(training_set.brats_name[j], slide_number_tts_post[j])
print('Average number of slides: ', np.mean(slide_number_tts_post))


