#######################
# v3.2 168*200 FullReluNetDSFour
#######################
import tensorflow as tf
from functools import reduce
import math
from tensorflow.python.framework import graph_util


def stats_graph(graph):
    flops = tf.profiler.profile(graph, options=tf.profiler.ProfileOptionBuilder.float_operation())
    params = tf.profiler.profile(graph, options=tf.profiler.ProfileOptionBuilder.trainable_variables_parameter())
    print('FLOPs: {};    Trainable params: {}'.format(flops.total_float_ops, params.total_parameters))


def load_pb(pb):
    with tf.gfile.GFile(pb, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
    with tf.Graph().as_default() as graph:
        tf.import_graph_def(graph_def, name='')
        return graph


#################################################################

##################################################################
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



def instance_norm(x):
    mean, variance = tf.nn.moments(x, axes=[1,2], keep_dims=True)
    epsilon = 1e-5
    inv = tf.rsqrt(variance + epsilon)
    normalized = (x-mean)*inv
    return normalized


################################################################

###################################################################

with tf.Graph().as_default() as graph:
    sess = tf.Session()

    img_H = 168  # 168
    img_W = 200  # 200

    flair_in = tf.placeholder("float", shape=[128, img_H, img_W, 1])
    t1_in = tf.placeholder("float", shape=[128, img_H, img_W, 1])
    t1ce_in = tf.placeholder("float", shape=[128, img_H, img_W, 1])
    t2_in = tf.placeholder("float", shape=[128, img_H, img_W, 1])
    seg_in = tf.placeholder("float", shape=[128, img_H, img_W, 1])
    # test flag for batch norm
    tst = tf.placeholder(tf.bool)
    iter = tf.placeholder(tf.int32)

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

    ##########################################


    y_softmax = tf.nn.softmax(h_conv7, axis=3)
    predict_image = tf.argmax(y_softmax, 3, name='output')  # if label equal 3, change to 4.

    update_ema = tf.group(update_ema1_2, update_ema2_2,
                          update_ema3, update_ema4, update_ema5, update_ema6)

    sess.run(tf.global_variables_initializer())

    print('stats before freezing')
    stats_graph(graph)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        # ***** (2) freeze graph *****
        output_graph = graph_util.convert_variables_to_constants(sess, graph.as_graph_def(), ['output'])
        with tf.gfile.GFile('graph.pb', "wb") as f:
            f.write(output_graph.SerializeToString())

# ***** (3) Load frozen graph *****
graph = load_pb('./graph.pb')
print('stats after freezing')
stats_graph(graph)
