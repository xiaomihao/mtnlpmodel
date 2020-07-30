from collections import Counter
import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K


def class_counter(data, encoder):
    cls_counter = Counter([sample.label for sample in data])
    classes_num = [0]*encoder.size()
    for label in cls_counter.keys():
        index = encoder.lookup(label)
        classes_num[index] = cls_counter[label]
    return classes_num


def get_class_weight(data, encoder):
    cls_weight = dict()
    data_len = len(data)
    cls_counter = class_counter(data, encoder)
    for key, val in enumerate(cls_counter):
        cls_weight[key] = pow(data_len/(val+1), 0.5)

    return cls_weight



def focal_loss(gamma=2, alpha=0.25):
    '''for binary classification
       focal loss paper:
       https://arxiv.org/pdf/1708.02002.pdf
    '''
    def focal_loss_fixed(y_true, y_pred):
        eps = 1e-12
        y_pred = K.clip(y_pred, eps, 1. - eps)  # free from log0
        pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
        pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))
        return -K.sum(alpha * K.pow(1. - pt_1, gamma) * K.log(pt_1)) - K.sum(
            (1 - alpha) * K.pow(pt_0, gamma) * K.log(1. - pt_0))

    return focal_loss_fixed


def focal_loss_mt(classes_num, gamma=2):
    '''focal loss paper:
       https://arxiv.org/pdf/1708.02002.pdf
    '''
    def mt_focal_loss_fixed(y_true, y_pred):
        from tensorflow.python.ops import array_ops
        eps = 1e-9
        y_pred = K.clip(y_pred, eps, 1. - eps)  # free from log0

        zeros = array_ops.zeros_like(y_pred, dtype=y_pred.dtype)
        classes_weight = array_ops.zeros_like(y_pred, dtype=y_pred.dtype)
        total_num = float(sum(classes_num))
        classes_w_t1 = np.power([total_num / (ff + 1) for ff in classes_num], 1/2)
        classes_w_t1 = np.power([ ff/(max(classes_w_t1)+eps) for ff in classes_w_t1 ], 1/2)
        classes_w_tensor = tf.convert_to_tensor(classes_w_t1, dtype=tf.float32)
        classes_weight += classes_w_tensor

        alpha = array_ops.where(tf.greater(y_true, zeros), classes_weight, zeros)  # alpha is related with ratio of each class

        weight = tf.multiply(y_true, tf.pow(tf.subtract(1., y_pred), gamma))
        ce = tf.multiply(y_true, -tf.log(y_pred))
        fl = tf.multiply(alpha, tf.multiply(weight, ce))
        reduce_fl = tf.reduce_max(fl, axis=1)

        return tf.reduce_mean(reduce_fl)

    return mt_focal_loss_fixed


def _abandon_focal_loss_mt(classes_num, gamma=2): # there are some bugs in this function, so abandon
    '''focal loss paper:
       https://arxiv.org/pdf/1708.02002.pdf
    '''
    def mt_focal_loss_fixed(y_true, y_pred):
        from tensorflow.python.ops import array_ops
        eps = 1e-9
        y_pred = K.clip(y_pred, eps, 1. - eps)  # free from log0

        zeros = array_ops.zeros_like(y_pred, dtype=y_pred.dtype)
        classes_weight = array_ops.zeros_like(y_pred, dtype=y_pred.dtype)
        total_num = float(sum(classes_num))
        classes_w_t1 = np.power([total_num / (ff + 1) for ff in classes_num], 1/2)
        classes_w_t1 = np.power([ ff/(max(classes_w_t1)+eps) for ff in classes_w_t1 ], 1/2)
        classes_w_tensor = tf.convert_to_tensor(classes_w_t1, dtype=tf.float32)
        classes_weight += classes_w_tensor

        alpha = array_ops.where(tf.greater(y_true, zeros), classes_weight, zeros)  # alpha is related with ratio of each class

        pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
        pt_0 = tf.where(tf.equal(y_true, 0), y_pred, (tf.zeros_like(y_pred)+K.epsilon()))
        return -K.sum(alpha * K.pow(1. - pt_1, gamma) * K.log(pt_1)) - K.sum(
            (1 - alpha) * K.pow(pt_0, gamma) * K.log(1. - pt_0))

    return mt_focal_loss_fixed


def triplet_loss(y_true, y_pred, alpha=0.2):
    """
    Implementation of the triplet loss as defined by formula (3)

    Arguments:
    y_true -- true labels, required when you define a loss in Keras, you don't need it in this function.
    y_pred -- python list containing three objects:
            anchor -- the encodings for the anchor images, of shape (None, 128)
            positive -- the encodings for the positive images, of shape (None, 128)
            negative -- the encodings for the negative images, of shape (None, 128)

    Returns:
    loss -- real number, value of the loss
    """

    anchor, positive, negative = y_pred[0], y_pred[1], y_pred[2]

    # Step 1: Compute the (encoding) distance between the anchor and the positive, you will need to sum over axis=-1
    pos_dist = K.sum(K.square(anchor - positive), axis=-1)
    # Step 2: Compute the (encoding) distance between the anchor and the negative, you will need to sum over axis=-1
    neg_dist = K.sum(K.square(anchor - negative), axis=-1)
    # Step 3: subtract the two previous distances and add alpha.
    basic_loss = pos_dist - neg_dist + alpha
    # Step 4: Take the maximum of basic_loss and 0.0. Sum over the training examples.
    loss = K.sum(K.maximum(basic_loss, 0))

    return loss
