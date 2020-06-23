# some models for classification
import math
import tensorflow as tf
from tf_crf_layer import keras_utils
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Layer
from tensorflow.python.keras import backend as K
from tensorflow.python.keras import constraints
from tensorflow.python.keras import initializers
from tensorflow.python.keras import regularizers
from tensorflow.keras.layers import Embedding
from tensorflow.python.ops import math_ops



def lstm_cls(input_layer, output_dim):
    from tensorflow.keras.layers import Dense, Dropout, Flatten
    cls_flat = Flatten()(input_layer)
    cls_dropout = Dropout(0.2)(cls_flat)
    output_layer = Dense(output_dim, activation='softmax', name='cls_Dense')(cls_dropout)

    return output_layer


def dilated_cnn_cls(input_layer, output_dim):
    from tensorflow.keras.layers import Conv1D, MaxPooling1D, BatchNormalization, Dense, Dropout, Flatten
    input_layer = BatchNormalization()(input_layer)
    cls_conv_emb = Conv1D(32, 3, activation='relu', padding='same')(input_layer)
    cls_conv_emb = Conv1D(64, 3, activation='relu', padding='same')(cls_conv_emb)
    cls_conv_emb = MaxPooling1D(2)(cls_conv_emb)
    cls_conv_emb = Conv1D(128, 3, activation='relu', dilation_rate=1, padding='same')(cls_conv_emb)
    cls_conv_emb = Conv1D(128, 3, activation='relu', dilation_rate=2, padding='same')(cls_conv_emb)
    cls_conv_emb = Conv1D(128, 3, activation='relu', dilation_rate=5, padding='same')(cls_conv_emb)
    cls_conv_emb = Conv1D(256, 1, activation='relu', padding='same')(cls_conv_emb)
    cls_conv_emb = MaxPooling1D(2)(cls_conv_emb)

    cls_flat = BatchNormalization()(cls_conv_emb)
    cls_flat = Flatten()(cls_flat)
    classification_dense = Dropout(0.2)(cls_flat)
    output_layer = Dense(output_dim, activation='softmax', name='CLS')(classification_dense)

    return output_layer


def textcnn_cls(input_layer, output_dim, outputlayer_name='CLS'):
    from tensorflow.keras.layers import Conv1D, MaxPooling1D, concatenate, Dense, Dropout, Flatten
    kernel_sizes = [2, 3, 4]
    pooling_out = []
    cls_conv_layer = input_layer
    for kernel_size in kernel_sizes:
        cls_conv_layer = Conv1D(filters=128, kernel_size=kernel_size, strides=1)(cls_conv_layer)
        cls_pooling_layer = MaxPooling1D(pool_size=int(cls_conv_layer.shape[1]))(cls_conv_layer)
        pooling_out.append(cls_pooling_layer)
        #print("kernel_size: %s \t c.shape: %s \t p.shape: %s" % (kernel_size, str(cls_conv_layer.shape), str(cls_pooling_layer.shape)))
    pool_output = concatenate([p for p in pooling_out])
    cls_flat = Flatten()(pool_output)
    cls_flat = Dropout(0.2)(cls_flat)
    cls_dense = Dense(output_dim, activation='softmax', name=outputlayer_name)(cls_flat)

    return cls_dense


def fasttext_cls(input_layer, output_dim):
    from tensorflow.keras.layers import Dense, Flatten, Conv1D
    # fast_text generally use embedding layer as its input_layer
    # directly connect dense to classify
    word_conv = Conv1D(64, 3, activation='relu', padding='same')(input_layer)
    word_conv = Conv1D(64, 3, activation='relu', padding='same')(word_conv)
    cls_flat = Flatten()(word_conv)
    return Dense(output_dim, activation='softmax')(cls_flat)


def get_paragraph_vector(input_layer):
    from tensorflow.keras.layers import Conv1D, Flatten ,MaxPooling1D
    embedding_layer = input_layer
    paragraph_vector = Conv1D(32, 3, activation='relu', padding='same')(embedding_layer)
    paragraph_vector = Conv1D(64, 3, activation='relu', padding='same')(paragraph_vector)
    paragraph_vector = MaxPooling1D(2)(paragraph_vector)
    paragraph_vector = Conv1D(128, 3, activation='relu', dilation_rate=1, padding='same')(paragraph_vector)
    paragraph_vector = Conv1D(128, 3, activation='relu', dilation_rate=2, padding='same')(paragraph_vector)
    paragraph_vector = Conv1D(128, 3, activation='relu', dilation_rate=5, padding='same')(paragraph_vector)
    paragraph_vector = Conv1D(256, 1, activation='relu', padding='same')(paragraph_vector)
    paragraph_vector = MaxPooling1D(2)(paragraph_vector)
    paragraph_vector = Flatten()(paragraph_vector)

    return paragraph_vector


def Discriminator(A_layer, onetask_output_shape, output_dtype='int32'):
    '''
        "A func to concate a task's output and another's input, in other words, tasks cascade."

    :param gen_output_shape:
            Define a layer's shape to generate a Input Layer to accept some layer's Output;
    :param A_layer:
            The layer you want to concate with, always be a Input Layer;
    :return:
            A Model you can use like a Layer, which concates a branch's output and another's input.
    '''
    from tensorflow.keras.layers import Input
    from tensorflow.keras.models import Model
    another_inputs = A_layer
    onetask_output = Input(onetask_output_shape, dtype=output_dtype)
    concatenated_input = tf.keras.layers.concatenate([onetask_output, another_inputs])
    # Now start applying Convolution Layers on concatenated_input
    # Deconvolution Layers
    return Model(inputs=[onetask_output, another_inputs], outputs=concatenated_input, name='Discriminator')


def Discriminator_new(onetask_output_shape, output_dtype='int32'):
    '''
        "A func to concate a task's output and another's input, in other words, tasks cascade."

    :param gen_output_shape:
            Define a layer's shape to generate a Input Layer to accept some layer's Output;
    :param A_layer:
            The layer you want to concate with, such as a Embedding Layer;
    :return:
            A Model you can use like a Layer, which concates a branch's output and another's input.
    '''
    from tensorflow.keras.layers import Input
    from tensorflow.keras.models import Model
    onetask_output = Input(onetask_output_shape, dtype=output_dtype)

    # Now start applying Convolution Layers on concatenated_input
    # Deconvolution Layers
    return Model(inputs=onetask_output, outputs=onetask_output, name='Discriminator')


def get_ner_cls_output_tensor(ner_cls_layer, mapping_size=5):
    ''' get ner branch's classification results, output is a Tensor.
    '''
    from tensorflow.python.ops import math_ops
    cls_index_tensor = math_ops.cast(tf.keras.backend.argmax(ner_cls_layer, axis=-1), 'int32')
    cls_index_tensor = tf.expand_dims(cls_index_tensor, -1)
    # print_op = tf.print(cls_index_tensor)
    # with tf.control_dependencies([print_op]):
    cls_index_tensor = tf.multiply(cls_index_tensor, tf.ones((1, mapping_size), dtype='int32'))
    return cls_index_tensor


# Vitual Embedding
class VirtualEmbedding(Embedding):
    def __init__(self,
                 input_dim,
                 output_dim,
                 embeddings_initializer='uniform',
                 embeddings_regularizer=None,
                 activity_regularizer=None,
                 embeddings_constraint=None,
                 mask_zero=True,
                 input_length=None,
                 mask_length=None,
                 **kwargs):
        if 'input_shape' not in kwargs:
            if input_length:
                kwargs['input_shape'] = (input_length,)
            else:
                kwargs['input_shape'] = (None,)
        dtype = kwargs.pop('dtype', K.floatx())
        # We set autocast to False, as we do not want to cast floating- point inputs
        # to self.dtype. In call(), we cast to int32, and casting to self.dtype
        # before casting to int32 might cause the int32 values to be different due
        # to a loss of precision.
        kwargs['autocast'] = False
        super(Embedding, self).__init__(dtype=dtype, **kwargs)

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.embeddings_initializer = initializers.get(embeddings_initializer)
        self.embeddings_regularizer = regularizers.get(embeddings_regularizer)
        self.activity_regularizer = regularizers.get(activity_regularizer)
        self.embeddings_constraint = constraints.get(embeddings_constraint)
        self.mask_zero = mask_zero
        self.supports_masking = mask_zero
        self.input_length = input_length
        self.mask_length = mask_length

    def compute_mask(self, inputs, mask=None):
        if not self.mask_zero:
            return None
        if not self.mask_length:
            return None
        if self.input_length>self.mask_length:
            mask_ones = tf.ones((tf.shape(inputs)[0], self.mask_length), dtype='int32')
            mask_zeros = tf.zeros((tf.shape(inputs)[0], self.input_length-self.mask_length), dtype='int32')
            tmp_mask = tf.concat([mask_ones, mask_zeros], axis=-1)
            return math_ops.not_equal(tmp_mask, 0)
        else:
            return math_ops.greater_equal(inputs, 0)
        # return self.keras_masks

    def get_config(self):
        config = {
            'input_dim': self.input_dim,
            'output_dim': self.output_dim,
            'embeddings_initializer':
                initializers.serialize(self.embeddings_initializer),
            'embeddings_regularizer':
                regularizers.serialize(self.embeddings_regularizer),
            'activity_regularizer':
                regularizers.serialize(self.activity_regularizer),
            'embeddings_constraint':
                constraints.serialize(self.embeddings_constraint),
            'mask_zero': self.mask_zero,
            'input_length': self.input_length,
            'mask_length': self.mask_length
        }
        base_config = super(Embedding, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))



# ArcFace
# Original paper: https://arxiv.org/pdf/1801.07698.pdf
# Original implementation: https://github.com/deepinsight/insightface
# Adapted from tensorflow implementation: https://github.com/luckycallor/InsightFace-tensorflow

# keras version
@keras_utils.register_keras_custom_object
class ArcFace(Layer):
    '''Custom Keras layer implementing ArcFace including:
    1. Generation of embeddings
    2. Loss function
    3. Accuracy function
    cite: https://github.com/ktjonsson/keras-ArcFace/tree/master/src
    '''

    def __init__(self, class_num, margin=0.5, scale=64., **kwargs):
        self.class_num = int(class_num)
        self.margin = margin
        self.scale = scale
        assert self.scale > 0.0
        assert self.margin >= 0.0
        assert self.margin < (math.pi / 2)

        self.cos_m = tf.math.cos(margin)
        self.sin_m = tf.math.sin(margin)
        self.mm = self.sin_m * margin
        self.threshold = tf.math.cos(tf.constant(math.pi) - margin)
        super(ArcFace, self).__init__(**kwargs)

    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        self.kernel = self.add_weight(name='kernel',
                                      shape=(input_shape[1], self.class_num),
                                      initializer='glorot_normal',
                                      trainable=True)
        super(ArcFace, self).build(input_shape)  # Be sure to call this at the end

    def call(self, x):
        embeddings = tf.nn.l2_normalize(x, axis=1, name='normed_embeddings')
        weights = tf.nn.l2_normalize(self.kernel, axis=0, name='normed_weights')
        cos_t = tf.matmul(embeddings, weights, name='cos_t')
        logits = self.get_logits(cos_t)
        softmax_prob = tf.nn.softmax(logits, axis=-1)
        # print_op = tf.print(logits, softmax_prob)
        # with tf.control_dependencies([print_op]):
            #softmax_prob=tf.identity(softmax_prob)
        return softmax_prob

    def get_logits(self, y_pred):
        cos_t = y_pred
        cos_t2 = tf.square(cos_t, name='cos_2')
        sin_t2 = tf.subtract(1., cos_t2, name='sin_2')
        sin_t = tf.sqrt(sin_t2, name='sin_t')
        cos_mt = self.scale * tf.subtract(tf.multiply(cos_t, self.cos_m), tf.multiply(sin_t, self.sin_m), name='cos_mt')
        cond_v = cos_t - self.threshold
        cond = tf.cast(tf.nn.relu(cond_v, name='if_else'), dtype=tf.bool)
        keep_val = self.scale * (cos_t - self.mm)
        cos_mt_temp = tf.where(cond, cos_mt, keep_val)
        labels = K.argmax(cos_t2, axis=-1)
        mask = tf.one_hot(labels, depth=self.class_num, name='one_hot_mask')
        inv_mask = tf.subtract(1., mask, name='inverse_mask')
        s_cos_t = tf.multiply(self.scale, cos_t, name='scalar_cos_t')
        logits = tf.add(tf.multiply(s_cos_t, inv_mask), tf.multiply(cos_mt_temp, mask), name='arcface_logits')
        return logits

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.class_num)

    def get_config(self):
        config = {
            'class_num': self.class_num,
            'margin': self.margin,
            'scale': self.scale,
        }

        base_config = super().get_config().copy()
        return dict(list(base_config.items()) + list(config.items()))


