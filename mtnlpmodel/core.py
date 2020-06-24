import tensorflow as tf
from tensorflow.keras import Input
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (Embedding,
                                     Flatten,
                                     Dropout,
                                     Dense,
                                     Lambda,
                                     Bidirectional,
                                     LSTM,
                                     LayerNormalization,)
from tf_crf_layer.layer import CRF
from tf_attention_layer.layers.global_attentioin_layer import GlobalAttentionLayer


def cls_branch(arcloss_param, output_dims, feature_extractor, cls_emb_layer, ner_emb_layer=None, outputlayer_name='CLS'):
    if arcloss_param:  # for arc-softmax loss
        from mtnlpmodel.utils.model_util import ArcFace
        with tf.keras.backend.name_scope("CLS_branch"):
            # cls branch
            cls_feature_layer = feature_extractor(cls_emb_layer)
            cls_flat_lstm = Flatten()(cls_feature_layer)
            cls_flat = Dropout(0.25)(cls_flat_lstm)
            cls_vec_layer = Dense(128, activation='relu', name='arc_vector')
            cls_dense = cls_vec_layer(cls_flat)
            cls_layer = ArcFace(output_dims, margin=0.2, name=outputlayer_name)
            cls_arc = cls_layer(cls_dense)
            cls_output = cls_arc
            # ner cls branch
            if ner_emb_layer is not None:
                ner_cls_feature_layer = feature_extractor(ner_emb_layer)
                ner_cls_flat_lstm = Flatten()(ner_cls_feature_layer)
                ner_cls_vec_layer = cls_vec_layer(ner_cls_flat_lstm)
                ner_cls_layer = cls_layer(ner_cls_vec_layer)
            else:
                ner_cls_layer = None
    else:   # for softmax loss
        with tf.keras.backend.name_scope("CLS_branch"):
            # cls branch
            cls_feature_layer = feature_extractor(cls_emb_layer)
            cls_flat_lstm = Flatten()(cls_feature_layer)
            cls_flat = Dropout(0.25)(cls_flat_lstm)
            cls_dense = Dense(output_dims, activation='softmax', name=outputlayer_name)
            cls_output = cls_dense(cls_flat)
            # ner cls branch
            if ner_emb_layer is not None:
                ner_cls_feature_layer = feature_extractor(ner_emb_layer)
                ner_cls_flat_lstm = Flatten()(ner_cls_feature_layer)
                ner_cls_layer = cls_dense(ner_cls_flat_lstm)
            else:
                ner_cls_layer = None

    return ner_cls_layer, cls_output



def build_model(model_choice, **hyperparams):
    from mtnlpmodel.utils.model_util import (get_ner_cls_output_tensor_merge_embedding,
                                             get_ner_cls_output_tensor_merge_input)
    # get hyperparams
    EMBED_DIM = hyperparams['EMBED_DIM']
    CRF_PARAMS = hyperparams['CRF_PARAMS']
    BiLSTM_STACK_CONFIG = hyperparams['BiLSTM_STACK_CONFIG']
    CLS2NER_KEYWORD_LEN = hyperparams['CLS2NER_KEYWORD_LEN']
    USE_ATTENTION_LAYER = hyperparams['USE_ATTENTION_LAYER']
    tag_size = hyperparams['ner_tag_lookuper'].size()
    label_size = hyperparams['cls_label_lookuper'].size()
    vocab_size = hyperparams['vocabulary_lookuper'].size()

    # input layer
    input_length = hyperparams['MAX_SENTENCE_LEN']
    ner_input_layer = Input(shape=(input_length,), dtype='int32', name='ner_input')
    cls_input_layer = Input(shape=(input_length,), dtype='int32', name='cls_input')

    # encoder
    if model_choice=='VIRTUAL_EMBEDDING':   # cls_out embedding merged to ner_input_embedding as virtual embedding
        from mtnlpmodel.utils.model_util import VirtualEmbedding, Discriminator_new
        with tf.keras.backend.name_scope("Encoder"):
            embedding_layer_vocab = Embedding(vocab_size,
                                              EMBED_DIM,
                                              mask_zero=True,
                                              input_length=input_length,
                                              name='embedding_vocab'
                                              )
            embedding_layer_virtual = VirtualEmbedding(label_size,
                                                       EMBED_DIM,
                                                       mask_zero=True,
                                                       input_length=CLS2NER_KEYWORD_LEN,
                                                       mask_length=CLS2NER_KEYWORD_LEN,
                                                       name='embedding_virtual',
                                                       )

            ner_embedding = embedding_layer_vocab(ner_input_layer)
            cls_embedding = embedding_layer_vocab(cls_input_layer)
            
            ner_embedding = Dropout(0.15)(ner_embedding)    # just like random erase
            cls_embedding = Dropout(0.15)(cls_embedding)

        with tf.keras.backend.name_scope("Feature_extractor"):
            for bilstm_config in BiLSTM_STACK_CONFIG:
                biLSTM = Bidirectional(LSTM(return_sequences=True, **bilstm_config, name='biLSTM'))
            bilstm_extrator = biLSTM

        # classification branch
        ner_cls_layer, cls_output = cls_branch(hyperparams['Arcloss'], 
                                               label_size, bilstm_extrator, 
                                               cls_embedding, ner_embedding,
                                               outputlayer_name='cls')
        ner_cls_output_shape = get_ner_cls_output_tensor_merge_embedding(CLS2NER_KEYWORD_LEN)(ner_cls_layer).shape
        ner_cls_output_layer = Lambda(get_ner_cls_output_tensor_merge_embedding(CLS2NER_KEYWORD_LEN), ner_cls_output_shape)(ner_cls_layer)

        # classification output will be used as a keyword adding to input of NER
        discriminator = Discriminator_new(onetask_output_shape=(CLS2NER_KEYWORD_LEN,),
                                          output_dtype='int32')
        ner_cls_input_layer = discriminator(ner_cls_output_layer)
        ner_virtual_embedding = embedding_layer_virtual(ner_cls_input_layer)
        ner_merged_embedding = tf.keras.layers.concatenate([ner_virtual_embedding, ner_embedding], axis=1)
        ner_branch_embedding = ner_merged_embedding

    elif model_choice=='CLS2NER_INPUT':   # cls_out merged to ner_input as virtual keywords
        from mtnlpmodel.utils.model_util import Discriminator
        from mtnlpmodel.utils.input_process_util import build_vacablookuper_from_list
        vocabs = list(hyperparams['vocabulary_lookuper'].inverse_index_table.values())
        cls_labels = list(hyperparams['cls_label_lookuper'].inverse_index_table.values())
        vocabs.extend(cls_labels)
        vocabulary_lookuper = build_vacablookuper_from_list(*vocabs)

        vocab_size = vocabulary_lookuper.size()
        with tf.keras.backend.name_scope("Encoder"):
            embedding_layer = Embedding(vocab_size,
                                        EMBED_DIM,
                                        mask_zero=True,
                                        input_length=input_length,
                                        )
            ner_embedding = embedding_layer(ner_input_layer)
            cls_embedding = embedding_layer(cls_input_layer)

            ner_embedding = Dropout(0.15)(ner_embedding)  # just like random erase
            cls_embedding = Dropout(0.15)(cls_embedding)

        with tf.keras.backend.name_scope("Feature_extractor"):
            for bilstm_config in BiLSTM_STACK_CONFIG:
                biLSTM = Bidirectional(LSTM(return_sequences=True, **bilstm_config, name='biLSTM'))
            bilstm_extrator = biLSTM

        # classification branch
        ner_cls_layer, cls_output = cls_branch(hyperparams['Arcloss'], 
                                               label_size, bilstm_extrator, 
                                               cls_embedding, ner_embedding,
                                               outputlayer_name='cls')
        ner_cls_output_shape = get_ner_cls_output_tensor_merge_input(CLS2NER_KEYWORD_LEN,
                                                                     **{"vocab_size":vocab_size,
                                                                        "label_size":label_size})(ner_cls_layer).shape

        ner_cls_output_layer = Lambda(get_ner_cls_output_tensor_merge_input(
                                         CLS2NER_KEYWORD_LEN,
                                         **{"vocab_size": vocab_size, "label_size": label_size}),
                                      ner_cls_output_shape)(ner_cls_layer)

        # classification output will be used as a keyword adding to input of NER
        discriminator = Discriminator(ner_input_layer, onetask_output_shape=(CLS2NER_KEYWORD_LEN,),
                                      output_dtype='int32')
        merged_ner_input_layer = discriminator([ner_cls_output_layer, ner_input_layer])
        ner_branch_embedding = embedding_layer(merged_ner_input_layer)

    else: # task independent
        with tf.keras.backend.name_scope("Encoder"):
            embedding_layer = Embedding(vocab_size,
                                        EMBED_DIM,
                                        mask_zero=True,
                                        input_length=input_length,
                                        )

            ner_embedding = embedding_layer(ner_input_layer)
            cls_embedding = embedding_layer(cls_input_layer)

            ner_embedding = Dropout(0.15)(ner_embedding)    # just like random erase
            cls_embedding = Dropout(0.15)(cls_embedding)

        with tf.keras.backend.name_scope("Feature_extractor"):
            for bilstm_config in BiLSTM_STACK_CONFIG:
                biLSTM = Bidirectional(LSTM(return_sequences=True, **bilstm_config, name='biLSTM'))
            bilstm_extrator = biLSTM

        # classification branch
        _, cls_output = cls_branch(hyperparams['Arcloss'], 
                                   label_size, bilstm_extrator,
                                   cls_embedding, outputlayer_name='cls')
        ner_branch_embedding = ner_embedding

    # NER branch
    with tf.keras.backend.name_scope("NER_branch"):
        # print_op = tf.print(ner_virtual_embedding._keras_mask, ner_embedding._keras_mask)
        # with tf.control_dependencies([print_op]):
        embedding_layer = LayerNormalization()(ner_branch_embedding)
        biLSTM = bilstm_extrator(embedding_layer)
        biLSTM = LayerNormalization()(biLSTM)
        if USE_ATTENTION_LAYER:
            biLSTM = GlobalAttentionLayer()(biLSTM)
        ner_output = CRF(tag_size, name="crf", **CRF_PARAMS)(biLSTM)

    # merge NER and Classification
    model = Model(inputs=[ner_input_layer, cls_input_layer], outputs=[ner_output, cls_output])
    return model


def finetune_model(model_choice, model_weights_path, freeze_list, **hyperparams):
    from mtnlpmodel.utils.model_util import (get_ner_cls_output_tensor_merge_embedding,
                                             get_ner_cls_output_tensor_merge_input)
    # get hyperparams
    EMBED_DIM = hyperparams['EMBED_DIM']
    CRF_PARAMS = hyperparams['CRF_PARAMS']
    BiLSTM_STACK_CONFIG = hyperparams['BiLSTM_STACK_CONFIG']
    CLS2NER_KEYWORD_LEN = hyperparams['CLS2NER_KEYWORD_LEN']
    USE_ATTENTION_LAYER = hyperparams['USE_ATTENTION_LAYER']
    tag_size = hyperparams['ner_tag_lookuper'].size()
    label_size = hyperparams['cls_label_lookuper'].size()
    vocab_size = hyperparams['vocabulary_lookuper'].size()

    # input layer
    input_length = hyperparams['MAX_SENTENCE_LEN']
    ner_input_layer = Input(shape=(input_length,), dtype='int32', name='ner_input')
    cls_input_layer = Input(shape=(input_length,), dtype='int32', name='cls_input')

    # encoder
    if model_choice=='VIRTUAL_EMBEDDING': # cls_out embedding merged to ner_input_embedding as virtual embedding
        from mtnlpmodel.utils.model_util import VirtualEmbedding, Discriminator_new
        with tf.keras.backend.name_scope("Encoder"):
            embedding_layer_vocab = Embedding(vocab_size,
                                              EMBED_DIM,
                                              mask_zero=True,
                                              input_length=input_length,
                                              name='embedding_vocab'
                                              )
            embedding_layer_virtual = VirtualEmbedding(label_size,
                                                       EMBED_DIM,
                                                       mask_zero=True,
                                                       input_length=CLS2NER_KEYWORD_LEN,
                                                       mask_length=CLS2NER_KEYWORD_LEN,
                                                       name='embedding_virtual_',
                                                       )

            ner_embedding = embedding_layer_vocab(ner_input_layer)
            cls_embedding = embedding_layer_vocab(cls_input_layer)

        with tf.keras.backend.name_scope("Feature_extractor"):
            for bilstm_config in BiLSTM_STACK_CONFIG:
                biLSTM = Bidirectional(LSTM(return_sequences=True, **bilstm_config, name='biLSTM'))
            bilstm_extrator = biLSTM

        # classification branch
        ner_cls_layer, cls_output = cls_branch(hyperparams['Arcloss'], 
                                               label_size, bilstm_extrator, 
                                               cls_embedding, ner_embedding,
                                               outputlayer_name='cls_')
        ner_cls_output_shape = get_ner_cls_output_tensor_merge_embedding(CLS2NER_KEYWORD_LEN)(ner_cls_layer).shape
        ner_cls_output_layer = Lambda(get_ner_cls_output_tensor_merge_embedding(CLS2NER_KEYWORD_LEN),
                                      ner_cls_output_shape)(ner_cls_layer)

        # classification output will be used as a keyword adding to input of NER
        discriminator = Discriminator_new(onetask_output_shape=(CLS2NER_KEYWORD_LEN,),
                                          output_dtype='int32')
        ner_cls_input_layer = discriminator(ner_cls_output_layer)
        ner_virtual_embedding = embedding_layer_virtual(ner_cls_input_layer)
        ner_merged_embedding = tf.keras.layers.concatenate([ner_virtual_embedding, ner_embedding], axis=1)
        ner_branch_embedding = ner_merged_embedding

    elif model_choice=='CLS2NER_INPUT':  # cls_out merged to ner_input as virtual keywords
        from mtnlpmodel.utils.model_util import Discriminator
        from mtnlpmodel.utils.input_process_util import build_vacablookuper_from_list
        vocabs = list(hyperparams['vocabulary_lookuper'].inverse_index_table.values())
        cls_labels = list(hyperparams['cls_label_lookuper'].inverse_index_table.values())
        vocabs.extend(cls_labels)
        vocabulary_lookuper = build_vacablookuper_from_list(*vocabs)

        vocab_size = vocabulary_lookuper.size()
        with tf.keras.backend.name_scope("Encoder"):
            embedding_layer = Embedding(vocab_size,
                                        EMBED_DIM,
                                        mask_zero=True,
                                        input_length=input_length,
                                        )
            ner_embedding = embedding_layer(ner_input_layer)
            cls_embedding = embedding_layer(cls_input_layer)

        with tf.keras.backend.name_scope("Feature_extractor"):
            for bilstm_config in BiLSTM_STACK_CONFIG:
                biLSTM = Bidirectional(LSTM(return_sequences=True, **bilstm_config, name='biLSTM'))
            bilstm_extrator = biLSTM

        # classification branch
        ner_cls_layer, cls_output = cls_branch(hyperparams['Arcloss'], 
                                               label_size, bilstm_extrator, 
                                               cls_embedding, ner_embedding,
                                               outputlayer_name='cls_')
        ner_cls_output_shape = get_ner_cls_output_tensor_merge_input(CLS2NER_KEYWORD_LEN,
                                                                     **{"vocab_size": vocab_size,
                                                                        "label_size": label_size})(ner_cls_layer).shape

        ner_cls_output_layer = Lambda(get_ner_cls_output_tensor_merge_input(
                                          CLS2NER_KEYWORD_LEN,
                                          **{"vocab_size": vocab_size,"label_size": label_size}),
                                      ner_cls_output_shape)(ner_cls_layer)

        # classification output will be used as a keyword adding to input of NER
        discriminator = Discriminator(ner_input_layer, onetask_output_shape=(CLS2NER_KEYWORD_LEN,),
                                      output_dtype='int32')
        merged_ner_input_layer = discriminator([ner_cls_output_layer, ner_input_layer])
        ner_branch_embedding = embedding_layer(merged_ner_input_layer)

    else: # task independent
        with tf.keras.backend.name_scope("Encoder"):
            embedding_layer = Embedding(vocab_size,
                                      EMBED_DIM,
                                      mask_zero=True,
                                      input_length=input_length,
                                      name='embedding'
                                      )

            ner_embedding = embedding_layer(ner_input_layer)
            cls_embedding = embedding_layer(cls_input_layer)

        with tf.keras.backend.name_scope("Feature_extractor"):
            for bilstm_config in BiLSTM_STACK_CONFIG:
                biLSTM = Bidirectional(LSTM(return_sequences=True, **bilstm_config, name='biLSTM'))
            bilstm_extrator = biLSTM

        # classification branch
        _, cls_output = cls_branch(hyperparams['Arcloss'], 
                                   label_size, bilstm_extrator,
                                   cls_embedding, ner_embedding,
                                   outputlayer_name='cls_')
        ner_branch_embedding = ner_embedding

    # NER branch
    with tf.keras.backend.name_scope("NER_branch"):
        # print_op = tf.print(ner_virtual_embedding._keras_mask, ner_embedding._keras_mask)
        # with tf.control_dependencies([print_op]):
        embedding_layer = LayerNormalization()(ner_branch_embedding)
        biLSTM = bilstm_extrator(embedding_layer)
        biLSTM = LayerNormalization()(biLSTM)
        if USE_ATTENTION_LAYER:
            biLSTM = GlobalAttentionLayer()(biLSTM)
        ner_output = CRF(tag_size, name="crf_", **CRF_PARAMS)(biLSTM)
    
    # merge NER and Classification
    model = Model(inputs=[ner_input_layer, cls_input_layer], outputs=[ner_output, cls_output])

    freeze_layers = [layer for layer in model.layers if layer.name in freeze_list]
    trainable_layers = [layer for layer in model.layers if layer.name not in freeze_list]
    
    for layer in freeze_layers:
        layer.trainable = False

    for layer in trainable_layers:
        layer.trainable = True
        
    model.load_weights(model_weights_path, by_name=True)
    
    return model


def get_freeze_list_for_finetuning(model_choice):
    '''Different structure models have different layers and layer names,
       through this func to get the corresponding recommendation frozen list.
       Layers in frozen_list are not trainable during the finetuning process. 
       You can modify the return list to customize your own frozen list.
    '''
    if model_choice=='VIRTUAL_EMBEDDING':
        return ['bidirectional', 'embedding_vocab']
    elif model_choice=='CLS2NER_INPUT':
        return ['bidirectional']
    else:
        return ['embedding', 'bidirectional']
    

def finetuning_logger(*args):
    import os
    print('Fine-tuning processing: ')
    for arg in args:
        try:
            if os.path.split(arg)[-1].startswith('weights'):
                print('Load model weights from {}'.format(arg))
        except:
            if isinstance(arg, list):
                print('Frozen list is [{}]'.format(', '.join(arg)))