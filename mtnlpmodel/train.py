import os
import sys
import tensorflow as tf
from tf_crf_layer.loss import ConditionalRandomFieldLoss
from tf_crf_layer.metrics import SequenceCorrectness
from seq2annotation.utils import create_dir_if_needed, create_file_dir_if_needed, create_or_rm_dir_if_needed
# tf.enable_eager_execution()
from mtnlpmodel.utils.input_process_util import _read_configure
from mtnlpmodel.utils.deliverablemodel_util import (ConverterForMTRequest,
                                                    ConverterForMTResponse_VirtualPad,
                                                    mtinput_export_as_deliverable_model, )

sys.path.append('.')
sys.path.append('..')


def main():
    # get configure
    config = _read_configure("./configure.yaml")

    # get Parameters (controller)
    EPOCHS = config.get("epochs", 10)
    PRETRAIN_EPOCHS =config.get("pretrain_cls", 5)
    BATCHSIZE = config.get("batch_size", 32)
    PRETRAIN_BATCHSIZE = config.get("pretrain_batchsize", 32)
    LEARNINGRATE = config.get("learning_rate", 0.001)
    MAX_SENTENCE_LEN = config.get("max_sentence_len", 25)
    LRDECAY = config.get('lr_decay', False)
    EARLYSTOP = config.get('early_stop', False)

    # get Parameters (model select)
    MODEL_CHOICE = config.get("model_choice", "VIRTUAL_EMBEDDING")
    FINETUNE = config.get("finetune", False)

    # get Parameters (model structure)
    CLS2NER_KEYWORD_LEN = config.get("cls2ner_keyword_len", 5)
    EMBED_DIM = config.get("embedding_dim", 128)
    ARCLOSS = config.get("Arcloss", True)
    USE_ATTENTION_LAYER = config.get("use_attention_layer", False)
    BiLSTM_STACK_CONFIG = config.get("bilstm_stack_config", [])
    CRF_PARAMS = config.get("crf_params", {})


    # get preprocessed input data dict
    from mtnlpmodel.utils.input_process_util import input_data_process
    # to build a fixed training environment, input data should be fixed.
    # input_data should be shuffled and remove duplication outside the trainer before running the program.
    # input data should be corpus(no duplication, shuffle well)

    if MODEL_CHOICE=='VIRTUAL_EMBEDDING' or MODEL_CHOICE=='CLS2NER_INPUT':  # different model structures have different input process way
        data_dict = input_data_process(config, **{'MAX_SENTENCE_LEN': MAX_SENTENCE_LEN,      # preprocess the input_data
                                       'CLS2NER_KEYWORD_LEN': CLS2NER_KEYWORD_LEN,})
    else:
        data_dict = input_data_process(config, **{'MAX_SENTENCE_LEN': MAX_SENTENCE_LEN,  # preprocess the input_data
                                                  'CLS2NER_KEYWORD_LEN': 0, })
        PRETRAIN_EPOCHS = 0

    # get lookupers
    ner_tag_lookuper = data_dict['ner_tag_lookuper']
    cls_label_lookuper = data_dict['cls_label_lookuper']
    vocabulary_lookuper = data_dict['vocabulary_lookuper']

    # get train/test data for training model
    ner_train_x, ner_train_y = data_dict['ner_train_x'], data_dict['ner_train_y']
    ner_test_x, ner_test_y = data_dict['ner_test_x'], data_dict['ner_test_y']

    cls_train_x, cls_train_y = data_dict['cls_train_x'], data_dict['cls_train_y']
    cls_test_x, cls_test_y = data_dict['cls_test_x'], data_dict['cls_test_y']


    # build model or finetuning
    from mtnlpmodel.core import build_model, finetune_model, get_freeze_list_for_finetuning
    params = {'EMBED_DIM': EMBED_DIM,
              'PRETRAIN_EPOCHS': PRETRAIN_EPOCHS,
              'BiLSTM_STACK_CONFIG': BiLSTM_STACK_CONFIG,
              'MAX_SENTENCE_LEN': MAX_SENTENCE_LEN,
              'CLS2NER_KEYWORD_LEN': CLS2NER_KEYWORD_LEN,
              'USE_ATTENTION_LAYER': USE_ATTENTION_LAYER,
              'Arcloss': ARCLOSS,
              'ner_tag_lookuper': ner_tag_lookuper,
              'cls_label_lookuper': cls_label_lookuper,
              'vocabulary_lookuper': vocabulary_lookuper,
              'CRF_PARAMS': CRF_PARAMS
             }
    model_choice = MODEL_CHOICE   # VIRTUAL_EMBEDDING, CLS2NER_INPUT, OTHER
    print("Model structure choosing {}".format(model_choice))

    from mtnlpmodel.core import finetuning_logger
    if FINETUNE:   # fine-tuning the model, load model by the weights
        recommend_freeze_list = get_freeze_list_for_finetuning(model_choice)   # you can modify this list to customize the freeze list
        model_weights_path = os.path.abspath('./results/h5_weights/weights.h5')   # use weight
        finetuning_logger(*(model_weights_path, recommend_freeze_list))   # print some log
        model = finetune_model(model_choice, model_weights_path, recommend_freeze_list, **params)

    else:         # train the model by random initializer(make a fresh start to train a model)
        model = build_model(model_choice, **params)     # to build the model
    
    model.summary()


    # build callbacks list
    callbacks_list = []

    tensorboard_callback = tf.keras.callbacks.TensorBoard(
        #log_dir=create_dir_if_needed(config["summary_log_dir"])
        log_dir='.\\results\\summary_log_dir',
        batch_size=BATCHSIZE,
    )
    callbacks_list.append(tensorboard_callback)

    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        os.path.join(create_dir_if_needed(config["model_dir"]), "cp-{epoch:04d}.ckpt"),
        load_weights_on_restart=True,
        verbose=1,
    )
    callbacks_list.append(checkpoint_callback)
    
    # early stop util
    if EARLYSTOP:
        early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss',  # early stop index
                                                      patience=3,          # early stop delay epoch
                                                      verbose=2,           # display mode
                                                      mode='auto')
        callbacks_list.append(early_stop)

    #learning rate decay util
    if LRDECAY:
        reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.75, patience=3, verbose=1,
                                                         mode='auto', epsilon=0.0001, cooldown=0, min_lr=0.00001)
        callbacks_list.append(reduce_lr)
    
    # ner_loss_func
    ner_loss_func = ConditionalRandomFieldLoss()
    
    # set optimizer
    adam_optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNINGRATE, beta_1=0.9, beta_2=0.999, amsgrad=True)


    if FINETUNE:
        NER_out_name = 'crf_'
        CLS_out_name = 'cls_'
    else:
        NER_out_name = 'crf'
        CLS_out_name = 'cls'

    # pretrain model -> train cls branch
    model.compile(optimizer=adam_optimizer,
                  loss={NER_out_name: ner_loss_func, CLS_out_name: 'categorical_crossentropy'},
                  loss_weights={NER_out_name: 0., CLS_out_name: 10.},  # set weight of loss
                  metrics={NER_out_name: SequenceCorrectness(), CLS_out_name: 'categorical_accuracy'})

    model.fit(
              {'ner_input': ner_train_x, 'cls_input': cls_train_x},
              {NER_out_name: ner_train_y, CLS_out_name: cls_train_y},
              epochs=PRETRAIN_EPOCHS,
              batch_size=PRETRAIN_BATCHSIZE,
              class_weight={NER_out_name: None, CLS_out_name: 'auto'},   # cls loss multiply the class weights
              validation_data=[{'ner_input': ner_test_x, 'cls_input': cls_test_x},
                               {NER_out_name: ner_test_y, CLS_out_name: cls_test_y}],
              callbacks=callbacks_list,)

    # train model
    model.compile(optimizer=adam_optimizer,
                  loss={NER_out_name: ner_loss_func, CLS_out_name: 'categorical_crossentropy'},
                  loss_weights={NER_out_name: 15., CLS_out_name: 10.},  # set weight of loss
                  metrics={NER_out_name: SequenceCorrectness(), CLS_out_name: 'categorical_accuracy'})
    
    model.fit(
            {'ner_input': ner_train_x, 'cls_input': cls_train_x},
            {NER_out_name: ner_train_y, CLS_out_name: cls_train_y},
            epochs=EPOCHS,
            batch_size=BATCHSIZE,
            class_weight={NER_out_name: None, CLS_out_name: 'auto'},   # cls loss multiply the class weights
            validation_data=[{'ner_input': ner_test_x, 'cls_input': cls_test_x},
                             {NER_out_name: ner_test_y, CLS_out_name: cls_test_y}],
            callbacks=callbacks_list,)


    # save model
    model.save(create_file_dir_if_needed(config["h5_model_file"]))

    model.save_weights(create_file_dir_if_needed(config["h5_weights_file"]))

    tf.keras.experimental.export_saved_model(
            model, create_or_rm_dir_if_needed(config["saved_model_dir"])
        )

    mtinput_export_as_deliverable_model(
        create_dir_if_needed(config["deliverable_model_dir"]),
        keras_saved_model=config["saved_model_dir"],
        converter_for_request=ConverterForMTRequest(),
        converter_for_response=ConverterForMTResponse_VirtualPad(prepad=CLS2NER_KEYWORD_LEN),
        lookup_tables={'vocab_lookup':vocabulary_lookuper,
                       'tag_lookup':ner_tag_lookuper,
                       'label_lookup':cls_label_lookuper},
        padding_parameter={"maxlen": MAX_SENTENCE_LEN, "value": 0, "padding": "post"},
        addition_model_dependency=["tf-crf-layer"],
        custom_object_dependency=["tf_crf_layer"],
    )



if __name__ == "__main__":
    main()