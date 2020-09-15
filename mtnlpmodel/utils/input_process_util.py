import random
import numpy as np
import tensorflow as tf
from tokenizer_tools.tagset.offset.corpus import Corpus
from tokenizer_tools.tagset.converter.offset_to_biluo import offset_to_biluo
from seq2annotation.input import generate_tagset, Lookuper, index_table_from_file


def read_from_file(path):
    with open(path, 'rt', encoding='utf-8') as f:
        output = [line.strip() for line in f.readlines()]
    return output


def _read_configure(config_filepath, return_empty=False):
    import os
    import pprint
    from pconf import Pconf
    from ioflow.configure.get_configure_path_from_argv import get_configure_path_from_argv
    from ioflow.configure.read_configure import find_best_file_candidate, guess_configure_file_type

    # set return_empty to True for not read config from env
    # which can prevent unexpected result
    # e.g. './configure.json' is not for this app, but for other using
    if return_empty:
        return {}

    config_fileprefix = os.path.splitext(os.path.split(config_filepath)[-1])[0]

    default_configure_candidate = [
        ".".join([config_fileprefix, ext]) for ext in ["yaml", "yml", "json"]
    ]
    builtin_configure_candidate = [
        ".".join(["./builtin_configure", ext]) for ext in ["yaml", "yml", "json"]
    ]

    default_configure = find_best_file_candidate(default_configure_candidate)
    builtin_configure = find_best_file_candidate(builtin_configure_candidate)

    active_configure_file = get_configure_path_from_argv()
    if not active_configure_file:
        active_configure_file = os.getenv("_DEFAULT_CONFIG_FILE", default_configure)

    builtin_configure_file = os.getenv("_BUILTIN_CONFIG_FILE", builtin_configure)

    # Note: this is a safeguard, before using any Pconf function, do execute this
    # In case former Pconf usage influence current usage
    # which will lead to a hidden and wired bug
    Pconf.clear()

    # disable read configure from environment
    # Pconf.env()

    active_configure_file_abs_path = os.path.realpath(active_configure_file)

    if not os.path.exists(active_configure_file):
        msg = "default configure file is not found! CWD: {}; activate_config: {}; builtin_configure: {}".format(
            os.getcwd(), active_configure_file, builtin_configure_file
        )
        print(msg)
        raise ValueError(msg)
    else:
        print(
            ">>> Using configure read from file: {}".format(
                active_configure_file_abs_path
            )
        )

    file_encoding = guess_configure_file_type(active_configure_file_abs_path)
    Pconf.file(active_configure_file, file_encoding)

    # try loading builtin configure file
    if builtin_configure_file and os.path.exists(builtin_configure_file):
        print("loading builtin configure from {}".format(builtin_configure_file))
        file_encoding = guess_configure_file_type(builtin_configure_file)
        Pconf.file(builtin_configure_file, encoding=file_encoding)
    else:
        print(">>> builtin configure file is not found!")

    # Get all the config values parsed from the sources
    config = Pconf.get()

    # NOTE: clean Pconf for later brand new use
    Pconf.clear()

    print("++" * 8, "configure", "++" * 8)
    pprint.pprint(config)

    return config


def get_label_from_corpus(corpus):
    labels = set()
    for sample in corpus:
        labels.add(sample.label)
    return sorted(list(labels))


def get_tag_from_corpus(corpus):
    tags = set()
    for sample in corpus:
        for span in sample.span_set:
            tags.add(span.entity)
    return sorted(list(tags))


def build_vacablookuper_from_list(*lists):
    char_list = []
    for ls in lists:
        char_list.extend(ls)
    char_list = sorted(list(set(char_list)))
    if char_list[0]!='<pad>':
        char_list.insert(0, '<pad>')

    index_table = {}
    index_counter = 1
    for key in char_list:
        index_table[key] = index_counter
        index_counter += 1

    return Lookuper(index_table)


def build_vacablookuper_from_corpus(*corpus_tuples):
    char_list = []
    for corpus in corpus_tuples:
        for sample in corpus:
            char_list.extend(sample.text)
    char_list = sorted(list(set(char_list)))
    char_list.insert(0, '<pad>')

    index_table = {}
    index_counter = 0
    for key in char_list:
        index_table[key] = index_counter
        index_counter += 1

    return Lookuper(index_table)


def index_table_from_corpus(corpus=None):
    from seq2annotation.input import Lookuper
    index_table = {}
    tmp_text_list = [sample.text for sample in corpus]
    text_list = []
    for text in tmp_text_list:
        text_list.extend(text)
    text_list = sorted(set(text_list))
    for index, word in enumerate(text_list):
        index_table[word] = index+1
    return Lookuper(index_table)


def random_padding_to_samesize(ner_data_tuple, cls_data_tuple):
    ner_train_data, ner_eval_data = ner_data_tuple
    cls_train_data, cls_eval_data = cls_data_tuple
    if len(ner_train_data)>len(cls_train_data):
        padding_samples = random.sample(cls_train_data, (len(ner_train_data)-len(cls_train_data)))
        cls_train_data.extend(padding_samples)
    else:
        padding_samples = random.sample(ner_train_data, (len(cls_train_data) - len(ner_train_data)))
        ner_train_data.extend(padding_samples)

    if len(ner_eval_data)>len(cls_eval_data):
        padding_samples = random.sample(cls_eval_data, (len(ner_eval_data)-len(cls_eval_data)))
        cls_eval_data.extend(padding_samples)
    else:
        padding_samples = random.sample(ner_eval_data, (len(cls_eval_data) - len(ner_eval_data)))
        ner_eval_data.extend(padding_samples)

    ner_processed_tuple = (ner_train_data, ner_eval_data)
    cls_processed_tuple = (cls_train_data, cls_eval_data)
    return ner_processed_tuple, cls_processed_tuple


def random_sampling_to_samesize(ner_data_tuple, cls_data_tuple):
    ner_train_data, ner_eval_data = ner_data_tuple
    cls_train_data, cls_eval_data = cls_data_tuple

    if len(ner_train_data)>len(cls_train_data):
        ner_train_data = random.sample(ner_train_data, len(cls_train_data))
    else:
        cls_train_data = random.sample(cls_train_data, len(ner_train_data))

    if len(ner_eval_data)>len(cls_eval_data):
        ner_eval_data = random.sample(ner_eval_data, len(cls_eval_data))
    else:
        cls_eval_data = random.sample(cls_eval_data, len(ner_eval_data))

    ner_processed_tuple = (ner_train_data, ner_eval_data)
    cls_processed_tuple = (cls_train_data, cls_eval_data)
    return ner_processed_tuple, cls_processed_tuple


def input_data_process(config, **hyperparams):
    # read NER/CLS individually (only support *.conllx)
    input_mode = config['input_mode']
    if input_mode == 'multi':
        # multi input
        data_ner = Corpus.read_from_file(config['ner_data'])
        data_cls = Corpus.read_from_file(config['cls_data'])
    else:
        # single input
        data = Corpus.read_from_file(config['data'])
        data_ner = data
        data_cls = data

    # get train/test corpus
    ner_tags = get_tag_from_corpus(data_ner)
    cls_labels = get_label_from_corpus(data_cls)

    test_ratio = config['test_ratio']
    ner_train_data, ner_eval_data = data_ner.train_test_split(test_size=test_ratio, random_state=50)
    cls_train_data, cls_eval_data = data_cls.train_test_split(test_size=test_ratio, random_state=50)

    ner_data_tuple, cls_data_tuple = random_sampling_to_samesize((ner_train_data, ner_eval_data), # mainly for multi input
                                                                 (cls_train_data,cls_eval_data))  # make sure cls & ner have
    # same size dataset
    ner_train_data, ner_eval_data = ner_data_tuple
    cls_train_data, cls_eval_data = cls_data_tuple

    # build lookupers
    ner_tag_lookuper = Lookuper({v: i for i, v in enumerate(generate_tagset(ner_tags))})
    cls_label_lookuper = Lookuper({v: i for i, v in enumerate(cls_labels)})

    vocab_data_file = config.get("vocabulary_file", None)

    if not vocab_data_file:
        # get vacab_data for corpus
        vocabulary_lookuper = build_vacablookuper_from_corpus(*(data_ner, data_cls))  # from corpus
    else:
        vocabulary_lookuper = index_table_from_file(vocab_data_file)  # from vocab_file

    # ner (data&tag) str->int
    def ner_preprocss(data, maxlen, cls_info_len):
        raw_x_ner = []
        raw_y_ner = []

        for offset_data in data:
            tags = offset_to_biluo(offset_data)

            if config['decoder_style']=='BIO':
                for index, tag in enumerate(tags):
                    tags[index] = tag.replace('U', 'B')
                    tags[index] = tags[index].replace('L', 'I')

            words = offset_data.text

            tag_ids = [ner_tag_lookuper.lookup(i) for i in tags]
            word_ids = [vocabulary_lookuper.lookup(i) for i in words]

            raw_x_ner.append(word_ids)
            raw_y_ner.append(tag_ids)

        if maxlen is None:
            maxlen = max(len(s) for s in raw_x_ner)

        maxlen_mt = maxlen + cls_info_len
        print(">>> maxlen: {}".format(maxlen))

        x_ner = tf.keras.preprocessing.sequence.pad_sequences(
            raw_x_ner, maxlen, padding="post"
        )  # right padding

        y_ner = tf.keras.preprocessing.sequence.pad_sequences(
            raw_y_ner, maxlen, value=0, padding="post"
        )

        y_ner = tf.keras.preprocessing.sequence.pad_sequences(
            y_ner, maxlen_mt, value=0, padding="pre"
        )

        return x_ner, y_ner

    # cls (data&label) str->int
    def cls_preprocss(data, maxlen, **kwargs):
        raw_x_cls = []
        raw_y_cls = []

        for offset_data in data:
            label = offset_data.label
            words = offset_data.text

            label_id = cls_label_lookuper.lookup(label)
            word_ids = [vocabulary_lookuper.lookup(i) for i in words]

            raw_x_cls.append(word_ids)
            raw_y_cls.append(label_id)

        if maxlen is None:
            maxlen = max(len(s) for s in raw_x_cls)

        print(">>> maxlen: {}".format(maxlen))

        x_cls = tf.keras.preprocessing.sequence.pad_sequences(
            raw_x_cls, maxlen, padding="post"
        )  # right padding

        from tensorflow.keras.utils import to_categorical
        y_cls = np.array(raw_y_cls)
        y_cls = y_cls[:, np.newaxis]
        y_cls = to_categorical(y_cls, kwargs.get('cls_dims', 81))

        return x_cls, y_cls

    ner_train_x, ner_train_y = ner_preprocss(ner_train_data, hyperparams['MAX_SENTENCE_LEN'],
                                             hyperparams['CLS2NER_KEYWORD_LEN'])
    ner_test_x, ner_test_y = ner_preprocss(ner_eval_data, hyperparams['MAX_SENTENCE_LEN'],
                                           hyperparams['CLS2NER_KEYWORD_LEN'])

    cls_train_x, cls_train_y = cls_preprocss(cls_train_data, hyperparams['MAX_SENTENCE_LEN'],
                                             **{'cls_dims': cls_label_lookuper.size()})
    cls_test_x, cls_test_y = cls_preprocss(cls_eval_data, hyperparams['MAX_SENTENCE_LEN'],
                                           **{'cls_dims': cls_label_lookuper.size()})

    #cls_class_weight = get_class_weight(cls_train_data, cls_label_lookuper)

    output_dict = {'ner_train_x': ner_train_x, 'ner_train_y': ner_train_y,
                   'ner_test_x': ner_test_x, 'ner_test_y': ner_test_y,
                   'cls_train_x': cls_train_x, 'cls_train_y': cls_train_y,
                   'cls_test_x': cls_test_x, 'cls_test_y': cls_test_y,
                   'ner_tag_lookuper': ner_tag_lookuper,
                   'cls_label_lookuper': cls_label_lookuper,
                   'vocabulary_lookuper': vocabulary_lookuper,
                   }

    return output_dict