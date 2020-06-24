import os
import pathlib
import shutil
from typing import Text, Any, Type
import random
from typing import Union, Callable, Dict

from deliverable_model.metacontent import MetaContent
from deliverable_model.builder import (
    DeliverableModelBuilder,
    MetadataBuilder,
    ProcessorBuilder,
    ModelBuilder,
)
from deliverable_model.builtin import LookupProcessor
from deliverable_model.builtin.processor import BILUOEncodeProcessor, PadProcessor
from deliverable_model.processor_base import ProcessorBase

import numpy as np
from typing import Any
from deliverable_model.response import Response
from deliverable_model.converter_base import ConverterBase
from deliverable_model.request import Request


class RequestProcessor(ProcessorBase):
    def preprocess(self, request: Request) -> Request:
        request['NER'] = request.query
        request['CLS'] = request.query

        return request

    @classmethod
    def load(cls, *args, **kwargs):
        return cls()


class ConverterForMTRequest(ConverterBase):
    '''Now only support NER and CLS'''
    def call(self, request: Request) -> Any:
        request["CLS"] = request.query
        request["NER"] = request.query

        return [request["CLS"], request["NER"]]


class ConverterForMTResponse(ConverterBase):
    def __call__(self, response) -> Response:
        response_data = response
        # ner branch
        ner_response = response_data[0].tolist()
        # cls branch
        cls_tmp_response = np.argmax(response_data[1])
        cls_tmp_restorer, cls_result_restorer = [], []
        cls_tmp_restorer.append(cls_tmp_response)
        cls_result_restorer.append(cls_tmp_restorer)
        result = Response([])
        result["cls"] = cls_result_restorer
        result.data = ner_response
        return result


class ConverterForMTResponse_VirtualPad(ConverterBase):
    def __init__(self, **kwargs):
        self.config = kwargs

    def __call__(self, response) -> Response:
        response_data = response
        # ner branch
        ner_response = response_data[0].tolist()
        ner_response[0][:self.config['prepad']] = []
        # cls branch
        cls_tmp_response = np.argmax(response_data[1])
        cls_tmp_restorer, cls_result_restorer = [], []
        cls_tmp_restorer.append(cls_tmp_response)
        cls_result_restorer.append(cls_tmp_restorer)

        result = Response([])
        result["cls"] = cls_result_restorer
        result.data = ner_response
        return result

    def get_config(self):
        return self.config


def mt_export_as_deliverable_model(
    output_dir,
    tensorflow_saved_model=None,
    converter_for_request: Union[None, Callable] = None,
    converter_for_response: Union[None, Callable] = None,
    keras_saved_model=None,
    keras_h5_model=None,
    meta_content_id="algorithmId-corpusId-configId-runId",
    lookup_tables: Dict = None,
    padding_parameter=None,
    addition_model_dependency=None,
    custom_object_dependency=None,
):
    # check parameters
    assert any(
        [tensorflow_saved_model, keras_saved_model, keras_h5_model]
    ), "one and only one of [tensorflow_saved_model, keras_saved_model, keras_h5_model] must be set up"
    assert (
        sum(
            int(bool(i))
            for i in [tensorflow_saved_model, keras_saved_model, keras_h5_model]
        )
        == 1
    ), "one and only one of [tensorflow_saved_model, keras_saved_model, keras_h5_model] must be set up"

    # default value
    addition_model_dependency = (
        [] if addition_model_dependency is None else addition_model_dependency
    )
    custom_object_dependency = (
        [] if custom_object_dependency is None else custom_object_dependency
    )

    # setup main object
    deliverable_model_builder = DeliverableModelBuilder(output_dir)

    # metadata builder
    metadata_builder = MetadataBuilder()

    meta_content = MetaContent(meta_content_id)

    metadata_builder.set_meta_content(meta_content)

    metadata_builder.save()

    # processor builder

    vocabulary_lookup_table = lookup_tables['vocab_lookup']
    tag_lookup_table = lookup_tables['tag_lookup']
    label_lookup_table = lookup_tables['label_lookup']

    processor_builder = ProcessorBuilder()

    decode_processor = BILUOEncodeProcessor()
    decoder_processor_handle = processor_builder.add_processor(decode_processor)

    pad_processor = PadProcessor(padding_parameter=padding_parameter)
    pad_processor_handle = processor_builder.add_processor(pad_processor)

    vocab_lookup_processor = LookupProcessor(vocabulary_lookup_table)
    vocab_lookup_processor_handle = processor_builder.add_processor(vocab_lookup_processor)

    tag_lookup_processor = LookupProcessor(tag_lookup_table)
    tag_lookup_processor_handle = processor_builder.add_processor(tag_lookup_processor)

    label_lookup_processor = LookupProcessor(label_lookup_table, **{"post_input_key": 'cls', "post_output_key": 'cls'})
    label_lookup_processor_handle = processor_builder.add_processor(label_lookup_processor)

    # # pre process: encoder[memory text] > lookup[str -> num] > pad[to fixed length]
    processor_builder.add_preprocess(decoder_processor_handle)
    processor_builder.add_preprocess(vocab_lookup_processor_handle)
    processor_builder.add_preprocess(pad_processor_handle)

    # # post process: lookup[num -> str] > encoder
    processor_builder.add_postprocess(tag_lookup_processor_handle)
    processor_builder.add_postprocess(label_lookup_processor_handle)
    processor_builder.add_postprocess(decoder_processor_handle)

    processor_builder.save()

    # model builder
    model_builder = ModelBuilder()
    model_builder.append_dependency(addition_model_dependency)
    model_builder.set_custom_object_dependency(custom_object_dependency)

    if converter_for_request:
        model_builder.add_converter_for_request(converter_for_request)

    if converter_for_response:
        model_builder.add_converter_for_response(converter_for_response)

    if tensorflow_saved_model:
        model_builder.add_tensorflow_saved_model(tensorflow_saved_model)
    elif keras_saved_model:
        model_builder.add_keras_saved_model(keras_saved_model)
    else:
        model_builder.add_keras_h5_model(keras_h5_model)

    model_builder.save()

    # compose all the parts
    deliverable_model_builder.add_processor(processor_builder)
    deliverable_model_builder.add_metadata(metadata_builder)
    deliverable_model_builder.add_model(model_builder)

    metadata = deliverable_model_builder.save()

    return metadata



def mtinput_export_as_deliverable_model(
    output_dir,
    tensorflow_saved_model=None,
    converter_for_request: Union[None, Callable] = None,
    converter_for_response: Union[None, Callable] = None,
    keras_saved_model=None,
    keras_h5_model=None,
    meta_content_id="algorithmId-corpusId-configId-runId",
    lookup_tables: Dict = None,
    padding_parameter=None,
    addition_model_dependency=None,
    custom_object_dependency=None,
):
    # check parameters
    assert any(
        [tensorflow_saved_model, keras_saved_model, keras_h5_model]
    ), "one and only one of [tensorflow_saved_model, keras_saved_model, keras_h5_model] must be set up"
    assert (
        sum(
            int(bool(i))
            for i in [tensorflow_saved_model, keras_saved_model, keras_h5_model]
        )
        == 1
    ), "one and only one of [tensorflow_saved_model, keras_saved_model, keras_h5_model] must be set up"

    # default value
    addition_model_dependency = (
        [] if addition_model_dependency is None else addition_model_dependency
    )
    custom_object_dependency = (
        [] if custom_object_dependency is None else custom_object_dependency
    )

    # setup main object
    deliverable_model_builder = DeliverableModelBuilder(output_dir)

    # metadata builder
    metadata_builder = MetadataBuilder()

    meta_content = MetaContent(meta_content_id)

    metadata_builder.set_meta_content(meta_content)

    metadata_builder.save()

    # processor builder

    vocabulary_lookup_table = lookup_tables['vocab_lookup']
    tag_lookup_table = lookup_tables['tag_lookup']
    label_lookup_table = lookup_tables['label_lookup']

    processor_builder = ProcessorBuilder()

    decode_processor = BILUOEncodeProcessor()
    decoder_processor_handle = processor_builder.add_processor(decode_processor)

    pad_processor = PadProcessor(padding_parameter=padding_parameter)
    pad_processor_handle = processor_builder.add_processor(pad_processor)

    request_processor = RequestProcessor()
    request_processor_handle = processor_builder.add_processor(request_processor)
    ner_vacab_lookup_processor = LookupProcessor(vocabulary_lookup_table)
    ner_vacab_lookup_processor.pre_input_key = 'NER'
    cls_vacab_lookup_processor = LookupProcessor(vocabulary_lookup_table)
    cls_vacab_lookup_processor.pre_input_key = 'CLS'
    ner_vocab_lookup_processor_handle = processor_builder.add_processor(ner_vacab_lookup_processor)
    cls_vocab_lookup_processor_handle = processor_builder.add_processor(cls_vacab_lookup_processor)

    tag_lookup_processor = LookupProcessor(tag_lookup_table)
    tag_lookup_processor_handle = processor_builder.add_processor(tag_lookup_processor)

    label_lookup_processor = LookupProcessor(label_lookup_table, **{"post_input_key": 'cls', "post_output_key": 'cls'})
    label_lookup_processor_handle = processor_builder.add_processor(label_lookup_processor)

    # # pre process: encoder[memory text] > lookup[str -> num] > pad[to fixed length]
    processor_builder.add_preprocess(request_processor_handle)
    processor_builder.add_preprocess(decoder_processor_handle)
    processor_builder.add_preprocess(ner_vocab_lookup_processor_handle)
    processor_builder.add_preprocess(cls_vocab_lookup_processor_handle)
    processor_builder.add_preprocess(pad_processor_handle)

    # # post process: lookup[num -> str] > encoder
    processor_builder.add_postprocess(tag_lookup_processor_handle)
    processor_builder.add_postprocess(label_lookup_processor_handle)
    processor_builder.add_postprocess(decoder_processor_handle)

    processor_builder.save()

    # model builder
    model_builder = ModelBuilder()
    model_builder.append_dependency(addition_model_dependency)
    model_builder.set_custom_object_dependency(custom_object_dependency)

    if converter_for_request:
        model_builder.add_converter_for_request(converter_for_request)

    if converter_for_response:
        model_builder.add_converter_for_response(converter_for_response)

    if tensorflow_saved_model:
        model_builder.add_tensorflow_saved_model(tensorflow_saved_model)
    elif keras_saved_model:
        model_builder.add_keras_saved_model(keras_saved_model)
    else:
        model_builder.add_keras_h5_model(keras_h5_model)

    model_builder.save()

    # compose all the parts
    deliverable_model_builder.add_processor(processor_builder)
    deliverable_model_builder.add_metadata(metadata_builder)
    deliverable_model_builder.add_model(model_builder)

    metadata = deliverable_model_builder.save()

    return metadata


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


def remove_files_in_dir(data_dir):
    input_file_list = [i.absolute() for i in pathlib.Path(data_dir).iterdir() if i.is_file()]
    for i in input_file_list:
        os.remove(i)


def remove_content_in_dir(data_dir):
    input_file_list = pathlib.Path(data_dir).iterdir()
    for i in input_file_list:
        file_path = str(i.absolute())
        if i.is_dir():
            shutil.rmtree(file_path)
        else:
            os.remove(file_path)

def create_dir_if_needed(directory):
    # copied from https://stackoverflow.com/questions/273192/how-can-i-safely-create-a-nested-directory-in-python
    import shutil
    # if not os.path.exists(directory):
    if not os.path.exists(directory):
        # os.makedirs(directory)
        os.makedirs(directory)

    return directory


def create_or_rm_dir_if_needed(directory):
    # copied from https://stackoverflow.com/questions/273192/how-can-i-safely-create-a-nested-directory-in-python
    import shutil
    # if not os.path.exists(directory):
    if not os.path.exists(directory):
        # os.makedirs(directory)
        os.makedirs(directory)
    else:
        shutil.rmtree(directory)
        try:
            os.makedirs(directory)
        except:
            pass

    return directory


def create_file_dir_if_needed(file):
    directory = os.path.dirname(file)

    create_dir_if_needed(directory)

    return file


def join_path(a, b):
    return os.path.join(a, str(pathlib.PurePosixPath(b)))


def class_from_module_path(module_path: Text) -> Type[Any]:
    # copied from rasa_nlu (https://github.com/RasaHQ/rasa) @ rasa_nlu/utils/__init__.py
    """Given the module name and path of a class, tries to retrieve the class.

    The loaded class can be used to instantiate new objects. """
    import importlib

    # load the module, will raise ImportError if module cannot be loaded
    if "." in module_path:
        module_name, _, class_name = module_path.rpartition('.')
        m = importlib.import_module(module_name)
        # get the class, will raise AttributeError if class cannot be found
        return getattr(m, class_name)
    else:
        return globals()[module_path]


def load_hook(hook_config):
    hook_instances = []
    for i in hook_config:
        class_ = class_from_module_path(i['class'])
        hook_instances.append(class_(**i.get('params', {})))

    return hook_instances


from tensorflow.keras.utils import Sequence
class MakeSequence(Sequence):
    def __init__(self, data_sets, batch_size):
        self.ner_x, self.ner_y, self.cls_x, self.cls_y = data_sets
        self.batch_size = batch_size

    def __len__(self):
        return int(np.ceil(len(self.ner_x) / float(self.batch_size)))

    def __getitem__(self, idx):
        ner_x_batch_data = self.ner_x[(idx * self.batch_size):((idx + 1) * self.batch_size)]
        ner_y_batch_data = self.ner_y[(idx * self.batch_size):((idx + 1) * self.batch_size)]
        cls_x_batch_data = self.cls_x[(idx * self.batch_size):((idx + 1) * self.batch_size)]
        cls_y_batch_data = self.cls_y[(idx * self.batch_size):((idx + 1) * self.batch_size)]

        return ({'ner_input': ner_x_batch_data,
                'cls_input': cls_x_batch_data },
               {'crf': ner_y_batch_data,
                'CLS': cls_y_batch_data})



def to_categorical_strat_from_one(y, num_classes=None, dtype='float32'):
    """Converts a class vector (integers) to binary class matrix.

    E.g. for use with categorical_crossentropy.

    # Arguments
        y: class vector to be converted into a matrix
            (integers from 1 to num_classes).
        num_classes: total number of classes.
        dtype: The data type expected by the input, as a string
            (`float32`, `float64`, `int32`...)

    # Returns
        A binary matrix representation of the input. The classes axis
        is placed last.

    # Example

    ```python
    # Consider an array of 5 labels out of a set of 3 classes {1, 2, 3}:
    > labels
    array([1, 3, 1, 3, 1])
    # `to_categorical` converts this into a matrix with as many
    # columns as there are classes. The number of rows
    # stays the same.
    > to_categorical(labels)
    array([[ 1.,  0.,  0.],
           [ 0.,  0.,  1.],
           [ 0.,  1.,  0.],
           [ 0.,  0.,  1.],
           [ 1.,  0.,  0.]], dtype=float32)
    ```
    """

    y = np.array(y, dtype='int')
    input_shape = y.shape
    if input_shape and input_shape[-1] == 1 and len(input_shape) > 1:
        input_shape = tuple(input_shape[:-1])
    y = y.ravel()
    if not num_classes:
        num_classes = np.max(y) + 1
    n = y.shape[0]
    categorical = np.zeros((n, num_classes), dtype=dtype)
    categorical[np.arange(n), y-1] = 1
    output_shape = input_shape + (num_classes,)
    categorical = np.reshape(categorical, output_shape)
    return categorical


def load_ckpt(ckpt_path, model):
    if not os.path.exists(ckpt_path):
        return
    else:
        ckpts = os.listdir(ckpt_path)
        if not ckpts:
            return
        else:
            import tensorflow as tf
            latest_ckpt = tf.train.latest_checkpoint(ckpt_path)
            model.load_weights(latest_ckpt)
            print('Load ckpt from {}'.format(latest_ckpt))