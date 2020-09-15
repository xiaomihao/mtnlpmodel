#-*- coding: UTF-8 -*

import warnings
warnings.filterwarnings('ignore')
import tqdm
import json, os, time
import tensorflow as tf
from deliverable_model import load
from deliverable_model.request import Request
from mtnlpmodel.utils.model_util import ArcFace
from mtnlpmodel.utils.model_util import VirtualEmbedding
from tokenizer_tools.tagset.offset.corpus import Corpus
from mtnlpmodel.utils.model_util import Mish
from tokenizer_tools.utils.batch_iterator import BatchingIterator

tf.keras.utils.get_custom_objects()[Mish.__name__] = Mish
tf.keras.utils.get_custom_objects()[ArcFace.__name__] = ArcFace
tf.keras.utils.get_custom_objects()[VirtualEmbedding.__name__] = VirtualEmbedding

'''
    调用模型对输入数据进行推理，即采用模型对输入数据做实体识别标注，
    推理结果供人工复查。
'''

class MtModelInference_Deliverable:
    def __init__(self, config_filepath):
        self.config_filepath = config_filepath
        self.config = self.get_config()

        if not os.path.exists(self.config['output_filepath']):
            os.mkdir(self.config['output_filepath'])


    def get_config(self):
        '''
            从配置文件中读取配置信息
        '''
        with open(self.config_filepath, 'rb') as f:
            self.config = json.load(f)
        return self.config


    @staticmethod
    def generate_batch_input(input_data, batch_size):
        length = len(input_data)
        if length<batch_size:
            yield input_data
        else:
            for i in range(0, length, batch_size):
                yield(input_data[i: i+batch_size])


    def _inference(self, model, input_data):
        output = []
        batch_size = 1
        batches = MtModelInference_Deliverable.generate_batch_input(input_data, batch_size)
        # batch_generator = BatchingIterator(batch_size)
        # batches = batch_generator(input_data)

        for batch in tqdm.tqdm(batches):
            request = Request(batch)
            response = model.inference(request)
            tmp_result = response['data'][0].sequence
            tmp_result.label = response['cls'][0][0]
            output.append(tmp_result)

        predict_result = Corpus(output)
        predict_result.write_to_file(os.path.join(self.config['output_filepath'], 'inference_out.conllx'))

        print('*** inference has been done, please check the result through the path below:')
        print('==> {}'.format(self.config['output_filepath']))

        return


    def call_inference(self):
        input_rawdata_filename = self.config['input_rawdata']
        input_file = os.path.join(self.config['data_filepath'], input_rawdata_filename)
        with open(input_file,'rt',encoding='utf-8') as f:
            input_data = [line.strip() for line in f.readlines()]
        model = load(self.config['model_filepath'])
        self._inference(model, input_data)


if __name__ == "__main__":
    print('*****The Inference program is running now, please wait.*****\n')
    start = time.time()
    config_filepath = './configure.json'
    MtModelInference_Deliverable(config_filepath).call_inference()
    end = time.time()
    print('==> Time cost is', end - start, 's')