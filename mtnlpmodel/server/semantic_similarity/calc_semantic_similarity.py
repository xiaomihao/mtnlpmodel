#-*- coding: UTF-8 -*

import warnings
warnings.filterwarnings('ignore')
import tqdm
import json, os, time
import numpy as np
import tensorflow as tf
from deliverable_model import load
from deliverable_model.request import Request
from mtnlpmodel.utils.model_util import ArcFace
from mtnlpmodel.utils.model_util import VirtualEmbedding
from tokenizer_tools.tagset.offset.corpus import Corpus
from mtnlpmodel.utils.model_util import Mish

tf.keras.utils.get_custom_objects()[Mish.__name__] = Mish
tf.keras.utils.get_custom_objects()[ArcFace.__name__] = ArcFace
tf.keras.utils.get_custom_objects()[VirtualEmbedding.__name__] = VirtualEmbedding

'''
    调用模型对输入数据进行推理，即采用模型对输入数据做实体识别标注，
    推理结果供人工复查。
'''

class CalcSemanticSimilarity:
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


    @staticmethod
    def calc_euclidean_distance(v1, v2):
        return np.linalg.norm(v1-v2)/(np.linalg.norm(v1)+np.linalg.norm(v2))


    def save_results_to_file(self, output):
        results_filepath = os.path.join(self.config['output_filepath'], 'similarity_out.txt')
        with open(results_filepath, 'wt', encoding='utf-8') as f:
            f.write('\n'.join(output))


    def _inference(self, model, input_data):
        output = []
        batch_size = 1
        batches = CalcSemanticSimilarity.generate_batch_input(input_data, batch_size)
        for batch in tqdm.tqdm(batches):
            input_A = batch[0][0]
            input_B = batch[0][1]
            request_A = Request(input_A)
            request_B = Request(input_B)
            vector_A = model.inference(request_A)
            vector_B = model.inference(request_B)
            distance = CalcSemanticSimilarity.calc_euclidean_distance(vector_A, vector_B)
            similarity = 1-distance
            print_str = str(input_A)+' : '+str(input_B)+' = '+'{}'.format(round(similarity*100, 2))+'%'
            output.append(print_str)

        self.save_results_to_file(output)
        print('*** inference has been done, please check the result through the path below:')
        print('==> {}'.format(self.config['output_filepath']))
        return


    def get_datatuples_from_file(self):
        input_rawdata_filename = self.config['input_rawdata']
        input_file = os.path.join(self.config['data_filepath'], input_rawdata_filename)
        with open(input_file,'rt',encoding='utf-8') as f:
            input_data = [line.strip() for line in f.readlines()]
        data_tuple_list = []
        for data in input_data:
            data_tuple = tuple(data.split('|'))
            data_tuple_list.append(data_tuple)
        return data_tuple_list
            

    def calc_semantic_similarity(self):
        data_tuples = self.get_datatuples_from_file()
        model = load(self.config['model_filepath'])
        self._inference(model, data_tuples)


if __name__ == "__main__":
    print('*****The Similarity Calculation program is running now, please wait.*****\n')
    start = time.time()
    config_filepath = './configure.json'
    CalcSemanticSimilarity(config_filepath).calc_semantic_similarity()
    end = time.time()
    print('==> Time cost is', end - start, 's')