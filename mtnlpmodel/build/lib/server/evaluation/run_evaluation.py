#-*- coding: UTF-8 -*

import warnings, time, os
warnings.filterwarnings('ignore')
from deliverable_model import load
from deliverable_model.request import Request
from tokenizer_tools.tagset.offset.corpus import Corpus
from tokenizer_tools.tagset.offset.document import Document
from mtnlpmodel.server.inference.run_inference import MtModelInference_Deliverable


class MtModelEvaluation_Deliverable(MtModelInference_Deliverable):
    @classmethod
    def inference_process(cls, model, input_data, input_ids):
        output = []
        for data, id in zip(input_data, input_ids):
            req = []
            req.append(data)
            request = Request(req)
            response = model.inference(request)
            tmp_result = response['data'][0].sequence
            if not isinstance(tmp_result, Document):
                tmp_result = Document(tmp_result.text, tmp_result.span_set)
            tmp_result.label = response['cls'][0][0]
            tmp_result.id = id
            output.append(tmp_result)
        corpus_inference = Corpus(output)
        return corpus_inference


    @classmethod
    def whats_different(cls, corpus1, corpus2):
        corpus1_differ_from_corpus2 = Corpus(sorted(corpus1.difference(corpus2), key=lambda doc: doc.id))
        corpus2_differ_from_corpus1 = Corpus(sorted(corpus2.difference(corpus1), key=lambda doc: doc.id))
        return (corpus1_differ_from_corpus2, corpus2_differ_from_corpus1)


    @classmethod
    def calc_acc(self, differ_corpus_tuples, test_corpus_length):
        eps = 1e-9
        test, infer = differ_corpus_tuples
        cls_differ_count = 0
        ner_differ_count = 0
        for test_sample, infer_sample in zip(test, infer):
            if test_sample.label != infer_sample.label and \
               test_sample.id == infer_sample.id:
                cls_differ_count += 1

            if test_sample.span_set != infer_sample.span_set and \
               test_sample.id == infer_sample.id:
                ner_differ_count += 1
        scores = (1-cls_differ_count/(test_corpus_length+eps), 1-ner_differ_count/(test_corpus_length+eps))
        return scores


    def _evaluation(self, corpus_test):
        model = load(self.config['model_filepath'])
        text_list = [''.join(sample.text) for sample in corpus_test]
        id_list = [sample.id for sample in corpus_test]
        corpus_infer = MtModelEvaluation_Deliverable.inference_process(model, text_list, id_list)

        # symantic_corpus = corpus_test.intersection(corpus_infer)
        differ_corpus_tuples = MtModelEvaluation_Deliverable.whats_different(corpus_test, corpus_infer)

        scores = MtModelEvaluation_Deliverable.calc_acc(differ_corpus_tuples, len(corpus_test))

        return scores, differ_corpus_tuples


    @classmethod
    def get_differ_info(cls, sample_tuple):
        test = sample_tuple[0]
        infer = sample_tuple[1]
        id = test.id
        label_test = test.label
        label_infer = infer.label
        infer_text = infer.text.copy()
        test_text = test.text.copy()
        test_spanset = test.span_set
        infer_spanset = infer.span_set

        count = 0
        for span in test_spanset:
            test_text[span.start+count] = '<' + test_text[span.start+count]
            test_text[span.end-1+count] += '>'
            test_text.insert(span.end+count, '(' + span.entity + ')')
            count += 1
        count =0
        for span in infer_spanset:
            infer_text[span.start+count] = '<' + infer_text[span.start+count]
            infer_text[span.end-1+count] += '>'
            infer_text.insert(span.end+count, '(' + span.entity + ')')
            count += 1

        test_text = ''.join(test_text)
        infer_text = ''.join(infer_text)
        output_test_line = 'id: {id} | ground: {test_text} => label: {label_test}\n'.format(id=id,
                                                                                            test_text=test_text,
                                                                                            label_test=label_test)
        output_infer_line = 'id: {id} | infer:  {infer_text} => label: {label_infer}\n'.format(id=id,
                                                                                               infer_text=infer_text,
                                                                                               label_infer=label_infer)
        return output_test_line, output_infer_line


    def save_result(self, scores, differ_corpus_tuples):
        output_metrics = []
        output_text = []
        output_metrics.append('CLS_ACC:  ' + str(scores[0]))
        output_metrics.append('NER_ACC:  ' + str(scores[1]))
        output_metrics.append('=='*40)
        output_metrics.append('Differences of Evaluation list below:')
        output_metrics.append('=='*40+'\n')
        output_metrics = '\n'.join(output_metrics)

        for sample_tuple in zip(differ_corpus_tuples[0], differ_corpus_tuples[1]):
            differ_info = MtModelEvaluation_Deliverable.get_differ_info(sample_tuple)
            output_text.append(''.join(differ_info))
        output_text = '\n'.join(output_text)

        output = []
        output.extend(output_metrics)
        output.extend(output_text)
        output_path = os.path.join(self.config['output_filepath'], 'evaluation_out.txt')
        with open(output_path, 'wt', encoding='utf-8') as f:
            f.writelines(output)
        print('Result has been saved in the path:')
        print('{}'.format(output_path))


    def __call__(self):
        corpus_test = Corpus.read_from_file(os.path.join(self.config['data_filepath'], self.config['data_filename']))
        scores, differ_corpus_tuples = self._evaluation(corpus_test)
        self.save_result(scores, differ_corpus_tuples)
        print('Evaluation has been done.')



if __name__ == '__main__':
    print('*****The Evaluation program is running now, please wait.*****\n')
    start = time.time()
    config_filepath = './configure.json'
    MtModelEvaluation_Deliverable(config_filepath).__call__()
    end = time.time()
    print('==>Time cost is', end - start, 's')