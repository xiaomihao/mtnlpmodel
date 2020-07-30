import tensorflow as tf
from seq2annotation.algorithms.model import Model

class EstimatorVirtualEmbeddingModel(Model):
    def label2id(self, labels, name=None):
        return labels

    def id2label(self, pred_ids, name=None):
        pred_strings = []
        return pred_strings

    def load_label_data(self):
        # data = np.loadtxt(self.params['tags'], dtype=np.unicode, encoding=None)
        data = self.params["labels_data"]
        mapping_strings = tf.Variable(data)

        return mapping_strings