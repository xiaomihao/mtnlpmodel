import tensorflow as tf
from tensorflow import keras
from tensorflow.python.keras.callbacks import Callback
import tensorflow.keras.backend as K


class SetLearningRate:
    """
        层的一个包装，用来设置当前层的学习率

        转载自：https://kexue.fm/archives/6418
    """

    def __init__(self, layer, lr=0.01, is_ada=False):
        self.layer = layer
        self.lamb = lr  # 学习率比例
        self.is_ada = is_ada  # 是否自适应学习率优化器

    def __call__(self, inputs):
        with tf.keras.backend.name_scope(self.layer.name):
            if not self.layer.built:
                input_shape = tf.keras.backend.int_shape(inputs)
                self.layer.build(input_shape)
                self.layer.built = True
                if self.layer._initial_weights is not None:
                    self.layer.set_weights(self.layer._initial_weights)
        for key in ['kernel', 'bias', 'embeddings', 'depthwise_kernel', 'pointwise_kernel', 'recurrent_kernel', 'gamma',
                    'beta']:
            if hasattr(self.layer, key):
                weight = getattr(self.layer, key)
                if self.is_ada:
                    lamb = self.lamb  # 自适应学习率优化器直接保持lamb比例
                else:
                    lamb = self.lamb ** 0.5  # SGD（包括动量加速），lamb要开平方
                tf.keras.backend.set_value(weight, tf.keras.backend.eval(weight) / lamb)  # 更改初始化
                # setattr(self.layer, key, weight*lamb)
                tf.keras.backend.set_value(weight, tf.keras.backend.eval(weight) * lamb)  # 修正

        return self.layer(inputs)


class LearningRateDecay(Callback):
    '''lr decay during training epoches
    '''

    def __init__(self, model):
        super().__init__()
        self.model = model

    def scheduler(self, epoch):
        # 每隔5个epoch，学习率减小为原来的1/10
        if epoch % 6 == 0 and epoch != 0:
            lr = K.get_value(self.model.optimizer.lr)
            # K.set_value(self.model.optimizer.lr, lr * 0.1)
            K.set_value(self.model.optimizer.lr, lr * 5)
            # print("Learning Rate changed to {}".format(lr * 0.1))
            print("Learning Rate changed to {}".format(lr * 5))
        return K.get_value(self.model.optimizer.lr)

    # if __name__ == '__main__':
    # how to use
    # from tensorflow.keras.callbacks import LearningRateScheduler
    # from mtnlp_model.mtnlpmodel.trainer.lrset_util import LearningRateDecay
    # reduce_lr = LearningRateScheduler(LearningRateDecay(model).scheduler)
    # callbacks_list.append(reduce_lr)




import numpy as np

class CyclicLR(Callback):
    '''
        https://github.com/bckenstler/CLR/blob/master/clr_callback.py
    '''

    """This callback implements a cyclical learning rate policy (CLR).
    The method cycles the learning rate between two boundaries with
    some constant frequency, as detailed in this paper (https://arxiv.org/abs/1506.01186).
    The amplitude of the cycle can be scaled on a per-iteration or
    per-cycle basis.
    This class has three built-in policies, as put forth in the paper.
    "triangular":
        A basic triangular cycle w/ no amplitude scaling.
    "triangular2":
        A basic triangular cycle that scales initial amplitude by half each cycle.
    "exp_range":
        A cycle that scales initial amplitude by gamma**(cycle iterations) at each
        cycle iteration.
    For more detail, please see paper.

    # Example
        ```python
            clr = CyclicLR(base_lr=0.001, max_lr=0.006,
                                step_size=2000., mode='triangular')
            model.fit(X_train, Y_train, callbacks=[clr])
        ```

    Class also supports custom scaling functions:
        ```python
            clr_fn = lambda x: 0.5*(1+np.sin(x*np.pi/2.))
            clr = CyclicLR(base_lr=0.001, max_lr=0.006,
                                step_size=2000., scale_fn=clr_fn,
                                scale_mode='cycle')
            model.fit(X_train, Y_train, callbacks=[clr])
        ```
    # Arguments
        base_lr: initial learning rate which is the
            lower boundary in the cycle.
        max_lr: upper boundary in the cycle. Functionally,
            it defines the cycle amplitude (max_lr - base_lr).
            The lr at any cycle is the sum of base_lr
            and some scaling of the amplitude; therefore
            max_lr may not actually be reached depending on
            scaling function.
        step_size: number of training iterations per
            half cycle. Authors suggest setting step_size
            2-8 x training iterations in epoch.
        mode: one of {triangular, triangular2, exp_range}.
            Default 'triangular'.
            Values correspond to policies detailed above.
            If scale_fn is not None, this argument is ignored.
        gamma: constant in 'exp_range' scaling function:
            gamma**(cycle iterations)
        scale_fn: Custom scaling policy defined by a single
            argument lambda function, where
            0 <= scale_fn(x) <= 1 for all x >= 0.
            mode paramater is ignored
        scale_mode: {'cycle', 'iterations'}.
            Defines whether scale_fn is evaluated on
            cycle number or cycle iterations (training
            iterations since start of cycle). Default is 'cycle'.
    """

    def __init__(self, base_lr=0.001, max_lr=0.006, step_size=2000., mode='triangular',
                 gamma=1., scale_fn=None, scale_mode='cycle'):
        super(CyclicLR, self).__init__()

        self.base_lr = base_lr
        self.max_lr = max_lr
        self.step_size = step_size
        self.mode = mode
        self.gamma = gamma
        if scale_fn == None:
            if self.mode == 'triangular':
                self.scale_fn = lambda x: 1.
                self.scale_mode = 'cycle'
            elif self.mode == 'triangular2':
                self.scale_fn = lambda x: 1 / (2. ** (x - 1))
                self.scale_mode = 'cycle'
            elif self.mode == 'exp_range':
                self.scale_fn = lambda x: gamma ** (x)
                self.scale_mode = 'iterations'
        else:
            self.scale_fn = scale_fn
            self.scale_mode = scale_mode
        self.clr_iterations = 0.
        self.trn_iterations = 0.
        self.history = {}

        self._reset()

    def _reset(self, new_base_lr=None, new_max_lr=None,
               new_step_size=None):
        """Resets cycle iterations.
        Optional boundary/step size adjustment.
        """
        if new_base_lr != None:
            self.base_lr = new_base_lr
        if new_max_lr != None:
            self.max_lr = new_max_lr
        if new_step_size != None:
            self.step_size = new_step_size
        self.clr_iterations = 0.

    def clr(self):
        cycle = np.floor(1 + self.clr_iterations / (2 * self.step_size))
        x = np.abs(self.clr_iterations / self.step_size - 2 * cycle + 1)
        if self.scale_mode == 'cycle':
            return self.base_lr + (self.max_lr - self.base_lr) * np.maximum(0, (1 - x)) * self.scale_fn(cycle)
        else:
            return self.base_lr + (self.max_lr - self.base_lr) * np.maximum(0, (1 - x)) * self.scale_fn(
                self.clr_iterations)

    def on_train_begin(self, logs={}):
        logs = logs or {}

        if self.clr_iterations == 0:
            K.set_value(self.model.optimizer.lr, self.base_lr)
        else:
            K.set_value(self.model.optimizer.lr, self.clr())

    def on_batch_end(self, epoch, logs=None):

        logs = logs or {}
        self.trn_iterations += 1
        self.clr_iterations += 1

        self.history.setdefault('lr', []).append(K.get_value(self.model.optimizer.lr))
        self.history.setdefault('iterations', []).append(self.trn_iterations)

        for k, v in logs.items():
            self.history.setdefault(k, []).append(v)

        K.set_value(self.model.optimizer.lr, self.clr())


class SGDRScheduler(Callback):
    '''Schedule learning rates with restarts
     A simple restart technique for stochastic gradient descent.
    The learning rate decays after each batch and peridically resets to its
    initial value. Optionally, the learning rate is additionally reduced by a
    fixed factor at a predifined set of epochs.
     # Arguments
        epochsize: Number of samples per epoch during training.
        batchsize: Number of samples per batch during training.
        start_epoch: First epoch where decay is applied.
        epochs_to_restart: Initial number of epochs before restarts.
        mult_factor: Increase of epochs_to_restart after each restart.
        lr_fac: Decrease of learning rate at epochs given in
                lr_reduction_epochs.
        lr_reduction_epochs: Fixed list of epochs at which to reduce
                             learning rate.
     # References
        - [SGDR: Stochastic Gradient Descent with Restarts](http://arxiv.org/abs/1608.03983)
    '''

    def __init__(self,
                 epochsize,
                 batchsize,
                 epochs_to_restart=2,
                 mult_factor=2,
                 lr_fac=0.1,
                 lr_reduction_epochs=(60, 120, 160)):
        super(SGDRScheduler, self).__init__()
        self.epoch = -1
        self.batch_since_restart = 0
        self.next_restart = epochs_to_restart
        self.epochsize = epochsize
        self.batchsize = batchsize
        self.epochs_to_restart = epochs_to_restart
        self.mult_factor = mult_factor
        self.batches_per_epoch = self.epochsize / self.batchsize
        self.lr_fac = lr_fac
        self.lr_reduction_epochs = lr_reduction_epochs
        self.lr_log = []

    def on_train_begin(self, logs={}):
        self.lr = K.get_value(self.model.optimizer.lr)

    def on_epoch_begin(self, epoch, logs={}):
        self.epoch += 1

    def on_batch_end(self, batch, logs={}):
        fraction_to_restart = self.batch_since_restart / \
                              (self.batches_per_epoch * self.epochs_to_restart)
        lr = 0.5 * self.lr * (1 + np.cos(fraction_to_restart * np.pi))
        K.set_value(self.model.optimizer.lr, lr)

        self.batch_since_restart += 1
        self.lr_log.append(lr)

    def on_epoch_end(self, epoch, logs={}):
        if self.epoch + 1 == self.next_restart:
            self.batch_since_restart = 0
            self.epochs_to_restart *= self.mult_factor
            self.next_restart += self.epochs_to_restart

        if (self.epoch + 1) in self.lr_reduction_epochs:
            self.lr *= self.lr_fac
