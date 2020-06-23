from tensorflow.python.ops import math_ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import state_ops
from tensorflow.python.framework import ops
from tensorflow.keras.optimizers import Nadam, Adam
import tensorflow as tf
from tensorflow.python.keras.callbacks import Callback

class _Nadam(Nadam):
    def __init__(self,info=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if info is None:
           info = dict()
        self.dataset_len = info.get('dataset_len',1)
        self.batch_size = info.get('batch_size',1)
        kwargs['decay'] = kwargs.get('schedule_decay', 0)


    def _prepare_local(self, var_device, var_dtype, apply_state):
        iter_transfer_actor = tf.constant(5 * (self.dataset_len // self.batch_size), dtype=self.iterations.dtype)
        local_iterations = tf.mod(self.iterations, iter_transfer_actor)
        lr_t = array_ops.identity(self._get_hyper('learning_rate', var_dtype))
        beta_1_t = array_ops.identity(self._get_hyper('beta_1', var_dtype))
        beta_2_t = array_ops.identity(self._get_hyper('beta_2', var_dtype))
        local_step = math_ops.cast(local_iterations + 1, var_dtype)
        next_step = math_ops.cast(local_iterations + 2, var_dtype)

        decay_base = math_ops.cast(0.96, var_dtype)

        m_t = beta_1_t * (1. - 0.5 * (
            math_ops.pow(decay_base, self._initial_decay * local_step)))
        m_t_1 = beta_1_t * (1. - 0.5 * (
            math_ops.pow(decay_base, self._initial_decay * next_step)))

        m_schedule_new = math_ops.cast(self._m_cache_read, var_dtype) * m_t
        if var_dtype is self._m_cache.dtype:
            m_schedule_new = array_ops.identity(state_ops.assign(
                self._m_cache, m_schedule_new, use_locking=self._use_locking))
        m_schedule_next = m_schedule_new * m_t_1

        apply_state[(var_device, var_dtype)] = dict(
            lr_t=lr_t,
            neg_lr_t=-lr_t,
            epsilon=ops.convert_to_tensor(self.epsilon, var_dtype),
            beta_1_t=beta_1_t,
            beta_2_t=beta_2_t,
            m_t=m_t,
            m_t_1=m_t_1,

            one_minus_beta_1_t=1 - beta_1_t,
            one_minus_beta_2_t=1 - beta_2_t,
            one_minus_m_t=1. - m_t,
            one_minus_m_schedule_new=1. - m_schedule_new,
            one_minus_m_schedule_next=1. - m_schedule_next,
            v_t_prime_denominator=1. - math_ops.pow(beta_2_t, local_step),
        )



class _Adam(Adam):

    def _prepare_local(self, var_device, var_dtype, apply_state):
        super(Adam, self)._prepare_local(var_device, var_dtype, apply_state)

        tmp_step = self.iterations % 1
        local_step = math_ops.cast(tmp_step + 1, var_dtype)
        beta_1_t = array_ops.identity(self._get_hyper('beta_1', var_dtype))
        beta_2_t = array_ops.identity(self._get_hyper('beta_2', var_dtype))
        beta_1_power = math_ops.pow(beta_1_t, local_step)
        beta_2_power = math_ops.pow(beta_2_t, local_step)
        lr = apply_state[(var_device, var_dtype)]['lr_t']
        apply_state[(var_device, var_dtype)].update(dict(
            lr=lr,
            epsilon=ops.convert_to_tensor(self.epsilon, var_dtype),
            beta_1_t=beta_1_t,
            beta_1_power=beta_1_power,
            one_minus_beta_1_t=1 - beta_1_t,
            beta_2_t=beta_2_t,
            beta_2_power=beta_2_power,
            one_minus_beta_2_t=1 - beta_2_t
        ))




