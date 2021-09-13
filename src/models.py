"""
Tensorflow related part
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np

from src import param
from utils import sigmoid


class TFParts(object):
    '''
    TensorFlow-related things.
    Keep TensorFlow-related components in a neat shell.
    '''

    def __init__(self, num_rels, num_cons, dim, batch_size, neg_per_positive, p_neg, psl, dim_r=None):
        self._num_rels = num_rels
        self._num_cons = num_cons
        self._dim = self._dim_r = dim  # dimension of both relation and ontology.
        if dim_r:
            self._dim_r = dim_r
        self._batch_size = batch_size
        self.batch_size_eval = batch_size*16
        self._neg_per_positive = neg_per_positive
        self._epoch_loss = 0
        self._p_neg = 1
        self._p_psl = 0.2
        self._soft_size = 1
        self._prior_psl = 0
        self._psl = psl

        assert psl == False

    def build_basics(self):
        tf.reset_default_graph()
        with tf.variable_scope("graph", initializer=tf.truncated_normal_initializer(0, 0.3)):
            # Variables (matrix of embeddings/transformations)
            self._ht = ht = tf.get_variable(
                name='ht',  # for t AND h
                shape=[self.num_cons, self._dim],
                dtype=tf.float32)

            self._r = r = tf.get_variable(
                name='r',
                shape=[self.num_rels, self._dim_r],
                dtype=tf.float32)

            self._A_h_index = A_h_index = tf.placeholder(
                dtype=tf.int64,
                shape=[None],
                name='A_h_index')
            self._A_r_index = A_r_index = tf.placeholder(
                dtype=tf.int64,
                shape=[None],
                name='A_r_index')
            self._A_t_index = A_t_index = tf.placeholder(
                dtype=tf.int64,
                shape=[None],
                name='A_t_index')

            # for uncertain graph
            self._A_w = tf.placeholder(
                dtype=tf.float32,
                shape=[None],
                name='_A_w')

            self._A_neg_hn_index = A_neg_hn_index = tf.placeholder(
                dtype=tf.int64,
                shape=(None, self._neg_per_positive),
                name='A_neg_hn_index')
            self._A_neg_rel_hn_index = A_neg_rel_hn_index = tf.placeholder(
                dtype=tf.int64,
                shape=(None, self._neg_per_positive),
                name='A_neg_rel_hn_index')
            self._A_neg_t_index = A_neg_t_index = tf.placeholder(
                dtype=tf.int64,
                shape=(None, self._neg_per_positive),
                name='A_neg_t_index')
            self._A_neg_h_index = A_neg_h_index = tf.placeholder(
                dtype=tf.int64,
                shape=(None, self._neg_per_positive),
                name='A_neg_h_index')
            self._A_neg_rel_tn_index = A_neg_rel_tn_index = tf.placeholder(
                dtype=tf.int64,
                shape=(None, self._neg_per_positive),
                name='A_neg_rel_tn_index')
            self._A_neg_tn_index = A_neg_tn_index = tf.placeholder(
                dtype=tf.int64,
                shape=(None, self._neg_per_positive),
                name='A_neg_tn_index')

            self._B_neg_hn_w = tf.placeholder(
                dtype=tf.float32,
                shape=[None, self._neg_per_positive],
                name='_B_neg_hn_w')

            self._B_neg_tn_w = tf.placeholder(
                dtype=tf.float32,
                shape=[None, self._neg_per_positive],
                name='_B_neg_tn_w')

            self._B_w = tf.placeholder(
                dtype=tf.float32,
                shape=[None],
                name='_B_w')

            self._A_semi_h_index = A_semi_h_index = tf.placeholder(
                dtype=tf.int64,
                shape=[None],
                name='A_semi_h_index')
            self._A_semi_r_index = A_semi_r_index = tf.placeholder(
                dtype=tf.int64,
                shape=[None],
                name='A_semi_r_index')
            self._A_semi_t_index = A_semi_t_index = tf.placeholder(
                dtype=tf.int64,
                shape=[None],
                name='A_semi_t_index')

            self._A_semi_w = tf.placeholder(
                dtype=tf.float32,
                shape=[None],
                name='_A_semi_w')
            

            # no normalization
            self._h_batch = tf.nn.embedding_lookup(ht, A_h_index)
            self._t_batch = tf.nn.embedding_lookup(ht, A_t_index)
            self._r_batch = tf.nn.embedding_lookup(r, A_r_index)

            self._neg_hn_con_batch = tf.nn.embedding_lookup(ht, A_neg_hn_index)
            self._neg_rel_hn_batch = tf.nn.embedding_lookup(r, A_neg_rel_hn_index)
            self._neg_t_con_batch = tf.nn.embedding_lookup(ht, A_neg_t_index)
            self._neg_h_con_batch = tf.nn.embedding_lookup(ht, A_neg_h_index)
            self._neg_rel_tn_batch = tf.nn.embedding_lookup(r, A_neg_rel_tn_index)
            self._neg_tn_con_batch = tf.nn.embedding_lookup(ht, A_neg_tn_index)

            self._semi_h_batch = tf.nn.embedding_lookup(ht, A_semi_h_index)
            self._semi_t_batch = tf.nn.embedding_lookup(ht, A_semi_t_index)
            self._semi_r_batch = tf.nn.embedding_lookup(r, A_semi_r_index)
            
            if self._psl:
                # psl batches
                self._soft_h_index = tf.placeholder(
                    dtype=tf.int64,
                    shape=[self._soft_size],
                    name='soft_h_index')
                self._soft_r_index = tf.placeholder(
                    dtype=tf.int64,
                    shape=[self._soft_size],
                    name='soft_r_index')
                self._soft_t_index = tf.placeholder(
                    dtype=tf.int64,
                    shape=[self._soft_size],
                    name='soft_t_index')

                # for uncertain graph and psl
                self._soft_w = tf.placeholder(
                    dtype=tf.float32,
                    shape=[self._soft_size],
                    name='soft_w_lower_bound')

                self._soft_h_batch = tf.nn.embedding_lookup(ht, self._soft_h_index)
                self._soft_t_batch = tf.nn.embedding_lookup(ht, self._soft_t_index)
                self._soft_r_batch = tf.nn.embedding_lookup(r, self._soft_r_index)


    def build_optimizer(self):
        if self._psl: 
            self._A_loss = tf.add(self.main_loss, self.psl_loss)
        else:
            self._A_loss = self.main_loss

        # Optimizer
        self._lr = lr = tf.placeholder(tf.float32)
        self._opt = opt = tf.train.AdamOptimizer(lr)

        # This can be replaced by
        # self._train_op_A = train_op_A = opt.minimize(A_loss)
        self._gradient = gradient = opt.compute_gradients(self._A_loss)  # splitted for debugging

        self._train_op = opt.apply_gradients(gradient)

        # Saver
        self._saver = tf.train.Saver(max_to_keep=1000)

    def build(self):
        self.build_basics()
        self.define_main_loss()  # abstract method. get self.main_loss
        if self._psl:
            self.define_psl_loss()  # abstract method. get self.psl_loss
        self.build_optimizer()

    def compute_psl_loss(self):
        self.prior_psl0 = tf.constant(self._prior_psl, tf.float32)
        self.psl_error_each = tf.square(tf.maximum(self._soft_w + self.prior_psl0 - self.psl_prob, 0))
        self.psl_mse = tf.reduce_mean(self.psl_error_each)
        self.psl_loss = self.psl_mse * self._p_psl

    @property
    def num_cons(self):
        return self._num_cons

    @property
    def num_rels(self):
        return self._num_rels

    @property
    def dim(self):
        return self._dim

    @property
    def batch_size(self):
        return self._batch_size

    @property
    def neg_batch_size(self):
        return self._neg_per_positive * self._batch_size


class UKGE_logi_TF(TFParts):
    def __init__(self, num_rels, num_cons, dim, batch_size, neg_per_positive, p_neg, psl = False):
        TFParts.__init__(self, num_rels, num_cons, dim, batch_size, neg_per_positive, p_neg, psl)
        self.build()

    def define_main_loss(self):
        # distmult on uncertain graph
        print('define main loss')

        self.w = tf.Variable(0.0, name="weights")
        self.b = tf.Variable(0.0, name="bias")

        self._htr = htr = tf.reduce_sum(
            tf.multiply(self._r_batch, tf.multiply(self._h_batch, self._t_batch, "element_wise_multiply"),
                        "r_product"),
            1)
        self._f_prob_h = f_prob_h = tf.sigmoid(self.w * htr + self.b)  # logistic regression
        self._f_score_h = f_score_h = tf.square(tf.subtract(f_prob_h, self._A_w))

        self._f_prob_hn = f_prob_hn = tf.sigmoid(self.w * (
            tf.reduce_sum(
                tf.multiply(self._neg_rel_hn_batch, tf.multiply(self._neg_hn_con_batch, self._neg_t_con_batch)), 2)
        ) + self.b)
        self._f_score_hn = f_score_hn = tf.reduce_mean(tf.square(f_prob_hn), 1)

        self._f_prob_tn = f_prob_tn = tf.sigmoid(self.w * (
            tf.reduce_sum(
                tf.multiply(self._neg_rel_tn_batch, tf.multiply(self._neg_h_con_batch, self._neg_tn_con_batch)), 2)
        ) + self.b)
        self._f_score_tn = f_score_tn = tf.reduce_mean(tf.square(f_prob_tn), 1)

        self.main_loss = (tf.reduce_sum(
            tf.add(tf.divide(tf.add(f_score_tn, f_score_hn), 2) * self._p_neg, f_score_h))) / self._batch_size

    def define_psl_loss(self):
        self.psl_prob = tf.sigmoid(self.w*tf.reduce_sum(
            tf.multiply(self._soft_r_batch,
                        tf.multiply(self._soft_h_batch, self._soft_t_batch)),
            1)+self.b)
        self.compute_psl_loss()  # in tf_parts


class UKGE_rect_TF(TFParts):
    def __init__(self, num_rels, num_cons, dim, batch_size, neg_per_positive, reg_scale, p_neg, psl):
        TFParts.__init__(self, num_rels, num_cons, dim, batch_size, neg_per_positive, p_neg, psl)
        self.reg_scale = reg_scale
        self.build()

    # override
    def define_main_loss(self):
        self.w = tf.Variable(0.0, name="weights")
        self.b = tf.Variable(0.0, name="bias")

        self._htr = htr = tf.reduce_sum(
            tf.multiply(self._r_batch, tf.multiply(self._h_batch, self._t_batch, "element_wise_multiply"),
                        "r_product"),
            1)

        self._f_prob_h = f_prob_h = self.w * htr + self.b
        self._f_score_h = f_score_h = tf.square(tf.subtract(f_prob_h, self._A_w))

        self._f_prob_hn = f_prob_hn = self.w * (
            tf.reduce_sum(
                tf.multiply(self._neg_rel_hn_batch, tf.multiply(self._neg_hn_con_batch, self._neg_t_con_batch)), 2)
        ) + self.b
        self._f_score_hn = f_score_hn = tf.reduce_mean(tf.square(f_prob_hn), 1)

        self._f_prob_tn = f_prob_tn = self.w * (
            tf.reduce_sum(
                tf.multiply(self._neg_rel_tn_batch, tf.multiply(self._neg_h_con_batch, self._neg_tn_con_batch)), 2)
        ) + self.b
        self._f_score_tn = f_score_tn = tf.reduce_mean(tf.square(f_prob_tn), 1)

        self.this_loss = this_loss = (tf.reduce_sum(
            tf.add(tf.divide(tf.add(f_score_tn, f_score_hn), 2) * self._p_neg, f_score_h))) / self._batch_size

        # L2 regularizer
        self._regularizer = regularizer = tf.add(tf.add(tf.divide(tf.nn.l2_loss(self._h_batch), self.batch_size),
                                                        tf.divide(tf.nn.l2_loss(self._t_batch), self.batch_size)),
                                                 tf.divide(tf.nn.l2_loss(self._r_batch), self.batch_size))

        self.main_loss = tf.add(this_loss, self.reg_scale * regularizer)

    # override
    def define_psl_loss(self):
        self.psl_prob = self.w * tf.reduce_sum(
            tf.multiply(self._soft_r_batch,
                        tf.multiply(self._soft_h_batch, self._soft_t_batch)),
            1) + self.b
        self.compute_psl_loss()  # in tf_parts









class TransE_m1_TF(TFParts):
    def __init__(self, num_rels, num_cons, dim, batch_size, neg_per_positive, p_neg, psl):
        TFParts.__init__(self, num_rels, num_cons, dim, batch_size, neg_per_positive, p_neg, psl)
        self.build()

    def score_function(self, h, t, r, numpy= False):
        if numpy:
            return np.sum((h + r - t)**2, axis=-1)
        else: #tf
            return tf.reduce_sum((h + r - t)**2, -1)


    def define_main_loss(self):
        # distmult on uncertain graph
        print('define main loss')

        self._f_score_h = f_score_h = self.score_function(self._h_batch, self._t_batch, self._r_batch)

        self._f_score_hn = f_score_hn = self.score_function(self._neg_hn_con_batch, self._neg_t_con_batch, self._neg_rel_hn_batch)

        self._f_score_tn = f_score_tn = self.score_function(self._neg_h_con_batch, self._neg_tn_con_batch, self._neg_rel_tn_batch)

        self.main_loss = (tf.reduce_sum(
                tf.maximum(
                    tf.reshape(tf.add(tf.subtract( 
                        tf.divide(tf.add(f_score_hn, f_score_tn), 2),
                        tf.tile(tf.expand_dims(f_score_h, -1),  [1, self._neg_per_positive])
                    ), tf.tile(tf.expand_dims(self._A_w, -1),  [1, self._neg_per_positive])), [self._batch_size*self._neg_per_positive])
                , 0), 
            )) / self._batch_size

    def define_psl_loss(self):
        self.w = tf.Variable(0.0, name="weights")
        self.b = tf.Variable(0.0, name="bias")

        self.psl_prob = tf.sigmoid(self.w*tf.reduce_sum(
            tf.multiply(self._soft_r_batch,
                        tf.multiply(self._soft_h_batch, self._soft_t_batch)),
            1)+self.b)
        self.compute_psl_loss()  # in tf_parts



class TransE_m2_TF(TFParts):
    def __init__(self, num_rels, num_cons, dim, batch_size, neg_per_positive, p_neg, psl):
        TFParts.__init__(self, num_rels, num_cons, dim, batch_size, neg_per_positive, p_neg, psl)
        self.build()

    def score_function(self, h, t, r):
        return tf.reduce_sum((h + r - t)**2, -1)


    def define_main_loss(self):
        # distmult on uncertain graph
        print('define main loss')

        self.w = tf.Variable(0.0, name="weights")
        self.b = tf.Variable(0.0, name="bias")

        self._htr = htr = tf.reduce_sum(
            tf.multiply(self._r_batch, tf.multiply(self._h_batch, self._t_batch, "element_wise_multiply"),
                        "r_product"),
            1)
        self._f_prob_h = f_prob_h = tf.sigmoid(self.w * htr + self.b)  # logistic regression
        self._f_score_h = f_score_h = tf.square(tf.subtract(f_prob_h, self._A_w))

        self._f_prob_hn = f_prob_hn = tf.sigmoid(self.w * (
            tf.reduce_sum(
                tf.multiply(self._neg_rel_hn_batch, tf.multiply(self._neg_hn_con_batch, self._neg_t_con_batch)), 2)
        ) + self.b)
        self._f_score_hn = f_score_hn = tf.reduce_mean(tf.square(f_prob_hn), 1)

        self._f_prob_tn = f_prob_tn = tf.sigmoid(self.w * (
            tf.reduce_sum(
                tf.multiply(self._neg_rel_tn_batch, tf.multiply(self._neg_h_con_batch, self._neg_tn_con_batch)), 2)
        ) + self.b)
        self._f_score_tn = f_score_tn = tf.reduce_mean(tf.square(f_prob_tn), 1)

        self.main_loss = (tf.reduce_sum(
                tf.maximum(
                    tf.reshape(tf.add(tf.subtract( 
                        tf.divide(tf.add(f_prob_hn, f_prob_tn), 2),
                        tf.tile(tf.expand_dims(f_prob_h, -1),  [1, self._neg_per_positive])
                    ), tf.tile(tf.expand_dims(self._A_w, -1),  [1, self._neg_per_positive])), [self._batch_size*self._neg_per_positive])
                , 0), 
            )) / self._batch_size

    def define_psl_loss(self):
        self.psl_prob = tf.sigmoid(self.w*tf.reduce_sum(
            tf.multiply(self._soft_r_batch,
                        tf.multiply(self._soft_h_batch, self._soft_t_batch)),
            1)+self.b)
        self.compute_psl_loss()  # in tf_parts




class TransE_m3_TF(TFParts):
    def __init__(self, num_rels, num_cons, dim, batch_size, neg_per_positive, p_neg, psl):
        TFParts.__init__(self, num_rels, num_cons, dim, batch_size, neg_per_positive, p_neg, psl)
        self.build()

    def score_function(self, h, t, r):
        return tf.sigmoid(self.w*(2 - tf.reduce_sum((h + r - t)**2, -1)) + self.b)


    def define_main_loss(self):
        # distmult on uncertain graph
        print('define main loss')

        self.w = tf.Variable(0.0, name="weights")
        self.b = tf.Variable(0.0, name="bias")

        self._f_score_h = f_score_h = self.score_function(
            self._h_batch,
            self._t_batch, 
            self._r_batch)

        f_score_h = tf.square(f_score_h - self._A_w)

        self._f_score_hn = f_score_hn = self.score_function(
            self._neg_hn_con_batch,
            self._neg_t_con_batch, 
            self._neg_rel_hn_batch)

        f_score_hn = tf.reduce_mean(tf.square(f_score_hn), 1)

        self._f_score_tn = f_score_tn = tf.reduce_mean(tf.square(self.score_function(
            self._neg_h_con_batch,
            self._neg_tn_con_batch, 
            self._neg_rel_tn_batch)), 1)
            
        self.main_loss = tf.reduce_mean(
            tf.add(tf.divide(tf.add(f_score_tn, f_score_hn), 2) * self._p_neg, f_score_h))

    def define_psl_loss(self):
        self.compute_psl_loss()  # in tf_parts


# no sigmoid
class TransE_m3_1_TF(TransE_m3_TF):

    def score_function(self, h, t, r):
        return self.w*(4 - tf.reduce_sum((h + r - t)**2, -1)) + self.b



# Uncertain TransE
# + semisupervised_v2
class TransE_m3_3_TF(TransE_m3_TF):


    def define_main_loss(self):
        # distmult on uncertain graph
        print('define main loss')

        self.w = tf.Variable(0.0, name="weights")
        self.b = tf.Variable(0.0, name="bias")

        self._f_score_h = f_score_h = self.score_function(
            self._h_batch,
            self._t_batch, 
            self._r_batch)

        f_score_h = tf.square(f_score_h - self._A_w)


        self._f_score_semi = f_score_semi = self.score_function(
            self._semi_h_batch,
            self._semi_t_batch, 
            self._semi_r_batch)

        f_score_semi = tf.square(f_score_semi - self._A_semi_w)

        # evaluation only
        self._f_score_hn = f_score_hn = self.score_function(
            self._neg_hn_con_batch,
            self._neg_t_con_batch, 
            self._neg_rel_hn_batch)
            
        self.main_loss = tf.add(tf.reduce_sum(f_score_semi), tf.reduce_sum(f_score_h))

    def define_psl_loss(self):
        self.compute_psl_loss()  # in tf_parts









class DistMult_m1_TF(TFParts):
    def __init__(self, num_rels, num_cons, dim, batch_size, neg_per_positive, p_neg, psl):
        TFParts.__init__(self, num_rels, num_cons, dim, batch_size, neg_per_positive, p_neg, psl)
        self.build()

    def define_main_loss(self):
        # distmult on uncertain graph
        print('define main loss')

        

        self._htr = htr = tf.reduce_sum(
            tf.multiply(self._r_batch, tf.multiply(self._h_batch, self._t_batch, "element_wise_multiply"),
                        "r_product"),
            1)
        self._f_score_h = f_score_h = htr # tf.sigmoid(self.w * htr + self.b)  # logistic regression

        # self._f_score_hn = f_score_hn = tf.sigmoid(self.w * (
        #     tf.reduce_sum(
        #         tf.multiply(self._neg_rel_hn_batch, tf.multiply(self._neg_hn_con_batch, self._neg_t_con_batch)), 2)
        # ) + self.b)
        self._f_score_hn = f_score_hn = tf.reduce_sum(
                tf.multiply(self._neg_rel_hn_batch, tf.multiply(self._neg_hn_con_batch, self._neg_t_con_batch)), 2)

        self._f_score_tn = f_score_tn = tf.reduce_sum(
                tf.multiply(self._neg_rel_tn_batch, tf.multiply(self._neg_h_con_batch, self._neg_tn_con_batch)), 2)

        self.main_loss = (tf.reduce_sum(
                tf.maximum(
                    tf.reshape(tf.add(tf.subtract( 
                        tf.divide(tf.add(f_score_hn, f_score_tn), 2),
                        tf.tile(tf.expand_dims(f_score_h, -1),  [1, self._neg_per_positive])
                    ), tf.tile(tf.expand_dims(self._A_w, -1),  [1, self._neg_per_positive])), [self._batch_size*self._neg_per_positive])
                , 0), 
            )) / self._batch_size

    def define_psl_loss(self):
        self.w = tf.Variable(0.0, name="weights")
        self.b = tf.Variable(0.0, name="bias")
        
        self.psl_prob = tf.sigmoid(self.w*tf.reduce_sum(
            tf.multiply(self._soft_r_batch,
                        tf.multiply(self._soft_h_batch, self._soft_t_batch)),
            1)+self.b)
        self.compute_psl_loss()  # in tf_parts



class DistMult_m2_TF(TFParts):
    def __init__(self, num_rels, num_cons, dim, batch_size, neg_per_positive, p_neg, psl):
        TFParts.__init__(self, num_rels, num_cons, dim, batch_size, neg_per_positive, p_neg, psl)
        self.build()

    def define_main_loss(self):
        # distmult on uncertain graph
        print('define main loss')

        self.w = tf.Variable(0.0, name="weights")
        self.b = tf.Variable(0.0, name="bias")

        self._htr = htr = tf.reduce_sum(
            tf.multiply(self._r_batch, tf.multiply(self._h_batch, self._t_batch, "element_wise_multiply"),
                        "r_product"),
            1)
        self._f_score_h = f_score_h = tf.sigmoid(self.w * htr + self.b)  # logistic regression

        self._f_score_hn = f_score_hn = tf.sigmoid(self.w * (
            tf.reduce_sum(
                tf.multiply(self._neg_rel_hn_batch, tf.multiply(self._neg_hn_con_batch, self._neg_t_con_batch)), 2)
        ) + self.b)

        self._f_score_tn = f_score_tn = tf.sigmoid(self.w * (
            tf.reduce_sum(
                tf.multiply(self._neg_rel_tn_batch, tf.multiply(self._neg_h_con_batch, self._neg_tn_con_batch)), 2)
        ) + self.b)

        self.main_loss = (tf.reduce_sum(
                tf.maximum(
                    tf.reshape(tf.add(tf.subtract( 
                        tf.divide(tf.add(f_score_hn, f_score_tn), 2),
                        tf.tile(tf.expand_dims(f_score_h, -1),  [1, self._neg_per_positive])
                    ), tf.tile(tf.expand_dims(self._A_w, -1),  [1, self._neg_per_positive])), [self._batch_size*self._neg_per_positive])
                , 0), 
            )) / self._batch_size

    def define_psl_loss(self):
        self.psl_prob = tf.sigmoid(self.w*tf.reduce_sum(
            tf.multiply(self._soft_r_batch,
                        tf.multiply(self._soft_h_batch, self._soft_t_batch)),
            1)+self.b)
        self.compute_psl_loss()  # in tf_parts




class ComplEx_m1_TF(TFParts):
    def __init__(self, num_rels, num_cons, dim, batch_size, neg_per_positive, p_neg, psl):
        TFParts.__init__(self, num_rels, num_cons, dim, batch_size, neg_per_positive, p_neg, psl)
        self.build()

    def split_embed_real_imag(self, vec, numpy = False):
        if numpy:
            return np.split(vec, [int(self._dim/2)], axis=-1)
        else: #tf
            split_size = [int(self._dim/2), int(self._dim/2)]
            return tf.split(vec,  split_size, -1)

    # override
    def build_basics(self):
        TFParts.build_basics(self)
        # split the embedding into real and imaginary part
        assert self._dim%2 == 0
        
        self._h_batch_real, self._h_batch_imag = self.split_embed_real_imag(self._h_batch)
        self._t_batch_real, self._t_batch_imag = self.split_embed_real_imag(self._t_batch)
        self._r_batch_real, self._r_batch_imag = self.split_embed_real_imag(self._r_batch)

        self._neg_hn_con_batch_real, self._neg_hn_con_batch_imag = self.split_embed_real_imag(self._neg_hn_con_batch)
        self._neg_rel_hn_batch_real, self._neg_rel_hn_batch_imag = self.split_embed_real_imag(self._neg_rel_hn_batch)
        self._neg_t_con_batch_real, self._neg_t_con_batch_imag   = self.split_embed_real_imag(self._neg_t_con_batch)
        self._neg_h_con_batch_real, self._neg_h_con_batch_imag   = self.split_embed_real_imag(self._neg_h_con_batch)
        self._neg_rel_tn_batch_real, self._neg_rel_tn_batch_imag = self.split_embed_real_imag(self._neg_rel_tn_batch)
        self._neg_tn_con_batch_real, self._neg_tn_con_batch_imag = self.split_embed_real_imag(self._neg_tn_con_batch)            


    def score_function(self, h_real, h_imag, t_real, t_image, r_real, r_imag, numpy = False):
        if numpy:
            return np.sum(h_real * t_real * r_real + h_imag * t_real * r_real + h_real * t_real * r_imag - h_imag * t_real * r_imag, axis=-1)
        else: #tf
            return tf.reduce_sum(h_real * t_real * r_real + h_imag * t_real * r_real + h_real * t_real * r_imag - h_imag * t_real * r_imag, -1, keep_dims = False)

    def define_main_loss(self):
        # distmult on uncertain graph
        print('define main loss')

        

        self._f_score_h = f_score_h = self.score_function(
            self._h_batch_real, self._h_batch_imag,
            self._t_batch_real, self._t_batch_imag, 
            self._r_batch_real, self._r_batch_imag)

        self._f_score_hn = f_score_hn = self.score_function(
            self._neg_hn_con_batch_real, self._neg_hn_con_batch_imag,
            self._neg_t_con_batch_real, self._neg_t_con_batch_imag, 
            self._neg_rel_hn_batch_real, self._neg_rel_hn_batch_imag)

        self._f_score_tn = f_score_tn = self.score_function(
            self._neg_h_con_batch_real, self._neg_h_con_batch_imag,
            self._neg_tn_con_batch_real, self._neg_tn_con_batch_imag, 
            self._neg_rel_tn_batch_real, self._neg_rel_tn_batch_imag)
            
        self.main_loss = (tf.reduce_sum(
                tf.maximum(
                    tf.reshape(tf.add(tf.subtract( 
                        tf.divide(tf.add(f_score_hn, f_score_tn), 2),
                        tf.tile(tf.expand_dims(f_score_h, -1),  [1, self._neg_per_positive])
                    ), tf.tile(tf.expand_dims(self._A_w, -1),  [1, self._neg_per_positive])), [self._batch_size*self._neg_per_positive])
                , 0), 
            )) / self._batch_size

    def define_psl_loss(self):

        self.w = tf.Variable(0.0, name="weights")
        self.b = tf.Variable(0.0, name="bias")
        
        self.psl_prob = tf.sigmoid(self.w*tf.reduce_sum(
            tf.multiply(self._soft_r_batch,
                        tf.multiply(self._soft_h_batch, self._soft_t_batch)),
            1)+self.b)
        self.compute_psl_loss()  # in tf_parts


class ComplEx_m1_1_TF(ComplEx_m1_TF):
    def score_function(self, h_real, h_imag, t_real, t_imag, r_real, r_imag, numpy = False):
        if numpy:
            return np.sum(h_real * t_real * r_real + h_imag * t_imag * r_real + h_real * t_imag * r_imag - h_imag * t_real * r_imag, axis=-1)
        else: #tf
            return tf.reduce_sum(h_real * t_real * r_real + h_imag * t_imag * r_real + h_real * t_imag * r_imag - h_imag * t_real * r_imag, -1, keep_dims = False)





class ComplEx_m3_TF(ComplEx_m1_TF):
    def __init__(self, num_rels, num_cons, dim, batch_size, neg_per_positive, p_neg, psl):
        ComplEx_m1_TF.__init__(self, num_rels, num_cons, dim, batch_size, neg_per_positive, p_neg, psl)

    # override
    def define_main_loss(self):
        # distmult on uncertain graph
        print('define main loss')

        self._f_score_h = f_score_h = tf.square(self.score_function(
            self._h_batch_real, self._h_batch_imag,
            self._t_batch_real, self._t_batch_imag, 
            self._r_batch_real, self._r_batch_imag) - self._A_w)

        self._f_score_hn = f_score_hn = tf.reduce_mean(tf.square(self.score_function(
            self._neg_hn_con_batch_real, self._neg_hn_con_batch_imag,
            self._neg_t_con_batch_real, self._neg_t_con_batch_imag, 
            self._neg_rel_hn_batch_real, self._neg_rel_hn_batch_imag)), 1)

        self._f_score_tn = f_score_tn = tf.reduce_mean(tf.square(self.score_function(
            self._neg_h_con_batch_real, self._neg_h_con_batch_imag,
            self._neg_tn_con_batch_real, self._neg_tn_con_batch_imag, 
            self._neg_rel_tn_batch_real, self._neg_rel_tn_batch_imag)), 1)


        self.main_loss = (tf.reduce_sum(
            tf.add(tf.divide(tf.add(f_score_tn, f_score_hn), 2) * self._p_neg, f_score_h))) / self._batch_size


class ComplEx_m4_TF(ComplEx_m1_TF):
    def __init__(self, num_rels, num_cons, dim, batch_size, neg_per_positive, p_neg, psl):
        ComplEx_m1_TF.__init__(self, num_rels, num_cons, dim, batch_size, neg_per_positive, p_neg, psl)


    def score_function(self, h_real, h_imag, t_real, t_image, r_real, r_imag, numpy = False, numpy_w = None, numpy_b = None):
        if numpy:
            return sigmoid(numpy_w*np.sum(h_real * t_real * r_real + h_imag * t_real * r_real + h_real * t_real * r_imag - h_imag * t_real * r_imag, axis=-1) + numpy_b)
        else: #tf
            return tf.sigmoid(
                self.w * 
                tf.reduce_sum(h_real * t_real * r_real + h_imag * t_real * r_real + h_real * t_real * r_imag - h_imag * t_real * r_imag, -1, keep_dims = False) +
                self.b)

    def define_main_loss(self):
        # distmult on uncertain graph
        print('define main loss')

        self.w = tf.Variable(0.0, name="weights")
        self.b = tf.Variable(0.0, name="bias")
        

        self._f_score_h = f_score_h = self.score_function(
            self._h_batch_real, self._h_batch_imag,
            self._t_batch_real, self._t_batch_imag, 
            self._r_batch_real, self._r_batch_imag)

        self._f_score_hn = f_score_hn = self.score_function(
            self._neg_hn_con_batch_real, self._neg_hn_con_batch_imag,
            self._neg_t_con_batch_real, self._neg_t_con_batch_imag, 
            self._neg_rel_hn_batch_real, self._neg_rel_hn_batch_imag)

        self._f_score_tn = f_score_tn = self.score_function(
            self._neg_h_con_batch_real, self._neg_h_con_batch_imag,
            self._neg_tn_con_batch_real, self._neg_tn_con_batch_imag, 
            self._neg_rel_tn_batch_real, self._neg_rel_tn_batch_imag)
            
        self.main_loss = (tf.reduce_sum(
                tf.maximum(
                    tf.reshape(tf.add(tf.subtract( 
                        tf.divide(tf.add(f_score_hn, f_score_tn), 2),
                        tf.tile(tf.expand_dims(f_score_h, -1),  [1, self._neg_per_positive])
                    ), tf.tile(tf.expand_dims(self._A_w, -1),  [1, self._neg_per_positive])), [self._batch_size*self._neg_per_positive])
                , 0), 
            )) / self._batch_size

    def define_psl_loss(self):        
        
        self.psl_prob = tf.sigmoid(self.w*tf.reduce_sum(
            tf.multiply(self._soft_r_batch,
                        tf.multiply(self._soft_h_batch, self._soft_t_batch)),
            1)+self.b)
        self.compute_psl_loss()  # in tf_parts



class ComplEx_m5_1_TF(ComplEx_m1_TF):
    def __init__(self, num_rels, num_cons, dim, batch_size, neg_per_positive, p_neg, psl):
        ComplEx_m1_TF.__init__(self, num_rels, num_cons, dim, batch_size, neg_per_positive, p_neg, psl)

    def score_function(self, h_real, h_imag, t_real, t_imag, r_real, r_imag, numpy = False, numpy_w = None, numpy_b = None):
        if numpy:
            return sigmoid(numpy_w*np.sum(h_real * t_real * r_real + h_imag * t_imag * r_real + h_real * t_imag * r_imag - h_imag * t_real * r_imag, axis=-1) + numpy_b)
        else: #tf
            return tf.sigmoid(
                self.w * 
                tf.reduce_sum(h_real * t_real * r_real + h_imag * t_imag * r_real + h_real * t_imag * r_imag - h_imag * t_real * r_imag, -1, keep_dims = False) +
                self.b)

    def define_main_loss(self):
        # distmult on uncertain graph
        print('define main loss')

        self.w = tf.Variable(0.0, name="weights")
        self.b = tf.Variable(0.0, name="bias")
        

        self._f_score_h = f_score_h = self.score_function(
            self._h_batch_real, self._h_batch_imag,
            self._t_batch_real, self._t_batch_imag, 
            self._r_batch_real, self._r_batch_imag)

        f_score_h = tf.square(f_score_h - self._A_w)

        self._f_score_hn = f_score_hn = self.score_function(
            self._neg_hn_con_batch_real, self._neg_hn_con_batch_imag,
            self._neg_t_con_batch_real, self._neg_t_con_batch_imag, 
            self._neg_rel_hn_batch_real, self._neg_rel_hn_batch_imag)

        f_score_hn = tf.reduce_mean(tf.square(f_score_hn), 1)

        self._f_score_tn = f_score_tn = tf.reduce_mean(tf.square(self.score_function(
            self._neg_h_con_batch_real, self._neg_h_con_batch_imag,
            self._neg_tn_con_batch_real, self._neg_tn_con_batch_imag, 
            self._neg_rel_tn_batch_real, self._neg_rel_tn_batch_imag)), 1)
            
        self.main_loss = (tf.reduce_sum(
            tf.add(tf.divide(tf.add(f_score_tn, f_score_hn), 2) * self._p_neg, f_score_h))) / self._batch_size

    def define_psl_loss(self):        
        
        self.psl_prob = tf.sigmoid(self.w*tf.reduce_sum(
            tf.multiply(self._soft_r_batch,
                        tf.multiply(self._soft_h_batch, self._soft_t_batch)),
            1)+self.b)
        self.compute_psl_loss()  # in tf_parts


    

# increase the weight of neg sample in loss
class ComplEx_m5_2_TF(ComplEx_m5_1_TF):

    def define_main_loss(self):
        # distmult on uncertain graph
        print('define main loss')

        self.w = tf.Variable(0.0, name="weights")
        self.b = tf.Variable(0.0, name="bias")
        

        self._f_score_h = f_score_h = self.score_function(
            self._h_batch_real, self._h_batch_imag,
            self._t_batch_real, self._t_batch_imag, 
            self._r_batch_real, self._r_batch_imag)

        f_score_h = tf.square(f_score_h - self._A_w)

        self._f_score_hn = f_score_hn = self.score_function(
            self._neg_hn_con_batch_real, self._neg_hn_con_batch_imag,
            self._neg_t_con_batch_real, self._neg_t_con_batch_imag, 
            self._neg_rel_hn_batch_real, self._neg_rel_hn_batch_imag)

        f_score_hn = tf.reduce_sum(tf.square(f_score_hn), 1)

        self._f_score_tn = f_score_tn = tf.reduce_sum(tf.square(self.score_function(
            self._neg_h_con_batch_real, self._neg_h_con_batch_imag,
            self._neg_tn_con_batch_real, self._neg_tn_con_batch_imag, 
            self._neg_rel_tn_batch_real, self._neg_rel_tn_batch_imag)), 1)
            
        self.main_loss = tf.reduce_mean(
            tf.add(tf.add(f_score_tn, f_score_hn), f_score_h))





# uncertain complex with semisupervised_v2
class ComplEx_m5_3_TF(ComplEx_m5_1_TF):

    # override
    def build_basics(self):
        ComplEx_m5_1_TF.build_basics(self)

        self._semi_h_batch_real, self._semi_h_batch_imag = self.split_embed_real_imag(self._semi_h_batch)
        self._semi_t_batch_real, self._semi_t_batch_imag = self.split_embed_real_imag(self._semi_t_batch)
        self._semi_r_batch_real, self._semi_r_batch_imag = self.split_embed_real_imag(self._semi_r_batch)


    def define_main_loss(self):
        # distmult on uncertain graph
        print('define main loss')

        self.w = tf.Variable(0.0, name="weights")
        self.b = tf.Variable(0.0, name="bias")
        

        self._f_score_h = f_score_h = self.score_function(
            self._h_batch_real, self._h_batch_imag,
            self._t_batch_real, self._t_batch_imag, 
            self._r_batch_real, self._r_batch_imag)

        f_score_h = tf.square(f_score_h - self._A_w)

        self._f_score_semi = f_score_semi = self.score_function(
            self._semi_h_batch_real, self._semi_h_batch_imag,
            self._semi_t_batch_real, self._semi_t_batch_imag, 
            self._semi_r_batch_real, self._semi_r_batch_imag)

        f_score_semi = tf.square(f_score_semi - self._A_semi_w)


        # evaluation only
        self._f_score_hn = f_score_hn = self.score_function(
            self._neg_hn_con_batch_real, self._neg_hn_con_batch_imag,
            self._neg_t_con_batch_real, self._neg_t_con_batch_imag, 
            self._neg_rel_hn_batch_real, self._neg_rel_hn_batch_imag)

            
        self.main_loss = tf.add(tf.reduce_sum(f_score_semi), tf.reduce_sum(f_score_h))



# semisupervised_v2
# 20200414 / batch_size in loss function
class ComplEx_m5_4_TF(ComplEx_m5_3_TF):


    def define_main_loss(self):
        # distmult on uncertain graph
        print('define main loss')

        self.w = tf.Variable(0.0, name="weights")
        self.b = tf.Variable(0.0, name="bias")
        

        self._f_score_h = f_score_h = self.score_function(
            self._h_batch_real, self._h_batch_imag,
            self._t_batch_real, self._t_batch_imag, 
            self._r_batch_real, self._r_batch_imag)

        f_score_h = tf.square(f_score_h - self._A_w)

        self._f_score_semi = f_score_semi = self.score_function(
            self._semi_h_batch_real, self._semi_h_batch_imag,
            self._semi_t_batch_real, self._semi_t_batch_imag, 
            self._semi_r_batch_real, self._semi_r_batch_imag)

        f_score_semi = tf.square(f_score_semi - self._A_semi_w)


        # evaluation only
        self._f_score_hn = f_score_hn = self.score_function(
            self._neg_hn_con_batch_real, self._neg_hn_con_batch_imag,
            self._neg_t_con_batch_real, self._neg_t_con_batch_imag, 
            self._neg_rel_hn_batch_real, self._neg_rel_hn_batch_imag)

            
        # self.main_loss = tf.add(tf.reduce_mean(f_score_semi)*self.batch_size, tf.reduce_sum(f_score_h))
        self.main_loss = tf.add(tf.reduce_mean(f_score_semi)*self.batch_size, tf.reduce_sum(f_score_h))/self.batch_size






# with --semisupervised_neg_v2
class ComplEx_m6_1_TF(ComplEx_m5_1_TF):

    def define_main_loss(self):
        # distmult on uncertain graph
        print('define main loss')

        self.w = tf.Variable(0.0, name="weights")
        self.b = tf.Variable(0.0, name="bias")
        

        self._f_score_h = f_score_h = self.score_function(
            self._h_batch_real, self._h_batch_imag,
            self._t_batch_real, self._t_batch_imag, 
            self._r_batch_real, self._r_batch_imag)
            
        f_score_h = tf.square(f_score_h - self._A_w)

        self._f_score_hn = f_score_hn =self.score_function(
            self._neg_hn_con_batch_real, self._neg_hn_con_batch_imag,
            self._neg_t_con_batch_real, self._neg_t_con_batch_imag, 
            self._neg_rel_hn_batch_real, self._neg_rel_hn_batch_imag)

        f_score_hn = tf.reduce_mean(tf.square(f_score_hn - self._B_neg_hn_w), 1)


        self._f_score_tn = f_score_tn = tf.reduce_mean(tf.square(self.score_function(
            self._neg_h_con_batch_real, self._neg_h_con_batch_imag,
            self._neg_tn_con_batch_real, self._neg_tn_con_batch_imag, 
            self._neg_rel_tn_batch_real, self._neg_rel_tn_batch_imag) - self._B_neg_hn_w), 1)
            
        self.main_loss = (tf.reduce_sum(
            tf.add(tf.divide(tf.add(f_score_tn, f_score_hn), 2) * self._p_neg, f_score_h))) / self._batch_size

# increase the weight of neg sample in loss
class ComplEx_m6_2_TF(ComplEx_m5_1_TF):

    def define_main_loss(self):
        # distmult on uncertain graph
        print('define main loss')

        self.w = tf.Variable(0.0, name="weights")
        self.b = tf.Variable(0.0, name="bias")
        

        self._f_score_h = f_score_h = self.score_function(
            self._h_batch_real, self._h_batch_imag,
            self._t_batch_real, self._t_batch_imag, 
            self._r_batch_real, self._r_batch_imag)
            
        f_score_h = tf.square(f_score_h - self._A_w)

        self._f_score_hn = f_score_hn =self.score_function(
            self._neg_hn_con_batch_real, self._neg_hn_con_batch_imag,
            self._neg_t_con_batch_real, self._neg_t_con_batch_imag, 
            self._neg_rel_hn_batch_real, self._neg_rel_hn_batch_imag)

        f_score_hn = tf.reduce_sum(tf.square(f_score_hn - self._B_neg_hn_w), 1)


        self._f_score_tn = f_score_tn = tf.reduce_sum(tf.square(self.score_function(
            self._neg_h_con_batch_real, self._neg_h_con_batch_imag,
            self._neg_tn_con_batch_real, self._neg_tn_con_batch_imag, 
            self._neg_rel_tn_batch_real, self._neg_rel_tn_batch_imag) - self._B_neg_hn_w), 1)
            
        self.main_loss = tf.reduce_mean(
            tf.add(tf.add(f_score_tn, f_score_hn), f_score_h))






# = m5 - MSE + negative logarithm loss
class ComplEx_m7_TF(ComplEx_m5_1_TF):

    def define_main_loss(self):
        # distmult on uncertain graph
        print('define main loss')

        self.w = tf.Variable(0.0, name="weights")
        self.b = tf.Variable(0.0, name="bias")

        self._f_score_h = f_score_h = self.score_function(
            self._h_batch_real, self._h_batch_imag,
            self._t_batch_real, self._t_batch_imag, 
            self._r_batch_real, self._r_batch_imag)

        self._f_score_hn = f_score_hn = self.score_function(
            self._neg_hn_con_batch_real, self._neg_hn_con_batch_imag,
            self._neg_t_con_batch_real, self._neg_t_con_batch_imag, 
            self._neg_rel_hn_batch_real, self._neg_rel_hn_batch_imag)

        self._f_score_tn = f_score_tn = self.score_function(
            self._neg_h_con_batch_real, self._neg_h_con_batch_imag,
            self._neg_tn_con_batch_real, self._neg_tn_con_batch_imag, 
            self._neg_rel_tn_batch_real, self._neg_rel_tn_batch_imag)
            

        self.main_loss = tf.reduce_sum(
                (
                    -tf.math.log(tf.sigmoid(f_score_h - tf.reduce_mean(tf.divide((f_score_hn + f_score_tn), 2), -1)))
                )*self._A_w 
                # (
                #     -tf.math.log(tf.sigmoid(f_score_h )) -
                #     tf.reduce_mean(tf.math.log(tf.sigmoid(-f_score_hn)), -1)   -
                #     tf.reduce_mean(tf.math.log(tf.sigmoid(-f_score_tn)), -1)
                # )*self._A_w 
            , -1) / self._batch_size



# = m5  + negative logarithm loss 
class ComplEx_m8_TF(ComplEx_m5_1_TF):

    def define_main_loss(self):
        # distmult on uncertain graph
        print('define main loss')

        self.w = tf.Variable(0.0, name="weights")
        self.b = tf.Variable(0.0, name="bias")

        self._f_score_h = f_score_h = self.score_function(
            self._h_batch_real, self._h_batch_imag,
            self._t_batch_real, self._t_batch_imag, 
            self._r_batch_real, self._r_batch_imag)

        self._f_score_hn = f_score_hn = self.score_function(
            self._neg_hn_con_batch_real, self._neg_hn_con_batch_imag,
            self._neg_t_con_batch_real, self._neg_t_con_batch_imag, 
            self._neg_rel_hn_batch_real, self._neg_rel_hn_batch_imag)

        self._f_score_tn = f_score_tn = self.score_function(
            self._neg_h_con_batch_real, self._neg_h_con_batch_imag,
            self._neg_tn_con_batch_real, self._neg_tn_con_batch_imag, 
            self._neg_rel_tn_batch_real, self._neg_rel_tn_batch_imag)
            

        self.main_loss = tf.reduce_sum(
                (
                    -tf.math.log(tf.sigmoid(f_score_h - tf.reduce_mean(tf.divide((f_score_hn + f_score_tn), 2), -1)))
                )*self._A_w 
            , -1) / self._batch_size + tf.reduce_mean(
                tf.add(tf.divide(tf.add(
                    tf.reduce_mean(tf.square(f_score_tn)), 
                    tf.reduce_mean(tf.square(f_score_hn)))
                , 2) * self._p_neg, tf.square(f_score_h - self._A_w)), -1)


# m5_1 + auto encoder (2 layers)
class ComplEx_m9_1_TF(ComplEx_m5_1_TF):
    
    def define_main_loss(self):
        ComplEx_m5_1_TF.define_main_loss(self)

        y_autoencoder_h = tf.one_hot(self._A_h_index, self._num_cons)

        h_batch = tf.concat([self._h_batch_real, self._h_batch_imag], -1)
        # t_batch = tf.concat([self._t_batch_real, self._t_batch_imag], -1)
        self._autoencoder_h_dense1 = tf.layers.dense(h_batch, 1024)
        self._autoencoder_h_output = self._autoencoder_h_dense2 = tf.layers.dense(self._autoencoder_h_dense1, self._num_cons)
        # self._autoencoder_h_output = tf.nn.softmax(self._autoencoder_h_dense2)

        lambda_autoencoder = 1.0

        self.main_loss += lambda_autoencoder*tf.nn.softmax_cross_entropy_with_logits(logits = self._autoencoder_h_output, labels = y_autoencoder_h) / self.batch_size

# m5_1 + auto encoder(1 layer)
class ComplEx_m9_2_TF(ComplEx_m5_1_TF):
    
    def define_main_loss(self):
        ComplEx_m5_1_TF.define_main_loss(self)

        y_autoencoder_h = tf.one_hot(self._A_h_index, self._num_cons)

        h_batch = tf.concat([self._h_batch_real, self._h_batch_imag], -1)
        # t_batch = tf.concat([self._t_batch_real, self._t_batch_imag], -1)
        # self._autoencoder_h_dense1 = tf.layers.dense(h_batch, 1024)
        self._autoencoder_h_output = self._autoencoder_h_dense2 = tf.layers.dense(h_batch, self._num_cons)
        # self._autoencoder_h_output = tf.nn.softmax(self._autoencoder_h_dense2)

        lambda_autoencoder = 1.0

        self.main_loss += lambda_autoencoder*tf.nn.softmax_cross_entropy_with_logits(logits = self._autoencoder_h_output, labels = y_autoencoder_h) / self.batch_size


# m5_1 + auto encoder (2 layers + relu)
class ComplEx_m9_3_TF(ComplEx_m5_1_TF):
    
    def define_main_loss(self):
        ComplEx_m5_1_TF.define_main_loss(self)

        y_autoencoder_h = tf.one_hot(self._A_h_index, self._num_cons)

        h_batch = tf.concat([self._h_batch_real, self._h_batch_imag], -1)
        # t_batch = tf.concat([self._t_batch_real, self._t_batch_imag], -1)
        self._autoencoder_h_dense1 = tf.nn.relu(tf.layers.dense(h_batch, 1024))
        self._autoencoder_h_output = self._autoencoder_h_dense2 = tf.layers.dense(self._autoencoder_h_dense1, self._num_cons)
        # self._autoencoder_h_output = tf.nn.softmax(self._autoencoder_h_dense2)

        lambda_autoencoder = 1.0

        self.main_loss += lambda_autoencoder*tf.nn.softmax_cross_entropy_with_logits(logits = self._autoencoder_h_output, labels = y_autoencoder_h) / self.batch_size




# semi supervised - confidence weighted
# run with --semisupervised_v1 or --semisupervised_v1_1
class ComplEx_m10_TF(ComplEx_m5_1_TF):

    def define_main_loss(self):
        # distmult on uncertain graph
        print('define main loss')

        self.w = tf.Variable(0.0, name="weights")
        self.b = tf.Variable(0.0, name="bias")
        

        self._f_score_h = f_score_h = self.score_function(
            self._h_batch_real, self._h_batch_imag,
            self._t_batch_real, self._t_batch_imag, 
            self._r_batch_real, self._r_batch_imag)
            
        f_score_h = tf.square((f_score_h - self._A_w)* self._B_w)


        self._f_score_hn = f_score_hn = self.score_function(
            self._neg_hn_con_batch_real, self._neg_hn_con_batch_imag,
            self._neg_t_con_batch_real, self._neg_t_con_batch_imag, 
            self._neg_rel_hn_batch_real, self._neg_rel_hn_batch_imag)
            
        f_score_hn = tf.reduce_mean(tf.square(f_score_hn*(1 - self._B_neg_hn_w) ), 1)

        self._f_score_tn = f_score_tn = tf.reduce_mean(tf.square(self.score_function(
            self._neg_h_con_batch_real, self._neg_h_con_batch_imag,
            self._neg_tn_con_batch_real, self._neg_tn_con_batch_imag, 
            self._neg_rel_tn_batch_real, self._neg_rel_tn_batch_imag)*(1 - self._B_neg_hn_w) ), 1)
            
        self.main_loss = (tf.reduce_sum(
            tf.add(tf.divide(tf.add(f_score_tn, f_score_hn), 2) * self._p_neg, f_score_h))) / self._batch_size



# semi supervised - confidence weighted
# run with --semisupervised_v1 or --semisupervised_v1_1
class ComplEx_m10_1_TF(ComplEx_m10_TF):

    def define_main_loss(self):
        # distmult on uncertain graph
        print('define main loss')

        self.w = tf.Variable(0.0, name="weights")
        self.b = tf.Variable(0.0, name="bias")
        

        self._f_score_h = f_score_h = self.score_function(
            self._h_batch_real, self._h_batch_imag,
            self._t_batch_real, self._t_batch_imag, 
            self._r_batch_real, self._r_batch_imag)
            
        f_score_h = tf.square((f_score_h - self._A_w))


        self._f_score_hn = f_score_hn = self.score_function(
            self._neg_hn_con_batch_real, self._neg_hn_con_batch_imag,
            self._neg_t_con_batch_real, self._neg_t_con_batch_imag, 
            self._neg_rel_hn_batch_real, self._neg_rel_hn_batch_imag)
            
        f_score_hn = tf.reduce_mean(tf.square(f_score_hn*(1 - self._B_neg_hn_w) ), 1)

        self._f_score_tn = f_score_tn = tf.reduce_mean(tf.square(self.score_function(
            self._neg_h_con_batch_real, self._neg_h_con_batch_imag,
            self._neg_tn_con_batch_real, self._neg_tn_con_batch_imag, 
            self._neg_rel_tn_batch_real, self._neg_rel_tn_batch_imag)*(1 - self._B_neg_tn_w) ), 1)
            
        self.main_loss = (tf.reduce_sum(
            tf.add(tf.divide(tf.add(f_score_tn, f_score_hn), 2) * self._p_neg, f_score_h))) / self._batch_size

# semi supervised - confidence weighted 2 - select those with higher weight
# run with --semisupervised_v1 or --semisupervised_v1_1
class ComplEx_m10_2_TF(ComplEx_m10_TF):

    def define_main_loss(self):
        # distmult on uncertain graph
        print('define main loss')

        self.w = tf.Variable(0.0, name="weights")
        self.b = tf.Variable(0.0, name="bias")
        

        self._f_score_h = f_score_h = self.score_function(
            self._h_batch_real, self._h_batch_imag,
            self._t_batch_real, self._t_batch_imag, 
            self._r_batch_real, self._r_batch_imag)
            
        f_score_h = tf.square((f_score_h - self._A_w))


        self._f_score_hn = f_score_hn = self.score_function(
            self._neg_hn_con_batch_real, self._neg_hn_con_batch_imag,
            self._neg_t_con_batch_real, self._neg_t_con_batch_imag, 
            self._neg_rel_hn_batch_real, self._neg_rel_hn_batch_imag)
            
        f_score_hn = tf.reduce_sum(tf.square(f_score_hn)*(self._B_neg_hn_w + 0.00001), 1) / (0.00001*self._batch_size + tf.reduce_sum(self._B_neg_hn_w, -1))

        self._f_score_tn = f_score_tn = tf.reduce_sum(tf.square(self.score_function(
            self._neg_h_con_batch_real, self._neg_h_con_batch_imag,
            self._neg_tn_con_batch_real, self._neg_tn_con_batch_imag, 
            self._neg_rel_tn_batch_real, self._neg_rel_tn_batch_imag))*(self._B_neg_tn_w + 0.00001), 1) / (0.00001*self._batch_size + tf.reduce_sum(self._B_neg_tn_w, -1))
            
        self.main_loss = (tf.reduce_sum(
            tf.add(tf.divide(tf.add(f_score_tn, f_score_hn), 2) * self._p_neg, f_score_h))) / self._batch_size


# semi supervised - confidence weighted 2 - select those with higher weight
# run with --semisupervised_v1 or --semisupervised_v1_1
class ComplEx_m10_3_TF(ComplEx_m10_TF):

    def define_main_loss(self):
        # distmult on uncertain graph
        print('define main loss')

        self.w = tf.Variable(0.0, name="weights")
        self.b = tf.Variable(0.0, name="bias")
        

        self._f_score_h = f_score_h = self.score_function(
            self._h_batch_real, self._h_batch_imag,
            self._t_batch_real, self._t_batch_imag, 
            self._r_batch_real, self._r_batch_imag)
            
        f_score_h = tf.square((f_score_h - self._A_w))


        self._f_score_hn = f_score_hn = self.score_function(
            self._neg_hn_con_batch_real, self._neg_hn_con_batch_imag,
            self._neg_t_con_batch_real, self._neg_t_con_batch_imag, 
            self._neg_rel_hn_batch_real, self._neg_rel_hn_batch_imag)
            
        f_score_hn = tf.reduce_sum(tf.square(f_score_hn)*(tf.math.exp(self._B_neg_hn_w + 0.00001)), 1) / tf.reduce_sum(tf.math.exp(0.00001 + self._B_neg_hn_w), -1) * self._neg_per_positive

        self._f_score_tn = f_score_tn = tf.reduce_sum(tf.square(self.score_function(
            self._neg_h_con_batch_real, self._neg_h_con_batch_imag,
            self._neg_tn_con_batch_real, self._neg_tn_con_batch_imag,  
            self._neg_rel_tn_batch_real, self._neg_rel_tn_batch_imag))*(tf.math.exp(self._B_neg_tn_w + 0.00001)), 1) / tf.reduce_sum(tf.math.exp(0.00001 + self._B_neg_tn_w), -1) * self._neg_per_positive
            
        self.main_loss = (tf.reduce_sum(
            tf.add(tf.divide(tf.add(f_score_tn, f_score_hn), 2) * self._p_neg, f_score_h))) / self._batch_size



# semi supervised - confidence weighted 2 - select those with higher weight
# run with --semisupervised_v1_1
class ComplEx_m10_4_TF(ComplEx_m5_1_TF):

    def define_main_loss(self):
        # distmult on uncertain graph
        print('define main loss')

        self.w = tf.Variable(0.0, name="weights")
        self.b = tf.Variable(0.0, name="bias")
        

        self._f_score_h = f_score_h = self.score_function(
            self._h_batch_real, self._h_batch_imag,
            self._t_batch_real, self._t_batch_imag, 
            self._r_batch_real, self._r_batch_imag)
            
        f_score_h = tf.square((f_score_h - self._A_w))*self._B_w


        self._f_score_hn = f_score_hn = self.score_function(
            self._neg_hn_con_batch_real, self._neg_hn_con_batch_imag,
            self._neg_t_con_batch_real, self._neg_t_con_batch_imag, 
            self._neg_rel_hn_batch_real, self._neg_rel_hn_batch_imag)
            
        f_score_hn = tf.reduce_mean(tf.square(f_score_hn), -1)

        self._f_score_tn = f_score_tn = tf.reduce_mean(tf.square(self.score_function(
            self._neg_h_con_batch_real, self._neg_h_con_batch_imag,
            self._neg_tn_con_batch_real, self._neg_tn_con_batch_imag,  
            self._neg_rel_tn_batch_real, self._neg_rel_tn_batch_imag)), -1)
            
        self.main_loss = (tf.reduce_sum(
            tf.add(tf.divide(tf.add(f_score_tn, f_score_hn), 2) * self._p_neg, f_score_h))) / self._batch_size








class RotatE_m1_TF(ComplEx_m1_TF):

    def __init__(self, num_rels, num_cons, dim, batch_size, neg_per_positive, p_neg, psl, gamma):
        TFParts.__init__(self, num_rels, num_cons, dim, batch_size, neg_per_positive, p_neg, psl, dim_r=int(dim/2))
        self._gamma = gamma # 24
        self._epsilon = 0.2
        self.build()

    def split_embed_real_imag(self, vec, numpy = False):
        if numpy:
            return np.split(vec, [int(self._dim/2)], axis=-1)
        else: #tf
            split_size = [int(self._dim/2), int(self._dim/2)]
            return tf.split(vec,  split_size, -1)


    # override
    def build_basics(self):
        TFParts.build_basics(self)
        # split the embedding into real and imaginary part
        assert self._dim%2 == 0
        
        self._h_batch_real, self._h_batch_imag = self.split_embed_real_imag(self._h_batch)
        self._t_batch_real, self._t_batch_imag = self.split_embed_real_imag(self._t_batch)

        self._neg_hn_con_batch_real, self._neg_hn_con_batch_imag = self.split_embed_real_imag(self._neg_hn_con_batch)
        self._neg_t_con_batch_real, self._neg_t_con_batch_imag   = self.split_embed_real_imag(self._neg_t_con_batch)
        self._neg_h_con_batch_real, self._neg_h_con_batch_imag   = self.split_embed_real_imag(self._neg_h_con_batch)
        self._neg_tn_con_batch_real, self._neg_tn_con_batch_imag = self.split_embed_real_imag(self._neg_tn_con_batch)            


    def score_function(self, h_real, h_imag, t_real, t_imag, r, numpy = False):
        pi = 3.14159265358979323846
        magic = (self._gamma + self._epsilon)/self._dim/pi
        
        if numpy:
            r_phrase = r/magic
            r_real = np.cos(r_phrase)
            r_imag = np.sin(r_phrase)

            score_real = h_real * r_real - h_imag * r_imag
            score_imag = h_real * r_imag + h_real * r_real
            score_real = score_real - t_real
            score_imag = score_imag - t_imag


            return self._gamma - np.sum(np.square(score_real) + np.square(score_imag), axis=-1)
        else: #tf
            
            r_phrase = r/magic
            r_real = tf.cos(r_phrase)
            r_imag = tf.sin(r_phrase)

            score_real = h_real * r_real - h_imag * r_imag
            score_imag = h_real * r_imag + h_real * r_real
            score_real = score_real - t_real
            score_imag = score_imag - t_imag

            return self._gamma - tf.reduce_sum(tf.square(score_real) + tf.square(score_imag), -1)

    def define_main_loss(self):
        # distmult on uncertain graph
        print('define main loss')

        self._f_score_h = f_score_h = self.score_function(
            self._h_batch_real, self._h_batch_imag,
            self._t_batch_real, self._t_batch_imag, 
            self._r_batch)

        f_score_h = tf.square(f_score_h - self._A_w)

        self._f_score_hn = f_score_hn = self.score_function(
            self._neg_hn_con_batch_real, self._neg_hn_con_batch_imag,
            self._neg_t_con_batch_real, self._neg_t_con_batch_imag, 
            self._neg_rel_hn_batch)

        f_score_hn = tf.reduce_mean(tf.square(f_score_hn), 1)

        self._f_score_tn = f_score_tn = tf.reduce_mean(tf.square(self.score_function(
            self._neg_h_con_batch_real, self._neg_h_con_batch_imag,
            self._neg_tn_con_batch_real, self._neg_tn_con_batch_imag, 
            self._neg_rel_tn_batch)), 1)
            
        self.main_loss = (tf.reduce_sum(
            tf.add(tf.divide(tf.add(f_score_tn, f_score_hn), 2) * self._p_neg, f_score_h))) / self._batch_size

    def define_psl_loss(self):

        self.w = tf.Variable(0.0, name="weights")
        self.b = tf.Variable(0.0, name="bias")
        
        self.psl_prob = tf.sigmoid(self.w*tf.reduce_sum(
            tf.multiply(self._soft_r_batch,
                        tf.multiply(self._soft_h_batch, self._soft_t_batch)),
            1)+self.b)
        self.compute_psl_loss()  # in tf_parts



class RotatE_m2_TF(RotatE_m1_TF):

    def score_function(self, h_real, h_imag, t_real, t_imag, r, numpy = False):
        pi = 3.14159265358979323846
        # magic = (self._gamma + self._epsilon)/self._dim/pi
        
        if numpy:
            # r_phrase = r/magic
            r_real = np.cos(r)
            r_imag = np.sin(r)

            score_real = h_real * r_real - h_imag * r_imag
            score_imag = h_real * r_imag + h_imag * r_real
            score_real = score_real - t_real
            score_imag = score_imag - t_imag


            return self._gamma - np.sum(np.square(score_real) + np.square(score_imag), axis=-1)
        else: #tf
            
            # r_phrase = r/magic
            r_real = tf.cos(r)
            r_imag = tf.sin(r)

            score_real = h_real * r_real - h_imag * r_imag
            score_imag = h_real * r_imag + h_imag * r_real
            score_real = score_real - t_real
            score_imag = score_imag - t_imag

            return self._gamma - tf.reduce_sum(tf.square(score_real) + tf.square(score_imag), -1)


class RotatE_m2_1_TF(RotatE_m2_TF):
    def define_main_loss(self):
        print('define main loss')
        self.w = tf.Variable(0.0, name="weights")
        self.b = tf.Variable(0.0, name="bias")

        self._f_score_h = f_score_h = self.score_function(
            self._h_batch_real, self._h_batch_imag,
            self._t_batch_real, self._t_batch_imag, 
            self._r_batch)

        f_score_h = tf.square(f_score_h - self._A_w)

        self._f_score_hn = f_score_hn = self.score_function(
            self._neg_hn_con_batch_real, self._neg_hn_con_batch_imag,
            self._neg_t_con_batch_real, self._neg_t_con_batch_imag, 
            self._neg_rel_hn_batch)

        f_score_hn = tf.reduce_sum(tf.square(f_score_hn), 1)

        self._f_score_tn = f_score_tn = tf.reduce_sum(tf.square(self.score_function(
            self._neg_h_con_batch_real, self._neg_h_con_batch_imag,
            self._neg_tn_con_batch_real, self._neg_tn_con_batch_imag, 
            self._neg_rel_tn_batch)), 1)
            
        self.main_loss = tf.reduce_mean(
            tf.add(tf.add(f_score_tn, f_score_hn), f_score_h))
    

class RotatE_m2_2_TF(RotatE_m2_1_TF):
    def score_function(self, h_real, h_imag, t_real, t_imag, r, numpy = False):
        pi = 3.14159265358979323846
        magic = (self._gamma + self._epsilon)/self._dim/pi
        
        if numpy:
            raise NotImplementedError
        else: #tf
            
            r_phrase = r/magic
            r_real = tf.cos(r)
            r_imag = tf.sin(r)

            score_real = h_real * r_real - h_imag * r_imag
            score_imag = h_real * r_imag + h_imag * r_real
            score_real = score_real - t_real
            score_imag = score_imag - t_imag

            return tf.sigmoid(self.w*(2 - tf.reduce_mean(tf.square(score_real) + tf.square(score_imag), -1)) + self.b)
    





# complex (ht -r)
class RotatE_m3_TF(ComplEx_m5_1_TF):

    def __init__(self, num_rels, num_cons, dim, batch_size, neg_per_positive, p_neg, psl, gamma=1):
        self._gamma = gamma # 24
        ComplEx_m5_1_TF.__init__(self, num_rels, num_cons, dim, batch_size, neg_per_positive, p_neg, psl)

    def score_function(self, h_real, h_imag, t_real, t_imag, r_real, r_imag, numpy = False, numpy_w = None, numpy_b = None):
        if numpy:

            score_real = h_real * r_real - h_imag * r_imag
            score_imag = h_real * r_imag + h_imag * r_real
            score_real = score_real - t_real
            score_imag = score_imag - t_imag


            return sigmoid(numpy_w*(self._gamma - np.sum(np.square(score_real) + np.square(score_imag), axis=-1)) + numpy_b)
        else: #tf
            
            score_real = h_real * r_real - h_imag * r_imag
            score_imag = h_real * r_imag + h_imag * r_real
            score_real = score_real - t_real
            score_imag = score_imag - t_imag

            return tf.sigmoid(self.w*(self._gamma - tf.reduce_sum(tf.square(score_real) + tf.square(score_imag), -1)) + self.b)


# Simplified Uncertain RotatE
# complex (ht -r)
# without semi-supervised
class RotatE_m3_1_TF(RotatE_m3_TF):

    def score_function(self, h_real, h_imag, t_real, t_imag, r_real, r_imag, numpy = False, numpy_w = None, numpy_b = None):
        if numpy:
            raise NotImplementedError
        else: #tf
            
            score_real = h_real * r_real - h_imag * r_imag
            score_imag = h_real * r_imag + h_imag * r_real
            score_real = score_real - t_real
            score_imag = score_imag - t_imag

            return tf.sigmoid(self.w*(2 - tf.reduce_mean(tf.square(score_real) + tf.square(score_imag), -1)) + self.b)

# Simplified Uncertain RotatE
# complex (ht -r)
# + semisupervised_v2_2
class RotatE_m3_2_TF(RotatE_m3_TF):
# override
    def build_basics(self):
        RotatE_m3_TF.build_basics(self)

        self._semi_h_batch_real, self._semi_h_batch_imag = self.split_embed_real_imag(self._semi_h_batch)
        self._semi_t_batch_real, self._semi_t_batch_imag = self.split_embed_real_imag(self._semi_t_batch)
        self._semi_r_batch_real, self._semi_r_batch_imag = self.split_embed_real_imag(self._semi_r_batch)



    def define_main_loss(self):
        # distmult on uncertain graph
        print('define main loss')

        self.w = tf.Variable(0.0, name="weights")
        self.b = tf.Variable(0.0, name="bias")
        

        self._f_score_h = f_score_h = self.score_function(
            self._h_batch_real, self._h_batch_imag,
            self._t_batch_real, self._t_batch_imag, 
            self._r_batch_real, self._r_batch_imag)

        f_score_h = tf.square(f_score_h - self._A_w)

        self._f_score_semi = f_score_semi = self.score_function(
            self._semi_h_batch_real, self._semi_h_batch_imag,
            self._semi_t_batch_real, self._semi_t_batch_imag, 
            self._semi_r_batch_real, self._semi_r_batch_imag)

        f_score_semi = tf.square(f_score_semi - self._A_semi_w)


        # evaluation only
        self._f_score_hn = f_score_hn = self.score_function(
            self._neg_hn_con_batch_real, self._neg_hn_con_batch_imag,
            self._neg_t_con_batch_real, self._neg_t_con_batch_imag, 
            self._neg_rel_hn_batch_real, self._neg_rel_hn_batch_imag)

            
        self.main_loss = tf.add(tf.reduce_sum(f_score_semi), tf.reduce_sum(f_score_h))

# Simplified Uncertain RotatE
# complex (ht -r) 
# + semisupervised_v2 
# positive:generated=1:1 weight

class RotatE_m3_3_TF(RotatE_m3_2_TF):

    def define_main_loss(self):
        # distmult on uncertain graph
        print('define main loss')

        self.w = tf.Variable(0.0, name="weights")
        self.b = tf.Variable(0.0, name="bias")
        

        self._f_score_h = f_score_h = self.score_function(
            self._h_batch_real, self._h_batch_imag,
            self._t_batch_real, self._t_batch_imag, 
            self._r_batch_real, self._r_batch_imag)

        f_score_h = tf.square(f_score_h - self._A_w)

        self._f_score_semi = f_score_semi = self.score_function(
            self._semi_h_batch_real, self._semi_h_batch_imag,
            self._semi_t_batch_real, self._semi_t_batch_imag, 
            self._semi_r_batch_real, self._semi_r_batch_imag)

        f_score_semi = tf.square(f_score_semi - self._A_semi_w)


        # evaluation only
        self._f_score_hn = f_score_hn = self.score_function(
            self._neg_hn_con_batch_real, self._neg_hn_con_batch_imag,
            self._neg_t_con_batch_real, self._neg_t_con_batch_imag, 
            self._neg_rel_hn_batch_real, self._neg_rel_hn_batch_imag)

            
        # self.main_loss = tf.add(tf.reduce_sum(f_score_semi)/self._neg_per_positive, tf.reduce_sum(f_score_h))
        self.main_loss = tf.add(tf.reduce_sum(f_score_semi)/self._neg_per_positive / 2, tf.reduce_sum(f_score_h)) / self.batch_size
        # 20200414 / batch_size in loss function





# complex (hr -t)
class RotatE_m4_TF(ComplEx_m5_1_TF):

    def __init__(self, num_rels, num_cons, dim, batch_size, neg_per_positive, p_neg, psl, gamma):
        self._gamma = gamma # 24
        ComplEx_m5_1_TF.__init__(self, num_rels, num_cons, dim, batch_size, neg_per_positive, p_neg, psl)

    def score_function(self, h_real, h_imag, t_real, t_imag, r_real, r_imag, numpy = False, numpy_w = None, numpy_b = None):
        if numpy:

            score_real = h_real * t_real - h_imag * t_imag
            score_imag = h_real * t_imag + h_imag * t_real
            score_real = score_real - r_real
            score_imag = score_imag - r_imag


            return sigmoid(numpy_w*(self._gamma - np.sum(np.square(score_real) + np.square(score_imag), axis=-1)) + numpy_b)
        else: #tf
            
            score_real = h_real * t_real - h_imag * t_imag
            score_imag = h_real * t_imag + h_imag * t_real
            score_real = score_real - r_real
            score_imag = score_imag - r_imag

            return tf.sigmoid(self.w*(self._gamma - tf.reduce_sum(tf.square(score_real) + tf.square(score_imag), -1)) + self.b)


# Uncertain RotatE
# "real" rotatE + MSE loss
# semisupervised_v2 positive:generated=1:1 weight
# self.r_batch_real and self.r_batch_imag are not used
# 20200414 / batch_size in loss function
class RotatE_m5_TF(RotatE_m1_TF):
    # override
    def build_basics(self):
        RotatE_m1_TF.build_basics(self)

        self._semi_h_batch_real, self._semi_h_batch_imag = self.split_embed_real_imag(self._semi_h_batch)
        self._semi_t_batch_real, self._semi_t_batch_imag = self.split_embed_real_imag(self._semi_t_batch)

    def score_function(self, h_real, h_imag, t_real, t_imag, r, numpy = False):
        pi = 3.14159265358979323846
        magic = (self._gamma + self._epsilon)/self._dim/pi
        
        if numpy:
            raise NotImplementedError
        else: #tf
            
            r_phrase = r/magic
            r_real = tf.cos(r_phrase)
            r_imag = tf.sin(r_phrase)

            score_real = h_real * r_real - h_imag * r_imag
            score_imag = h_real * r_imag + h_imag * r_real
            score_real = score_real - t_real
            score_imag = score_imag - t_imag

            return tf.sigmoid(self.w*(2 - tf.reduce_mean(tf.square(score_real) + tf.square(score_imag), -1)) + self.b)
    
    def define_main_loss(self):
        # distmult on uncertain graph
        print('define main loss')

        self.w = tf.Variable(0.0, name="weights")
        self.b = tf.Variable(0.0, name="bias")
        

        self._f_score_h = f_score_h = self.score_function(
            self._h_batch_real, self._h_batch_imag,
            self._t_batch_real, self._t_batch_imag, 
            self._r_batch)

        f_score_h = tf.square(f_score_h - self._A_w)

        self._f_score_semi = f_score_semi = self.score_function(
            self._semi_h_batch_real, self._semi_h_batch_imag,
            self._semi_t_batch_real, self._semi_t_batch_imag, 
            self._semi_r_batch)

        f_score_semi = tf.square(f_score_semi - self._A_semi_w)


        # evaluation only
        self._f_score_hn = f_score_hn = self.score_function(
            self._neg_hn_con_batch_real, self._neg_hn_con_batch_imag,
            self._neg_t_con_batch_real, self._neg_t_con_batch_imag, 
            self._neg_rel_hn_batch)

            
        # self.main_loss = tf.add(tf.reduce_sum(f_score_semi)/self._neg_per_positive, tf.reduce_sum(f_score_h))
        self.main_loss = tf.add(tf.reduce_sum(f_score_semi) / self._neg_per_positive / 2, tf.reduce_sum(f_score_h)) / self.batch_size

# Uncertain RotatE
# rotatE m5 without semi-supervised
class RotatE_m5_1_TF(RotatE_m5_TF):
    
    def define_main_loss(self):
        # distmult on uncertain graph
        print('define main loss')

        self.w = tf.Variable(0.0, name="weights")
        self.b = tf.Variable(0.0, name="bias")
        

        self._f_score_h = f_score_h = self.score_function(
            self._h_batch_real, self._h_batch_imag,
            self._t_batch_real, self._t_batch_imag, 
            self._r_batch)

        f_score_h = tf.square(f_score_h - self._A_w)

        self._f_score_hn = f_score_hn = self.score_function(
            self._neg_hn_con_batch_real, self._neg_hn_con_batch_imag,
            self._neg_t_con_batch_real, self._neg_t_con_batch_imag, 
            self._neg_rel_hn_batch)

        f_score_hn = tf.reduce_mean(tf.square(f_score_hn), 1)

        self._f_score_tn = f_score_tn = tf.reduce_mean(tf.square(self.score_function(
            self._neg_h_con_batch_real, self._neg_h_con_batch_imag,
            self._neg_tn_con_batch_real, self._neg_tn_con_batch_imag, 
            self._neg_rel_tn_batch)), 1)
            
        self.main_loss = (tf.reduce_sum(
            tf.add(tf.divide(tf.add(f_score_tn, f_score_hn), 2) * self._p_neg, f_score_h))) / self._batch_size







# + semisupervised_v2_2
class UKGE_logi_m1_TF(TFParts):
    def __init__(self, num_rels, num_cons, dim, batch_size, neg_per_positive, p_neg, psl = False):
        TFParts.__init__(self, num_rels, num_cons, dim, batch_size, neg_per_positive, p_neg, psl)
        self.build()


    def score_function(self, h, t, r, numpy=False):
        htr = tf.reduce_sum(
            tf.multiply(r, tf.multiply(h, t, "element_wise_multiply"),
                        "r_product"),
            1)

        return tf.sigmoid(self.w * htr + self.b)

    def define_main_loss(self):
        # distmult on uncertain graph
        print('define main loss')

        self.w = tf.Variable(0.0, name="weights")
        self.b = tf.Variable(0.0, name="bias")

        
        self._f_score_h = f_score_h = self.score_function(
            self._h_batch,
            self._t_batch, 
            self._r_batch)
        f_score_h = tf.square(tf.subtract(f_score_h, self._A_w))

        self._f_score_semi = f_score_semi = self.score_function(
            self._semi_h_batch,
            self._semi_t_batch, 
            self._semi_r_batch)

        f_score_semi = tf.square(f_score_semi - self._A_semi_w)


        # evaluation only
        self._f_score_hn = _f_score_hn = tf.sigmoid(self.w * (
            tf.reduce_sum(
                tf.multiply(self._neg_rel_hn_batch, tf.multiply(self._neg_hn_con_batch, self._neg_t_con_batch)), 2)
        ) + self.b)


        self.main_loss = tf.add(tf.reduce_sum(f_score_semi), tf.reduce_sum(f_score_h))

    def define_psl_loss(self):
        self.psl_prob = tf.sigmoid(self.w*tf.reduce_sum(
            tf.multiply(self._soft_r_batch,
                        tf.multiply(self._soft_h_batch, self._soft_t_batch)),
            1)+self.b)
        self.compute_psl_loss()  # in tf_parts


# UKGE logi with semi-supervised (uncertain DistMult)
# + semisupervised_v2
class UKGE_logi_m2_TF(UKGE_logi_m1_TF):


    def define_main_loss(self):
        # distmult on uncertain graph
        print('define main loss')

        self.w = tf.Variable(0.0, name="weights")
        self.b = tf.Variable(0.0, name="bias")

        
        self._f_score_h = f_score_h = self.score_function(
            self._h_batch,
            self._t_batch, 
            self._r_batch)
        f_score_h = tf.square(tf.subtract(f_score_h, self._A_w))

        self._f_score_semi = f_score_semi = self.score_function(
            self._semi_h_batch,
            self._semi_t_batch, 
            self._semi_r_batch)

        f_score_semi = tf.square(f_score_semi - self._A_semi_w)


        # evaluation only
        self._f_score_hn = _f_score_hn = tf.sigmoid(self.w * (
            tf.reduce_sum(
                tf.multiply(self._neg_rel_hn_batch, tf.multiply(self._neg_hn_con_batch, self._neg_t_con_batch)), 2)
        ) + self.b)


        # self.main_loss = tf.add(tf.reduce_sum(f_score_semi)/self._neg_per_positive, tf.reduce_sum(f_score_h))
        self.main_loss = tf.add(tf.reduce_sum(f_score_semi)/ 2 /self._neg_per_positive, tf.reduce_sum(f_score_h)) / self.batch_size
        # 20200414 / batch_size in loss function
