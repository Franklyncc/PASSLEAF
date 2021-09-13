''' Module for held-out test.'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
from numpy import linalg as LA
import heapq as HP
import pandas as pd
import os

# from scipy.special import expit as sigmoid
from utils import sigmoid

import sys

if '../' not in sys.path:  # src folder
    sys.path.append('../')

from os.path import join
import data
import time
import pickle
import random

import sklearn
from sklearn import tree

import scipy

from src.models import * #UKGE_logi_TF, UKGE_rect_TF, TransE_m1_TF, DistMult_m1_TF


# This class is used to load and combine a TF_Parts and a Data object, and provides some useful methods for training
class Tester(object):
    class IndexScore:
        """
        The score of a tail when h and r is given.
        It's used in the ranking task to facilitate comparison and sorting.
        Print w as 3 digit precision float.
        """

        def __init__(self, index, score):
            self.index = index
            self.score = score

        def __lt__(self, other):
            return self.score < other.score

        def __repr__(self):
            # return "(index: %d, w:%.3f)" % (self.index, self.score)
            return "(%d, %.3f)" % (self.index, self.score)

        def __str__(self):
            return "(index: %d, w:%.3f)" % (self.index, self.score)

    def __init__(self):
        self.tf_parts = None
        self.this_data = None
        self.vec_c = np.array([0])
        self.vec_r = np.array([0])
        # below for test data
        self.test_triples = np.array([0])
        self.test_triples_group = {}

    # completed by child class
    def build_by_file(self, test_data_file, model_dir, model_filename='xcn-distmult.ckpt', data_filename='xc-data.bin'):
        # load the saved Data()
        self.this_data = data.Data()
        data_save_path = join(model_dir, data_filename)
        self.this_data.load(data_save_path)

        # load testing data
        self.load_test_data(test_data_file)

        self.model_dir = model_dir  # used for saving

    # completed by child class
    def build_by_file_no_id_mapping(self, test_data_file, model_dir, model_filename='xcn-distmult.ckpt', data_filename='xc-data.bin'):
        # load the saved Data()
        self.this_data = data.DataNoIdMapping()
        data_save_path = join(model_dir, data_filename)
        self.this_data.load(data_save_path)

        # load testing data
        self.load_test_data(test_data_file)

        self.model_dir = model_dir  # used for saving

    # abstract method
    def build_by_var(self, test_data, tf_model, this_data, sess):
        raise NotImplementedError("Fatal Error: This model' tester didn't implement its build_by_var() function!")

    def build_by_pickle(self, test_data_file, model_dir, data_filename, pickle_file, loadComplEx=False):
        """
        :param pickle_file: pickle embedding
        :return:
        """
        # load the saved Data()
        self.this_data = data.Data()
        data_save_path = join(model_dir, data_filename)
        self.this_data.load(data_save_path)

        # load testing data
        self.load_test_data(test_data_file)

        self.model_dir = model_dir  # used for saving

        with open(pickle_file, 'rb') as f:
            ht, r = pickle.load(f)  # unpickle
        self.vec_c = ht
        self.vec_r = r

    def load_hr_map(self, data_dir, hr_base_file, supplement_t_files, splitter='\t', line_end='\n'):
        """
        Initialize self.hr_map.
        Load self.hr_map={h:{r:t:w}}}, not restricted to test data
        :param hr_base_file: Get self.hr_map={h:r:{t:w}}} from the file.
        :param supplement_t_files: Add t(only t) to self.hr_map. Don't add h or r.
        :return:
        """
        self.hr_map = {}
        with open(join(data_dir, hr_base_file)) as f:
            for line in f:
                line = line.rstrip(line_end).split(splitter)
                h = self.this_data.con_str2index(line[0])
                r = self.this_data.rel_str2index(line[1])
                t = self.this_data.con_str2index(line[2])
                w = float(line[3])
                # construct hr_map
                if self.hr_map.get(h) == None:
                    self.hr_map[h] = {}
                if self.hr_map[h].get(r) == None:
                    self.hr_map[h][r] = {t: w}
                else:
                    self.hr_map[h][r][t] = w

        count = 0
        for h in self.hr_map:
            count += len(self.hr_map[h])
        print('Loaded ranking test queries. Number of (h,r,?t) queries: %d' % count)

        for file in supplement_t_files:
            with open(join(data_dir, file)) as f:
                for line in f:
                    line = line.rstrip(line_end).split(splitter)
                    h = self.this_data.con_str2index(line[0])
                    r = self.this_data.rel_str2index(line[1])
                    t = self.this_data.con_str2index(line[2])
                    w = float(line[3])

                    # update hr_map
                    if h in self.hr_map and r in self.hr_map[h]:
                        self.hr_map[h][r][t] = w

    def save_hr_map(self, outputfile):
        """
        Print to file for debugging. (not applicable for reloading)
        Prerequisite: self.hr_map has been loaded.
        :param outputfile:
        :return:
        """
        if self.hr_map is None:
            raise ValueError("Tester.hr_map hasn't been loaded! Use Tester.load_hr_map() to load it.")

        with open(outputfile, 'w') as f:
            for h in self.hr_map:
                for r in self.hr_map[h]:
                    tw_truth = self.hr_map[h][r]  # {t:w}
                    tw_list = [self.IndexScore(t, w) for t, w in tw_truth.items()]
                    tw_list.sort(reverse=True)  # descending on w
                    f.write('h: %d, r: %d\n' % (h, r))
                    f.write(str(tw_list) + '\n')

    def load_test_data(self, filename, splitter='\t', line_end='\n'):
        num_lines = 0
        triples = []
        for line in open(filename):
            line = line.rstrip(line_end).split(splitter)
            if len(line) < 4:
                continue
            num_lines += 1
            h = self.this_data.con_str2index(line[0])
            r = self.this_data.rel_str2index(line[1])
            t = self.this_data.con_str2index(line[2])
            w = float(line[3])
            if h is None or r is None or t is None or w is None:
                continue
            triples.append([h, r, t, w])

            # add to group
            if self.test_triples_group.get(r) == None:
                self.test_triples_group[r] = [(h, r, t, w)]
            else:
                self.test_triples_group[r].append((h, r, t, w))

        # Note: test_triples will be a np.float64 array! (because of the type of w)
        # Take care of the type of hrt when unpacking.
        self.test_triples = np.array(triples)

        print("Loaded test data from %s, %d out of %d." % (filename, len(triples), num_lines))
        # print("Rel each cat:", self.rel_num_cases)

    def load_triples_into_df(self, filename):
        num_lines = 0
        triples = []
        for line in open(filename):
            line = line.rstrip('\n').split('\t')
            if len(line) < 4:
                continue
            num_lines += 1
            h = self.this_data.con_str2index(line[0])
            r = self.this_data.rel_str2index(line[1])
            t = self.this_data.con_str2index(line[2])
            w = float(line[3])
            if h is None or r is None or t is None or w is None:
                continue
            triples.append([h, r, t, w])
        return pd.DataFrame(triples, columns=['v1','relation','v2','w'])

    def con_index2vec(self, c):
        return self.vec_c[c]

    def rel_index2vec(self, r):
        return self.vec_r[r]

    def con_str2vec(self, str):
        this_index = self.this_data.con_str2index(str)
        if this_index == None:
            return None
        return self.vec_c[this_index]

    def rel_str2vec(self, str):
        this_index = self.this_data.rel_str2index(str)
        if this_index == None:
            return None
        return self.vec_r[this_index]

    class index_dist:
        def __init__(self, index, dist):
            self.dist = dist
            self.index = index
            return

        def __lt__(self, other):
            return self.dist > other.dist

    def con_index2str(self, str):
        return self.this_data.con_index2str(str)

    def rel_index2str(self, str):
        return self.this_data.rel_index2str(str)

    # input must contain a pool of pre- or post-projected vecs. return a list of indices and dist
    def kNN(self, vec, vec_pool, topk=10, self_id=None):
        q = []
        for i in range(len(vec_pool)):
            # skip self
            if i == self_id:
                continue
            dist = np.dot(vec, vec_pool[i])
            if len(q) < topk:
                HP.heappush(q, self.index_dist(i, dist))
            else:
                # indeed it fetches the biggest
                # as the index_dist "lt" is defined as larger dist
                tmp = HP.nsmallest(1, q)[0]
                if tmp.dist < dist:
                    HP.heapreplace(q, self.index_dist(i, dist))
        rst = []
        while len(q) > 0:
            item = HP.heappop(q)
            rst.insert(0, (item.index, item.dist))
        return rst

    # input must contain a pool of pre- or post-projected vecs. return a list of indices and dist. rank an index in a vec_pool from 
    def rank_index_from(self, vec, vec_pool, index, self_id=None):
        dist = np.dot(vec, vec_pool[index])
        rank = 1
        for i in range(len(vec_pool)):
            if i == index or i == self_id:
                continue
            if dist < np.dot(vec, vec_pool[i]):
                rank += 1
        return rank

    def rel_cat_id(self, r):
        if r in self.relnn:
            return 3
        elif r in self.rel1n:
            return 1
        elif r in self.reln1:
            return 2
        else:
            return 0

    def dissimilarity(self, h, r, t):
        h_vec = self.vec_c[h]
        t_vec = self.vec_c[t]
        r_vec = self.vec_r[r]
        return np.dot(r_vec, np.multiply(h_vec, t_vec))

    def dissimilarity2(self, h, r, t):
        # h_vec = self.vec_c[h]
        # t_vec = self.vec_c[t]
        r_vec = self.vec_r[r]
        return np.dot(r_vec, np.multiply(h, t))

    def get_info(self, triple):
        """
        convert the float h, r, t to int, and return
        :param triple: triple: np.array[4], dtype=np.float64: h,r,t,w
        :return: h, r, t(index), w(float)
        """
        h_, r_, t_, w = triple  # h_, r_, t_: float64
        return int(h_), int(r_), int(t_), w

    def vecs_from_triples(self, h, r, t):
        """
        :param h,r,t: int index
        :return: h_vec, r_vec, t_vec
        """
        h, r, t = int(h), int(r), int(t)  # just in case of float
        hvec = self.con_index2vec(h)
        rvec = self.rel_index2vec(r)
        tvec = self.con_index2vec(t)
        return hvec, rvec, tvec

    # Abstract method. Different scoring function for different models.
    def get_score(self, h, r, t):
        raise NotImplementedError("get_score() is not defined in this model's tester")

    # Abstract method. Different scoring function for different models.
    def get_score_batch(self, h_batch, r_batch, t_batch, isneg2Dbatch=False):
        raise NotImplementedError("get_score_batch() is not defined in this model's tester")

    def get_bound_score(self, h, r, t):
        # for most models, just return the original score
        # may be overwritten by child class
        return self.get_score(h, r, t)

    def get_bound_score_batch(self, h_batch, r_batch, t_batch, isneg2Dbatch=False):
        # for non-rect models, just return the original score
        # rect models will override it
        return self.get_score_batch(h_batch, r_batch, t_batch, isneg2Dbatch)


    def get_mse(self, toprint=True, save_dir='', epoch=0):
        test_triples = self.test_triples
        N = test_triples.shape[0]

        # existing triples
        # (score - w)^2
        h_batch = test_triples[:, 0].astype(int)
        r_batch = test_triples[:, 1].astype(int)
        t_batch = test_triples[:, 2].astype(int)
        w_batch = test_triples[:, 3]
        scores = self.get_score_batch(h_batch, r_batch, t_batch)
        mse = np.sum(np.square(scores - w_batch))
        mse = mse / N

        return mse

    def get_mae(self, verbose=False, save_dir='', epoch=0):
        test_triples = self.test_triples
        N = test_triples.shape[0]

        # existing triples
        # (score - w)^2
        h_batch = test_triples[:, 0].astype(int)
        r_batch = test_triples[:, 1].astype(int)
        t_batch = test_triples[:, 2].astype(int)
        w_batch = test_triples[:, 3]
        scores = self.get_score_batch(h_batch, r_batch, t_batch)
        mae = np.sum(np.absolute(scores - w_batch))

        mae = mae / N

        return mae

    def get_mse_neg(self, neg_per_positive):
        test_triples = self.test_triples
        N = test_triples.shape[0]

        # negative samples
        # (score - 0)^2
        all_neg_hn_batch = self.this_data.corrupt_batch(test_triples, neg_per_positive, "h")
        all_neg_tn_batch = self.this_data.corrupt_batch(test_triples, neg_per_positive, "t")
        neg_hn_batch, neg_rel_hn_batch, \
        neg_t_batch, neg_h_batch, \
        neg_rel_tn_batch, neg_tn_batch \
            = all_neg_hn_batch[:, :, 0].astype(int), \
              all_neg_hn_batch[:, :, 1].astype(int), \
              all_neg_hn_batch[:, :, 2].astype(int), \
              all_neg_tn_batch[:, :, 0].astype(int), \
              all_neg_tn_batch[:, :, 1].astype(int), \
              all_neg_tn_batch[:, :, 2].astype(int)
        scores_hn = self.get_score_batch(neg_hn_batch, neg_rel_hn_batch, neg_t_batch, isneg2Dbatch=True)
        scores_tn = self.get_score_batch(neg_h_batch, neg_rel_tn_batch, neg_tn_batch, isneg2Dbatch=True)
        mse_hn = np.sum(np.mean(np.square(scores_hn - 0), axis=1)) / N
        mse_tn = np.sum(np.mean(np.square(scores_tn - 0), axis=1)) / N

        mse = (mse_hn + mse_tn) / 2
        return mse

    def get_mae_neg(self, neg_per_positive):
        test_triples = self.test_triples
        N = test_triples.shape[0]

        # negative samples
        # (score - 0)^2
        all_neg_hn_batch = self.this_data.corrupt_batch(test_triples, neg_per_positive, "h")
        all_neg_tn_batch = self.this_data.corrupt_batch(test_triples, neg_per_positive, "t")
        neg_hn_batch, neg_rel_hn_batch, \
        neg_t_batch, neg_h_batch, \
        neg_rel_tn_batch, neg_tn_batch \
            = all_neg_hn_batch[:, :, 0].astype(int), \
              all_neg_hn_batch[:, :, 1].astype(int), \
              all_neg_hn_batch[:, :, 2].astype(int), \
              all_neg_tn_batch[:, :, 0].astype(int), \
              all_neg_tn_batch[:, :, 1].astype(int), \
              all_neg_tn_batch[:, :, 2].astype(int)
        scores_hn = self.get_score_batch(neg_hn_batch, neg_rel_hn_batch, neg_t_batch, isneg2Dbatch=True)
        scores_tn = self.get_score_batch(neg_h_batch, neg_rel_tn_batch, neg_tn_batch, isneg2Dbatch=True)
        mae_hn = np.sum(np.mean(np.absolute(scores_hn - 0), axis=1)) / N
        mae_tn = np.sum(np.mean(np.absolute(scores_tn - 0), axis=1)) / N

        mae_neg = (mae_hn + mae_tn) / 2
        return mae_neg

    def con_index2vec_batch(self, indices):
        return np.squeeze(self.vec_c[[indices], :])

    def rel_index2vec_batch(self, indices):
        return np.squeeze(self.vec_r[[indices], :])

    def pred_top_k_tail(self, k, h, r):
        """
        Predict top k tail.
        The #returned items <= k.
        Consider add tail_pool to limit the range of searching.
        :param k: how many results to return
        :param h: index of head
        :param r: index of relation
        :return:
        """
        q = []  # min heap
        N = self.vec_c.shape[0]  # the total number of concepts
        
        scores_cache = self.get_score_batch(
            np.repeat(h, N), 
            np.repeat(r, N), 
            np.arange(0, N)
        )
        for t_idx in range(N):
            score = scores_cache[t_idx]
            
            if len(q) < k:
                HP.heappush(q, self.IndexScore(t_idx, score))
            else:
                tmp = q[0]  # smallest score
                if tmp.score < score:
                    HP.heapreplace(q, self.IndexScore(t_idx, score))

        indices = np.zeros(len(q), dtype=int)
        scores = np.ones(len(q), dtype=float)
        i = len(q) - 1  # largest score first
        while len(q) > 0:
            item = HP.heappop(q)
            indices[i] = item.index
            scores[i] = item.score
            i -= 1

        return indices, scores

    def get_t_ranks(self, h, r, ts, accurate_mode=False):
        """
        :param accurate_mode: set this flag to disable "Fast Ranking"
        Given some t index, return the ranks for each t
        :return:
        """
        # prediction
        assert len(ts) == len(np.unique(ts))
        # scores = np.array([self.get_score(h, r, t) for t in ts])  # predict scores for t from ground truth
        # scores = self.get_score_batch(
        #             np.repeat(h, len(ts)), 
        #             np.repeat(r, len(ts)), 
        #             ts
        #         )

        # ranks = np.ones(len(ts), dtype=int)  # initialize rank as all 1

        N = self.vec_c.shape[0]  # pool of t: all concept vectors

        scores_cache = self.get_score_batch(
            np.repeat(h, N), 
            np.repeat(r, N), 
            np.arange(0, N)
        )
        scores = scores_cache[ts]
        

        # import time
        # tStart = time.time()
        # for i in range(N):  # compute scores for all concept vectors as t
        #     # score_i = self.get_score(h, r, i)
        #     score_i = scores_cache[i]
        #     rankplus = (scores < score_i).astype(int)  # rank+1 if score<score_i
        #     ranks += rankplus

        #     # if score_i == 1:
        #     #     print('[WARN] score_i is 1!')
        #     #     print(ranks)

        # tEnd = time.time()
        # print("1. It cost %f sec" % (tEnd - tStart))

        # tStart = time.time()

        if accurate_mode:
            ranks = scipy.stats.rankdata(scores_cache, method='ordinal')[ts]
            ranks = N - ranks + 1
        else: # fast ranking (erroneous when there are ties in the scores)
            ranks = np.sum(scores < np.tile(scores_cache, (len(ts),1)).transpose(), axis=0) + 1

        # tEnd = time.time()
        # print("2. It cost %f sec" % (tEnd - tStart))
        

        return ranks

    def ndcg(self, h, r, tw_truth, verbose, accurate_mode=False):
        """
        Compute nDCG(normalized discounted cummulative gain)
        sum(score_ground_truth / log2(rank+1)) / max_possible_dcg
        :param tw_truth: [IndexScore1, IndexScore2, ...], soreted by IndexScore.score descending
        :param accurate_mode: set this flag to disable "Fast Ranking"
        :return:
        """
        # prediction
        ts = [tw.index for tw in tw_truth]
        ranks = self.get_t_ranks(h, r, ts, accurate_mode=accurate_mode)

        # linear gain
        gains = np.array([tw.score for tw in tw_truth])
        # print('gains')
        # print(gains)
        discounts = np.log2(ranks + 1)
        discounted_gains = gains / discounts

        dcg = np.sum(discounted_gains)  # discounted cumulative gain
        # normalize
        max_possible_dcg = np.sum(gains / np.log2(np.arange(len(gains)) + 2))  # when ranks = [1, 2, ...len(truth)]
        ndcg = dcg / max_possible_dcg  # normalized discounted cumulative gain

        # exponential gain
        exp_gains = np.array([2 ** tw.score - 1 for tw in tw_truth])
        # print('exp_gains')
        # print(exp_gains)
        exp_discounted_gains = exp_gains / discounts
        # print('discounts')
        # print(discounts)
        exp_dcg = np.sum(exp_discounted_gains)
        # print('exp_dcg')
        # print(exp_dcg)
        # normalize
        exp_max_possible_dcg = np.sum(
            exp_gains / np.log2(np.arange(len(exp_gains)) + 2))  # when ranks = [1, 2, ...len(truth)]
        # print('exp_max_possible_dcg')
        # print(exp_max_possible_dcg)
        exp_ndcg = exp_dcg / exp_max_possible_dcg  # normalized discounted cumulative gain

        

        if verbose:
            if ndcg > 1.001:
                print('[Error] nDCG > 1. (', ndcg, ')')
            if exp_ndcg > 1.001:
                print('[Error] exp nDCG > 1. (', exp_ndcg, ')')

            print(ranks)
            print(gains)

        return ndcg, exp_ndcg

    def mean_ndcg(self, hr_map, verbose=False, accurate_mode=False):
        """
        :param hr_map: {h:{r:{t:w}}}
        :param accurate_mode: set this flag to disable "Fast Ranking"
        :return:
        """
        ndcg_sum = 0  # nDCG with linear gain
        exp_ndcg_sum = 0  # nDCG with exponential gain
        count = 0

        t0 = time.time()

        # debug ndcg
        res = []  # [(h,r,tw_truth, ndcg)]

        r_N = self.vec_r.shape[0]
        ndcg_sum_r = np.zeros(r_N)
        count_r = np.zeros(r_N)

        all_ndcg = []


        for h in hr_map:
            for r in hr_map[h]:
                tw_dict = hr_map[h][r]  # {t:w}
                tw_truth = [self.IndexScore(t, w) for t, w in tw_dict.items()]
                tw_truth.sort(reverse=True)  # descending on w
                ndcg, exp_ndcg = self.ndcg(h, r, tw_truth, verbose, accurate_mode=accurate_mode)  # nDCG with linear gain and exponential gain
                ndcg_sum += ndcg
                exp_ndcg_sum += exp_ndcg
                count += 1
                all_ndcg.append([h, r, len(hr_map[h][r].values()), max(hr_map[h][r].values()), ndcg, exp_ndcg])

                ndcg_sum_r[r] += ndcg
                count_r[r] += 1

                if verbose:
                    print(h, r, ndcg, exp_ndcg)
                    
                # if count % 100 == 0:
                #     print('Processed %d, time %s' % (count, (time.time() - t0)))
                #     print('mean ndcg (linear gain) now: %f' % (ndcg_sum / count))
                #     print('mean ndcg (exponential gain) now: %f' % (exp_ndcg_sum / count))

                # debug
                # ranks = self.get_t_ranks(h, r, [tw.index for tw in tw_truth])
                # res.append((h,r,tw_truth, ndcg, ranks))

        return ndcg_sum / count, exp_ndcg_sum / count, ndcg_sum_r / count_r, count_r, all_ndcg

    def mean_rank(self, hr_map, weighted = False, verbose=False, accurate_mode=False):
        """
        :param hr_map: {h:{r:{t:w}}}
        :param weighted: use linearly weighted metrics
        :param accurate_mode: set this flag to disable "Fast Ranking"
        :return:
        """
        rank_sum = 0  # nDCG with linear gain
        count = 0

        t0 = time.time()

        # debug ndcg
        res = []  # [(h,r,tw_truth, ndcg)]


        all_rank = []


        for h in hr_map:
            for r in hr_map[h]:
                tw_dict = hr_map[h][r]  # {t:w}
                tw_truth = [self.IndexScore(t, w) for t, w in tw_dict.items()]
                tw_truth.sort(reverse=True)  # descending on w
                ts = [tw.index for tw in tw_truth]
                ranks = self.get_t_ranks(h, r, ts, accurate_mode=accurate_mode)
                
                if weighted: 
                    weight = np.array([tw.score for tw in tw_truth])
                    rank_sum += sum(ranks*weight)
                    count += sum(weight)

                else:
                    rank_sum += sum(ranks)
                    count += ranks.shape[0]
                
                all_rank.append(ranks)


                if verbose:
                    print(h, r, ranks)
                    


        return rank_sum / count, all_rank

    def mean_hitAtK(self, hr_map, ks, weighted = None, verbose=False, accurate_mode=False):
        """
        :param hr_map: {h:{r:{t:w}}}
        :param weighted: use linearly weighted metrics
        :param accurate_mode: set this flag to disable "Fast Ranking"
        :return:
        """
        # hitAt10_sum = 0  # nDCG with linear gain
        # count = 0

        hitAtK = np.zeros((len(ks),))
        count = np.zeros((len(ks),))

        if weighted == None:
            weighted = [False for _ in range(len(ks))]
        else:
            assert len(weighted) == len(ks)


        all_rank = []
        t0 = time.time()

        debug_count = 0
        for h in hr_map:
            for r in hr_map[h]:
                tw_dict = hr_map[h][r]  # {t:w}
                tw_truth = [self.IndexScore(t, w) for t, w in tw_dict.items()]
                tw_truth.sort(reverse=True)  # descending on w
                ts = [tw.index for tw in tw_truth]
                ranks = self.get_t_ranks(h, r, ts, accurate_mode=accurate_mode)

                for k_idx, k in enumerate(ks):
                    if weighted[k_idx]: 
                        weight = np.array([tw.score for tw in tw_truth])
                        hitAtK[k_idx] += sum((ranks <= k)*weight)
                        count[k_idx] += sum(weight)
                    else:
                        hitAtK[k_idx] += sum(ranks <= k)
                        count[k_idx] += ranks.shape[0]

                    # all_rank.append(ranks)


                if verbose:
                    print(h, r, ranks)
                

        # print('count: ',count)
        return hitAtK / count, all_rank

    def mean_precision(self, hr_map, verbose=False, accurate_mode=False):
        """
        :param hr_map: {h:{r:{t:w}}}
        :param accurate_mode: set this flag to disable "Fast Ranking"
        :return:
        """
        precision_sum = 0  # nDCG with linear gain
        count = 0

        t0 = time.time()

        all_rank = []


        N = self.vec_c.shape[0]  # the total number of concepts
        for h in hr_map:
            for r in hr_map[h]:
                tw_dict = hr_map[h][r]  # {t:w}
                tw_truth = [self.IndexScore(t, w) for t, w in tw_dict.items()]
                tw_truth.sort(reverse=True)  # descending on w
                ts = [tw.index for tw in tw_truth]

                scores_cache = self.get_score_batch(
                    np.repeat(h, N), 
                    np.repeat(r, N), 
                    np.arange(0, N)
                )
                scores = scores_cache[ts]
                precision_sum += sum(scores >= 0.5)
                count += scores.shape[0]
                all_rank.append(scores)


                if verbose:
                    print(h, r)
                    
                    if count % 50 == 0:
                        print('Processed %d, time %s' % (count, (time.time() - t0)))
                        print('mean rank now: %f' % (precision_sum / count))
                        print('ts', ts)
                        print('scores: ', scores)

        print('count: ',count)
        return precision_sum / count, all_rank



    def classify_triples(self, confT, plausTs):
        """
        Classify high-confidence relation facts
        :param confT: the threshold of ground truth confidence score
        :param plausTs: the list of proposed thresholds of computed plausibility score
        :return:
        """
        test_triples = self.test_triples

        h_batch = test_triples[:, 0].astype(int)
        r_batch = test_triples[:, 1].astype(int)
        t_batch = test_triples[:, 2].astype(int)
        w_batch = test_triples[:, 3]

        # ground truth
        high_gt = set(np.squeeze(np.argwhere(w_batch > confT)))  # positive
        low_gt = set(np.squeeze(np.argwhere(w_batch <= confT)))  # negative

        P = []
        R = []
        Acc = []

        # prediction
        scores = self.get_score_batch(h_batch, r_batch, t_batch)
        # print(scores[:1000])
        print('The mean of prediced scores: %f' % np.mean(scores))
        # pred_thres = np.arange(0, 1, 0.05)
        for pthres in plausTs:

            high_pred = set(np.squeeze(np.argwhere(scores > pthres)))
            low_pred = set(np.squeeze(np.argwhere(scores <= pthres)))

            # precision-recall
            TP = high_gt & high_pred  # union intersection
            if len(high_pred) == 0:
                precision = 1
            else:
                precision = len(TP) / len(high_pred)


            recall = len(TP) / len(high_gt)
            P.append(precision)
            R.append(recall)

            # accuracy
            TPTN = (len(TP) + len(low_gt & low_pred))
            accuracy = TPTN / test_triples.shape[0]
            Acc.append(accuracy)

        P = np.array(P)
        R = np.array(R)
        F1 = 2 * np.multiply(P, R) / (P + R)
        Acc = np.array(Acc)

        return scores, P, R, F1, Acc


    def decision_tree_classify(self, confT, train_data):
        """
        :param confT: :param confT: the threshold of ground truth confidence score
        :param train_data: dataframe['v1','relation','v2','w']
        :return:
        """
        # train_data = pd.read_csv(os.path.join(data_dir,'train.tsv'), sep='\t', header=None, names=['v1','relation','v2','w'])

        test_triples = self.test_triples

        # train
        train_h, train_r, train_t = train_data['v1'].values.astype(int), train_data['relation'].values.astype(int), train_data['v2'].values.astype(int)
        train_X = self.get_score_batch(train_h, train_r, train_t)[:, np.newaxis]  # feature(2D, n*1)
        train_Y = train_data['w']>confT  # label (high confidence/not)
        clf = tree.DecisionTreeClassifier()
        clf.fit(train_X, train_Y)

        # predict
        test_triples = self.test_triples
        test_h, test_r, test_t = test_triples[:, 0].astype(int), test_triples[:, 1].astype(int), test_triples[:, 2].astype(int)
        test_X = self.get_score_batch(test_h, test_r, test_t)[:, np.newaxis]
        test_Y_truth = test_triples[:, 3]>confT
        test_Y_pred = clf.predict(test_X)
        print('Number of true positive: %d' % np.sum(test_Y_truth))
        print('Number of predicted positive: %d'%np.sum(test_Y_pred))


        precision, recall, F1, _ = sklearn.metrics.precision_recall_fscore_support(test_Y_truth, test_Y_pred)
        accu = sklearn.metrics.accuracy_score(test_Y_truth, test_Y_pred)

        # P-R curve
        P, R, thres = sklearn.metrics.precision_recall_curve(test_Y_truth, test_X)

        return test_X, precision, recall, F1, accu, P, R

    def get_fixed_hr(self, outputdir=None, n=500):
        hr_map500 = {}
        dict_keys = []
        for h in self.hr_map.keys():
            for r in self.hr_map[h].keys():
                dict_keys.append([h, r])

        dict_keys = sorted(dict_keys, key=lambda x: len(self.hr_map[x[0]][x[1]]), reverse=True)
        dict_final_keys = []

        for i in range(2525):
            dict_final_keys.append(dict_keys[i])

        count = 0
        for i in range(n):
            temp_key = random.choice(dict_final_keys)
            h = temp_key[0]
            r = temp_key[1]
            for t in self.hr_map[h][r]:
                w = self.hr_map[h][r][t]
                if hr_map500.get(h) == None:
                    hr_map500[h] = {}
                if hr_map500[h].get(r) == None:
                    hr_map500[h][r] = {t: w}
                else:
                    hr_map500[h][r][t] = w

        for h in hr_map500.keys():
            for r in hr_map500[h].keys():
                count = count + 1

        self.hr_map_sub = hr_map500

        if outputdir is not None:
            with open(outputdir, 'wb') as handle:
                pickle.dump(hr_map500, handle, protocol=pickle.HIGHEST_PROTOCOL)

        return hr_map500



    def find_false_negatives(self, hr_map):
        """
        :param hr_map: {h:{r:{t:w}}}
        :return:
        """

        results = []

        N = self.vec_c.shape[0]  # the total number of concepts
        for h in hr_map:
            for r in hr_map[h]:
                tw_dict = hr_map[h][r]  # {t:w}
                existing_items = [t for t, w in tw_dict.items()]
                # existing_items_mask = np.ones((N,), dtype=bool)
                # existing_items_mask[existing_items] = False
                scores_cache = self.get_score_batch(
                    np.repeat(h, N), 
                    np.repeat(r, N), 
                    np.arange(0, N)
                )
                scores_cache[existing_items] = -1 # remove existing item
                candidate_false_negatives = np.argwhere(scores_cache > 0.7).flatten()

                # print('h: %s, r: %s:'%(self.con_index2str(h), self.rel_index2str(r)))
                for candidate in candidate_false_negatives:
                    results.append([self.con_index2str(h), self.rel_index2str(r), self.con_index2str(candidate), scores_cache[candidate]])
                    # print('%s: %.4f'%(self.con_index2str(candidate), scores_cache[candidate]))

        return results
                




class UKGE_logi_Tester(Tester):
    def __init__(self, ):
        Tester.__init__(self)

    # override
    def build_by_file(self, test_data_file, model_dir, model_filename='model.bin', data_filename='data.bin'):
        """
        load data and model from files
        get self.vec_c (vectors for concepts), and self.vec_r(vectors for relations)
        :return:
        """
        Tester.build_by_file(self, test_data_file, model_dir, model_filename,
                             data_filename)

        # load tf model and embeddings
        model_save_path = join(model_dir, model_filename)
        self.tf_parts = UKGE_logi_TF(num_rels=self.this_data.num_rels(),
                                     num_cons=self.this_data.num_cons(),
                                     dim=self.this_data.dim,
                                     batch_size=self.this_data.batch_size, neg_per_positive=10, p_neg=1)
        # neg_per_positive, reg_scale and p_neg are not used in testing
        self.sess = sess = tf.Session()
        self.tf_parts._saver.restore(sess, model_save_path)  # load it
        value_ht, value_r, w, b = sess.run(
            [self.tf_parts._ht, self.tf_parts._r, self.tf_parts.w, self.tf_parts.b])  # extract values.
        self.vec_c = np.array(value_ht)
        self.vec_r = np.array(value_r)
        self.w = w
        self.b = b
        # when a model doesn't have Mt, suppose it should pass Mh

    # override
    def build_by_var(self, test_data, tf_model, this_data, sess=tf.Session()):
        """
        use data and model in memory.
        get self.vec_c (vectors for concepts), and self.vec_r(vectors for relations)
        :return:
        """
        self.this_data = this_data  # data.Data()

        self.test_triples = test_data
        self.tf_parts = tf_model

        self.sess = sess

        value_ht, value_r, w, b = sess.run(
            [self.tf_parts._ht, self.tf_parts._r, self.tf_parts.w, self.tf_parts.b])  # extract values.
        self.vec_c = np.array(value_ht)
        self.vec_r = np.array(value_r)
        self.w = w
        self.b = b

    # override
    def get_score(self, h, r, t):
        hvec, rvec, tvec = self.vecs_from_triples(h, r, t)
        return sigmoid(self.w*np.sum(np.multiply(np.multiply(hvec, tvec), rvec))+self.b)

    # override
    def get_score_batch(self, h_batch, r_batch, t_batch, isneg2Dbatch=False):
        N = h_batch.shape[0] # batch size
        scores = np.zeros(h_batch.shape) # [batch size] or [batch size, neg sample]
        for base_idx in range(0, N, self.tf_parts.batch_size_eval):
            effective_bs = min(self.tf_parts.batch_size_eval, N - base_idx)

            if isneg2Dbatch:
                scores_tmp = self.sess.run([self.tf_parts._f_prob_hn], feed_dict={
                    self.tf_parts._A_neg_hn_index:     h_batch[base_idx:base_idx + effective_bs],
                    self.tf_parts._A_neg_rel_hn_index: r_batch[base_idx:base_idx + effective_bs],
                    self.tf_parts._A_neg_t_index:      t_batch[base_idx:base_idx + effective_bs]
                    })
            else:
                scores_tmp = self.sess.run([self.tf_parts._f_prob_h], feed_dict={
                    self.tf_parts._A_h_index: h_batch[base_idx:base_idx + effective_bs],
                    self.tf_parts._A_r_index: r_batch[base_idx:base_idx + effective_bs],
                    self.tf_parts._A_t_index: t_batch[base_idx:base_idx + effective_bs]
                    })

            scores[base_idx:base_idx + effective_bs] = scores_tmp[0]

        return scores


class UKGE_rect_Tester(Tester):
    def __init__(self, ):
        Tester.__init__(self)

    # override
    def build_by_file(self, test_data_file, model_dir, model_filename='model.bin',
                      data_filename='data.bin'):
        """
        load data and model from files
        get self.vec_c (vectors for concepts), and self.vec_r(vectors for relations)
        :return:
        """

        # grandparent class: Tester
        Tester.build_by_file(self, test_data_file, model_dir, model_filename, data_filename)

        # load tf model and embeddings
        model_save_path = join(model_dir, model_filename)

        # reg_scale and neg_per_pos won't be used in the tf model during testing
        # just give any to build
        self.tf_parts = UKGE_rect_TF(num_rels=self.this_data.num_rels(),
                                     num_cons=self.this_data.num_cons(),
                                     dim=self.this_data.dim,
                                     batch_size=self.this_data.batch_size, neg_per_positive=10, reg_scale=0.1,
                                     p_neg=1)
        # neg_per_positive, reg_scale and p_neg are not used in testing
        sess = tf.Session()
        self.tf_parts._saver.restore(sess, model_save_path)  # load it
        value_ht, value_r, w, b = sess.run(
            [self.tf_parts._ht, self.tf_parts._r, self.tf_parts.w, self.tf_parts.b])  # extract values.
        sess.close()
        self.vec_c = np.array(value_ht)
        self.vec_r = np.array(value_r)
        self.w = w
        self.b = b

    # override
    def build_by_var(self, test_data, tf_model, this_data, sess=tf.Session()):
        """
        use data and model in memory.
        get self.vec_c (vectors for concepts), and self.vec_r(vectors for relations)
        :return:
        """
        self.this_data = this_data  # data.Data()

        self.test_triples = test_data
        self.tf_parts = tf_model

        value_ht, value_r, w, b = sess.run(
            [self.tf_parts._ht, self.tf_parts._r, self.tf_parts.w, self.tf_parts.b])  # extract values.
        self.vec_c = np.array(value_ht)
        self.vec_r = np.array(value_r)
        self.w = w
        self.b = b

    # override
    def get_score(self, h, r, t):
        # no sigmoid
        hvec, rvec, tvec = self.vecs_from_triples(h, r, t)
        return self.w * np.sum(np.multiply(np.multiply(hvec, tvec), rvec)) + self.b

    # override
    def get_score_batch(self, h_batch, r_batch, t_batch, isneg2Dbatch=False):
        # no sigmoid
        hvecs = self.con_index2vec_batch(h_batch)
        rvecs = self.rel_index2vec_batch(r_batch)
        tvecs = self.con_index2vec_batch(t_batch)
        if isneg2Dbatch:
            axis = 2  # axis for reduce_sum
        else:
            axis = 1
        return self.w * np.sum(np.multiply(np.multiply(hvecs, tvecs), rvecs), axis=axis) + self.b


    def bound_score(self, scores):
        """
        scores<0 =>0
        score>1 => 1
        :param scores:
        :return:
        """
        return np.minimum(np.maximum(scores, 0), 1)

    # override
    def get_bound_score(self, h, r, t):
        score = self.get_score(h, r, t)
        return self.bound_score(score)

    # override
    def get_bound_score_batch(self, h_batch, r_batch, t_batch, isneg2Dbatch=False):
        scores = self.get_score_batch(h_batch, r_batch, t_batch, isneg2Dbatch)
        return self.bound_score(scores)

    def get_mse(self, toprint=False, save_dir='', epoch=0):
        test_triples = self.test_triples
        N = test_triples.shape[0]

        # existing triples
        # (score - w)^2
        h_batch = test_triples[:, 0].astype(int)
        r_batch = test_triples[:, 1].astype(int)
        t_batch = test_triples[:, 2].astype(int)
        w_batch = test_triples[:, 3]
        scores = self.get_score_batch(h_batch, r_batch, t_batch)
        scores = self.bound_score(scores)
        mse = np.sum(np.square(scores - w_batch))

        mse = mse / N

        return mse

    def get_mae(self, verbose=False, save_dir='', epoch=0):
        test_triples = self.test_triples
        N = test_triples.shape[0]

        # existing triples
        # (score - w)^2
        h_batch = test_triples[:, 0].astype(int)
        r_batch = test_triples[:, 1].astype(int)
        t_batch = test_triples[:, 2].astype(int)
        w_batch = test_triples[:, 3]
        scores = self.get_score_batch(h_batch, r_batch, t_batch)
        scores = self.bound_score(scores)
        mae = np.sum(np.absolute(scores - w_batch))

        mae = mae / N
        return mae

    def get_mse_neg(self, neg_per_positive):
        test_triples = self.test_triples
        N = test_triples.shape[0]

        # negative samples
        # (score - 0)^2
        all_neg_hn_batch = self.this_data.corrupt_batch(test_triples, neg_per_positive, "h")
        all_neg_tn_batch = self.this_data.corrupt_batch(test_triples, neg_per_positive, "t")
        neg_hn_batch, neg_rel_hn_batch, \
        neg_t_batch, neg_h_batch, \
        neg_rel_tn_batch, neg_tn_batch \
            = all_neg_hn_batch[:, :, 0].astype(int), \
              all_neg_hn_batch[:, :, 1].astype(int), \
              all_neg_hn_batch[:, :, 2].astype(int), \
              all_neg_tn_batch[:, :, 0].astype(int), \
              all_neg_tn_batch[:, :, 1].astype(int), \
              all_neg_tn_batch[:, :, 2].astype(int)
        scores_hn = self.get_score_batch(neg_hn_batch, neg_rel_hn_batch, neg_t_batch, isneg2Dbatch=True)
        scores_tn = self.get_score_batch(neg_h_batch, neg_rel_tn_batch, neg_tn_batch, isneg2Dbatch=True)

        scores_hn = self.bound_score(scores_hn)
        scores_tn = self.bound_score(scores_tn)

        mse_hn = np.sum(np.mean(np.square(scores_hn - 0), axis=1)) / N
        mse_tn = np.sum(np.mean(np.square(scores_tn - 0), axis=1)) / N

        mse = (mse_hn + mse_tn) / 2
        return mse




class TransE_m1_Tester(Tester):
    def __init__(self, ):
        Tester.__init__(self)

    # override
    def build_by_file(self, test_data_file, model_dir, model_filename='model.bin', data_filename='data.bin'):
        """
        load data and model from files
        get self.vec_c (vectors for concepts), and self.vec_r(vectors for relations)
        :return:
        """
        Tester.build_by_file(self, test_data_file, model_dir, model_filename,
                             data_filename)

        # load tf model and embeddings
        model_save_path = join(model_dir, model_filename)
        self.tf_parts = TransE_m1_TF(num_rels=self.this_data.num_rels(),
                                     num_cons=self.this_data.num_cons(),
                                     dim=self.this_data.dim,
                                     batch_size=self.this_data.batch_size, neg_per_positive=10, p_neg=1)
        # neg_per_positive, reg_scale and p_neg are not used in testing
        sess = tf.Session()
        self.tf_parts._saver.restore(sess, model_save_path)  # load it
        value_ht, value_r = sess.run(
            [self.tf_parts._ht, self.tf_parts._r])  # extract values.
        sess.close()
        self.vec_c = np.array(value_ht)
        self.vec_r = np.array(value_r)
        # when a model doesn't have Mt, suppose it should pass Mh

    # override
    def build_by_var(self, test_data, tf_model, this_data, sess=tf.Session()):
        """
        use data and model in memory.
        get self.vec_c (vectors for concepts), and self.vec_r(vectors for relations)
        :return:
        """
        self.this_data = this_data  # data.Data()

        self.test_triples = test_data
        self.tf_parts = tf_model

        value_ht, value_r = sess.run(
            [self.tf_parts._ht, self.tf_parts._r])  # extract values.
        self.vec_c = np.array(value_ht)
        self.vec_r = np.array(value_r)

    # override
    def get_score(self, h, r, t):
        hvec, rvec, tvec = self.vecs_from_triples(h, r, t)
        return self.tf_parts.score_function(hvec, rvec, tvec,numpy=True)

    # override
    def get_score_batch(self, h_batch, r_batch, t_batch, isneg2Dbatch=False):
        hvecs = self.con_index2vec_batch(h_batch)
        rvecs = self.rel_index2vec_batch(r_batch)
        tvecs = self.con_index2vec_batch(t_batch)
        # if isneg2Dbatch:
        #     axis = 2  # axis for reduce_sum
        # else:
        #     axis = 1
        return self.tf_parts.score_function(hvecs, rvecs, tvecs,numpy=True)




class DistMult_m1_Tester(Tester):
    def __init__(self, ):
        Tester.__init__(self)

    # override
    def build_by_file(self, test_data_file, model_dir, model_filename='model.bin', data_filename='data.bin'):
        """
        load data and model from files
        get self.vec_c (vectors for concepts), and self.vec_r(vectors for relations)
        :return:
        """
        Tester.build_by_file(self, test_data_file, model_dir, model_filename,
                             data_filename)

        # load tf model and embeddings
        model_save_path = join(model_dir, model_filename)
        self.tf_parts = Distmult_m1_TF(num_rels=self.this_data.num_rels(),
                                     num_cons=self.this_data.num_cons(),
                                     dim=self.this_data.dim,
                                     batch_size=self.this_data.batch_size, neg_per_positive=10, p_neg=1)
        # neg_per_positive, reg_scale and p_neg are not used in testing
        sess = tf.Session()
        self.tf_parts._saver.restore(sess, model_save_path)  # load it
        value_ht, value_r = sess.run(
            [self.tf_parts._ht, self.tf_parts._r])  # extract values.
        sess.close()
        self.vec_c = np.array(value_ht)
        self.vec_r = np.array(value_r)
        # when a model doesn't have Mt, suppose it should pass Mh

    # override
    def build_by_var(self, test_data, tf_model, this_data, sess=tf.Session()):
        """
        use data and model in memory.
        get self.vec_c (vectors for concepts), and self.vec_r(vectors for relations)
        :return:
        """
        self.this_data = this_data  # data.Data()

        self.test_triples = test_data
        self.tf_parts = tf_model

        value_ht, value_r= sess.run(
            [self.tf_parts._ht, self.tf_parts._r])  # extract values.
        self.vec_c = np.array(value_ht)
        self.vec_r = np.array(value_r)

    # override
    def get_score(self, h, r, t):
        hvec, rvec, tvec = self.vecs_from_triples(h, r, t)
        return (np.sum(np.multiply(np.multiply(hvec, tvec), rvec)))

    # override
    def get_score_batch(self, h_batch, r_batch, t_batch, isneg2Dbatch=False):
        hvecs = self.con_index2vec_batch(h_batch)
        rvecs = self.rel_index2vec_batch(r_batch)
        tvecs = self.con_index2vec_batch(t_batch)
        if isneg2Dbatch:
            axis = 2  # axis for reduce_sum
        else:
            axis = 1
        return (np.sum(np.multiply(np.multiply(hvecs, tvecs), rvecs), axis=axis))




class DistMult_m2_Tester(Tester):
    def __init__(self, ):
        Tester.__init__(self)

    # override
    def build_by_file(self, test_data_file, model_dir, model_filename='model.bin', data_filename='data.bin'):
        """
        load data and model from files
        get self.vec_c (vectors for concepts), and self.vec_r(vectors for relations)
        :return:
        """
        Tester.build_by_file(self, test_data_file, model_dir, model_filename,
                             data_filename)

        # load tf model and embeddings
        model_save_path = join(model_dir, model_filename)
        self.tf_parts = Distmult_m2_TF(num_rels=self.this_data.num_rels(),
                                     num_cons=self.this_data.num_cons(),
                                     dim=self.this_data.dim,
                                     batch_size=self.this_data.batch_size, neg_per_positive=10, p_neg=1)
        # neg_per_positive, reg_scale and p_neg are not used in testing
        sess = tf.Session()
        self.tf_parts._saver.restore(sess, model_save_path)  # load it
        value_ht, value_r, w, b = sess.run(
            [self.tf_parts._ht, self.tf_parts._r, self.tf_parts.w, self.tf_parts.b])  # extract values.
        sess.close()
        self.vec_c = np.array(value_ht)
        self.vec_r = np.array(value_r)
        self.w = w
        self.b = b
        # when a model doesn't have Mt, suppose it should pass Mh

    # override
    def build_by_var(self, test_data, tf_model, this_data, sess=tf.Session()):
        """
        use data and model in memory.
        get self.vec_c (vectors for concepts), and self.vec_r(vectors for relations)
        :return:
        """
        self.this_data = this_data  # data.Data()

        self.test_triples = test_data
        self.tf_parts = tf_model

        value_ht, value_r, w, b = sess.run(
            [self.tf_parts._ht, self.tf_parts._r, self.tf_parts.w, self.tf_parts.b])  # extract values.
        self.vec_c = np.array(value_ht)
        self.vec_r = np.array(value_r)
        self.w = w
        self.b = b

    # override
    def get_score(self, h, r, t):
        hvec, rvec, tvec = self.vecs_from_triples(h, r, t)
        return sigmoid(self.w*np.sum(np.multiply(np.multiply(hvec, tvec), rvec))+self.b)

    # override
    def get_score_batch(self, h_batch, r_batch, t_batch, isneg2Dbatch=False):
        hvecs = self.con_index2vec_batch(h_batch)
        rvecs = self.rel_index2vec_batch(r_batch)
        tvecs = self.con_index2vec_batch(t_batch)
        if isneg2Dbatch:
            axis = 2  # axis for reduce_sum
        else:
            axis = 1
        return sigmoid(self.w*np.sum(np.multiply(np.multiply(hvecs, tvecs), rvecs), axis=axis)+self.b)




class ComplEx_m1_Tester(Tester):
    def __init__(self, ):
        Tester.__init__(self)

    # override
    def build_by_file(self, test_data_file, model_dir, model_filename='model.bin', data_filename='data.bin'):
        """
        load data and model from files
        get self.vec_c (vectors for concepts), and self.vec_r(vectors for relations)
        :return:
        """
        Tester.build_by_file(self, test_data_file, model_dir, model_filename,
                             data_filename)

        # load tf model and embeddings
        model_save_path = join(model_dir, model_filename)
        self.tf_parts = ComplEx_m1_TF(num_rels=self.this_data.num_rels(),
                                     num_cons=self.this_data.num_cons(),
                                     dim=self.this_data.dim,
                                     batch_size=self.this_data.batch_size, neg_per_positive=10, p_neg=1)
        # neg_per_positive, reg_scale and p_neg are not used in testing
        sess = tf.Session()
        self.tf_parts._saver.restore(sess, model_save_path)  # load it
        value_ht, value_r, w, b = sess.run(
            [self.tf_parts._ht, self.tf_parts._r])  # extract values.
        sess.close()
        self.vec_c = np.array(value_ht)
        self.vec_r = np.array(value_r)
        # when a model doesn't have Mt, suppose it should pass Mh

    # override
    def build_by_var(self, test_data, tf_model, this_data, sess=tf.Session()):
        """
        use data and model in memory.
        get self.vec_c (vectors for concepts), and self.vec_r(vectors for relations)
        :return:
        """
        self.this_data = this_data  # data.Data()

        self.test_triples = test_data
        self.tf_parts = tf_model

        value_ht, value_r = sess.run(
            [self.tf_parts._ht, self.tf_parts._r])  # extract values.
        self.vec_c = np.array(value_ht)
        self.vec_r = np.array(value_r)

    # override
    def get_score(self, h, r, t):
        hvec, rvec, tvec = self.vecs_from_triples(h, r, t)

        hvec_real, hvec_imag = self.tf_parts.split_embed_real_imag(hvec, numpy=True)
        rvec_real, rvec_imag = self.tf_parts.split_embed_real_imag(rvec, numpy=True)
        tvec_real, tvec_imag = self.tf_parts.split_embed_real_imag(tvec, numpy=True)
        
        
        return self.tf_parts.score_function(
            hvec_real, hvec_imag, 
            tvec_real, tvec_imag, 
            rvec_real, rvec_imag, numpy=True)

    # override
    def get_score_batch(self, h_batch, r_batch, t_batch, isneg2Dbatch=False):
        hvecs = self.con_index2vec_batch(h_batch)
        rvecs = self.rel_index2vec_batch(r_batch)
        tvecs = self.con_index2vec_batch(t_batch)

        hvecs_real, hvecs_imag = self.tf_parts.split_embed_real_imag(hvecs, numpy=True)
        rvecs_real, rvecs_imag = self.tf_parts.split_embed_real_imag(rvecs, numpy=True)
        tvecs_real, tvecs_imag = self.tf_parts.split_embed_real_imag(tvecs, numpy=True)
        
        
        # if isneg2Dbatch:
        #     axis = 2  # axis for reduce_sum
        # else:
        #     axis = 1
        return self.tf_parts.score_function(
            hvecs_real, hvecs_imag, 
            tvecs_real, tvecs_imag, 
            rvecs_real, rvecs_imag, numpy=True)


class ComplEx_m3_Tester(ComplEx_m1_Tester):
    def __init__(self, ):
        Tester.__init__(self)

    # override
    def build_by_file(self, test_data_file, model_dir, model_filename='model.bin', data_filename='data.bin'):
        """
        load data and model from files
        get self.vec_c (vectors for concepts), and self.vec_r(vectors for relations)
        :return:
        """
        Tester.build_by_file(self, test_data_file, model_dir, model_filename,
                             data_filename)

        # load tf model and embeddings
        model_save_path = join(model_dir, model_filename)
        self.tf_parts = ComplEx_m3_TF(num_rels=self.this_data.num_rels(),
                                     num_cons=self.this_data.num_cons(),
                                     dim=self.this_data.dim,
                                     batch_size=self.this_data.batch_size, neg_per_positive=10, p_neg=1)
        # neg_per_positive, reg_scale and p_neg are not used in testing
        sess = tf.Session()
        self.tf_parts._saver.restore(sess, model_save_path)  # load it
        value_ht, value_r = sess.run(
            [self.tf_parts._ht, self.tf_parts._r])  # extract values.
        sess.close()
        self.vec_c = np.array(value_ht)
        self.vec_r = np.array(value_r)
        # when a model doesn't have Mt, suppose it should pass Mh



class ComplEx_m4_Tester(Tester):
    def __init__(self, ):
        Tester.__init__(self)

    # override
    def build_by_file(self, test_data_file, model_dir, model_filename='model.bin', data_filename='data.bin'):
        """
        load data and model from files
        get self.vec_c (vectors for concepts), and self.vec_r(vectors for relations)
        :return:
        """
        Tester.build_by_file(self, test_data_file, model_dir, model_filename,
                             data_filename)

        # load tf model and embeddings
        model_save_path = join(model_dir, model_filename)
        self.tf_parts = ComplEx_m4_TF(num_rels=self.this_data.num_rels(),
                                     num_cons=self.this_data.num_cons(),
                                     dim=self.this_data.dim,
                                     batch_size=self.this_data.batch_size, neg_per_positive=10, p_neg=1)
        # neg_per_positive, reg_scale and p_neg are not used in testing
        self.sess = sess = tf.Session()
        self.tf_parts._saver.restore(sess, model_save_path)  # load it
        value_ht, value_r, w, b = sess.run(
            [self.tf_parts._ht, self.tf_parts._r, self.tf_parts.w, self.tf_parts.b])  # extract values.
        sess.close()
        self.vec_c = np.array(value_ht)
        self.vec_r = np.array(value_r)
        self.w = w
        self.b = b
        # when a model doesn't have Mt, suppose it should pass Mh

    # override
    def build_by_var(self, test_data, tf_model, this_data, sess=tf.Session()):
        """
        use data and model in memory.
        get self.vec_c (vectors for concepts), and self.vec_r(vectors for relations)
        :return:
        """
        self.this_data = this_data  # data.Data()

        self.test_triples = test_data
        self.tf_parts = tf_model

        self.sess = sess
        value_ht, value_r, w, b = sess.run(
            [self.tf_parts._ht, self.tf_parts._r, self.tf_parts.w, self.tf_parts.b])  # extract values.
        self.vec_c = np.array(value_ht)
        self.vec_r = np.array(value_r)
        self.w = w
        self.b = b

    # override
    def get_score(self, h, r, t):
        hvec, rvec, tvec = self.vecs_from_triples(h, r, t)

        hvec_real, hvec_imag = self.tf_parts.split_embed_real_imag(hvec, numpy=True)
        rvec_real, rvec_imag = self.tf_parts.split_embed_real_imag(rvec, numpy=True)
        tvec_real, tvec_imag = self.tf_parts.split_embed_real_imag(tvec, numpy=True)
        
        return self.tf_parts.score_function(
            hvec_real, hvec_imag, 
            tvec_real, tvec_imag, 
            rvec_real, rvec_imag, numpy=True, numpy_w=self.w, numpy_b=self.b)

    # override
    def get_score_batch(self, h_batch, r_batch, t_batch, isneg2Dbatch=False):
        N = h_batch.shape[0] # batch size
        scores = np.zeros(h_batch.shape) # [batch size] or [batch size, neg sample]
        for base_idx in range(0, N, self.tf_parts.batch_size_eval):
            effective_bs = min(self.tf_parts.batch_size_eval, N - base_idx)

            if isneg2Dbatch:
                scores_tmp = self.sess.run([self.tf_parts._f_score_hn], feed_dict={
                    self.tf_parts._A_neg_hn_index:     h_batch[base_idx:base_idx + effective_bs],
                    self.tf_parts._A_neg_rel_hn_index: r_batch[base_idx:base_idx + effective_bs],
                    self.tf_parts._A_neg_t_index:      t_batch[base_idx:base_idx + effective_bs]
                    })
            else:
                scores_tmp = self.sess.run([self.tf_parts._f_score_h], feed_dict={
                    self.tf_parts._A_h_index: h_batch[base_idx:base_idx + effective_bs],
                    self.tf_parts._A_r_index: r_batch[base_idx:base_idx + effective_bs],
                    self.tf_parts._A_t_index: t_batch[base_idx:base_idx + effective_bs]
                    })

            scores[base_idx:base_idx + effective_bs] = scores_tmp[0]

        return scores
        
        
        # if isneg2Dbatch:
        #     axis = 2  # axis for reduce_sum
        # else:
        #     axis = 1




class ComplEx_m5_1_Tester(ComplEx_m4_Tester):
    def __init__(self, ):
        Tester.__init__(self)

    # override
    def build_by_file(self, test_data_file, model_dir, model_filename='model.bin', data_filename='data.bin'):
        """
        load data and model from files
        get self.vec_c (vectors for concepts), and self.vec_r(vectors for relations)
        :return:
        """
        Tester.build_by_file(self, test_data_file, model_dir, model_filename,
                             data_filename)

        # load tf model and embeddings
        model_save_path = join(model_dir, model_filename)
        self.tf_parts = ComplEx_m5_1_TF(num_rels=self.this_data.num_rels(),
                                     num_cons=self.this_data.num_cons(),
                                     dim=self.this_data.dim,
                                     batch_size=self.this_data.batch_size, neg_per_positive=param.neg_per_pos, p_neg=1, psl=(param.n_psl > 0))
        # neg_per_positive, reg_scale and p_neg are not used in testing
        self.sess = sess = tf.Session()
        self.tf_parts._saver.restore(sess, model_save_path)  # load it
        value_ht, value_r, w, b = sess.run(
            [self.tf_parts._ht, self.tf_parts._r, self.tf_parts.w, self.tf_parts.b])  # extract values.
        self.vec_c = np.array(value_ht)
        self.vec_r = np.array(value_r)
        self.w = w
        self.b = b
        # when a model doesn't have Mt, suppose it should pass Mh


class ComplEx_m5_2_Tester(ComplEx_m4_Tester):
    def __init__(self, ):
        Tester.__init__(self)

    # override
    def build_by_file(self, test_data_file, model_dir, model_filename='model.bin', data_filename='data.bin'):
        """
        load data and model from files
        get self.vec_c (vectors for concepts), and self.vec_r(vectors for relations)
        :return:
        """
        Tester.build_by_file(self, test_data_file, model_dir, model_filename,
                             data_filename)

        # load tf model and embeddings
        model_save_path = join(model_dir, model_filename)
        self.tf_parts = ComplEx_m5_2_TF(num_rels=self.this_data.num_rels(),
                                     num_cons=self.this_data.num_cons(),
                                     dim=self.this_data.dim,
                                     batch_size=self.this_data.batch_size, neg_per_positive=param.neg_per_pos, p_neg=1, psl=(param.n_psl > 0))
        # neg_per_positive, reg_scale and p_neg are not used in testing
        self.sess = sess = tf.Session()
        self.tf_parts._saver.restore(sess, model_save_path)  # load it
        value_ht, value_r, w, b = sess.run(
            [self.tf_parts._ht, self.tf_parts._r, self.tf_parts.w, self.tf_parts.b])  # extract values.
        self.vec_c = np.array(value_ht)
        self.vec_r = np.array(value_r)
        self.w = w
        self.b = b

class ComplEx_m5_3_Tester(ComplEx_m4_Tester):
    def __init__(self, ):
        Tester.__init__(self)

    # override
    def build_by_file(self, test_data_file, model_dir, model_filename='model.bin', data_filename='data.bin'):
        """
        load data and model from files
        get self.vec_c (vectors for concepts), and self.vec_r(vectors for relations)
        :return:
        """
        Tester.build_by_file(self, test_data_file, model_dir, model_filename,
                             data_filename)

        # load tf model and embeddings
        model_save_path = join(model_dir, model_filename)
        self.tf_parts = ComplEx_m5_3_TF(num_rels=self.this_data.num_rels(),
                                     num_cons=self.this_data.num_cons(),
                                     dim=self.this_data.dim,
                                     batch_size=self.this_data.batch_size, neg_per_positive=param.neg_per_pos, p_neg=1, psl=(param.n_psl > 0))
        # neg_per_positive, reg_scale and p_neg are not used in testing
        self.sess = sess = tf.Session()
        self.tf_parts._saver.restore(sess, model_save_path)  # load it
        value_ht, value_r, w, b = sess.run(
            [self.tf_parts._ht, self.tf_parts._r, self.tf_parts.w, self.tf_parts.b])  # extract values.
        self.vec_c = np.array(value_ht)
        self.vec_r = np.array(value_r)
        self.w = w
        self.b = b


class ComplEx_m5_4_Tester(ComplEx_m4_Tester):
    def __init__(self, ):
        Tester.__init__(self)

    # override
    def build_by_file(self, test_data_file, model_dir, model_filename='model.bin', data_filename='data.bin'):
        """
        load data and model from files
        get self.vec_c (vectors for concepts), and self.vec_r(vectors for relations)
        :return:
        """
        Tester.build_by_file(self, test_data_file, model_dir, model_filename,
                             data_filename)

        # load tf model and embeddings
        model_save_path = join(model_dir, model_filename)
        self.tf_parts = ComplEx_m5_4_TF(num_rels=self.this_data.num_rels(),
                                     num_cons=self.this_data.num_cons(),
                                     dim=self.this_data.dim,
                                     batch_size=self.this_data.batch_size, neg_per_positive=param.neg_per_pos, p_neg=1, psl=(param.n_psl > 0))
        # neg_per_positive, reg_scale and p_neg are not used in testing
        self.sess = sess = tf.Session()
        self.tf_parts._saver.restore(sess, model_save_path)  # load it
        value_ht, value_r, w, b = sess.run(
            [self.tf_parts._ht, self.tf_parts._r, self.tf_parts.w, self.tf_parts.b])  # extract values.
        self.vec_c = np.array(value_ht)
        self.vec_r = np.array(value_r)
        self.w = w
        self.b = b




class ComplEx_m6_1_Tester(ComplEx_m4_Tester):
    def __init__(self, ):
        Tester.__init__(self)

    # override
    def build_by_file(self, test_data_file, model_dir, model_filename='model.bin', data_filename='data.bin'):
        """
        load data and model from files
        get self.vec_c (vectors for concepts), and self.vec_r(vectors for relations)
        :return:
        """
        Tester.build_by_file(self, test_data_file, model_dir, model_filename,
                             data_filename)

        # load tf model and embeddings
        model_save_path = join(model_dir, model_filename)
        self.tf_parts = ComplEx_m6_1_TF(num_rels=self.this_data.num_rels(),
                                     num_cons=self.this_data.num_cons(),
                                     dim=self.this_data.dim,
                                     batch_size=self.this_data.batch_size, neg_per_positive=param.neg_per_pos, p_neg=1, psl=(param.n_psl > 0))
        # neg_per_positive, reg_scale and p_neg are not used in testing
        self.sess = sess = tf.Session()
        self.tf_parts._saver.restore(sess, model_save_path)  # load it
        value_ht, value_r, w, b = sess.run(
            [self.tf_parts._ht, self.tf_parts._r, self.tf_parts.w, self.tf_parts.b])  # extract values.
        self.vec_c = np.array(value_ht)
        self.vec_r = np.array(value_r)
        self.w = w
        self.b = b

class ComplEx_m6_2_Tester(ComplEx_m4_Tester):
    def __init__(self, ):
        Tester.__init__(self)

    # override
    def build_by_file(self, test_data_file, model_dir, model_filename='model.bin', data_filename='data.bin'):
        """
        load data and model from files
        get self.vec_c (vectors for concepts), and self.vec_r(vectors for relations)
        :return:
        """
        Tester.build_by_file(self, test_data_file, model_dir, model_filename,
                             data_filename)

        # load tf model and embeddings
        model_save_path = join(model_dir, model_filename)
        self.tf_parts = ComplEx_m6_2_TF(num_rels=self.this_data.num_rels(),
                                     num_cons=self.this_data.num_cons(),
                                     dim=self.this_data.dim,
                                     batch_size=self.this_data.batch_size, neg_per_positive=param.neg_per_pos, p_neg=1, psl=(param.n_psl > 0))
        # neg_per_positive, reg_scale and p_neg are not used in testing
        self.sess = sess = tf.Session()
        self.tf_parts._saver.restore(sess, model_save_path)  # load it
        value_ht, value_r, w, b = sess.run(
            [self.tf_parts._ht, self.tf_parts._r, self.tf_parts.w, self.tf_parts.b])  # extract values.
        self.vec_c = np.array(value_ht)
        self.vec_r = np.array(value_r)
        self.w = w
        self.b = b
        # when a model doesn't have Mt, suppose it should pass Mh



class ComplEx_m9_2_Tester(ComplEx_m4_Tester):
    def __init__(self, ):
        Tester.__init__(self)

    # override
    def build_by_file(self, test_data_file, model_dir, model_filename='model.bin', data_filename='data.bin'):
        """
        load data and model from files
        get self.vec_c (vectors for concepts), and self.vec_r(vectors for relations)
        :return:
        """
        Tester.build_by_file(self, test_data_file, model_dir, model_filename,
                             data_filename)

        # load tf model and embeddings
        model_save_path = join(model_dir, model_filename)
        self.tf_parts = ComplEx_m9_2_TF(num_rels=self.this_data.num_rels(),
                                     num_cons=self.this_data.num_cons(),
                                     dim=self.this_data.dim,
                                     batch_size=self.this_data.batch_size, neg_per_positive=param.neg_per_pos, p_neg=1, psl=(param.n_psl > 0))
        # neg_per_positive, reg_scale and p_neg are not used in testing
        self.sess = sess = tf.Session()
        self.tf_parts._saver.restore(sess, model_save_path)  # load it
        value_ht, value_r, w, b = sess.run(
            [self.tf_parts._ht, self.tf_parts._r, self.tf_parts.w, self.tf_parts.b])  # extract values.
        self.vec_c = np.array(value_ht)
        self.vec_r = np.array(value_r)
        self.w = w
        self.b = b
        # when a model doesn't have Mt, suppose it should pass Mh


class ComplEx_m9_3_Tester(ComplEx_m4_Tester):
    def __init__(self, ):
        Tester.__init__(self)

    # override
    def build_by_file(self, test_data_file, model_dir, model_filename='model.bin', data_filename='data.bin'):
        """
        load data and model from files
        get self.vec_c (vectors for concepts), and self.vec_r(vectors for relations)
        :return:
        """
        Tester.build_by_file(self, test_data_file, model_dir, model_filename,
                             data_filename)

        # load tf model and embeddings
        model_save_path = join(model_dir, model_filename)
        self.tf_parts = ComplEx_m9_3_TF(num_rels=self.this_data.num_rels(),
                                     num_cons=self.this_data.num_cons(),
                                     dim=self.this_data.dim,
                                     batch_size=self.this_data.batch_size, neg_per_positive=param.neg_per_pos, p_neg=1, psl=(param.n_psl > 0))
        # neg_per_positive, reg_scale and p_neg are not used in testing
        self.sess = sess = tf.Session()
        self.tf_parts._saver.restore(sess, model_save_path)  # load it
        value_ht, value_r, w, b = sess.run(
            [self.tf_parts._ht, self.tf_parts._r, self.tf_parts.w, self.tf_parts.b])  # extract values.
        self.vec_c = np.array(value_ht)
        self.vec_r = np.array(value_r)
        self.w = w
        self.b = b
        # when a model doesn't have Mt, suppose it should pass Mh






class ComplEx_m10_Tester(ComplEx_m4_Tester):
    def __init__(self, ):
        Tester.__init__(self)

    # override
    def build_by_file(self, test_data_file, model_dir, model_filename='model.bin', data_filename='data.bin'):
        """
        load data and model from files
        get self.vec_c (vectors for concepts), and self.vec_r(vectors for relations)
        :return:
        """
        Tester.build_by_file(self, test_data_file, model_dir, model_filename,
                             data_filename)

        # load tf model and embeddings
        model_save_path = join(model_dir, model_filename)
        self.tf_parts = ComplEx_m10_TF(num_rels=self.this_data.num_rels(),
                                     num_cons=self.this_data.num_cons(),
                                     dim=self.this_data.dim,
                                     batch_size=self.this_data.batch_size, neg_per_positive=param.neg_per_pos, p_neg=1, psl=(param.n_psl > 0))
        # neg_per_positive, reg_scale and p_neg are not used in testing
        self.sess = sess = tf.Session()
        self.tf_parts._saver.restore(sess, model_save_path)  # load it
        value_ht, value_r, w, b = sess.run(
            [self.tf_parts._ht, self.tf_parts._r, self.tf_parts.w, self.tf_parts.b])  # extract values.
        self.vec_c = np.array(value_ht)
        self.vec_r = np.array(value_r)
        self.w = w
        self.b = b
        # when a model doesn't have Mt, suppose it should pass Mh


class ComplEx_m10_1_Tester(ComplEx_m10_Tester):
    def __init__(self, ):
        Tester.__init__(self)

    # override
    def build_by_file(self, test_data_file, model_dir, model_filename='model.bin', data_filename='data.bin'):
        """
        load data and model from files
        get self.vec_c (vectors for concepts), and self.vec_r(vectors for relations)
        :return:
        """
        Tester.build_by_file(self, test_data_file, model_dir, model_filename,
                             data_filename)

        # load tf model and embeddings
        model_save_path = join(model_dir, model_filename)
        self.tf_parts = ComplEx_m10_1_TF(num_rels=self.this_data.num_rels(),
                                     num_cons=self.this_data.num_cons(),
                                     dim=self.this_data.dim,
                                     batch_size=self.this_data.batch_size, neg_per_positive=param.neg_per_pos, p_neg=1, psl=(param.n_psl > 0))
        # neg_per_positive, reg_scale and p_neg are not used in testing
        self.sess = sess = tf.Session()
        self.tf_parts._saver.restore(sess, model_save_path)  # load it
        value_ht, value_r, w, b = sess.run(
            [self.tf_parts._ht, self.tf_parts._r, self.tf_parts.w, self.tf_parts.b])  # extract values.
        self.vec_c = np.array(value_ht)
        self.vec_r = np.array(value_r)
        self.w = w
        self.b = b
        # when a model doesn't have Mt, suppose it should pass Mh

class ComplEx_m10_2_Tester(ComplEx_m10_Tester):
    def __init__(self, ):
        Tester.__init__(self)

    # override
    def build_by_file(self, test_data_file, model_dir, model_filename='model.bin', data_filename='data.bin'):
        """
        load data and model from files
        get self.vec_c (vectors for concepts), and self.vec_r(vectors for relations)
        :return:
        """
        Tester.build_by_file(self, test_data_file, model_dir, model_filename,
                             data_filename)

        # load tf model and embeddings
        model_save_path = join(model_dir, model_filename)
        self.tf_parts = ComplEx_m10_2_TF(num_rels=self.this_data.num_rels(),
                                     num_cons=self.this_data.num_cons(),
                                     dim=self.this_data.dim,
                                     batch_size=self.this_data.batch_size, neg_per_positive=param.neg_per_pos, p_neg=1, psl=(param.n_psl > 0))
        # neg_per_positive, reg_scale and p_neg are not used in testing
        self.sess = sess = tf.Session()
        self.tf_parts._saver.restore(sess, model_save_path)  # load it
        value_ht, value_r, w, b = sess.run(
            [self.tf_parts._ht, self.tf_parts._r, self.tf_parts.w, self.tf_parts.b])  # extract values.
        self.vec_c = np.array(value_ht)
        self.vec_r = np.array(value_r)
        self.w = w
        self.b = b
        # when a model doesn't have Mt, suppose it should pass Mh



class ComplEx_m10_3_Tester(ComplEx_m10_Tester):
    def __init__(self, ):
        Tester.__init__(self)

    # override
    def build_by_file(self, test_data_file, model_dir, model_filename='model.bin', data_filename='data.bin'):
        """
        load data and model from files
        get self.vec_c (vectors for concepts), and self.vec_r(vectors for relations)
        :return:
        """
        Tester.build_by_file(self, test_data_file, model_dir, model_filename,
                             data_filename)

        # load tf model and embeddings
        model_save_path = join(model_dir, model_filename)
        self.tf_parts = ComplEx_m10_3_TF(num_rels=self.this_data.num_rels(),
                                     num_cons=self.this_data.num_cons(),
                                     dim=self.this_data.dim,
                                     batch_size=self.this_data.batch_size, neg_per_positive=param.neg_per_pos, p_neg=1, psl=(param.n_psl > 0))
        # neg_per_positive, reg_scale and p_neg are not used in testing
        self.sess = sess = tf.Session()
        self.tf_parts._saver.restore(sess, model_save_path)  # load it
        value_ht, value_r, w, b = sess.run(
            [self.tf_parts._ht, self.tf_parts._r, self.tf_parts.w, self.tf_parts.b])  # extract values.
        self.vec_c = np.array(value_ht)
        self.vec_r = np.array(value_r)
        self.w = w
        self.b = b
        # when a model doesn't have Mt, suppose it should pass Mh

class ComplEx_m10_4_Tester(ComplEx_m10_Tester):
    def __init__(self, ):
        Tester.__init__(self)

    # override
    def build_by_file(self, test_data_file, model_dir, model_filename='model.bin', data_filename='data.bin'):
        """
        load data and model from files
        get self.vec_c (vectors for concepts), and self.vec_r(vectors for relations)
        :return:
        """
        Tester.build_by_file(self, test_data_file, model_dir, model_filename,
                             data_filename)

        # load tf model and embeddings
        model_save_path = join(model_dir, model_filename)
        self.tf_parts = ComplEx_m10_4_TF(num_rels=self.this_data.num_rels(),
                                     num_cons=self.this_data.num_cons(),
                                     dim=self.this_data.dim,
                                     batch_size=self.this_data.batch_size, neg_per_positive=param.neg_per_pos, p_neg=1, psl=(param.n_psl > 0))
        # neg_per_positive, reg_scale and p_neg are not used in testing
        self.sess = sess = tf.Session()
        self.tf_parts._saver.restore(sess, model_save_path)  # load it
        value_ht, value_r, w, b = sess.run(
            [self.tf_parts._ht, self.tf_parts._r, self.tf_parts.w, self.tf_parts.b])  # extract values.
        self.vec_c = np.array(value_ht)
        self.vec_r = np.array(value_r)
        self.w = w
        self.b = b
        # when a model doesn't have Mt, suppose it should pass Mh







class RotatE_m1_Tester(Tester):
    def __init__(self, ):
        Tester.__init__(self)

    # override
    def build_by_file(self, test_data_file, model_dir, model_filename='model.bin', data_filename='data.bin'):
        """
        load data and model from files
        get self.vec_c (vectors for concepts), and self.vec_r(vectors for relations)
        :return:
        """
        Tester.build_by_file(self, test_data_file, model_dir, model_filename,
                             data_filename)

        # load tf model and embeddings
        model_save_path = join(model_dir, model_filename)
        self.tf_parts = RotatE_m1_TF(num_rels=self.this_data.num_rels(),
                                     num_cons=self.this_data.num_cons(),
                                     dim=self.this_data.dim,
                                     batch_size=self.this_data.batch_size, neg_per_positive=10, p_neg=1)
        # neg_per_positive, reg_scale and p_neg are not used in testing
        self.sess = sess = tf.Session()
        self.tf_parts._saver.restore(sess, model_save_path)  # load it
        value_ht, value_r = sess.run(
            [self.tf_parts._ht, self.tf_parts._r])  # extract values.
        self.vec_c = np.array(value_ht)
        self.vec_r = np.array(value_r)
        # when a model doesn't have Mt, suppose it should pass Mh

    # override
    def build_by_var(self, test_data, tf_model, this_data, sess=tf.Session()):
        """
        use data and model in memory.
        get self.vec_c (vectors for concepts), and self.vec_r(vectors for relations)
        :return:
        """
        self.this_data = this_data  # data.Data()

        self.test_triples = test_data
        self.tf_parts = tf_model

        self.sess = sess

        value_ht, value_r = sess.run(
            [self.tf_parts._ht, self.tf_parts._r])  # extract values.
        self.vec_c = np.array(value_ht)
        self.vec_r = np.array(value_r)

    # override
    def get_score(self, h, r, t):
        hvec, rvec, tvec = self.vecs_from_triples(h, r, t)

        hvec_real, hvec_imag = self.tf_parts.split_embed_real_imag(hvec, numpy=True)
        tvec_real, tvec_imag = self.tf_parts.split_embed_real_imag(tvec, numpy=True)
        
        
        return self.tf_parts.score_function(
            hvec_real, hvec_imag, 
            tvec_real, tvec_imag, 
            rvec, numpy=True)

    # override
    def get_score_batch(self, h_batch, r_batch, t_batch, isneg2Dbatch=False):
        N = h_batch.shape[0] # batch size
        scores = np.zeros(h_batch.shape) # [batch size] or [batch size, neg sample]
        for base_idx in range(0, N, self.tf_parts.batch_size_eval):
            effective_bs = min(self.tf_parts.batch_size_eval, N - base_idx)

            if isneg2Dbatch:
                scores_tmp = self.sess.run([self.tf_parts._f_score_hn], feed_dict={
                    self.tf_parts._A_neg_hn_index:     h_batch[base_idx:base_idx + effective_bs],
                    self.tf_parts._A_neg_rel_hn_index: r_batch[base_idx:base_idx + effective_bs],
                    self.tf_parts._A_neg_t_index:      t_batch[base_idx:base_idx + effective_bs]
                    })
            else:
                scores_tmp = self.sess.run([self.tf_parts._f_score_h], feed_dict={
                    self.tf_parts._A_h_index: h_batch[base_idx:base_idx + effective_bs],
                    self.tf_parts._A_r_index: r_batch[base_idx:base_idx + effective_bs],
                    self.tf_parts._A_t_index: t_batch[base_idx:base_idx + effective_bs]
                    })

            scores[base_idx:base_idx + effective_bs] = scores_tmp[0]

        return scores
        
class RotatE_m2_Tester(RotatE_m1_Tester):
    def __init__(self, ):
        Tester.__init__(self)

    # override
    def build_by_file(self, test_data_file, model_dir, model_filename='model.bin', data_filename='data.bin'):
        """
        load data and model from files
        get self.vec_c (vectors for concepts), and self.vec_r(vectors for relations)
        :return:
        """
        Tester.build_by_file(self, test_data_file, model_dir, model_filename,
                             data_filename)

        # load tf model and embeddings
        model_save_path = join(model_dir, model_filename)
        self.tf_parts = RotatE_m2_TF(num_rels=self.this_data.num_rels(),
                                     num_cons=self.this_data.num_cons(),
                                     dim=self.this_data.dim,
                                     batch_size=self.this_data.batch_size, 
                                     neg_per_positive=param.neg_per_pos, 
                                     p_neg=1, psl=(param.n_psl > 0),
                                     gamma=1)

        # neg_per_positive, reg_scale and p_neg are not used in testing
        self.sess = sess = tf.Session()
        self.tf_parts._saver.restore(sess, model_save_path)  # load it
        value_ht, value_r = sess.run(
            [self.tf_parts._ht, self.tf_parts._r])  # extract values.
        self.vec_c = np.array(value_ht)
        self.vec_r = np.array(value_r)

class RotatE_m2_1_Tester(RotatE_m1_Tester):
    def __init__(self, ):
        Tester.__init__(self)

    # override
    def build_by_file(self, test_data_file, model_dir, model_filename='model.bin', data_filename='data.bin'):
        """
        load data and model from files
        get self.vec_c (vectors for concepts), and self.vec_r(vectors for relations)
        :return:
        """
        Tester.build_by_file(self, test_data_file, model_dir, model_filename,
                             data_filename)

        # load tf model and embeddings
        model_save_path = join(model_dir, model_filename)
        self.tf_parts = RotatE_m2_1_TF(num_rels=self.this_data.num_rels(),
                                     num_cons=self.this_data.num_cons(),
                                     dim=self.this_data.dim,
                                     batch_size=self.this_data.batch_size, 
                                     neg_per_positive=param.neg_per_pos, 
                                     p_neg=1, psl=(param.n_psl > 0),
                                     gamma=1)
        # neg_per_positive, reg_scale and p_neg are not used in testing
        self.sess = sess = tf.Session()
        self.tf_parts._saver.restore(sess, model_save_path)  # load it
        value_ht, value_r = sess.run(
            [self.tf_parts._ht, self.tf_parts._r])  # extract values.
        self.vec_c = np.array(value_ht)
        self.vec_r = np.array(value_r)






class UKGE_logi_m1_Tester(ComplEx_m4_Tester):
    def __init__(self, ):
        Tester.__init__(self)

    # override
    def build_by_file(self, test_data_file, model_dir, model_filename='model.bin', data_filename='data.bin'):
        """
        load data and model from files
        get self.vec_c (vectors for concepts), and self.vec_r(vectors for relations)
        :return:
        """
        Tester.build_by_file(self, test_data_file, model_dir, model_filename,
                             data_filename)

        # load tf model and embeddings
        model_save_path = join(model_dir, model_filename)
        self.tf_parts = UKGE_logi_m1_TF(num_rels=self.this_data.num_rels(),
                                     num_cons=self.this_data.num_cons(),
                                     dim=self.this_data.dim,
                                     batch_size=self.this_data.batch_size, neg_per_positive=param.neg_per_pos, p_neg=1, psl=(param.n_psl > 0))
        # neg_per_positive, reg_scale and p_neg are not used in testing
        self.sess = sess = tf.Session()
        self.tf_parts._saver.restore(sess, model_save_path)  # load it
        value_ht, value_r, w, b = sess.run(
            [self.tf_parts._ht, self.tf_parts._r, self.tf_parts.w, self.tf_parts.b])  # extract values.
        self.vec_c = np.array(value_ht)
        self.vec_r = np.array(value_r)
        self.w = w
        self.b = b



class UKGE_logi_m2_Tester(ComplEx_m4_Tester):
    def __init__(self, ):
        Tester.__init__(self)

    # override
    def build_by_file(self, test_data_file, model_dir, model_filename='model.bin', data_filename='data.bin'):
        """
        load data and model from files
        get self.vec_c (vectors for concepts), and self.vec_r(vectors for relations)
        :return:
        """
        Tester.build_by_file(self, test_data_file, model_dir, model_filename,
                             data_filename)

        # load tf model and embeddings
        model_save_path = join(model_dir, model_filename)
        self.tf_parts = UKGE_logi_m2_TF(num_rels=self.this_data.num_rels(),
                                     num_cons=self.this_data.num_cons(),
                                     dim=self.this_data.dim,
                                     batch_size=self.this_data.batch_size, neg_per_positive=param.neg_per_pos, p_neg=1, psl=(param.n_psl > 0))
        # neg_per_positive, reg_scale and p_neg are not used in testing
        self.sess = sess = tf.Session()
        self.tf_parts._saver.restore(sess, model_save_path)  # load it
        value_ht, value_r, w, b = sess.run(
            [self.tf_parts._ht, self.tf_parts._r, self.tf_parts.w, self.tf_parts.b])  # extract values.
        self.vec_c = np.array(value_ht)
        self.vec_r = np.array(value_r)
        self.w = w
        self.b = b


class RotatE_m3_1_Tester(ComplEx_m4_Tester):
    def __init__(self, ):
        Tester.__init__(self)

    # override
    def build_by_file(self, test_data_file, model_dir, model_filename='model.bin', data_filename='data.bin'):
        """
        load data and model from files
        get self.vec_c (vectors for concepts), and self.vec_r(vectors for relations)
        :return:
        """
        Tester.build_by_file(self, test_data_file, model_dir, model_filename,
                             data_filename)

        # load tf model and embeddings
        model_save_path = join(model_dir, model_filename)
        self.tf_parts = RotatE_m3_1_TF(num_rels=self.this_data.num_rels(),
                                     num_cons=self.this_data.num_cons(),
                                     dim=self.this_data.dim,
                                     batch_size=self.this_data.batch_size, neg_per_positive=param.neg_per_pos, p_neg=1, psl=(param.n_psl > 0))
        # neg_per_positive, reg_scale and p_neg are not used in testing
        self.sess = sess = tf.Session()
        self.tf_parts._saver.restore(sess, model_save_path)  # load it
        value_ht, value_r, w, b = sess.run(
            [self.tf_parts._ht, self.tf_parts._r, self.tf_parts.w, self.tf_parts.b])  # extract values.
        self.vec_c = np.array(value_ht)
        self.vec_r = np.array(value_r)
        self.w = w
        self.b = b


class RotatE_m3_2_Tester(ComplEx_m4_Tester):
    def __init__(self, ):
        Tester.__init__(self)

    # override
    def build_by_file(self, test_data_file, model_dir, model_filename='model.bin', data_filename='data.bin'):
        """
        load data and model from files
        get self.vec_c (vectors for concepts), and self.vec_r(vectors for relations)
        :return:
        """
        Tester.build_by_file(self, test_data_file, model_dir, model_filename,
                             data_filename)

        # load tf model and embeddings
        model_save_path = join(model_dir, model_filename)
        self.tf_parts = RotatE_m3_2_TF(num_rels=self.this_data.num_rels(),
                                     num_cons=self.this_data.num_cons(),
                                     dim=self.this_data.dim,
                                     batch_size=self.this_data.batch_size, neg_per_positive=param.neg_per_pos, p_neg=1, psl=(param.n_psl > 0))
        # neg_per_positive, reg_scale and p_neg are not used in testing
        self.sess = sess = tf.Session()
        self.tf_parts._saver.restore(sess, model_save_path)  # load it
        value_ht, value_r, w, b = sess.run(
            [self.tf_parts._ht, self.tf_parts._r, self.tf_parts.w, self.tf_parts.b])  # extract values.
        self.vec_c = np.array(value_ht)
        self.vec_r = np.array(value_r)
        self.w = w
        self.b = b


class RotatE_m2_2_Tester(ComplEx_m4_Tester):
    def __init__(self, ):
        Tester.__init__(self)

    # override
    def build_by_file(self, test_data_file, model_dir, model_filename='model.bin', data_filename='data.bin'):
        """
        load data and model from files
        get self.vec_c (vectors for concepts), and self.vec_r(vectors for relations)
        :return:
        """
        Tester.build_by_file(self, test_data_file, model_dir, model_filename,
                             data_filename)

        # load tf model and embeddings
        model_save_path = join(model_dir, model_filename)
        self.tf_parts = RotatE_m2_2_TF(num_rels=self.this_data.num_rels(),
                                     num_cons=self.this_data.num_cons(),
                                     dim=self.this_data.dim,
                                     batch_size=self.this_data.batch_size, neg_per_positive=param.neg_per_pos, p_neg=1, psl=(param.n_psl > 0))
        # neg_per_positive, reg_scale and p_neg are not used in testing
        self.sess = sess = tf.Session()
        self.tf_parts._saver.restore(sess, model_save_path)  # load it
        value_ht, value_r, w, b = sess.run(
            [self.tf_parts._ht, self.tf_parts._r, self.tf_parts.w, self.tf_parts.b])  # extract values.
        self.vec_c = np.array(value_ht)
        self.vec_r = np.array(value_r)
        self.w = w
        self.b = b


class RotatE_m3_3_Tester(ComplEx_m4_Tester):
    def __init__(self, ):
        Tester.__init__(self)

    # override
    def build_by_file(self, test_data_file, model_dir, model_filename='model.bin', data_filename='data.bin'):
        """
        load data and model from files
        get self.vec_c (vectors for concepts), and self.vec_r(vectors for relations)
        :return:
        """
        Tester.build_by_file(self, test_data_file, model_dir, model_filename,
                             data_filename)

        # load tf model and embeddings
        model_save_path = join(model_dir, model_filename)
        self.tf_parts = RotatE_m3_3_TF(num_rels=self.this_data.num_rels(),
                                     num_cons=self.this_data.num_cons(),
                                     dim=self.this_data.dim,
                                     batch_size=self.this_data.batch_size, neg_per_positive=param.neg_per_pos, p_neg=1, psl=(param.n_psl > 0))
        # neg_per_positive, reg_scale and p_neg are not used in testing
        self.sess = sess = tf.Session()
        self.tf_parts._saver.restore(sess, model_save_path)  # load it
        value_ht, value_r, w, b = sess.run(
            [self.tf_parts._ht, self.tf_parts._r, self.tf_parts.w, self.tf_parts.b])  # extract values.
        self.vec_c = np.array(value_ht)
        self.vec_r = np.array(value_r)
        self.w = w
        self.b = b



class RotatE_m5_Tester(ComplEx_m4_Tester):
    def __init__(self, ):
        Tester.__init__(self)

    # override
    def build_by_file(self, test_data_file, model_dir, model_filename='model.bin', data_filename='data.bin'):
        """
        load data and model from files
        get self.vec_c (vectors for concepts), and self.vec_r(vectors for relations)
        :return:
        """
        Tester.build_by_file(self, test_data_file, model_dir, model_filename,
                             data_filename)

        # load tf model and embeddings
        model_save_path = join(model_dir, model_filename)
        self.tf_parts = RotatE_m5_TF(num_rels=self.this_data.num_rels(),
                                     num_cons=self.this_data.num_cons(),
                                     dim=self.this_data.dim,
                                     batch_size=self.this_data.batch_size, neg_per_positive=param.neg_per_pos, p_neg=1, psl=(param.n_psl > 0), gamma=2)
        # neg_per_positive, reg_scale and p_neg are not used in testing
        self.sess = sess = tf.Session()
        self.tf_parts._saver.restore(sess, model_save_path)  # load it
        value_ht, value_r, w, b = sess.run(
            [self.tf_parts._ht, self.tf_parts._r, self.tf_parts.w, self.tf_parts.b])  # extract values.
        self.vec_c = np.array(value_ht)
        self.vec_r = np.array(value_r)
        self.w = w
        self.b = b


class RotatE_m5_1_Tester(ComplEx_m4_Tester):
    def __init__(self, ):
        Tester.__init__(self)

    # override
    def build_by_file(self, test_data_file, model_dir, model_filename='model.bin', data_filename='data.bin'):
        """
        load data and model from files
        get self.vec_c (vectors for concepts), and self.vec_r(vectors for relations)
        :return:
        """
        Tester.build_by_file(self, test_data_file, model_dir, model_filename,
                             data_filename)

        # load tf model and embeddings
        model_save_path = join(model_dir, model_filename)
        self.tf_parts = RotatE_m5_1_TF(num_rels=self.this_data.num_rels(),
                                     num_cons=self.this_data.num_cons(),
                                     dim=self.this_data.dim,
                                     batch_size=self.this_data.batch_size, neg_per_positive=param.neg_per_pos, p_neg=1, psl=(param.n_psl > 0), gamma=2)
        # neg_per_positive, reg_scale and p_neg are not used in testing
        self.sess = sess = tf.Session()
        self.tf_parts._saver.restore(sess, model_save_path)  # load it
        value_ht, value_r, w, b = sess.run(
            [self.tf_parts._ht, self.tf_parts._r, self.tf_parts.w, self.tf_parts.b])  # extract values.
        self.vec_c = np.array(value_ht)
        self.vec_r = np.array(value_r)
        self.w = w
        self.b = b



class TransE_m3_Tester(ComplEx_m4_Tester):
    def __init__(self, ):
        Tester.__init__(self)

    # override
    def build_by_file(self, test_data_file, model_dir, model_filename='model.bin', data_filename='data.bin'):
        """
        load data and model from files
        get self.vec_c (vectors for concepts), and self.vec_r(vectors for relations)
        :return:
        """
        Tester.build_by_file(self, test_data_file, model_dir, model_filename,
                             data_filename)

        # load tf model and embeddings
        model_save_path = join(model_dir, model_filename)
        self.tf_parts = TransE_m3_TF(num_rels=self.this_data.num_rels(),
                                     num_cons=self.this_data.num_cons(),
                                     dim=self.this_data.dim,
                                     batch_size=self.this_data.batch_size, neg_per_positive=param.neg_per_pos, p_neg=1, psl=(param.n_psl > 0))
        # neg_per_positive, reg_scale and p_neg are not used in testing
        self.sess = sess = tf.Session()
        self.tf_parts._saver.restore(sess, model_save_path)  # load it
        value_ht, value_r, w, b = sess.run(
            [self.tf_parts._ht, self.tf_parts._r, self.tf_parts.w, self.tf_parts.b])  # extract values.
        self.vec_c = np.array(value_ht)
        self.vec_r = np.array(value_r)
        self.w = w
        self.b = b

class TransE_m3_1_Tester(ComplEx_m4_Tester):
    def __init__(self, ):
        Tester.__init__(self)

    # override
    def build_by_file(self, test_data_file, model_dir, model_filename='model.bin', data_filename='data.bin'):
        """
        load data and model from files
        get self.vec_c (vectors for concepts), and self.vec_r(vectors for relations)
        :return:
        """
        Tester.build_by_file(self, test_data_file, model_dir, model_filename,
                             data_filename)

        # load tf model and embeddings
        model_save_path = join(model_dir, model_filename)
        self.tf_parts = TransE_m3_1_TF(num_rels=self.this_data.num_rels(),
                                     num_cons=self.this_data.num_cons(),
                                     dim=self.this_data.dim,
                                     batch_size=self.this_data.batch_size, neg_per_positive=param.neg_per_pos, p_neg=1, psl=(param.n_psl > 0))
        # neg_per_positive, reg_scale and p_neg are not used in testing
        self.sess = sess = tf.Session()
        self.tf_parts._saver.restore(sess, model_save_path)  # load it
        value_ht, value_r, w, b = sess.run(
            [self.tf_parts._ht, self.tf_parts._r, self.tf_parts.w, self.tf_parts.b])  # extract values.
        self.vec_c = np.array(value_ht)
        self.vec_r = np.array(value_r)
        self.w = w
        self.b = b

class TransE_m3_3_Tester(ComplEx_m4_Tester):
    def __init__(self, ):
        Tester.__init__(self)

    # override
    def build_by_file(self, test_data_file, model_dir, model_filename='model.bin', data_filename='data.bin'):
        """
        load data and model from files
        get self.vec_c (vectors for concepts), and self.vec_r(vectors for relations)
        :return:
        """
        Tester.build_by_file(self, test_data_file, model_dir, model_filename,
                             data_filename)

        # load tf model and embeddings
        model_save_path = join(model_dir, model_filename)
        self.tf_parts = TransE_m3_3_TF(num_rels=self.this_data.num_rels(),
                                     num_cons=self.this_data.num_cons(),
                                     dim=self.this_data.dim,
                                     batch_size=self.this_data.batch_size, neg_per_positive=param.neg_per_pos, p_neg=1, psl=(param.n_psl > 0))
        # neg_per_positive, reg_scale and p_neg are not used in testing
        self.sess = sess = tf.Session()
        self.tf_parts._saver.restore(sess, model_save_path)  # load it
        value_ht, value_r, w, b = sess.run(
            [self.tf_parts._ht, self.tf_parts._r, self.tf_parts.w, self.tf_parts.b])  # extract values.
        self.vec_c = np.array(value_ht)
        self.vec_r = np.array(value_r)
        self.w = w
        self.b = b


class OpenKE_Tester(ComplEx_m4_Tester):
    def __init__(self, ):
        Tester.__init__(self)
    
    # virtual function
    def init_tf_parts(self):
        raise NotImplementedError

    # override
    def build_by_file(self, test_data_file, model_dir, model_filename='complEx.ckpt', data_filename='data.bin'):
        """
        load data and model from files
        get self.vec_c (vectors for concepts), and self.vec_r(vectors for relations)
        :return:
        """

        
        import torch

        Tester.build_by_file_no_id_mapping(self, test_data_file, model_dir, model_filename,
                             data_filename)


        # load tf model and embeddings
        model_save_path = join(model_dir, model_filename)
        self.init_tf_parts()
        
        self.tf_parts.batch_size_eval = 2048

        self.tf_parts.cuda()
        self.tf_parts.load_checkpoint(model_save_path)  # load it


        self.vec_c = np.zeros((self.this_data.num_cons(), ))
        self.vec_r = np.zeros((self.this_data.num_rels(), ))
        # self.w = w
        # self.b = b


    def to_var(self, x, use_gpu):

        import torch
        from torch.autograd import Variable

        if use_gpu:
            return Variable(torch.from_numpy(x).cuda())
        else:
            return Variable(torch.from_numpy(x))

    # override
    def get_score(self, h, r, t):
        raise NotImplementedError

    # override
    def get_score_batch(self, h_batch, r_batch, t_batch, isneg2Dbatch=False):
        N = h_batch.shape[0] # batch size
        scores = np.zeros(h_batch.shape) # [batch size] or [batch size, neg sample]
        for base_idx in range(0, N, self.tf_parts.batch_size_eval):
            effective_bs = min(self.tf_parts.batch_size_eval, N - base_idx)

            if isneg2Dbatch:
                # print(r_batch.flatten()[:1000])
                shape = (effective_bs, h_batch.shape[1])
                scores_tmp = -self.tf_parts.predict({
                    'batch_h': self.to_var(h_batch[base_idx:base_idx + effective_bs].flatten(), 1),
                    'batch_t': self.to_var(t_batch[base_idx:base_idx + effective_bs].flatten(), 1),
                    'batch_r': self.to_var(r_batch[base_idx:base_idx + effective_bs].flatten(), 1),
                    'mode': 'tail_batch'
                })
                scores_tmp = np.reshape(scores_tmp, shape)
            else:
                scores_tmp = -self.tf_parts.predict({
                    'batch_h': self.to_var(h_batch[base_idx:base_idx + effective_bs], 1),
                    'batch_t': self.to_var(t_batch[base_idx:base_idx + effective_bs], 1),
                    'batch_r': self.to_var(r_batch[base_idx:base_idx + effective_bs], 1),
                    'mode': 'tail_batch'
                })

            scores[base_idx:base_idx + effective_bs] = scores_tmp

        return scores

class ComplEx_Tester(OpenKE_Tester):
    # override
    def init_tf_parts(self):
        from openke.module.model import ComplEx
        
        self.tf_parts = ComplEx(
            ent_tot = self.this_data.num_cons(),
            rel_tot = self.this_data.num_rels(),
            dim = 512
        )


class DistMult_Tester(OpenKE_Tester):
    # override
    def init_tf_parts(self):
        from openke.module.model import DistMult

        self.tf_parts = DistMult(
            ent_tot = self.this_data.num_cons(),
            rel_tot = self.this_data.num_rels(),
            dim = 512
        )


class RotatE_Tester(OpenKE_Tester):
    # override
    def init_tf_parts(self):
        from openke.module.model import RotatE

        self.tf_parts = RotatE(
            ent_tot = self.this_data.num_cons(),
            rel_tot = self.this_data.num_rels(),
            dim = 512
        )