''' Module for training TF parts.'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import pandas as pd
from os.path import join

from src import param

import sys

if '../src' not in sys.path:
    sys.path.append('../src')

import numpy as np
import tensorflow as tf
import time
from src.data import BatchLoader

from src.utils import vec_length
from src.list import ModelList
from src.models import * #UKGE_logi_TF, UKGE_rect_TF, TransE_m1_TF, DistMult_m1_TF
from src.testers import * #UKGE_logi_Tester, UKGE_rect_Tester, TransE_m1_Tester, DistMult_m1_Tester



class Trainer(object):
    def __init__(self):
        self.batch_size = 128
        self.dim = 64
        self.this_data = None
        self.tf_parts = None
        self.save_path = 'this-distmult.ckpt'
        self.data_save_path = 'this-data.bin'
        self.file_val = ""
        self.L1 = False

    def build(self, data_obj, save_dir,
              model_save='model.bin',
              data_save='data.bin', psl=True, semisupervised_negative_sample=False):
        """
        All files are stored in save_dir.
        output files:
        1. tf model
        2. this_data (Data())
        3. training_loss.csv, val_loss.csv
        :param model_save: filename for model
        :param data_save: filename for self.this_data
        :param knn_neg: use kNN negative sampling
        :return:
        """
        self.verbose = param.verbose  # print extra information
        self.this_data = data_obj
        self.dim = self.this_data.dim = param.dim
        self.batch_size = self.this_data.batch_size = param.batch_size
        self.neg_per_positive = param.neg_per_pos
        self.reg_scale = param.reg_scale
        self.psl = psl
        self.semisupervised_negative_sample = semisupervised_negative_sample
        self.semisupervised_negative_sample_v2 = param.semisupervised_negative_sample_v2
        self.semisupervised_v1 = param.semisupervised_v1
        self.semisupervised_v1_1 = param.semisupervised_v1_1

        self.ss_pool = None # for semisupervised v2, for negative samples (h index, r index, t index, w)
        # self.ss_pool = np.zeros((100000000, 4)) # larger_pool1
        # self.ss_pool2 = np.zeros((, 4)) # for semisupervised v2_3, for negative samples (h index, r index, t index, w)
        self.ss_pool_base = 0 # for semisupervised v2, the base of the ring
        self.ss_pool_end = 1 # for semisupervised v2, the base of the ring

        self.batchloader = BatchLoader(self.this_data, self.batch_size, self.neg_per_positive)

        self.p_neg = param.p_neg
        self.p_psl = param.p_psl

        # paths for saving
        self.save_dir = save_dir
        self.save_path = join(save_dir, model_save)  # tf model
        self.data_save_path = join(save_dir, data_save)  # this_data (Data())
        self.train_loss_path = join(save_dir, 'trainig_loss.csv')
        self.val_loss_path = join(save_dir, 'val_loss.csv')
        self.val_bin_class_path = join(save_dir, 'val_score_binary_classification.csv')
        self.test_loss_path = join(save_dir, 'test_loss.csv')
        self.test_bin_class_path = join(save_dir, 'test_score_binary_classification.csv')

        print('Now using model: ', param.whichmodel)

        self.whichmodel = param.whichmodel

        self.build_tf_parts()  # could be overrided

        # if param.transitional_v0:
        #     self.tester.load_hr_map(param.data_dir(), 'train.tsv')
        #     self.hr_map_train = self.tester.hr_map
        #     self.tester.hr_map = None

    def build_tf_parts(self):
        """
        Build tfparts (model) and validator.
        Different for every model.
        :return:
        """
        if self.whichmodel == ModelList.LOGI:
            self.tf_parts = UKGE_logi_TF(num_rels=self.this_data.num_rels(),
                                         num_cons=self.this_data.num_cons(),
                                         dim=self.dim,
                                         batch_size=self.batch_size,
                                         neg_per_positive=self.neg_per_positive, p_neg=self.p_neg, psl=self.psl)
            self.validator = UKGE_logi_Tester()
            self.tester = UKGE_logi_Tester()

        elif self.whichmodel == ModelList.RECT:
            self.tf_parts = UKGE_rect_TF(num_rels=self.this_data.num_rels(),
                                         num_cons=self.this_data.num_cons(),
                                         dim=self.dim,
                                         batch_size=self.batch_size,
                                         neg_per_positive=self.neg_per_positive, p_neg=self.p_neg, psl=self.psl, reg_scale=self.reg_scale)
            self.validator = UKGE_rect_Tester()
            self.tester = UKGE_rect_Tester()

        elif self.whichmodel == ModelList.TransE_m1 or self.whichmodel == ModelList.TransE:
            self.tf_parts = TransE_m1_TF(num_rels=self.this_data.num_rels(),
                                         num_cons=self.this_data.num_cons(),
                                         dim=self.dim,
                                         batch_size=self.batch_size,
                                         neg_per_positive=self.neg_per_positive, p_neg=self.p_neg, psl=self.psl)
            self.validator = TransE_m1_Tester()
            self.tester = TransE_m1_Tester()


        elif self.whichmodel == ModelList.TransE_m3 or self.whichmodel == ModelList.TransE:
            self.tf_parts = TransE_m3_TF(num_rels=self.this_data.num_rels(),
                                         num_cons=self.this_data.num_cons(),
                                         dim=self.dim,
                                         batch_size=self.batch_size,
                                         neg_per_positive=self.neg_per_positive, p_neg=self.p_neg, psl=self.psl)
            self.validator = TransE_m3_Tester()
            self.tester = TransE_m3_Tester()

        elif self.whichmodel == ModelList.TransE_m3_1:
            self.tf_parts = TransE_m3_1_TF(num_rels=self.this_data.num_rels(),
                                         num_cons=self.this_data.num_cons(),
                                         dim=self.dim,
                                         batch_size=self.batch_size,
                                         neg_per_positive=self.neg_per_positive, p_neg=self.p_neg, psl=self.psl)
            self.validator = TransE_m3_1_Tester()
            self.tester = TransE_m3_1_Tester()


        elif self.whichmodel == ModelList.TransE_m3_3:
            self.tf_parts = TransE_m3_3_TF(num_rels=self.this_data.num_rels(),
                                         num_cons=self.this_data.num_cons(),
                                         dim=self.dim,
                                         batch_size=self.batch_size,
                                         neg_per_positive=self.neg_per_positive, p_neg=self.p_neg, psl=self.psl)
            self.validator = TransE_m3_3_Tester()
            self.tester = TransE_m3_3_Tester()



        elif self.whichmodel == ModelList.DistMult_m1 or self.whichmodel == ModelList.DistMult:
            self.tf_parts = DistMult_m1_TF(num_rels=self.this_data.num_rels(),
                                         num_cons=self.this_data.num_cons(),
                                         dim=self.dim,
                                         batch_size=self.batch_size,
                                         neg_per_positive=self.neg_per_positive, p_neg=self.p_neg, psl=self.psl)
            self.validator = DistMult_m1_Tester()
            self.tester = DistMult_m1_Tester()

        elif self.whichmodel == ModelList.DistMult_m2:
            self.tf_parts = DistMult_m2_TF(num_rels=self.this_data.num_rels(),
                                         num_cons=self.this_data.num_cons(),
                                         dim=self.dim,
                                         batch_size=self.batch_size,
                                         neg_per_positive=self.neg_per_positive, p_neg=self.p_neg, psl=self.psl)
            self.validator = DistMult_m2_Tester()
            self.tester = DistMult_m2_Tester()


        elif self.whichmodel == ModelList.ComplEx_m1 or self.whichmodel == ModelList.ComplEx:
            self.tf_parts = ComplEx_m1_TF(num_rels=self.this_data.num_rels(),
                                         num_cons=self.this_data.num_cons(),
                                         dim=self.dim,
                                         batch_size=self.batch_size,
                                         neg_per_positive=self.neg_per_positive, p_neg=self.p_neg, psl=self.psl)
            self.validator = ComplEx_m1_Tester()
            self.tester = ComplEx_m1_Tester()

        elif self.whichmodel == ModelList.ComplEx_m1_1:
            self.tf_parts = ComplEx_m1_1_TF(num_rels=self.this_data.num_rels(),
                                         num_cons=self.this_data.num_cons(),
                                         dim=self.dim,
                                         batch_size=self.batch_size,
                                         neg_per_positive=self.neg_per_positive, p_neg=self.p_neg, psl=self.psl)
            self.validator = ComplEx_m1_Tester()
            self.tester = ComplEx_m1_Tester()

        elif self.whichmodel == ModelList.ComplEx_m3:
            self.tf_parts = ComplEx_m3_TF(num_rels=self.this_data.num_rels(),
                                         num_cons=self.this_data.num_cons(),
                                         dim=self.dim,
                                         batch_size=self.batch_size,
                                         neg_per_positive=self.neg_per_positive, p_neg=self.p_neg, psl=self.psl)
            self.validator = ComplEx_m3_Tester()
            self.tester = ComplEx_m3_Tester()

        elif self.whichmodel == ModelList.ComplEx_m4:
            self.tf_parts = ComplEx_m4_TF(num_rels=self.this_data.num_rels(),
                                         num_cons=self.this_data.num_cons(),
                                         dim=self.dim,
                                         batch_size=self.batch_size,
                                         neg_per_positive=self.neg_per_positive, p_neg=self.p_neg, psl=self.psl)
            self.validator = ComplEx_m4_Tester()
            self.tester = ComplEx_m4_Tester()


        elif self.whichmodel == ModelList.ComplEx_m5_1:
            self.tf_parts = ComplEx_m5_1_TF(num_rels=self.this_data.num_rels(),
                                         num_cons=self.this_data.num_cons(),
                                         dim=self.dim,
                                         batch_size=self.batch_size,
                                         neg_per_positive=self.neg_per_positive, p_neg=self.p_neg, psl=self.psl)
            self.validator = ComplEx_m4_Tester()
            self.tester = ComplEx_m5_1_Tester()

        elif self.whichmodel == ModelList.ComplEx_m5_2:
            self.tf_parts = ComplEx_m5_2_TF(num_rels=self.this_data.num_rels(),
                                         num_cons=self.this_data.num_cons(),
                                         dim=self.dim,
                                         batch_size=self.batch_size,
                                         neg_per_positive=self.neg_per_positive, p_neg=self.p_neg, psl=self.psl)
            self.validator = ComplEx_m5_2_Tester()
            self.tester = ComplEx_m5_2_Tester()
        
        elif self.whichmodel == ModelList.ComplEx_m5_3:
            self.tf_parts = ComplEx_m5_3_TF(num_rels=self.this_data.num_rels(),
                                         num_cons=self.this_data.num_cons(),
                                         dim=self.dim,
                                         batch_size=self.batch_size,
                                         neg_per_positive=self.neg_per_positive, p_neg=self.p_neg, psl=self.psl)
            self.validator = ComplEx_m5_3_Tester()
            self.tester = ComplEx_m5_3_Tester()

        elif self.whichmodel == ModelList.ComplEx_m5_4:
            self.tf_parts = ComplEx_m5_4_TF(num_rels=self.this_data.num_rels(),
                                         num_cons=self.this_data.num_cons(),
                                         dim=self.dim,
                                         batch_size=self.batch_size,
                                         neg_per_positive=self.neg_per_positive, p_neg=self.p_neg, psl=self.psl)
            self.validator = ComplEx_m5_4_Tester()
            self.tester = ComplEx_m5_4_Tester()


        elif self.whichmodel == ModelList.ComplEx_m6_1:
            self.tf_parts = ComplEx_m6_1_TF(num_rels=self.this_data.num_rels(),
                                         num_cons=self.this_data.num_cons(),
                                         dim=self.dim,
                                         batch_size=self.batch_size,
                                         neg_per_positive=self.neg_per_positive, p_neg=self.p_neg, psl=self.psl)
            self.validator = ComplEx_m6_1_Tester()
            self.tester = ComplEx_m6_1_Tester()

        elif self.whichmodel == ModelList.ComplEx_m6_2:
            self.tf_parts = ComplEx_m6_2_TF(num_rels=self.this_data.num_rels(),
                                         num_cons=self.this_data.num_cons(),
                                         dim=self.dim,
                                         batch_size=self.batch_size,
                                         neg_per_positive=self.neg_per_positive, p_neg=self.p_neg, psl=self.psl)
            self.validator = ComplEx_m6_2_Tester()
            self.tester = ComplEx_m6_2_Tester()

        elif self.whichmodel == ModelList.ComplEx_m7:
            self.tf_parts = ComplEx_m7_TF(num_rels=self.this_data.num_rels(),
                                         num_cons=self.this_data.num_cons(),
                                         dim=self.dim,
                                         batch_size=self.batch_size,
                                         neg_per_positive=self.neg_per_positive, p_neg=self.p_neg, psl=self.psl)
            self.validator = ComplEx_m4_Tester()
            self.tester = ComplEx_m4_Tester()

        elif self.whichmodel == ModelList.ComplEx_m8:
            self.tf_parts = ComplEx_m8_TF(num_rels=self.this_data.num_rels(),
                                         num_cons=self.this_data.num_cons(),
                                         dim=self.dim,
                                         batch_size=self.batch_size,
                                         neg_per_positive=self.neg_per_positive, p_neg=self.p_neg, psl=self.psl)
            self.validator = ComplEx_m4_Tester()
            self.tester = ComplEx_m4_Tester()

        elif self.whichmodel == ModelList.ComplEx_m9_1:
            self.tf_parts = ComplEx_m9_1_TF(num_rels=self.this_data.num_rels(),
                                         num_cons=self.this_data.num_cons(),
                                         dim=self.dim,
                                         batch_size=self.batch_size,
                                         neg_per_positive=self.neg_per_positive, p_neg=self.p_neg, psl=self.psl)
            self.validator = ComplEx_m4_Tester()
            self.tester = ComplEx_m4_Tester()

        elif self.whichmodel == ModelList.ComplEx_m9_2:
            self.tf_parts = ComplEx_m9_2_TF(num_rels=self.this_data.num_rels(),
                                         num_cons=self.this_data.num_cons(),
                                         dim=self.dim,
                                         batch_size=self.batch_size,
                                         neg_per_positive=self.neg_per_positive, p_neg=self.p_neg, psl=self.psl)
            self.validator = ComplEx_m9_2_Tester()
            self.tester = ComplEx_m9_2_Tester()

        elif self.whichmodel == ModelList.ComplEx_m9_3:
            self.tf_parts = ComplEx_m9_3_TF(num_rels=self.this_data.num_rels(),
                                         num_cons=self.this_data.num_cons(),
                                         dim=self.dim,
                                         batch_size=self.batch_size,
                                         neg_per_positive=self.neg_per_positive, p_neg=self.p_neg, psl=self.psl)
            self.validator = ComplEx_m9_3_Tester()
            self.tester = ComplEx_m9_3_Tester()


        elif self.whichmodel == ModelList.ComplEx_m10:
            self.tf_parts = ComplEx_m10_TF(num_rels=self.this_data.num_rels(),
                                         num_cons=self.this_data.num_cons(),
                                         dim=self.dim,
                                         batch_size=self.batch_size,
                                         neg_per_positive=self.neg_per_positive, p_neg=self.p_neg, psl=self.psl)
            self.validator = ComplEx_m10_Tester()
            self.tester = ComplEx_m10_Tester()

        elif self.whichmodel == ModelList.ComplEx_m10_1:
            self.tf_parts = ComplEx_m10_1_TF(num_rels=self.this_data.num_rels(),
                                         num_cons=self.this_data.num_cons(),
                                         dim=self.dim,
                                         batch_size=self.batch_size,
                                         neg_per_positive=self.neg_per_positive, p_neg=self.p_neg, psl=self.psl)
            self.validator = ComplEx_m10_1_Tester()
            self.tester = ComplEx_m10_1_Tester()

        elif self.whichmodel == ModelList.ComplEx_m10_2:
            self.tf_parts = ComplEx_m10_2_TF(num_rels=self.this_data.num_rels(),
                                         num_cons=self.this_data.num_cons(),
                                         dim=self.dim,
                                         batch_size=self.batch_size,
                                         neg_per_positive=self.neg_per_positive, p_neg=self.p_neg, psl=self.psl)
            self.validator = ComplEx_m10_2_Tester()
            self.tester = ComplEx_m10_2_Tester()
        
        elif self.whichmodel == ModelList.ComplEx_m10_3:
            self.tf_parts = ComplEx_m10_3_TF(num_rels=self.this_data.num_rels(),
                                         num_cons=self.this_data.num_cons(),
                                         dim=self.dim,
                                         batch_size=self.batch_size,
                                         neg_per_positive=self.neg_per_positive, p_neg=self.p_neg, psl=self.psl)
            self.validator = ComplEx_m10_3_Tester()
            self.tester = ComplEx_m10_3_Tester()

        elif self.whichmodel == ModelList.ComplEx_m10_4:
            self.tf_parts = ComplEx_m10_4_TF(num_rels=self.this_data.num_rels(),
                                         num_cons=self.this_data.num_cons(),
                                         dim=self.dim,
                                         batch_size=self.batch_size,
                                         neg_per_positive=self.neg_per_positive, p_neg=self.p_neg, psl=self.psl)
            self.validator = ComplEx_m10_4_Tester()
            self.tester = ComplEx_m10_4_Tester()

        




        elif self.whichmodel == ModelList.RotatE_m1:
            self.tf_parts = RotatE_m1_TF(num_rels=self.this_data.num_rels(),
                                         num_cons=self.this_data.num_cons(),
                                         dim=self.dim,
                                         batch_size=self.batch_size,
                                         neg_per_positive=self.neg_per_positive, p_neg=self.p_neg, psl=self.psl, gamma=1)
            self.validator = RotatE_m1_Tester()
            self.tester = RotatE_m1_Tester()

        elif self.whichmodel == ModelList.RotatE_m2:
            self.tf_parts = RotatE_m2_TF(num_rels=self.this_data.num_rels(),
                                         num_cons=self.this_data.num_cons(),
                                         dim=self.dim,
                                         batch_size=self.batch_size,
                                         neg_per_positive=self.neg_per_positive, p_neg=self.p_neg, psl=self.psl, gamma=1)
            self.validator = RotatE_m2_Tester()
            self.tester = RotatE_m2_Tester()

        elif self.whichmodel == ModelList.RotatE_m2_1:
            self.tf_parts = RotatE_m2_1_TF(num_rels=self.this_data.num_rels(),
                                         num_cons=self.this_data.num_cons(),
                                         dim=self.dim,
                                         batch_size=self.batch_size,
                                         neg_per_positive=self.neg_per_positive, p_neg=self.p_neg, psl=self.psl, gamma=1)
            self.validator = RotatE_m2_1_Tester()
            self.tester = RotatE_m2_1_Tester()

        elif self.whichmodel == ModelList.RotatE_m2_2:
            self.tf_parts = RotatE_m2_2_TF(num_rels=self.this_data.num_rels(),
                                         num_cons=self.this_data.num_cons(),
                                         dim=self.dim,
                                         batch_size=self.batch_size,
                                         neg_per_positive=self.neg_per_positive, p_neg=self.p_neg, psl=self.psl, gamma=1)
            self.validator = RotatE_m2_2_Tester()
            self.tester = RotatE_m2_2_Tester()

        elif self.whichmodel == ModelList.RotatE_m3:
            self.tf_parts = RotatE_m3_TF(num_rels=self.this_data.num_rels(),
                                         num_cons=self.this_data.num_cons(),
                                         dim=self.dim,
                                         batch_size=self.batch_size,
                                         neg_per_positive=self.neg_per_positive, p_neg=self.p_neg, psl=self.psl, gamma=1)
            self.validator = ComplEx_m4_Tester()
            self.tester = ComplEx_m4_Tester()

        
        elif self.whichmodel == ModelList.RotatE_m3_1:
            self.tf_parts = RotatE_m3_1_TF(num_rels=self.this_data.num_rels(),
                                         num_cons=self.this_data.num_cons(),
                                         dim=self.dim,
                                         batch_size=self.batch_size,
                                         neg_per_positive=self.neg_per_positive, p_neg=self.p_neg, psl=self.psl, gamma=1)
            self.validator =RotatE_m3_1_Tester()
            self.tester =RotatE_m3_1_Tester()

        elif self.whichmodel == ModelList.RotatE_m3_2:
            self.tf_parts = RotatE_m3_2_TF(num_rels=self.this_data.num_rels(),
                                         num_cons=self.this_data.num_cons(),
                                         dim=self.dim,
                                         batch_size=self.batch_size,
                                         neg_per_positive=self.neg_per_positive, p_neg=self.p_neg, psl=self.psl, gamma=1)
            self.validator =RotatE_m3_2_Tester()
            self.tester =RotatE_m3_2_Tester()

        elif self.whichmodel == ModelList.RotatE_m3_3:
            self.tf_parts = RotatE_m3_3_TF(num_rels=self.this_data.num_rels(),
                                         num_cons=self.this_data.num_cons(),
                                         dim=self.dim,
                                         batch_size=self.batch_size,
                                         neg_per_positive=self.neg_per_positive, p_neg=self.p_neg, psl=self.psl, gamma=1)
            self.validator =RotatE_m3_3_Tester()
            self.tester =RotatE_m3_3_Tester()

        elif self.whichmodel == ModelList.RotatE_m4:
            self.tf_parts = RotatE_m4_TF(num_rels=self.this_data.num_rels(),
                                         num_cons=self.this_data.num_cons(),
                                         dim=self.dim,
                                         batch_size=self.batch_size,
                                         neg_per_positive=self.neg_per_positive, p_neg=self.p_neg, psl=self.psl, gamma=1)
            self.validator = ComplEx_m4_Tester()
            self.tester = ComplEx_m4_Tester()

        elif self.whichmodel == ModelList.RotatE_m5:
            self.tf_parts = RotatE_m5_TF(num_rels=self.this_data.num_rels(),
                                         num_cons=self.this_data.num_cons(),
                                         dim=self.dim,
                                         batch_size=self.batch_size,
                                         neg_per_positive=self.neg_per_positive, p_neg=self.p_neg, psl=self.psl, gamma=2)
            self.validator = RotatE_m5_Tester()
            self.tester = RotatE_m5_Tester()

        elif self.whichmodel == ModelList.RotatE_m5_1:
            self.tf_parts = RotatE_m5_1_TF(num_rels=self.this_data.num_rels(),
                                         num_cons=self.this_data.num_cons(),
                                         dim=self.dim,
                                         batch_size=self.batch_size,
                                         neg_per_positive=self.neg_per_positive, p_neg=self.p_neg, psl=self.psl, gamma=2)
            self.validator = RotatE_m5_1_Tester()
            self.tester = RotatE_m5_1_Tester()



        elif self.whichmodel == ModelList.UKGE_logi_m1:
            self.tf_parts = UKGE_logi_m1_TF(num_rels=self.this_data.num_rels(),
                                         num_cons=self.this_data.num_cons(),
                                         dim=self.dim,
                                         batch_size=self.batch_size,
                                         neg_per_positive=self.neg_per_positive, p_neg=self.p_neg, psl=self.psl)
            self.validator = UKGE_logi_m1_Tester()
            self.tester = UKGE_logi_m1_Tester()

        elif self.whichmodel == ModelList.UKGE_logi_m2:
            self.tf_parts = UKGE_logi_m2_TF(num_rels=self.this_data.num_rels(),
                                         num_cons=self.this_data.num_cons(),
                                         dim=self.dim,
                                         batch_size=self.batch_size,
                                         neg_per_positive=self.neg_per_positive, p_neg=self.p_neg, psl=self.psl)
            self.validator = UKGE_logi_m2_Tester()
            self.tester = UKGE_logi_m2_Tester()

    def gen_batch(self, forever=False, shuffle=True, negsampler=None):
        """
        :param ht_embedding: for kNN negative sampling
        :return:
        """
        l = self.this_data.triples.shape[0]
        while True:
            triples = self.this_data.triples  # np.float64 [[h,r,t,w]]
            if shuffle:
                np.random.shuffle(triples)
            for i in range(0, l, self.batch_size):

                batch = triples[i: i + self.batch_size, :]
                if batch.shape[0] < self.batch_size:
                    batch = np.concatenate((batch, self.this_data.triples[:self.batch_size - batch.shape[0]]), axis=0)
                    assert batch.shape[0] == self.batch_size

                h_batch, r_batch, t_batch, w_batch = batch[:, 0], batch[:, 1], batch[:, 2], batch[:, 3]

                hrt_batch = batch[:, 0:3].astype(int)

                all_neg_hn_batch = self.this_data.corrupt_batch(hrt_batch, self.neg_per_positive, "h")
                all_neg_tn_batch = self.this_data.corrupt_batch(hrt_batch, self.neg_per_positive, "t")

                neg_hn_batch, neg_rel_hn_batch, \
                neg_t_batch, neg_h_batch, \
                neg_rel_tn_batch, neg_tn_batch \
                    = all_neg_hn_batch[:, :, 0], \
                      all_neg_hn_batch[:, :, 1], \
                      all_neg_hn_batch[:, :, 2], \
                      all_neg_tn_batch[:, :, 0], \
                      all_neg_tn_batch[:, :, 1], \
                      all_neg_tn_batch[:, :, 2]
                yield h_batch.astype(np.int64), r_batch.astype(np.int64), t_batch.astype(np.int64), w_batch.astype(
                    np.float32), \
                      neg_hn_batch.astype(np.int64), neg_rel_hn_batch.astype(np.int64), \
                      neg_t_batch.astype(np.int64), neg_h_batch.astype(np.int64), \
                      neg_rel_tn_batch.astype(np.int64), neg_tn_batch.astype(np.int64)
            if not forever:
                break


    def pool_based_semisupervised(self, sess, batch, n_generated_samples, n_new_samples, n_semi_samples, pool_size, sample_balance_for_semisuper_v0 = False):
        # @param n_generated_samples  # of negative and semisupervised samples used for training
        # @param n_new_samples  new sample for the pool
        # @param n_semi_samples  # of semisupervised samples used for training

        A_h_index, A_r_index, A_t_index, A_w, \
            A_neg_hn_index, A_neg_rel_hn_index, \
            A_neg_t_index, A_neg_h_index, A_neg_rel_tn_index, A_neg_tn_index = batch

        if self.ss_pool is None:
            self.ss_pool = np.zeros((pool_size, 4))
        
        
        # generate new semisupervised sample
        self.validator.build_by_var(None, self.tf_parts, None, sess = sess)
        
        new_semi_samples = np.zeros((n_generated_samples, 4))
        # new_semi_samples[:, 0] = new_samples_h_index = np.random.choice(self.this_data.num_cons(), n_generated_samples, replace=True)
        # new_semi_samples[:, 1] = new_samples_r_index = np.random.choice(self.this_data.num_rels(), n_generated_samples, replace=True)
        # new_semi_samples[:, 2] = new_samples_t_index = np.random.choice(self.this_data.num_cons(), n_generated_samples, replace=True)
        new_semi_samples[:, 0] = new_samples_h_index = np.concatenate((A_neg_hn_index.flatten(), A_neg_h_index.flatten()))
        new_semi_samples[:, 1] = new_samples_r_index = np.concatenate((A_neg_rel_hn_index.flatten(), A_neg_rel_tn_index.flatten()))
        new_semi_samples[:, 2] = new_samples_t_index = np.concatenate((A_neg_t_index.flatten(), A_neg_tn_index.flatten()))
        new_semi_samples[:, 3] = self.validator.get_score_batch(new_samples_h_index, new_samples_r_index, new_samples_t_index, isneg2Dbatch=False)

        if sample_balance_for_semisuper_v0:
            mask = new_samples_r_index != 0
            n_filtered_samples = sum(mask)
            
            new_semi_samples[0:n_filtered_samples] = new_semi_samples[mask]

            n_new_samples = n_filtered_samples


        


        # copy new samples to pool
        new_neg_pool_end = self.ss_pool_end + n_new_samples
        remained = max(0, new_neg_pool_end - self.ss_pool.shape[0])
        self.ss_pool[self.ss_pool_end:min(new_neg_pool_end, self.ss_pool.shape[0])] = new_semi_samples[0:n_new_samples-remained]
        self.ss_pool[0:remained] = new_semi_samples[n_new_samples-remained:n_new_samples]
        # update base and end of pool
        if self.ss_pool_end > self.ss_pool_base:
            self.ss_pool_base += self.ss_pool.shape[0]
        self.ss_pool_end = new_neg_pool_end
        self.ss_pool_base = max(self.ss_pool_base, self.ss_pool_end)%self.ss_pool.shape[0]
        self.ss_pool_end %= self.ss_pool.shape[0]

        # fetch existing semisupervised sample from the pool
        # WARNING: assume no deletion from pool!!!
        tmp = np.random.randint(
                (self.ss_pool_end if self.ss_pool_base < self.ss_pool_end else self.ss_pool.shape[0]) - 
                n_semi_samples
            )
        selected_samples = np.arange(tmp, tmp + n_semi_samples)
        # if self.ss_pool_base < self.ss_pool_end: 
        #     selected_samples = np.random.choice(self.ss_pool_end, n_semi_samples, replace=False)
        # else: # == 
        #     selected_samples = np.random.choice(self.ss_pool.shape[0], n_semi_samples, replace=False)
        
        A_semi_h_index = new_samples_h_index
        A_semi_r_index = new_samples_r_index
        A_semi_t_index = new_samples_t_index
        A_semi_w = np.zeros((n_generated_samples,))
        if n_semi_samples > 0:
            A_semi_h_index[0:n_semi_samples] = self.ss_pool[selected_samples, 0]
            A_semi_r_index[0:n_semi_samples] = self.ss_pool[selected_samples, 1]
            A_semi_t_index[0:n_semi_samples] = self.ss_pool[selected_samples, 2]
            A_semi_w[:n_semi_samples] = self.ss_pool[selected_samples, 3]
        feed_dict_gen = {self.tf_parts._A_semi_h_index: A_semi_h_index,
            self.tf_parts._A_semi_r_index: A_semi_r_index,
            self.tf_parts._A_semi_t_index: A_semi_t_index,
            self.tf_parts._A_semi_w: A_semi_w
            }
    
        # feed_dict = {**feed_dict, **feed_dict_psl}

        # if batch_id == 0:
        #     print(self.ss_pool_end, self.ss_pool_base)

        return feed_dict_gen


    def train(self, epochs=20, save_every_epoch=20, lr=0.001, data_dir="", resume_model_path=None):
        sess = tf.Session()  # show device info
        # gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.3)
        # gpu_options = tf.GPUOptions(allow_growth=True)
        # sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True, gpu_options=gpu_options))

        sess.run(tf.global_variables_initializer())

        if resume_model_path is not None:
            self.tf_parts._saver.restore(sess, resume_model_path)

        num_batch = self.this_data.triples.shape[0] // self.batch_size
        print('Number of batches per epoch: %d' % num_batch)


        train_losses = []  # [[every epoch, loss]]
        val_losses = []  # [[saver epoch, loss]]
        val_losses_bin_classify = []  # [[saver epoch, loss]]
        test_losses = [] # same with val_loss
        test_losses_bin_classify = [] # same with val_loss

        for epoch in range(1, epochs + 1):
            epoch_loss = self.train1epoch(sess, num_batch, lr, epoch)
            train_losses.append([epoch, epoch_loss])

            if np.isnan(epoch_loss):
                print("Nan loss. Training collapsed.")
                return

            if epoch % save_every_epoch == 10:
                # save model
                this_save_path = self.tf_parts._saver.save(sess, self.save_path, global_step=epoch)  # save model
                self.this_data.save(self.data_save_path)  # save data
                print('VALIDATE AND SAVE MODELS:')
                print("Model saved in file: %s. Data saved in file: %s" % (this_save_path, self.data_save_path))

                # validation error
                val_loss, val_loss_neg, mean_ndcg, mean_exp_ndcg, bin_classify_score, mean_ndcg_r = self.get_val_loss(epoch,
                                                                                          sess)  # loss for testing triples and negative samples
                val_losses.append([epoch, val_loss, val_loss_neg, mean_ndcg, mean_exp_ndcg])
                val_losses_bin_classify.append(np.insert(bin_classify_score, 0, epoch))


                # test_loss, test_loss_neg, test_mean_ndcg, test_mean_exp_ndcg, test_bin_classify_score, test_mean_ndcg_r = self.get_test_loss(epoch,
                #                                                                           sess)  # loss for testing triples and negative samples
                # test_losses.append([epoch, test_loss, test_loss_neg, test_mean_ndcg, test_mean_exp_ndcg])
                # test_losses_bin_classify.append(np.insert(test_bin_classify_score, 0, epoch))

                
                # save and print metrics
                self.save_loss(train_losses, self.train_loss_path, columns=['epoch', 'training_loss'])
                self.save_loss(val_losses, self.val_loss_path,
                               columns=['val_epoch', 'mse', 'mse_neg', 'ndcg(linear)', 'ndcg(exp)'])
                self.save_loss(val_losses_bin_classify, self.val_bin_class_path,
                               columns=['val_epoch', 'P_0.5', 'P_0.6', 'P_0.7', 'R_0.5', 'R_0.6', 'R_0.7', 
                                        'F1_0.5', 'F1_0.6', 'F1_0.7', 'Acc_0.5', 'Acc_0.6', 'Acc_0.7'])
                # self.save_loss(test_losses, self.test_loss_path,
                #                columns=['test_epoch', 'mse', 'mse_neg', 'ndcg(linear)', 'ndcg(exp)'])
                # self.save_loss(test_losses_bin_classify, self.test_bin_class_path,
                #                columns=['test_epoch', 'P_0.5', 'P_0.6', 'P_0.7', 'R_0.5', 'R_0.6', 'R_0.7', 
                #                         'F1_0.5', 'F1_0.6', 'F1_0.7', 'Acc_0.5', 'Acc_0.6', 'Acc_0.7'])



                

        this_save_path = self.tf_parts._saver.save(sess, self.save_path)
        with sess.as_default():
            ht_embeddings = self.tf_parts._ht.eval()
            r_embeddings = self.tf_parts._r.eval()
        print("Model saved in file: %s" % this_save_path)
        sess.close()
        return ht_embeddings, r_embeddings

    def get_val_loss(self, epoch, sess):
        # validation error

        self.validator.build_by_var(self.this_data.val_triples, self.tf_parts, self.this_data, sess=sess)

        if not hasattr(self.validator, 'hr_map'):
            self.validator.load_hr_map(param.data_dir(), 'val.tsv', ['train.tsv', 'val.tsv',
                                                                          'test.tsv'])
        if not hasattr(self.validator, 'hr_map_sub'):
            hr_map200 = self.validator.get_fixed_hr(n=200)  # use smaller size for faster validation
        else:
            hr_map200 = self.validator.hr_map_sub

        mean_ndcg, mean_exp_ndcg, mean_ndcg_r, _, _ = self.validator.mean_ndcg(hr_map200)

        # metrics: mse
        mse = self.validator.get_mse(save_dir=self.save_dir, epoch=epoch, toprint=self.verbose)
        mse_neg = self.validator.get_mse_neg(self.neg_per_positive)

        # metrics: triple classification
        scores, P, R, F1, Acc = self.validator.classify_triples(0.7, [0.5, 0.6, 0.7])


        return mse, mse_neg, mean_ndcg, mean_exp_ndcg, np.ndarray.flatten(np.array([P, R, F1, Acc])), mean_ndcg_r
    def get_test_loss(self, epoch, sess):
        # validation error

        self.tester.build_by_var(self.this_data.test_triples, self.tf_parts, self.this_data, sess=sess)

        if not hasattr(self.tester, 'hr_map'):
            self.tester.load_hr_map(param.data_dir(), 'test.tsv', ['train.tsv', 'val.tsv',
                                                                          'test.tsv'])
        if not hasattr(self.tester, 'hr_map_sub'):
            hr_map200 = self.tester.get_fixed_hr(n=200)  # use smaller size for faster validation
        else:
            hr_map200 = self.tester.hr_map_sub

        mean_ndcg, mean_exp_ndcg, mean_ndcg_r, _, _ = self.tester.mean_ndcg(hr_map200)

        # metrics: mse
        mse = self.tester.get_mse(save_dir=self.save_dir, epoch=epoch, toprint=self.verbose)
        mse_neg = self.tester.get_mse_neg(self.neg_per_positive)

        scores, P, R, F1, Acc = self.tester.classify_triples(0.7, [0.5, 0.6, 0.7])

        return mse, mse_neg, mean_ndcg, mean_exp_ndcg, np.ndarray.flatten(np.array([P, R, F1, Acc])), mean_ndcg_r

    def save_loss(self, losses, filename, columns):
        df = pd.DataFrame(losses, columns=columns)
        print(df.tail(5))
        df.to_csv(filename, index=False)

    def train1epoch(self, sess, num_batch, lr, epoch):
        batch_time = 0

        epoch_batches = self.batchloader.gen_batch(forever=True)

        epoch_loss = []

        for batch_id in range(num_batch):

            batch = next(epoch_batches)
            A_h_index, A_r_index, A_t_index, A_w, \
            A_neg_hn_index, A_neg_rel_hn_index, \
            A_neg_t_index, A_neg_h_index, A_neg_rel_tn_index, A_neg_tn_index = batch

            time00 = time.time()


            
            
            feed_dict = {self.tf_parts._A_h_index: A_h_index,
                           self.tf_parts._A_r_index: A_r_index,
                           self.tf_parts._A_t_index: A_t_index,
                           self.tf_parts._A_w: A_w,
                           self.tf_parts._A_neg_hn_index: A_neg_hn_index,
                           self.tf_parts._A_neg_rel_hn_index: A_neg_rel_hn_index,
                           self.tf_parts._A_neg_t_index: A_neg_t_index,
                           self.tf_parts._A_neg_h_index: A_neg_h_index,
                           self.tf_parts._A_neg_rel_tn_index: A_neg_rel_tn_index,
                           self.tf_parts._A_neg_tn_index: A_neg_tn_index,
                           self.tf_parts._lr: lr,
                           
                           }
            # drop samples of relations that is too frequently seen
            if param.sample_balance_v0:
                mask1 = A_r_index != 1
                mask2 = A_r_index == 1
                mask3 = np.random.choice(sum(mask2), int(sum(mask2)*0.2))
                A_h_index = np.concatenate((A_h_index[mask1], A_h_index[mask2][mask3]))
                A_r_index = np.concatenate((A_r_index[mask1], A_r_index[mask2][mask3]))
                A_t_index = np.concatenate((A_t_index[mask1], A_t_index[mask2][mask3]))
                A_w = np.concatenate((A_w[mask1], A_w[mask2][mask3]))
                A_neg_hn_index = np.concatenate((A_neg_hn_index[mask1], A_neg_hn_index[mask2][mask3]))
                A_neg_rel_hn_index = np.concatenate((A_neg_rel_hn_index[mask1], A_neg_rel_hn_index[mask2][mask3]))
                A_neg_t_index = np.concatenate((A_neg_t_index[mask1], A_neg_t_index[mask2][mask3]))
                A_neg_h_index = np.concatenate((A_neg_h_index[mask1], A_neg_h_index[mask2][mask3]))
                A_neg_rel_tn_index = np.concatenate((A_neg_rel_tn_index[mask1], A_neg_rel_tn_index[mask2][mask3]))
                A_neg_tn_index = np.concatenate((A_neg_tn_index[mask1], A_neg_tn_index[mask2][mask3]))

                feed_dict = {self.tf_parts._A_h_index: A_h_index,
                           self.tf_parts._A_r_index: A_r_index,
                           self.tf_parts._A_t_index: A_t_index,
                           self.tf_parts._A_w: A_w,
                           self.tf_parts._A_neg_hn_index: A_neg_hn_index,
                           self.tf_parts._A_neg_rel_hn_index: A_neg_rel_hn_index,
                           self.tf_parts._A_neg_t_index: A_neg_t_index,
                           self.tf_parts._A_neg_h_index: A_neg_h_index,
                           self.tf_parts._A_neg_rel_tn_index: A_neg_rel_tn_index,
                           self.tf_parts._A_neg_tn_index: A_neg_tn_index,
                           self.tf_parts._lr: lr,
                        }

            if param.sample_balance_v0_1:
                mask1 = A_r_index != 1
                mask2 = A_r_index == 1
                mask3 = np.random.choice(sum(mask2), int(sum(mask2)*0.5))
                
                A_h_index = np.concatenate((A_h_index[mask1], A_h_index[mask2][mask3]))
                A_r_index = np.concatenate((A_r_index[mask1], A_r_index[mask2][mask3]))
                A_t_index = np.concatenate((A_t_index[mask1], A_t_index[mask2][mask3]))
                A_w = np.concatenate((A_w[mask1], A_w[mask2][mask3]))
                A_neg_hn_index = np.concatenate((A_neg_hn_index[mask1], A_neg_hn_index[mask2][mask3]))
                A_neg_rel_hn_index = np.concatenate((A_neg_rel_hn_index[mask1], A_neg_rel_hn_index[mask2][mask3]))
                A_neg_t_index = np.concatenate((A_neg_t_index[mask1], A_neg_t_index[mask2][mask3]))
                A_neg_h_index = np.concatenate((A_neg_h_index[mask1], A_neg_h_index[mask2][mask3]))
                A_neg_rel_tn_index = np.concatenate((A_neg_rel_tn_index[mask1], A_neg_rel_tn_index[mask2][mask3]))
                A_neg_tn_index = np.concatenate((A_neg_tn_index[mask1], A_neg_tn_index[mask2][mask3]))

                feed_dict = {self.tf_parts._A_h_index: A_h_index,
                           self.tf_parts._A_r_index: A_r_index,
                           self.tf_parts._A_t_index: A_t_index,
                           self.tf_parts._A_w: A_w,
                           self.tf_parts._A_neg_hn_index: A_neg_hn_index,
                           self.tf_parts._A_neg_rel_hn_index: A_neg_rel_hn_index,
                           self.tf_parts._A_neg_t_index: A_neg_t_index,
                           self.tf_parts._A_neg_h_index: A_neg_h_index,
                           self.tf_parts._A_neg_rel_tn_index: A_neg_rel_tn_index,
                           self.tf_parts._A_neg_tn_index: A_neg_tn_index,
                           self.tf_parts._lr: lr,
                        }
            
            if self.semisupervised_negative_sample:
                # semi-supervised negative sampling
                self.validator.build_by_var(None, self.tf_parts, None, sess=sess)
                B_neg_hn_w = self.validator.get_score_batch(A_neg_hn_index, A_neg_rel_hn_index, A_neg_t_index, isneg2Dbatch=True)
                B_neg_tn_w = self.validator.get_score_batch(A_neg_h_index, A_neg_rel_tn_index, A_neg_tn_index, isneg2Dbatch=True)
                
                non_semi_supervised_sample = round(max(4, (1- epoch/100)*self.neg_per_positive))
                B_neg_hn_w[np.random.choice(self.neg_per_positive, non_semi_supervised_sample, replace=False)] = 0
                B_neg_tn_w[np.random.choice(self.neg_per_positive, non_semi_supervised_sample, replace=False)] = 0

                feed_dict_semi = {self.tf_parts._B_neg_hn_w: B_neg_hn_w,
                           self.tf_parts._B_neg_tn_w: B_neg_tn_w}

                feed_dict = {**feed_dict, **feed_dict_semi}

            if self.semisupervised_negative_sample_v2:
                # semi-supervised negative sampling
                self.validator.build_by_var(None, self.tf_parts, None, sess=sess)
                B_neg_hn_w = self.validator.get_score_batch(A_neg_hn_index, A_neg_rel_hn_index, A_neg_t_index, isneg2Dbatch=True)
                B_neg_tn_w = self.validator.get_score_batch(A_neg_h_index, A_neg_rel_tn_index, A_neg_tn_index, isneg2Dbatch=True)
                
                non_semi_supervised_sample = round(max(4, (1- epoch/100)*self.neg_per_positive))
                B_neg_hn_w[:, np.random.choice(self.neg_per_positive, non_semi_supervised_sample, replace=False)] = 0
                B_neg_tn_w[:, np.random.choice(self.neg_per_positive, non_semi_supervised_sample, replace=False)] = 0

                feed_dict_semi = {self.tf_parts._B_neg_hn_w: B_neg_hn_w,
                           self.tf_parts._B_neg_tn_w: B_neg_tn_w}

                feed_dict = {**feed_dict, **feed_dict_semi}


            if param.semisupervised_v1:
                # semi-supervised negative sampling
                self.validator.build_by_var(None, self.tf_parts, None, sess=sess)
                B_neg_hn_w = self.validator.get_score_batch(A_neg_hn_index, A_neg_rel_hn_index, A_neg_t_index, isneg2Dbatch=True)
                B_neg_tn_w = self.validator.get_score_batch(A_neg_h_index, A_neg_rel_tn_index, A_neg_tn_index, isneg2Dbatch=True)
                B_w = self.validator.get_score_batch(A_h_index, A_r_index, A_t_index)
                
                feed_dict_semi = {self.tf_parts._B_neg_hn_w: B_neg_hn_w,
                           self.tf_parts._B_neg_tn_w: B_neg_tn_w,
                           self.tf_parts._B_w: B_w}

                feed_dict = {**feed_dict, **feed_dict_semi}

            if param.semisupervised_v1_1:
                # semi-supervised negative sampling
                self.validator.build_by_var(None, self.tf_parts, None, sess=sess)
                B_neg_hn_w = self.validator.get_score_batch(A_neg_hn_index, A_neg_rel_hn_index, A_neg_t_index, isneg2Dbatch=True)
                B_neg_tn_w = self.validator.get_score_batch(A_neg_h_index, A_neg_rel_tn_index, A_neg_tn_index, isneg2Dbatch=True)
                B_w = self.validator.get_score_batch(A_h_index, A_r_index, A_t_index)
                
                non_semi_supervised_sample_neg = round(max(4, (1- epoch/100)*self.neg_per_positive))
                non_semi_supervised_sample_pos = round(max(int(self.batch_size*0.2), (1- epoch/100)*self.batch_size))
                B_neg_hn_w[:, np.random.choice(self.neg_per_positive, non_semi_supervised_sample_neg, replace=False)] = 0
                B_neg_tn_w[:, np.random.choice(self.neg_per_positive, non_semi_supervised_sample_neg, replace=False)] = 0
                B_w[np.random.choice(self.batch_size, non_semi_supervised_sample_pos, replace=False)] = 1

                feed_dict_semi = {self.tf_parts._B_neg_hn_w: B_neg_hn_w,
                           self.tf_parts._B_neg_tn_w: B_neg_tn_w,
                           self.tf_parts._B_w: B_w}

                feed_dict = {**feed_dict, **feed_dict_semi}
                

            # pool-based semi-supervised learning
            
            if param.semisupervised_v2:
                n_generated_samples = self.batch_size*self.neg_per_positive*2                     # semi sample used for training
                n_new_samples       = self.batch_size*self.neg_per_positive if epoch >= 20 else 0  # new sample for the pool
                # original (M_0.8)
                n_semi_samples      = int(min(n_generated_samples*0.8, max(0, -30 + epoch)*n_generated_samples*0.02))
                # M_1.0
                # n_semi_samples      = int(min(n_generated_samples, max(0, -30 + epoch)*n_generated_samples*0.02))
                # M_0.9
                # n_semi_samples      = int(min(n_generated_samples*0.9, max(0, -30 + epoch)*n_generated_samples*0.02))
                # M_0.5
                # n_semi_samples      = int(min(n_generated_samples*0.5, max(0, -30 + epoch)*n_generated_samples*0.02))
                # M_0.3
                # n_semi_samples      = int(min(n_generated_samples*0.3, max(0, -30 + epoch)*n_generated_samples*0.02))
                

                if param.sample_balance_for_semisuper_v0:
                    feed_dict_gen = self.pool_based_semisupervised(sess, batch, n_generated_samples, n_new_samples, n_semi_samples, 10000000, True)
                else:  # normal
                    feed_dict_gen = self.pool_based_semisupervised(sess, batch, n_generated_samples, n_new_samples, n_semi_samples, 10000000)

                feed_dict = {**feed_dict, **feed_dict_gen}
                    


            # pool-based semi-supervised learning - constant N_{semi}
            if param.semisupervised_v2_2:

                n_generated_samples = self.batch_size*self.neg_per_positive*2                     # semi sample used for training
                n_new_samples       = self.batch_size*self.neg_per_positive  # new sample for the pool
                n_semi_samples      = int(n_semi_samples*0.8) if epoch > 10 else 0
                
                feed_dict_gen = self.pool_based_semisupervised(sess, batch, n_generated_samples, n_new_samples, n_semi_samples, 10000000)

                feed_dict = {**feed_dict, **feed_dict_gen}
                

                
                
            if param.semisupervised_v2_3:
                n_generated_samples = self.batch_size*self.neg_per_positive*2                     # semi sample used for training
                n_new_samples       = self.batch_size*self.neg_per_positive if epoch >= 200 else 0  # new sample for the pool
                # original
                # n_semi_samples = int(min(n_semi_samples*0.8, max(0, -220 + epoch)*n_semi_samples*0.0025))
                # v20200222
                # n_semi_samples = int(min(n_semi_samples*0.8, max(0, -1000 + epoch)*n_semi_samples*0.00125))
                # v20200223
                # n_semi_samples = int(min(n_semi_samples*0.8, max(0, -2000 + epoch)*n_semi_samples*0.00125))
                # v20200226
                n_semi_samples = int(min(n_semi_samples*0.8, max(0, -3000 + epoch)*n_semi_samples*0.00125))
                
                feed_dict_gen = self.pool_based_semisupervised(sess, batch, n_generated_samples, n_new_samples, n_semi_samples, self.batch_size*self.neg_per_positive*900*10)

                feed_dict = {**feed_dict, **feed_dict_gen}

                
            # if param.transitional_v0:
            #     #
            #     # (a) --------w-------> (c)
            #     #    --r1--> (b) --r2-->
            #     pass



            if self.psl:
                soft_h_index, soft_r_index, soft_t_index, soft_w_index = self.batchloader.gen_psl_samples()  # length: param.n_psl
                batch_time += time.time() - time00
                
                feed_dict_psl = {self.tf_parts._soft_h_index: soft_h_index,
                    self.tf_parts._soft_r_index: soft_r_index,
                    self.tf_parts._soft_t_index: soft_t_index,
                    self.tf_parts._soft_w: soft_w_index
                    }
            
                feed_dict = {**feed_dict, **feed_dict_psl}
                
                _, gradient, batch_loss, psl_mse, mse_pos, mse_neg, main_loss, psl_prob, psl_mse_each, rule_prior = sess.run(
                [self.tf_parts._train_op, self.tf_parts._gradient,
                 self.tf_parts._A_loss, self.tf_parts.psl_mse, 
                 self.tf_parts._f_score_h, self.tf_parts._f_score_hn,
                 self.tf_parts.main_loss, self.tf_parts.psl_prob, self.tf_parts.psl_error_each,
                 self.tf_parts.prior_psl0],
                feed_dict=feed_dict)
                
                param.prior_psl = rule_prior
                
            else:
                batch_time += time.time() - time00
                _, gradient, batch_loss, mse_pos, mse_neg, main_loss = sess.run(
                [self.tf_parts._train_op, self.tf_parts._gradient,
                 self.tf_parts._A_loss, self.tf_parts._f_score_h, self.tf_parts._f_score_hn,
                 self.tf_parts.main_loss],
                feed_dict=feed_dict)
                
            

            
            
            epoch_loss.append(batch_loss)

            if ((batch_id + 1) % 50 == 0) or batch_id == num_batch - 1:
                print('process: %d / %d. Epoch %d' % (batch_id + 1, num_batch, epoch))

        this_total_loss = np.sum(epoch_loss) / len(epoch_loss)
        print("Loss of epoch %d = %s" % (epoch, np.sum(this_total_loss)))
        # print('MSE on positive instances: %f, MSE on negative samples: %f' % (np.mean(mse_pos), np.mean(mse_neg)))

        return this_total_loss