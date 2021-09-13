import os
import sys
from analysisUtil import *

"""
generate summary data

results_dirs: the directory to include in the summary
openke_models: subfolder names of the openke models to include in the summary
models: subfolder names of the PASSLEAF models to include in the summary

output: csv to stdout
"""
# result_dirs = ['trained_model_batch512_dim_512_semisupervised_neg_v2', 'trained_model_batch512_dim_512_semisupervised_v2_2',
# 'trained_model_batch512_dim_512', 'trained_model_openke']
# # , 'RotatE_m3_3_0306' 'RotatE_m3_3_0304',  'RotatE_m5_0412', 
# result_dirs = ['trained_model_batch512_dim_512_semisupervised_v2_M_1.0', 'trained_model_batch512_dim_512_semisupervised_v2', 'trained_model_batch512_dim_512']
# 'trained_model_batch512_dim_512_semisupervised_v2_M_0.3', 
result_dirs = ['trained_model_batch512_dim_512_semisupervised_v2', 'trained_model_batch512_dim_512']

openke_models = ['complEx', 'distMult', 'rotatE']

# models = ['RotatE_m5_0415', 'RotatE_m5_1_0416', 'RotatE_m3_1_0116', 'RotatE_m3_3_0414', 
# 'logi_0117', 'UKGE_logi_m2_0306', 'UKGE_logi_m2_0301', 'UKGE_logi_m2_0415', 
# 'ComplEx_m5_4_0306', 'ComplEx_m5_4_0301', 'ComplEx_m5_4_0414', 'ComplEx_m5_4_0422', 'ComplEx_m5_1_0117', 
# 'ComplEx_m6_1_0304', 'ComplEx_m6_1_0308']
# models = ['ComplEx_m5_4_0531', 'ComplEx_m5_4_0527', 'ComplEx_m5_4_0306', 'ComplEx_m5_4_0301', 'ComplEx_m5_4_0414', 'ComplEx_m5_4_0422', 'ComplEx_m5_1_0117']
models = ['ComplEx_m5_4_0531', 'ComplEx_m5_4_0527', 'ComplEx_m5_4_0306', 'ComplEx_m5_4_0301', 'ComplEx_m5_4_0414', 'ComplEx_m5_4_0422', 'ComplEx_m5_1_0117',
'logi_0117', 'UKGE_logi_m2_0306', 'UKGE_logi_m2_0301', 'UKGE_logi_m2_0415', 
 'RotatE_m5_0415', 'RotatE_m5_1_0416', 'RotatE_m3_1_0116', 'RotatE_m3_3_0414', 
]

print(','.join(SUMMARY_COLUMNS))
for result_dir in result_dirs:
    for dataset in os.listdir(result_dir):

        dataset_path = os.path.join(result_dir, dataset)
        dataset = dataset[0:5]
        
        for model in models:

            model_path = os.path.join(dataset_path, model)
            if not os.path.exists(model_path):
                continue


            for result in os.listdir(model_path):

                # selection criteria: nDCG
                if 'val_loss' in result:

                    loss_val_file_path = os.path.join(dataset_path, model, result)
                    
                    best_epoch, _ = getBestRecord(loss_val_file_path, 'ndcg', 'ndcg(linear)', True)

                    loss_test_file_path = os.path.join(dataset_path, model, 'test_mean_rank_accurate.csv')
                    if not os.path.exists(loss_test_file_path):
                        print(loss_test_file_path, 'not exists!', file=sys.stderr)
                        continue

                    best_record_mean_rank = getRecord(loss_test_file_path, 'mean_rank', best_epoch)

                    loss_test_file_path = os.path.join(dataset_path, model, 'test_loss_accurate.csv')
                    if not os.path.exists(loss_test_file_path):
                        print(loss_test_file_path, 'not exists!', file=sys.stderr)
                        continue

                    best_record_ndcg = getRecord(loss_test_file_path, 'ndcg', best_epoch)


                    loss_test_file_path = os.path.join(dataset_path, model, 'test_loss_training_included.csv')
                    if not os.path.exists(loss_test_file_path):
                        print(loss_test_file_path, 'not exists!', file=sys.stderr)
                        continue

                    best_record_ndcg_training_included = getRecord(loss_test_file_path, 'ndcg', best_epoch)

                    if best_record_ndcg_training_included is None: best_record_ndcg_training_included = ['-1','-1']

                    


                    if best_record_mean_rank != None:
                        print(','.join(['0', 'nDCG', dataset, model] + best_record_mean_rank + best_record_ndcg + best_record_ndcg_training_included[-2:]))



                
                # selection criteria: hit@20
                if ('val_mean_rank' in result):
                

                    loss_val_file_path = os.path.join(dataset_path, model, result)
                    
                    best_epoch, _ = getBestRecord(loss_val_file_path, 'mean_rank', 'mean_hit@20')

                    loss_test_file_path = os.path.join(dataset_path, model, 'test_mean_rank_accurate.csv')
                    if not os.path.exists(loss_test_file_path):
                        print(loss_test_file_path, 'not exists!', file=sys.stderr)
                        continue

                    best_record_mean_rank = getRecord(loss_test_file_path, 'mean_rank', best_epoch)


                    loss_test_file_path = os.path.join(dataset_path, model, 'test_loss_accurate.csv')
                    if not os.path.exists(loss_test_file_path):
                        print(loss_test_file_path, 'not exists!', file=sys.stderr)
                        continue

                    best_record_ndcg = getRecord(loss_test_file_path, 'ndcg', best_epoch)


                    loss_test_file_path = os.path.join(dataset_path, model, 'test_loss_training_included.csv')
                    if not os.path.exists(loss_test_file_path):
                        print(loss_test_file_path, 'not exists!', file=sys.stderr)
                        continue

                    best_record_ndcg_training_included = getRecord(loss_test_file_path, 'ndcg', best_epoch)

                    if best_record_ndcg_training_included is None: best_record_ndcg_training_included = ['-1','-1']

                    


                    if best_record_mean_rank != None:
                        print(','.join(['0', 'hit@20', dataset, model] + best_record_mean_rank + best_record_ndcg + best_record_ndcg_training_included[-2:]))


                # selection criteria: hit@20
                if ('_mean_rank' in result and 'val_threshold_' in result and (not (model in openke_models and '_loss.csv' in result) )):
                    # val_threshold_0.x_mean_rank_accurate.csv for openke
                    # or val_threshold_0.x_mean_rank.csv for others



                    t = len('val_threshold_')
                    threshold = result[t: t+3]


                    loss_val_file_path = os.path.join(dataset_path, model, result)


                    best_epoch, _ = getBestRecord(loss_val_file_path, 'mean_rank', 'mean_hit@20')

                    loss_test_file_path = os.path.join(dataset_path, model, 'test_threshold_%s_mean_rank_accurate.csv'%(threshold))
                    if not os.path.exists(loss_test_file_path):
                        print(loss_test_file_path, 'not exists!!!', file=sys.stderr)
                        continue

                    best_record_mean_rank = getRecord(loss_test_file_path, 'mean_rank', best_epoch)
                    

                    loss_test_file_path = os.path.join(dataset_path, model, 'test_threshold_%s_loss_accurate.csv'%(threshold))
                    if not os.path.exists(loss_test_file_path):
                        print(loss_test_file_path, 'not exists!!!', file=sys.stderr)
                        continue

                    best_record_ndcg = getRecord(loss_test_file_path, 'ndcg', best_epoch)


                    if best_record_mean_rank != None:
                        print(','.join([str(threshold), 'hit@20', dataset, model] + best_record_mean_rank + best_record_ndcg))



                # selection criteria: mse
                if 'val_loss' in result:

                    loss_val_file_path = os.path.join(dataset_path, model, result)
                    
                    best_epoch, _ = getBestRecord(loss_val_file_path, 'ndcg', 'mse', False)

                    loss_test_file_path = os.path.join(dataset_path, model, 'test_mean_rank_accurate.csv')
                    if not os.path.exists(loss_test_file_path):
                        print(loss_test_file_path, 'not exists!', file=sys.stderr)
                        continue

                    best_record_mean_rank = getRecord(loss_test_file_path, 'mean_rank', best_epoch)

                    loss_test_file_path = os.path.join(dataset_path, model, 'test_loss_accurate.csv')
                    if not os.path.exists(loss_test_file_path):
                        print(loss_test_file_path, 'not exists!', file=sys.stderr)
                        continue

                    best_record_ndcg = getRecord(loss_test_file_path, 'ndcg', best_epoch)


                    if best_record_mean_rank != None:
                        print(','.join(['0', 'mse', dataset, model] + best_record_mean_rank + best_record_ndcg))

                    


