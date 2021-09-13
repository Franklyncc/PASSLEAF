MEAN_RANK_COLUMNS = ['test_epoch', 'mean_rank', 'mean_hit@10', 'mean_hit@20', 'mean_hit@40', 'mean_rank_weighted', 'mean_hit@10_weighted', 'mean_hit@20_weighted', 'mean_hit@40_weighted']
NDCG_COLUMNS = ['test_epoch', 'mse', 'mse_neg', 'mse_neg(second)', 'ndcg(linear)', 'ndcg(exp)']

SUMMARY_COLUMNS = ['threshold', 'criteria', 'dataset', 'model'] + MEAN_RANK_COLUMNS + NDCG_COLUMNS + ['ndcg(linear) training included', 'ndcg(exp) training included']
import os


def getColumnMapping(fmt):
    if fmt == 'mean_rank': # mean_rank
        return  MEAN_RANK_COLUMNS

    elif fmt == 'ndcg': # loss_accuracy
        return  NDCG_COLUMNS

    elif fmt == 'summary':
        return SUMMARY_COLUMNS

def getBestRecord(file, fmt, metric, isMax=True):
    if not os.path.exists(file): return None
    columns = getColumnMapping(fmt)

    metric_index = columns.index(metric)
    epoch_index  = 0
    

    best_score = -999 if isMax else 999
    best_epoch = None
    best_record = None
    with open(file) as f:
        for line in f.read().split('\n')[0:-1]:
            record = line.split(',')

            # score = float(record[1])
            try: 
                score = float(record[metric_index])
            except ValueError: # header row
                continue
            
            # if score < best_score:
            if (isMax and score > best_score) or (not isMax and score < best_score):
                best_epoch = int(record[epoch_index])
                best_score = score
                best_record = record

    return best_epoch, best_record

def getRecord(file, fmt, epoch, isMax=True):
    if not os.path.exists(file): return None
    columns = getColumnMapping(fmt)


    epoch_index  = 0
    

    best_record = None
    with open(file) as f:
        for line in f.read().split('\n')[0:-1]:
            record = line.split(',')
            
            if int(record[epoch_index]) == epoch:
                best_record = record

    return best_record


