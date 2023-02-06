import pandas as pd
import os
import numpy as np
import random
import torch
import logging
from sklearn.metrics import matthews_corrcoef, f1_score, precision_recall_curve, auc, roc_curve, accuracy_score
import numpy as np
import json
import pickle
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

def auc_metrics(preds, labels):
    y = np.array(labels)
    pred = np.array(preds)
    fpr, tpr, thresholds = roc_curve(y, pred, pos_label=1)
    precision, recall, _thresholds = precision_recall_curve(y, pred)
    roc_auc = auc(fpr, tpr)
    pr_auc = auc(recall, precision)
    return {
      "roc_auc": roc_auc,
      "pr_auc": pr_auc
    }
def acc_metrics(preds, labels):
    preds = [int(_ >= 0) for _ in preds]
    return {
      "acc_score": accuracy_score(labels, preds),
      "f1_score": f1_score(labels, preds)
    }


def load_uqkdt_tsv_data(file_path, add_time=False):
    data_list = []
    with open(file_path, 'r') as f:
        for line in f:
            u, q, k, d, t = line.strip().split('\t')
            sample = [u, q, k, d]
            if add_time:
                sample = [u, q, k, d, t]
            data_list.append(sample)
    logger.info(f"load from {file_path} with {len(data_list)} samples")
    return data_list    

def load_tsv_list(file_path):
    node_dict = {}
    with open(file_path, 'r') as f:
        for i, line in enumerate(f):
            node_dict[line.strip()] = i
    logger.info(f"load from {file_path} with {len(node_dict)} samples")
    return node_dict

def load_data_list(file_path):
    data_list = []
    with open(file_path, 'r') as f:
        for i, line in enumerate(f):
            data_list.append(line.strip().split('\t'))
    logger.info(f"load from {file_path} with {len(data_list)} samples")
    return data_list

def load_node2id_list(file_path):
    nodeid2text = {}
    qtext2nodeid = {}
    ktext2nodeid = {}
    udtext2nodeid = {}
    with open(file_path, 'r') as f:
        for i, line in enumerate(f):
            text, nodeid = line.strip().split('\t')
            nodeid2text[nodeid] = text
            if nodeid.startswith('q'):
                qtext2nodeid[text] = nodeid
            elif nodeid.startswith('k'):
                ktext2nodeid[text] = nodeid
            else:
                udtext2nodeid[text] = nodeid
    logger.info(f"load from {file_path} with {len(nodeid2text)} samples")
    return nodeid2text, qtext2nodeid, ktext2nodeid, udtext2nodeid

def load_edge2id_list(file_path):
    edge2id_list = []
    with open(file_path, 'r') as f:
        for i, line in enumerate(f):
            edgeid, nodeid = line.strip().split('\t')
            edge2id_list.append((edgeid, nodeid))
    logger.info(f"load from {file_path} with {len(edge2id_list)} samples")
    return edge2id_list

def load_train_dev_data(data_args):
    # graph node: nodeid \t nodetext
    # train_edge: node1 \t node2 \t edgetype \t edgeweight
    # test: node1 \t node2 
    # dev: node1 \t node2 
    node_info_dict = {}
    with open(data_args.node_info_path, 'r') as f:
        for line in f:
            nodeid, nodetext = line.strip().split('\t')
            node_info_dict[nodeid] = nodetext
    
    
    train_data_list = []
    with open(data_args.train_data, 'r') as f:
        for line in f:
            try:
                node1id, node2id, edge_type, edge_weight = line.strip().split('\t')
            except:
                node1id, node2id = line.strip().split('\t')
                edge_type = 'un'
                edge_weight = '0'
            train_data_list.append([node1id, node2id, edge_type, float(edge_weight)])

    dev_data_list = []
    with open(data_args.dev_data, 'r') as f:
        for line in f:
            try:
                node1id, node2id, edge_type, edge_weight = line.strip().split('\t')
            except:
                node1id, node2id = line.strip().split('\t')
                edge_type = 'un'
                edge_weight = '0'
            dev_data_list.append([node1id, node2id, edge_type, float(edge_weight)])
    
    print('node count : ', len(node_info_dict))
    print('train data count : ', len(train_data_list))
    print('dev data count : ', len(dev_data_list))
    return node_info_dict, train_data_list, dev_data_list

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# def recall_k(pred_rank, ground_turth, k=1):
#     count = 0
#     for preds, t in zip(pred_rank, ground_turth):
#         if t in preds[:k]:
#             count += 1
#     return count / len(pred_rank)

def recall_k(pred_rank, ground_turth, k=10):
    recall_list = []
    for pred, gt in zip(pred_rank, ground_turth):
        topk = pred[:k]
        recall_list.append(len(set(topk) & set(gt)) / len(set(gt)))
    return sum(recall_list) / len(recall_list)


def precision_k(pred_rank, ground_turth, k=10):
    precision_list = []
    for pred, gt in zip(pred_rank, ground_turth):
        topk = pred[:k]
        precision_list.append(len(set(topk) & set(gt)) / len(set(topk)))
    return sum(precision_list) / len(precision_list)


def ndcg_k(actual, predicted, topk):
    res = 0
    for user_id in range(len(actual)):
        k = min(topk, len(actual[user_id]))
        idcg = idcg_k(k)
        dcg_k = sum([int(predicted[user_id][j] in
                         set(actual[user_id])) / np.log2(j+2) for j in range(topk)])
        res += dcg_k / idcg
    return res / float(len(actual))

def idcg_k(k):
    res = sum([1.0/np.log2(i+2) for i in range(k)])
    if not res:
        return 1.0
    else:
        return res

def calc_recall_precision(pred_rank, ground_turth, k_list=[10], types=["precision", "recall", "mrr", "ndcg"]):
    result = {}
    for k in k_list:
        recall_list = []
        precision_list = []
        position_list = []
        ndcg_list = []
        for pred, gt in zip(pred_rank, ground_turth):
            topk = pred[:k]
            gt_set = set(gt)
            recall_list.append(len(set(topk) & gt_set) / len(gt_set))
            precision_list.append(len(set(topk) & gt_set) / len(set(topk)))
            dcg_k = sum([int(topk[j] in gt_set) / np.log2(j+2) for j in range(len(topk))])
            ndcg_list.append(dcg_k / idcg_k(len(gt_set)))
            for i, p in enumerate(topk):
                if p in gt:
                    position_list.append( 1 / (i + 1))
                    break
                if i == len(topk) - 1:
                    position_list.append(0)
        if "recall" in types:
            result[f"recall_at_{k}"] = sum(recall_list) / len(recall_list)
        if "precision" in types:
            result[f"precision_at_{k}"] = sum(precision_list) / len(precision_list)
        if "mrr" in types:
            result[f"mrr_at_{k}"] = sum(position_list) / len(position_list)
        if "ndcg" in types:
            result[f"ndcg_at_{k}"] = sum(ndcg_list) / len(ndcg_list)
    return result
        

def load_json_data(file_name):
    with open(file_name, 'r') as f:
        data = json.load(f)
    return data


def pickle_file_load(file_path):
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    logger.info(f"pickle load from {file_path}")
    return data

def pickle_file_dump(data, file_path):
    with open(file_path, 'wb') as f:
        pickle.dump(data, f)
    logger.info(f"pickle dump to {file_path}")