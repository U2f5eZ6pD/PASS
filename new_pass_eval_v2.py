from re import L
import numpy as np
import pandas as pd
import transformers
import torch
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler
from torch.nn.utils import clip_grad_norm_
from transformers import BertTokenizer, BertModel, BertConfig, get_linear_schedule_with_warmup, AdamW
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
import logging
from new_pass_model_v2 import create_node_dict, HyperGraphDataset, Pass, transtext2nodeid, BPRloss, HyperNodeExample, construct_sample_input_u, construct_sample_input_q, construct_sample_input_k, construct_sample_input_d
import time
from utils import load_data_list, calc_recall_precision, load_edge2id_list, load_node2id_list, load_uqkdt_tsv_data, load_tsv_list, set_seed, auc_metrics, acc_metrics, recall_k, precision_k, load_json_data, pickle_file_dump, pickle_file_load
from tensorboardX import SummaryWriter
import os
import collections
import json
from tqdm import tqdm
import random
from new_pass_model_v2 import transnodeid2type, transedgeid2type

U_TYPE = 0
D_TYPE = 1
Q_TYPE = 2
K_TYPE = 3

UT_TYPE = 0
DO_TYPE = 1
IN_TYPE = 2
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


def construct_input_pair(pair_batch, graph_batch, node2example, device):
    node2id = {}
    id2node = {}
    nodesize = 0
    batch_graph_nodeid = [[],[]]
    batch_graph_adj = [[],[]]
    batch_graph_edgetpye = [[],[]]

    for i, (graph1, graph2) in enumerate(graph_batch):
        assert pair_batch[i][0] == graph1[0][0], (pair_batch[i][0],graph1[0][0])
        assert pair_batch[i][1] == graph2[0][0], (pair_batch[i][1],graph2[0][0])
        for nodeid in graph1[0] + graph2[0]:
            if nodeid not in node2id:
                node2id[nodeid] = nodesize
                id2node[nodesize] = nodeid
                nodesize += 1
        
        for i, graph in enumerate([graph1, graph2]):
            batch_graph_nodeid[i].append([node2id[nodeid] for nodeid in graph[0]])
            batch_graph_adj[i].append(graph[1])
            batch_graph_edgetpye[i].append(graph[2])

    input_ids = []
    input_masks = []
    input_token_type_ids = []
    node_types = []
    for i in range(nodesize):
        example = node2example[id2node[i]]
        input_ids.append(example.input_ids)
        input_masks.append(example.input_masks)
        input_token_type_ids.append(example.segement_ids)
        node_types.append(example.nodetype)  

    input_ids = torch.tensor(input_ids, dtype=torch.long).to(device)
    input_masks = torch.tensor(input_masks, dtype=torch.long).to(device)
    input_token_type_ids = torch.tensor(input_token_type_ids, dtype=torch.long).to(device)
    node_types = torch.tensor(node_types, dtype=torch.long).to(device)

    batch_graph_nodeid = [torch.tensor(_, dtype=torch.long).to(device) for _ in batch_graph_nodeid]
    batch_graph_adj = [torch.stack(_).long().to(device) for _ in batch_graph_adj]
    batch_graph_edgetpye = [torch.tensor(_, dtype=torch.long).to(device) for _ in batch_graph_edgetpye]       

    return input_ids, input_masks, input_token_type_ids, node_types, batch_graph_nodeid, batch_graph_adj, batch_graph_edgetpye


def to_eval_model(model, node2examples, dev_id_data_list, presample_graph_list_dev, device, candidates_kd_list=None, pred_save_file=None):
    uq2ground = collections.defaultdict(list)
    uq2session = collections.defaultdict(list)
    dk2order = collections.defaultdict(list) 
    index = 0
    kd2index = {}
    index2kd = {}
    index_uq = 0
    uq2index = {}
    index2uq = {}
    if candidates_kd_list is None:
        for i, d in enumerate(dev_id_data_list):
            if (d[2], d[3]) not in kd2index:
                kd2index[(d[2], d[3])] = index
                index2kd[index] = (d[2], d[3])
                index += 1
                dk2order[(d[3], d[2])] = (presample_graph_list_dev[i][3] , presample_graph_list_dev[i][2])
            if (d[0], d[1]) not in uq2index:
                uq2index[(d[0], d[1])] = index_uq
                index2uq[index_uq] = (d[0], d[1])
                index_uq += 1
                uq2session[(d[0], d[1])] = (presample_graph_list_dev[i][0] , presample_graph_list_dev[i][1])
            uq2ground[(d[0], d[1])].append(kd2index[(d[2], d[3])])
    else:
        for kd in candidates_kd_list:
            kd = (kd[0], kd[1])
            if kd not in kd2index:
                kd2index[kd] = index
                index2kd[index] = kd
                index += 1
        for d in dev_id_data_list:
            if (d[0], d[1]) not in uq2index:
                uq2index[(d[0], d[1])] = index_uq
                index2uq[index_uq] = (d[0], d[1])
                index_uq += 1
            uq2ground[(d[0], d[1])].append(kd2index[(d[2], d[3])])
    for uq in uq2ground:
        uq2ground[uq] = sorted(list(set(uq2ground[uq])))
    
    logger.info(f"size of candidates_kd_list: {len(kd2index)}")
    logger.info(f"size of uq pair: {len(uq2index)}")

    infer_batch_size = 128
    all_candidates = [index2kd[i] for i in range(len(kd2index))]
    all_candidates_embs = []
    for i in tqdm(range(len(all_candidates) // infer_batch_size + 1), total=(len(all_candidates) // infer_batch_size + 1)):
        batch = all_candidates[i*infer_batch_size: (i+1)*infer_batch_size]
        dk_batch = [[_[1], _[0]] for _ in batch]
        graph_batch = [dk2order[(dk[0], dk[1])] for dk in dk_batch]
        batch_input = construct_input_pair(dk_batch, graph_batch, node2examples, device)
        input_ids, input_masks, input_token_type_ids, node_types, batch_graph_nodeid, batch_graph_adj, batch_graph_edgetpye = batch_input
        with torch.no_grad():
            emb_batch = model.encode_domain_keyword(input_ids, input_masks, input_token_type_ids, node_types, batch_graph_nodeid, batch_graph_adj, batch_graph_edgetpye)
        all_candidates_embs.append(emb_batch)
    all_candidates_embs = torch.cat(all_candidates_embs, dim=0)

    all_uqs = [index2uq[i] for i in range(len(uq2index))]
    all_uq_embs = []
    for i in tqdm(range(len(all_uqs) // infer_batch_size + 1), total=(len(all_uqs) // infer_batch_size + 1)):
        batch = all_uqs[i*infer_batch_size: (i+1)*infer_batch_size]
        graph_batch = [uq2session[(uq[0], uq[1])] for uq in batch]
        batch_input = construct_input_pair(batch, graph_batch, node2examples, device)
        input_ids, input_masks, input_token_type_ids, node_types, batch_graph_nodeid, batch_graph_adj, batch_graph_edgetpye = batch_input
        with torch.no_grad():
            emb_batch = model.encode_user_query(input_ids, input_masks, input_token_type_ids, node_types, batch_graph_nodeid, batch_graph_adj, batch_graph_edgetpye)
        all_uq_embs.append(emb_batch)
    all_uq_embs = torch.cat(all_uq_embs, dim=0)



    preds = []
    gts = []
    predict_result = {}
    all_up_for_calc = list(uq2ground.keys())
    calc_batch_size = 10
    for i in tqdm(range(len(all_up_for_calc) // calc_batch_size + 1), total=(len(all_up_for_calc) // calc_batch_size + 1)):
        batch_uq = all_up_for_calc[i*calc_batch_size: (i+1)*calc_batch_size]
        batch_src_index = [uq2index[uq] for uq in batch_uq]
        batch_labels = [uq2ground[uq] for uq in batch_uq]
        emb1 = all_uq_embs[batch_src_index]
        emb2 = all_candidates_embs
        score = torch.matmul(F.normalize(emb1, dim=-1), F.normalize(emb2, dim=-1).transpose(1, 0))
        ranks = torch.argsort(score, dim=-1, descending=True)[:, :200]
        ranks = ranks.detach().cpu().numpy().tolist()
        preds.extend(ranks)
        gts.extend(batch_labels)
        if pred_save_file:
            for i, uq in enumerate(batch_uq):
                predict_result['\t'.join(uq)] = {
                    'gt': [index2kd[_] for _ in batch_labels[i]], 
                    'pred': [index2kd[_] for _ in ranks[i][:50]]
                }

    if pred_save_file:
        with open(pred_save_file, 'w') as f:
            json.dump(predict_result, f, indent=2)

    result = calc_recall_precision(preds, gts, k_list=[5,10, 20,50,100], types=["recall", "ndcg"])
    for i in range(5):
        logger.info(f"dev example {i}")
        logger.info(f"u q pair: {(node2examples[all_up_for_calc[i][0]].nodetext, node2examples[all_up_for_calc[i][1]].nodetext)}")
        logger.info(f"ground truth: {[(node2examples[index2kd[_][0]].nodetext, node2examples[index2kd[_][1]].nodetext) for _ in gts[i]]}")
        logger.info(f"pred top 5: {[(node2examples[index2kd[_][0]].nodetext, node2examples[index2kd[_][1]].nodetext) for _ in preds[i][:5]]}")
    return result
    

import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='parameters')
    parser.add_argument('--max_len', type=int, default=10)
    parser.add_argument('--epoch', type=int, default=5)
    parser.add_argument('--lr', type=float, default=1e-04)
    parser.add_argument('--encoder', type=str, default='bert-base-uncased')
    parser.add_argument('--weight_decay', type=float, default=0)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--seed', type=int, default=213214)
    parser.add_argument('--local_rank', type=int, default=-1)
    parser.add_argument('--node_info_path', type=str, default='data/node2id.tsv')
    parser.add_argument('--train_file_path', type=str, default='data/train.tsv')
    parser.add_argument('--dev_file_path', type=str, default='data/dev.tsv')
    parser.add_argument('--user_list', type=str, default='data/user_list.tsv')
    parser.add_argument('--domain_list', type=str, default='data/domain_list.tsv')
    parser.add_argument('--reprocessed', type=str, default='false')
    parser.add_argument('--processed_train_node2examples', type=str, default='data/train_node2examples.pkl')
    parser.add_argument('--candidates_dev_file_path', type=str, default=None)
    parser.add_argument('--train_batch_size', type=int, default=512)
    parser.add_argument('--eval_batch_size', type=int, default=512)
    parser.add_argument('--output_dir', type=str, default='pass_outputs')
    parser.add_argument('--load_from_checkpoint', type=str, default='new_pass_outputs/output1/run_1/pytorch_model.96550.bin')

    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    print(args)
    set_seed(args.seed)
    args.no_cuda = False


    if args.local_rank == -1 or args.no_cuda:
        device = torch.device(
            "cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        n_gpu = torch.cuda.device_count()
    else:
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        n_gpu = 1
        # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.distributed.init_process_group(backend='nccl')

    nodeid2text, qtext2nodeid, ktext2nodeid, udtext2nodeid = load_node2id_list(args.node_info_path)
    user_dict = load_tsv_list(args.user_list)
    domain_dict = load_tsv_list(args.domain_list)
    if args.processed_train_node2examples:
        if os.path.exists(args.processed_train_node2examples) and args.reprocessed != 'true':
            node2examples = pickle_file_load(args.processed_train_node2examples)
        else:
            tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
            node2examples = create_node_dict(nodeid2text, user_dict, domain_dict, tokenizer, args.max_len)
            pickle_file_dump(node2examples, args.processed_train_node2examples)

    node2examples['UNK'] = HyperNodeExample('UNK', 0, 'UNK', [0] * 10, [0] * 10, [0] * 10)

    dev_data_list = load_uqkdt_tsv_data(args.dev_file_path)
    dev_id_data_list = transtext2nodeid(dev_data_list, qtext2nodeid, ktext2nodeid, udtext2nodeid)

    if args.candidates_dev_file_path:
        candidates_kd_list = load_data_list(args.candidates_dev_file_path)
        candidates_kd_list = [(ktext2nodeid[kd[0]], udtext2nodeid[kd[1]]) for kd in candidates_kd_list]
    else:
        candidates_kd_list = None


    model = Pass(text_encoder='bert-base-uncased', 
                    encoder_dim=768, 
                    node_dim=256, 
                    user_size=len(user_dict), 
                    domain_size=len(domain_dict), 
                    user_agg_type='mean', 
                    adver_agg_type='mean', 
                    query_agg_type='mean', 
                    keyword_agg_type='mean',
                    pretrain_user=None, 
                    pretrain_domain=None)

    # model = nn.DataParallel(model, device_ids=gpus, output_device=device)
    if args.load_from_checkpoint and os.path.exists(args.load_from_checkpoint):
        model.load_state_dict(torch.load(args.load_from_checkpoint))
        logger.info(f"load from {args.load_from_checkpoint}")
    model.to(device)

    presample_graph_list_dev = pickle_file_load("presample_graph_list_dev.pkl")
    eval_result = to_eval_model(model, node2examples, dev_id_data_list, presample_graph_list_dev, device, pred_save_file='tmp_graph.json')
    for k, v in eval_result.items():
        logger.info(f"{k}: {v}")

                    
            
