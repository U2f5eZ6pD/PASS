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
from new_pass_model_v2 import create_node_dict, HyperGraphDataset, Pass
import time
from utils import load_node2id_list, load_edge2id_list, load_uqkdt_tsv_data, load_tsv_list, set_seed, auc_metrics, acc_metrics, recall_k, precision_k, load_json_data, calc_recall_precision
from tensorboardX import SummaryWriter
import os
import collections
import json
from tqdm import tqdm
import random

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


def infer_text_embedding(model, all_text, tokenizer, device, maxlen=10, cls_token='[CLS]', sep_token='[SEP]'):
    all_text_list = list(all_text.keys())
    infer_batch_size = 4096
    all_candidates_embs = []
    for i in tqdm(range(len(all_text_list) // infer_batch_size + 1), total=(len(all_text_list) // infer_batch_size + 1)):
        batch = all_text_list[i*infer_batch_size: (i+1)*infer_batch_size]
        batch_input_ids = []
        batch_input_masks = []
        batch_input_token_type_ids = []
        for text in batch:
            tokens = tokenizer.tokenize(text)
            tokens = [cls_token] + tokens + [sep_token]
            input_ids = tokenizer.convert_tokens_to_ids(tokens)[:maxlen]
            token_type_ids = [0] * len(input_ids)
            input_mask = [1] * len(input_ids)
            padding = [0] * (maxlen - len(input_ids))
            input_ids += padding
            input_mask += padding
            token_type_ids += padding
            batch_input_ids.append(input_ids)
            batch_input_masks.append(input_mask)
            batch_input_token_type_ids.append(token_type_ids)

        batch_input_ids = torch.tensor(batch_input_ids, dtype=torch.long).to(device)
        batch_input_masks = torch.tensor(batch_input_masks, dtype=torch.long).to(device)
        batch_input_token_type_ids = torch.tensor(batch_input_token_type_ids, dtype=torch.long).to(device)
        with torch.no_grad():
            emb_batch = model.nodeEncoder.q_model(batch_input_ids, batch_input_masks, batch_input_token_type_ids)[0][:, 0]
        all_candidates_embs.append(emb_batch.cpu())
    all_candidates_embs = torch.cat(all_candidates_embs, dim=0)
    text2embedding = {}
    for i, text in enumerate(all_text_list):
        text2embedding[text] = all_candidates_embs[i]
    return text2embedding



import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='parameters')
    parser.add_argument('--seed', type=int, default=213214)
    parser.add_argument('--max_len', type=int, default=10)
    parser.add_argument('--local_rank', type=int, default=-1)
    parser.add_argument('--train_file_path', type=str, default='data/train.tsv')
    parser.add_argument('--dev_file_path', type=str, default='data/dev.tsv')
    parser.add_argument('--test_file_path', type=str, default='data/test.tsv')
    parser.add_argument('--user_list', type=str, default='data/user_list.tsv')
    parser.add_argument('--domain_list', type=str, default='data/domain_list.tsv')  
    parser.add_argument('--train_batch_size', type=int, default=256)
    parser.add_argument('--eval_batch_size', type=int, default=256)
    parser.add_argument('--output_dir', type=str, default='qkmodel_outputs')
    parser.add_argument('--load_from_checkpoint', type=str, default='new_pass_outputs/data_sample/run_31/pytorch_model.22795.bin')
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

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    user_dict = load_tsv_list(args.user_list)
    domain_dict = load_tsv_list(args.domain_list)
    train_data_list = load_uqkdt_tsv_data(args.train_file_path)
    dev_data_list = load_uqkdt_tsv_data(args.dev_file_path)
    test_data_list = load_uqkdt_tsv_data(args.test_file_path)
    all_data_list = train_data_list + dev_data_list + test_data_list

    print(len(set([_[0] for _ in all_data_list])))
    print(len(set([_[3] for _ in all_data_list])))
    print(len(user_dict))
    print(len(domain_dict))


    all_query = {}
    all_keyword = {}
    u2q = collections.defaultdict(list)
    d2k = collections.defaultdict(list)

    for i, uqkd in tqdm(enumerate(train_data_list), total=len(train_data_list)):
        u, q, k, d = uqkd
        d2k[d].append(k)
        all_keyword[k] = 1
        u2q[u].append(q)
        all_query[q] = 1   

    for i, uqkd in tqdm(enumerate(dev_data_list), total=len(dev_data_list)):
        u, q, k, d = uqkd
        d2k[d].append(k)
        all_keyword[k] = 1
        u2q[u].append(q)
        all_query[q] = 1   

    for i, uqkd in tqdm(enumerate(test_data_list), total=len(test_data_list)):
        u, q, k, d = uqkd
        d2k[d].append(k)
        all_keyword[k] = 1
        u2q[u].append(q)
        all_query[q] = 1   


    print("avg u2cq:", sum([len(u2q[_]) for _ in u2q]) / len(u2q))
    print("avg d2ck:", sum([len(d2k[_]) for _ in d2k]) / len(d2k))
    print(len(u2q))
    print(len(d2k))
    
    model = Pass(text_encoder='bert-base-uncased', 
                    encoder_dim=768, 
                    node_dim=256, 
                    user_size=len(user_dict), 
                    domain_size=len(domain_dict), 
                    user_agg_type='mean', 
                    adver_agg_type='mean', 
                    query_agg_type='mean', 
                    keyword_agg_type='mean')

    #model = TwinBertWithUD(encoder='bert-base-uncased', encoder_dim=768, projection_dim=256)

    # model = nn.DataParallel(model, device_ids=gpus, output_device=device)
    # if args.load_from_checkpoint and os.path.exists(args.load_from_checkpoint):
    #     model.load_state_dict(torch.load(args.load_from_checkpoint))
    #     logger.info(f"load from {args.load_from_checkpoint}")
    model.to(device)
    user2embedding = {}
    domain2embedding = {}
    keyword2embedding = infer_text_embedding(model, all_keyword, tokenizer, device, maxlen=10, cls_token='[CLS]', sep_token='[SEP]')
    for domain in d2k:
        keywords_embedding = [keyword2embedding[k] for k in d2k[domain]]
        keywords_embedding = torch.mean(torch.cat(keywords_embedding).reshape(-1, 768), 0)
        domain2embedding[domain] = keywords_embedding.numpy()

    query2embedding = infer_text_embedding(model, all_query, tokenizer, device, maxlen=10, cls_token='[CLS]', sep_token='[SEP]')
    for user in u2q:
        querys_embedding = [query2embedding[q] for q in u2q[user]]
        users_embedding = torch.mean(torch.cat(querys_embedding).reshape(-1, 768), 0)
        user2embedding[user] = users_embedding.numpy()

    
    numpy_userembeddings = np.zeros((len(user2embedding), 768))
    numpy_domainembeddings = np.zeros((len(domain2embedding), 768))
    print(numpy_userembeddings.shape)
    print(numpy_domainembeddings.shape)
    for user in user_dict:
        numpy_userembeddings[user_dict[user]] = user2embedding[user]
    for domain in domain_dict:
        numpy_domainembeddings[domain_dict[domain]] = domain2embedding[domain]

    print(numpy_userembeddings.shape)
    print(numpy_domainembeddings.shape)
    numpy_userembeddings = F.normalize(torch.from_numpy(numpy_userembeddings), dim=-1).numpy()
    numpy_domainembeddings = F.normalize(torch.from_numpy(numpy_domainembeddings), dim=-1).numpy()
    np.save('pretrained_userembeddings.npy',numpy_userembeddings)
    np.save('pretrained_domainembeddings.npy',numpy_domainembeddings)

                    
            
