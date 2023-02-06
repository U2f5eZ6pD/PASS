import numpy as np
# import pandas as pd
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
from utils import load_edge2id_list, load_node2id_list, load_uqkdt_tsv_data, load_tsv_list, set_seed, auc_metrics, acc_metrics, recall_k, precision_k, load_json_data, pickle_file_dump, pickle_file_load
from new_pass_parse import parse_args
from tqdm import tqdm
from tensorboardX import SummaryWriter
import os
from new_pass_eval_v2 import to_eval_model


logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

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

# ************** construct dataset ***************
# pre tokenize

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
else:
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    node2examples = create_node_dict(nodeid2text, user_dict, domain_dict, tokenizer, args.max_len)
    pickle_file_dump(node2examples, args.processed_train_node2examples)
node2examples['UNK'] = HyperNodeExample('UNK', 0, 'UNK', [0] * 10, [0] * 10, [0] * 10)

train_data_list = load_uqkdt_tsv_data(args.train_file_path)
train_id_data_list = transtext2nodeid(train_data_list, qtext2nodeid, ktext2nodeid, udtext2nodeid)

dev_data_list = load_uqkdt_tsv_data(args.dev_file_path)
dev_id_data_list = transtext2nodeid(dev_data_list, qtext2nodeid, ktext2nodeid, udtext2nodeid)

datadir = args.data_dir

# # train_id_order_edge_list = [[] for data in train_id_data_list]
# # train_id_session_edge_list = [[] for data in train_id_data_list]
train_order_edge_list = []
with open(datadir + '/train_order_edge_list.tsv', 'r') as f:
    for line in f:
        train_order_edge_list.append(line.strip().split('\t'))
train_id_order_edge_list = [[ktext2nodeid[_.lstrip()] for _ in data[:3]] for data in train_order_edge_list]

train_session_edge_list = []
with open(datadir + '/train_session_edge_list.tsv', 'r') as f:
    for line in f:
        train_session_edge_list.append(line.strip().split('\t'))
train_id_session_edge_list = [[qtext2nodeid[_.lstrip()] for _ in data[:3]] for data in train_session_edge_list]

train_personal_edge_list = []
with open(datadir + '/train_personal_click_edge_list.tsv', 'r') as f:
    for line in f:
        train_personal_edge_list.append(line.strip().split('\t'))

train_id_personal_edge_list = []
for i in range(len(train_personal_edge_list)):
    id_edge_i_list = []
    edge_i_list = train_personal_edge_list[i]
    if edge_i_list[0] != '': 
        for qkd in edge_i_list[:2]:
            qkd_pair = qkd.split("###")
            id_edge_i_list.append([train_id_data_list[i][0], 
                                qtext2nodeid[qkd_pair[0]], 
                                ktext2nodeid[qkd_pair[1]]])
    train_id_personal_edge_list.append(id_edge_i_list)


train_advertiser_edge_list = []
with open(datadir + '/train_advertiser_click_edge.tsv', 'r') as f:
    for line in f:
        train_advertiser_edge_list.append(line.strip().split('\t'))

train_id_advertiser_edge_list = []
for i in range(len(train_advertiser_edge_list)):
    id_edge_i_list = []
    edge_i_list = train_advertiser_edge_list[i]
    if edge_i_list[0] != '': 
        for uq in edge_i_list[:2]:
            uq_pair = uq.split("###")
            id_edge_i_list.append([ 
                                qtext2nodeid[uq_pair[1]], 
                                train_id_data_list[i][2], 
                                train_id_data_list[i][3]])
    train_id_advertiser_edge_list.append(id_edge_i_list)


train_qclick_edge_list = []
with open(datadir + '/train_query_click_edge_list.tsv', 'r') as f:
    for line in f:
        if line.strip().split('\t')[0] == "":
            train_qclick_edge_list.append([])
        else:
            train_qclick_edge_list.append(line.strip().split('\t'))
train_id_qclick_edge_list = [[ktext2nodeid[_.lstrip()] for _ in data[:3]] for data in train_qclick_edge_list]

train_kclick_edge_list = []
with open(datadir + '/train_keyword_click_edge_list.tsv', 'r') as f:
    for line in f:
        if line.strip().split('\t')[0] == "":
            train_kclick_edge_list.append([])
        else:
            train_kclick_edge_list.append(line.strip().split('\t'))
train_id_kclick_edge_list = [[qtext2nodeid[_.lstrip()] for _ in data[:3]] for data in train_kclick_edge_list]

# # dev_id_order_edge_list = [[] for data in dev_id_data_list]
# # dev_id_session_edge_list = [[] for data in dev_id_data_list]


dev_order_edge_list = []
with open(datadir + '/dev_order_edge_list.tsv', 'r') as f:
    for line in f:
        dev_order_edge_list.append(line.strip().split('\t'))
dev_id_order_edge_list = [[ktext2nodeid[_.lstrip()] for _ in data[:3]] for data in dev_order_edge_list]

dev_session_edge_list = []
with open(datadir + '/dev_session_edge_list.tsv', 'r') as f:
    for line in f:
        dev_session_edge_list.append(line.strip().split('\t'))
dev_id_session_edge_list = [[qtext2nodeid[_.lstrip()] for _ in data[:3]] for data in dev_session_edge_list]

dev_personal_edge_list = []
with open(datadir + '/dev_personal_click_edge_list.tsv', 'r') as f:
    for line in f:
        dev_personal_edge_list.append(line.strip().split('\t'))

dev_id_personal_edge_list = []
for i in range(len(dev_personal_edge_list)):
    id_edge_i_list = []
    edge_i_list = dev_personal_edge_list[i]
    if edge_i_list[0] != '': 
        for qkd in edge_i_list[:2]:
            qkd_pair = qkd.split("###")
            id_edge_i_list.append([dev_id_data_list[i][0], 
                                qtext2nodeid[qkd_pair[0]], 
                                ktext2nodeid[qkd_pair[1]]])
    dev_id_personal_edge_list.append(id_edge_i_list)


dev_advertiser_edge_list = []
with open(datadir + '/dev_advertiser_click_edge.tsv', 'r') as f:
    for line in f:
        dev_advertiser_edge_list.append(line.strip().split('\t'))

dev_id_advertiser_edge_list = []
for i in range(len(dev_advertiser_edge_list)):
    id_edge_i_list = []
    edge_i_list = dev_advertiser_edge_list[i]
    if edge_i_list[0] != '': 
        for uq in edge_i_list[:2]:
            uq_pair = uq.split("###")
            id_edge_i_list.append([
                                qtext2nodeid[uq_pair[1]], 
                                dev_id_data_list[i][2], 
                                dev_id_data_list[i][3]])
    dev_id_advertiser_edge_list.append(id_edge_i_list)


dev_qclick_edge_list = []
with open(datadir + '/dev_query_click_edge_list.tsv', 'r') as f:
    for line in f:
        if line.strip().split('\t')[0] == "":
            dev_qclick_edge_list.append([])
        else:
            dev_qclick_edge_list.append(line.strip().split('\t'))
dev_id_qclick_edge_list = [[ktext2nodeid[_.lstrip()] for _ in data[:3]] for data in dev_qclick_edge_list]

dev_kclick_edge_list = []
with open(datadir + '/dev_keyword_click_edge_list.tsv', 'r') as f:
    for line in f:
        if line.strip().split('\t')[0] == "":
            dev_kclick_edge_list.append([])
        else:
            dev_kclick_edge_list.append(line.strip().split('\t'))
dev_id_kclick_edge_list = [[qtext2nodeid[_.lstrip()] for _ in data[:3]] for data in dev_kclick_edge_list]


presample_graph_list = []
for i, sample in tqdm(enumerate(train_id_data_list), total=len(train_id_data_list)):
    u, q, k, d = sample[:4]
    ugraph_tuple = construct_sample_input_u(u, train_id_personal_edge_list[i], 5, 3)
    qgraph_tuple = construct_sample_input_q(q, train_id_session_edge_list[i], train_id_qclick_edge_list[i], 7, 2)
    kgraph_tuple = construct_sample_input_k(k, train_id_order_edge_list[i], train_id_kclick_edge_list[i], 7, 2)
    dgraph_tuple = construct_sample_input_d(d, train_id_advertiser_edge_list[i], 4, 3)
    presample_graph_list.append([ugraph_tuple, qgraph_tuple, kgraph_tuple, dgraph_tuple])


presample_graph_list_dev = []
for i, sample in tqdm(enumerate(dev_id_data_list), total=len(dev_id_data_list)):
    u, q, k, d = sample[:4]
    ugraph_tuple = construct_sample_input_u(u, dev_id_personal_edge_list[i], 5, 3)
    qgraph_tuple = construct_sample_input_q(q, dev_id_session_edge_list[i], dev_id_qclick_edge_list[i], 7, 2)
    kgraph_tuple = construct_sample_input_k(k, dev_id_order_edge_list[i], dev_id_kclick_edge_list[i], 7, 2)
    dgraph_tuple = construct_sample_input_d(d, dev_id_advertiser_edge_list[i], 4, 3)
    presample_graph_list_dev.append([ugraph_tuple, qgraph_tuple, kgraph_tuple, dgraph_tuple])


# presample_graph_list = []
# for i, sample in tqdm(enumerate(train_id_data_list), total=len(train_id_data_list)):
#     u, q, k, d = sample[:4]
#     ugraph_tuple = construct_sample_input_u(u, [], 1, 1)
#     qgraph_tuple = construct_sample_input_q(q, [], [], 1, 1)
#     kgraph_tuple = construct_sample_input_k(k, [], [], 1, 1)
#     dgraph_tuple = construct_sample_input_d(d, [], 1, 1)
#     presample_graph_list.append([ugraph_tuple, qgraph_tuple, kgraph_tuple, dgraph_tuple])


# presample_graph_list_dev = []
# for i, sample in tqdm(enumerate(dev_id_data_list), total=len(dev_id_data_list)):
#     u, q, k, d = sample[:4]
#     ugraph_tuple = construct_sample_input_u(u, [], 1, 1)
#     qgraph_tuple = construct_sample_input_q(q, [], [], 1, 1)
#     kgraph_tuple = construct_sample_input_k(k, [], [], 1, 1)
#     dgraph_tuple = construct_sample_input_d(d, [], 1, 1)
#     presample_graph_list_dev.append([ugraph_tuple, qgraph_tuple, kgraph_tuple, dgraph_tuple])

# pickle_file_dump(presample_graph_list, "presample_graph_list.pkl")
# pickle_file_dump(presample_graph_list_dev, "presample_graph_list_dev.pkl")
# presample_graph_list = pickle_file_load("presample_graph_list.pkl")
# presample_graph_list_dev = pickle_file_load("presample_graph_list_dev.pkl")

# for i in range(10):
#     print([node2examples[_].nodetext for _ in train_id_data_list[i]])
#     for j in range(4):
#         print([node2examples[_].nodetext for _ in presample_graph_list[i][j][0]])
#         print(presample_graph_list[i][j][1])
#         print(presample_graph_list[i][j][2])  



train_dataset = HyperGraphDataset(node2examples, train_id_data_list, presample_graph_list)
train_sampler = RandomSampler(train_dataset)
train_dataloader = DataLoader(
    train_dataset, sampler=train_sampler, batch_size=args.train_batch_size, collate_fn=train_dataset.collate_fn)


model = Pass(text_encoder='bert-base-uncased', 
                encoder_dim=768, 
                node_dim=256, 
                user_size=len(user_dict), 
                domain_size=len(domain_dict), 
                user_agg_type='mean', 
                adver_agg_type='mean', 
                query_agg_type='mean', 
                keyword_agg_type='mean',
                pretrain_user=args.pretrain_user, 
                pretrain_domain=args.pretrain_domain)



if not os.path.exists(args.output_dir):
    os.makedirs(args.output_dir)
run_id = len(os.listdir(args.output_dir))
args.output_dir = os.path.join(args.output_dir, "run_" + str(run_id))
os.makedirs(args.output_dir)

logger.info(f"output_dir: {args.output_dir}")

os.makedirs(os.path.join(args.output_dir, "log"), exist_ok=True)
summary_writer = SummaryWriter(os.path.join(args.output_dir, "log"))

global_step = 0
# model = nn.DataParallel(model, device_ids=gpus, output_device=device)
if args.load_from_checkpoint and os.path.exists(args.load_from_checkpoint):
    # state_dict = torch.load(args.load_from_checkpoint)
    # old_keys = []
    # new_keys = []
    # init_state_dict = model.state_dict()
    # new_state_dict = {}
    # for key in init_state_dict.keys():
    #     if key.startswith("nodeEncoder.d") or key.startswith("nodeEncoder.u"):
    #         continue
    #     else:
    #         new_state_dict[key] = state_dict.pop(key)
    # init_state_dict = model.state_dict()
    # for key in init_state_dict.keys():
    #     if key not in new_state_dict:
    #         new_state_dict[key] = init_state_dict[key] 
    # model.load_state_dict(new_state_dict) 
    # for key in state_dict.keys():
    #     if key.startswith('nodeEncoder.q') or key.startswith('nodeEncoder.k'):
    #         continue
    #     print(key)
    #     print("=====================")
    #     if key in init_state_dict:
    #         print("YES")
    #     else:
    #         print("NO")
    #     print("=====================")
    # exit()
    # model.load_state_dict(torch.load(args.load_from_checkpoint))
    # logger.info(f"load from {args.load_from_checkpoint}")
    # model.to(device)
    # model.eval()
    # logger.info("***** Dev results *****")
    # # result = to_eval_model(model, node2examples, dev_id_data_list, presample_graph_list_dev, device)
    # # for k, v in result.items():
    # #     logger.info(f"{k}: {v}")
    # #     summary_writer.add_scalar(f'eval/{k}', v, global_step)
    model.to(device)
else:
    model.to(device)

criterion = BPRloss()

no_decay = ['bias', 'LayerNorm.weight']
optimizer_grouped_parameters = [
    {'params' : [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
     'weight_decay' : args.weight_decay
    },
    {'params' : [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
     'weight_decay' : 0.0
    }
]
# optimizer = optim.Adam(model.parameters(),lr = LEARNING_RATE, weight_decay=args.weight_decay)
optimizer = AdamW(optimizer_grouped_parameters, lr = args.lr)
# print(model)
epochs = args.epoch
total_steps = len(train_dataloader) * args.epoch
model.train()

eval_per_step =  total_steps // (args.epoch * args.eval_per_epoch)

for e in range(epochs):
    for batch in train_dataloader:
        model.train()
        input_ids, input_masks, input_token_type_ids, node_types, batch_graph_nodeid, batch_graph_adj, batch_graph_edgetpye = batch

        input_ids = input_ids.to(device)
        input_masks = input_masks.to(device)
        input_token_type_ids = input_token_type_ids.to(device)
        node_types = node_types.to(device)
        batch_graph_nodeid = [_.to(device) for _ in batch_graph_nodeid]
        batch_graph_adj = [_.to(device) for _ in batch_graph_adj]
        batch_graph_edgetpye = [_.to(device) for _ in batch_graph_edgetpye]
        
        logits = model(input_ids, input_masks, input_token_type_ids, node_types, 
                        batch_graph_nodeid, batch_graph_adj, batch_graph_edgetpye)
        labels = torch.arange(logits.shape[0])
        labels = labels.to(device)
        loss = criterion(logits, labels)
        # print(loss)
        # if global_step == 1:
        #     exit()
        loss.backward()
        summary_writer.add_scalar('loss/BPRLoss', loss, global_step)
        optimizer.step()
        optimizer.zero_grad()
        global_step += 1
        if global_step % 10 == 0:
            logger.info(f"training loss: {loss} at global step {global_step} at epoch {e}")

        if global_step % eval_per_step == 0:
            model.eval()
            logger.info("***** Dev results *****")
            result = to_eval_model(model, node2examples, dev_id_data_list, presample_graph_list_dev, device, pred_save_file='overfit_graph.json')
            for k, v in result.items():
                logger.info(f"{k}: {v}")
                summary_writer.add_scalar(f'eval/{k}', v, global_step)

            model_to_save = model.module if hasattr(
                model, 'module') else model  # Only save the model it-self
            output_model_file = os.path.join(
                args.output_dir, "pytorch_model." + str(global_step) + ".bin")
            torch.save(model_to_save.state_dict(),
                        output_model_file)

            # output_config_file = os.path.join(
            #     args.output_dir, "config.json")
            # with open(output_config_file, 'w') as f:
            #     f.write(model_to_save.config.to_json_string())