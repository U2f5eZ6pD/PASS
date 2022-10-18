import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler
import transformers
from transformers import BertTokenizer, BertModel, BertConfig, DistilBertConfig, DistilBertModel, DistilBertTokenizer, RobertaConfig
import numpy as np
from tqdm import tqdm
#from torch_scatter import scatter
import logging
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

U_TYPE = 0
D_TYPE = 1
Q_TYPE = 2
K_TYPE = 3


UT_TYPE = 0
DO_TYPE = 1
IN_TYPE = 2
USERSELF_TYPE = 3
ADVESELF_TYPE = 4
PC_TYPE = 5
AC_TYPE = 6

def transnodeid2type(nodeid):
    if nodeid.startswith('u'):
        return U_TYPE
    if nodeid.startswith('q'):
        return Q_TYPE
    if nodeid.startswith('k'):
        return K_TYPE
    if nodeid.startswith('d'):
        return D_TYPE

def transedgeid2type(edgeid):
    if edgeid.startswith('ut'):
        return UT_TYPE
    if edgeid.startswith('do'):
        return DO_TYPE
    if edgeid.startswith('uqkd'):
        return IN_TYPE
    return -1


# def construct_sample_input_u(u, hyperedge, max_node_size, max_hyperedge_size):
#     sample_node2id = {}
#     sample_id2node = {}
#     sample_nodes = [u]
#     u_index = 0
#     sample_hyperAdjs = torch.zeros(max_node_size, max_hyperedge_size)
#     if len(hyperedge) > 0:
#         for edge in hyperedge:
#             for node in edge:
#                 if node not in sample_nodes:
#                     sample_nodes.append(node)

#         for id, node in enumerate(sample_nodes):
#             sample_node2id[node] = id
#             sample_id2node[id] = node

#         while len(sample_nodes) < max_node_size:
#             sample_nodes.append("UNK")

#         for i, edge in enumerate(hyperedge):
#             for node in edge:
#                 sample_hyperAdjs[sample_node2id[node]][i] = 1
#         sample_edge_type = [PC_TYPE for _ in range(max_hyperedge_size)]
#     else:
#         sample_hyperAdjs[0][0] = 1
#         while len(sample_nodes) < max_node_size:
#             sample_nodes.append("UNK")
#         sample_edge_type = [USERSELF_TYPE for _ in range(max_hyperedge_size)]
#     return sample_nodes, sample_hyperAdjs, sample_edge_type


def construct_sample_input_u(u, hyperedge, max_node_size, max_hyperedge_size):
    sample_node2id = {}
    sample_id2node = {}
    sample_nodes = [u]
    u_index = 0
    sample_hyperAdjs = torch.zeros(max_node_size, max_hyperedge_size)
    if len(hyperedge) > 0:
        for edge in hyperedge:
            for node in edge:
                if node not in sample_nodes:
                    sample_nodes.append(node)

        for id, node in enumerate(sample_nodes):
            sample_node2id[node] = id
            sample_id2node[id] = node

        while len(sample_nodes) < max_node_size:
            sample_nodes.append("UNK")
        
        sample_hyperAdjs[0][0] = 1
        for i, edge in enumerate(hyperedge):
            for node in edge:
                sample_hyperAdjs[sample_node2id[node]][i + 1] = 1
        sample_edge_type = [PC_TYPE for _ in range(max_hyperedge_size)]
    else:
        sample_hyperAdjs[0][0] = 1
        while len(sample_nodes) < max_node_size:
            sample_nodes.append("UNK")
        sample_edge_type = [USERSELF_TYPE for _ in range(max_hyperedge_size)]
    return sample_nodes, sample_hyperAdjs, sample_edge_type


# def construct_sample_input_d(d, hyperedge, max_node_size, max_hyperedge_size):
#     sample_node2id = {}
#     sample_id2node = {}
#     sample_nodes = [d]
#     d_index = 0
#     sample_hyperAdjs = torch.zeros(max_node_size, max_hyperedge_size)
#     if len(hyperedge) > 0:
#         for edge in hyperedge:
#             for node in edge:
#                 if node not in sample_nodes:
#                     sample_nodes.append(node)

#         for id, node in enumerate(sample_nodes):
#             sample_node2id[node] = id
#             sample_id2node[id] = node

#         while len(sample_nodes) < max_node_size:
#             sample_nodes.append("UNK")

#         for i, edge in enumerate(hyperedge):
#             for node in edge:
#                 sample_hyperAdjs[sample_node2id[node]][i] = 1

#         # can be change
#         sample_edge_type = [AC_TYPE for _ in range(max_hyperedge_size)]
#     else:
#         sample_hyperAdjs[0][0] = 1
#         while len(sample_nodes) < max_node_size:
#             sample_nodes.append("UNK")
#         sample_edge_type = [USERSELF_TYPE for _ in range(max_hyperedge_size)]        
#     return sample_nodes, sample_hyperAdjs, sample_edge_type

def construct_sample_input_d(d, hyperedge, max_node_size, max_hyperedge_size):
    sample_node2id = {}
    sample_id2node = {}
    sample_nodes = [d]
    d_index = 0
    sample_hyperAdjs = torch.zeros(max_node_size, max_hyperedge_size)
    if len(hyperedge) > 0:
        for edge in hyperedge:
            for node in edge:
                if node not in sample_nodes:
                    sample_nodes.append(node)

        for id, node in enumerate(sample_nodes):
            sample_node2id[node] = id
            sample_id2node[id] = node

        while len(sample_nodes) < max_node_size:
            sample_nodes.append("UNK")

        sample_hyperAdjs[0][0] = 1
        for i, edge in enumerate(hyperedge):
            for node in edge:
                sample_hyperAdjs[sample_node2id[node]][i + 1] = 1

        # can be change
        sample_edge_type = [AC_TYPE for _ in range(max_hyperedge_size)]
    else:
        sample_hyperAdjs[0][0] = 1
        while len(sample_nodes) < max_node_size:
            sample_nodes.append("UNK")
        sample_edge_type = [USERSELF_TYPE for _ in range(max_hyperedge_size)]        
    return sample_nodes, sample_hyperAdjs, sample_edge_type


def construct_sample_input_q(q, sessionedge, clickedge, max_node_size, max_hyperedge_size):
    sample_node2id = {}
    sample_id2node = {}
    sample_nodes = [q]
    q_index = 0
    for node in sessionedge:
        if node not in sample_nodes:
            sample_nodes.append(node)

    for node in clickedge:
        if node not in sample_nodes:
            sample_nodes.append(node)

    for id, node in enumerate(sample_nodes):
        sample_node2id[node] = id
        sample_id2node[id] = node

    while len(sample_nodes) < max_node_size:
        sample_nodes.append("UNK")

    # can be change
    sample_hyperAdjs = torch.zeros(max_node_size, max_hyperedge_size)
    sample_hyperAdjs[q_index][0] = 1
    for node in sessionedge:
        sample_hyperAdjs[sample_node2id[node]][0] = 1
    
    if len(clickedge) > 0:
        sample_hyperAdjs[q_index][1] = 1
        for node in clickedge:
            sample_hyperAdjs[sample_node2id[node]][1] = 1

    sample_edge_type = [UT_TYPE for _ in range(max_hyperedge_size)]
    return sample_nodes, sample_hyperAdjs, sample_edge_type


def construct_sample_input_k(k, orderedge, clickedge, max_node_size, max_hyperedge_size):
    sample_node2id = {}
    sample_id2node = {}
    sample_nodes = [k]
    k_index = 0
    for node in orderedge:
        if node not in sample_nodes:
            sample_nodes.append(node)

    for node in clickedge:
        if node not in sample_nodes:
            sample_nodes.append(node)

    for id, node in enumerate(sample_nodes):
        sample_node2id[node] = id
        sample_id2node[id] = node

    while len(sample_nodes) < max_node_size:
        sample_nodes.append("UNK")
    # can be change

    sample_hyperAdjs = torch.zeros(max_node_size, max_hyperedge_size)
    sample_hyperAdjs[k_index][0] = 1
    for node in orderedge:
        sample_hyperAdjs[sample_node2id[node]][0] = 1
    
    if len(clickedge) > 0:
        sample_hyperAdjs[k_index][1] = 1
        for node in clickedge:
            sample_hyperAdjs[sample_node2id[node]][1] = 1

    sample_edge_type = [DO_TYPE for _ in range(max_hyperedge_size)]
    return sample_nodes, sample_hyperAdjs, sample_edge_type

def create_node_dict(nodeid2text, user_dict, domain_dict, tokenizer, maxlen=10, cls_token='[CLS]', sep_token='[SEP]'):
    tokenizer = tokenizer
    node2example = {}
    for nodeid in tqdm(nodeid2text, total=len(nodeid2text)):
        node_text = nodeid2text[nodeid]
        node_type = transnodeid2type(nodeid)
        if node_type == U_TYPE:
            input_ids = [user_dict[node_text]] + [0] * (maxlen - 1)
            segment_ids = [0] * len(input_ids)
            input_mask = [0] * len(input_ids)
        if node_type == D_TYPE:
            input_ids = [domain_dict[node_text]] + [0] * (maxlen - 1)
            segment_ids = [0] * len(input_ids)
            input_mask = [0] * len(input_ids)
        if node_type == Q_TYPE or node_type == K_TYPE:
            tokens = tokenizer.tokenize(node_text)
            tokens = [cls_token] + tokens + [sep_token]
            input_ids = tokenizer.convert_tokens_to_ids(tokens)[:maxlen]
            segment_ids = [0] * len(input_ids)
            input_mask = [1] * len(input_ids)
            padding = [0] * (maxlen - len(input_ids))
            input_ids += padding
            input_mask += padding
            segment_ids += padding
        node2example[nodeid] = HyperNodeExample(nodeid, node_type, node_text, input_ids, input_mask, segment_ids)

    return node2example

def create_edge_info(edge2id_list, node2example, edge2example=None, is_train=True):
    if not edge2example:
        edge2example = {}
    for edgeid, nodeid in tqdm(edge2id_list, total=len(edge2id_list)):
        edgetype = transedgeid2type(edgeid)
        if edgeid not in edge2example:
            edge2example[edgeid] = HyperEdgeExample(edgeid, edgetype)
        edge2example[edgeid].add_node(nodeid, is_train)
        node2example[nodeid].add_edge(edgeid, is_train)        
    return edge2example, node2example

def transtext2nodeid(data_list, qtext2nodeid, ktext2nodeid, udtext2nodeid):
    id_data_list = []
    for r in data_list:
        id_data_list.append((udtext2nodeid[r[0]], qtext2nodeid[r[1]], ktext2nodeid[r[2]], udtext2nodeid[r[3]]))
    return id_data_list


class HyperNodeExample(object):
    def __init__(self, nodeid, nodetype, nodetext, input_ids, input_masks, segement_ids):
        self.nodeid = nodeid
        self.nodetext = nodetext
        self.input_ids = input_ids
        self.input_masks = input_masks
        self.segement_ids = segement_ids
        self.nodetype = nodetype
        self.edgel = {
            UT_TYPE: [],
            DO_TYPE: [],
            IN_TYPE: []
        }
        self.dev_edgel = {
            UT_TYPE: [],
            DO_TYPE: [],
            IN_TYPE: []
        }
    
    def add_edge(self, edgeid, is_train=True):
        edgetype = transedgeid2type(edgeid)
        if is_train:
            self.edgel[edgetype].append(edgeid)
        else:
            self.dev_edgel[edgetype].append(edgeid)

class HyperEdgeExample(object):
    def __init__(self, edgeid, edgetype):
        self.edgeid = edgeid
        self.edgetype = edgetype
        self.nodel = {
            U_TYPE: [],
            D_TYPE: [],
            Q_TYPE: [],
            K_TYPE: []
        }
        self.dev_nodel = {
            U_TYPE: [],
            D_TYPE: [],
            Q_TYPE: [],
            K_TYPE: []
        }
    
    def add_node(self, nodeid, is_train=True):
        nodetype = transnodeid2type(nodeid)
        if is_train:
            self.nodel[nodetype].append(nodeid)
        else:
            self.dev_nodel[nodetype].append(nodeid)

class HyperGraphDataset(Dataset):
    def __init__(self, node2example, data_list, presample_graph_list):
        self.data_list = data_list
        self.presample_graph_list = presample_graph_list
        self.node2example = node2example

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        item = self.data_list[index]
        sample_graph = self.presample_graph_list[index]
        return item, sample_graph

    def collate_fn(self, examples):
        batch_item = [_[0] for _ in examples]
        batch_sample_graph = [_[1] for _ in examples]

        node2id = {}
        id2node = {}
        nodesize = 0
        batch_graph_nodeid = [[],[],[],[]]
        batch_graph_adj = [[],[],[],[]]
        batch_graph_edgetpye = [[],[],[],[]]

        for ugraph, qgraph, kgraph, dgraph in batch_sample_graph:
            for nodeid in ugraph[0] + qgraph[0] + kgraph[0] + dgraph[0]:
                if nodeid not in node2id:
                    node2id[nodeid] = nodesize
                    id2node[nodesize] = nodeid
                    nodesize += 1
            
            for i, graph in enumerate([ugraph, qgraph, kgraph, dgraph]):
                batch_graph_nodeid[i].append([node2id[nodeid] for nodeid in graph[0]])
                batch_graph_adj[i].append(graph[1])
                batch_graph_edgetpye[i].append(graph[2])

        input_ids = []
        input_masks = []
        input_token_type_ids = []
        node_types = []

        for i in range(nodesize):
            example = self.node2example[id2node[i]]
            input_ids.append(example.input_ids)
            input_masks.append(example.input_masks)
            input_token_type_ids.append(example.segement_ids)
            node_types.append(example.nodetype)  

        input_ids = torch.tensor(input_ids, dtype=torch.long)
        input_masks = torch.tensor(input_masks, dtype=torch.long)
        input_token_type_ids = torch.tensor(input_token_type_ids, dtype=torch.long)
        node_types = torch.tensor(node_types, dtype=torch.long)

        batch_graph_nodeid = [torch.tensor(_, dtype=torch.long) for _ in batch_graph_nodeid]
        batch_graph_adj = [torch.stack(_).long() for _ in batch_graph_adj]
        batch_graph_edgetpye = [torch.tensor(_, dtype=torch.long) for _ in batch_graph_edgetpye]       

        return input_ids, input_masks, input_token_type_ids, node_types, batch_graph_nodeid, batch_graph_adj, batch_graph_edgetpye

    

class AttentionEdgeLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super(AttentionEdgeLayer, self).__init__()
        self.in_features = in_features   
        self.out_features = out_features   
        #self.dropout = dropout    
        # self.alpha = alpha     

        self.W = nn.Parameter(torch.zeros(size=(in_features, out_features)))  
        nn.init.xavier_uniform_(self.W.data, gain=1.414) 
        self.a = nn.Parameter(torch.zeros(size=(2*out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)   
        self.leakyrelu = nn.LeakyReLU()
    
    def forward(self, X, A):

        # b * n * d
        # 
        H = torch.matmul(X, self.W)   # [N, out_features]
        B = A.shape[0]
        N = A.shape[1]    # N 
        M = A.shape[2]
        # print(N, M)
        # b * m * d
        degree = torch.sum(A, dim=1).unsqueeze(-1)
        degree = degree.masked_fill(degree == 0, 1)
        Em = torch.matmul(A.transpose(-1, -2).type_as(H), H) / degree
        a_input = torch.cat([H.repeat(1, 1, M).view(B, N*M, -1), Em.repeat(1, N, 1)], dim=1).view(B, N, -1, 2*self.out_features)
        # [B, N, M, 2*out_features]
        e = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(3))
        # [B, N, M, 1] => [B, N, M] 
        zero_vec = -1e12 * torch.ones_like(e)
        attention = torch.where(A > 0, e, zero_vec)   # [B, N, M]
        attention = F.softmax(attention, dim=2)    #  [B, N, M]
        #attention = F.dropout(attention, self.dropout, training=self.training)   # dropout
        h_prime = torch.matmul(attention.transpose(-1, -2).type_as(H), H)  # [B, M, N].[B, N, out_features] => [M, out_features]
        return h_prime 


class SingleAttentionNodeLayer(nn.Module):
    def __init__(self, in_features=256, out_features=256):
        super(SingleAttentionNodeLayer, self).__init__()
        self.in_features = in_features   
        self.out_features = out_features   
        #self.dropout = dropout    
        # self.alpha = alpha     

        self.W = nn.Parameter(torch.zeros(size=(in_features, out_features)))  
        nn.init.xavier_uniform_(self.W.data, gain=1.414) 
        self.a = nn.Parameter(torch.zeros(size=(2*out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)   
        self.leakyrelu = nn.LeakyReLU()
    
    def forward(self, X, A, centernode):

        # b * n * d
        # print(A.shape)
        # print(A[0])
        H = torch.matmul(X, self.W)   # [N, out_features]
        # b * 1 * d
        centerH = torch.matmul(centernode, self.W) 
        B = A.shape[0]
        N = A.shape[1]    # N 
        M = A.shape[2]
        # print(N, M)
        # b * n * d
        edge_mask = A[:, :, :1]
        # b * n * 1
        # degree = torch.sum(A, dim=1).unsqueeze(-1)
        # degree = degree.masked_fill(degree == 0, 1)
        # Em = torch.matmul(A.transpose(-1, -2).type_as(H), H) / degree
        a_input = torch.cat([H, centerH.repeat(1, N, 1)], dim=-1).view(B, N, -1, 2*self.out_features)
        # [B, N, 1, 2*out_features]
        e = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(3))
        # [B, N, 1, 1] => [B, N, 1] 
        zero_vec = -1e12 * torch.ones_like(e)
        attention = torch.where(edge_mask > 0, e, zero_vec)   # [B, N, 1]
        attention = F.softmax(attention, dim=1)    #  [B, N, 1]
        #attention = F.dropout(attention, self.dropout, training=self.training)   # dropout
        h_prime = torch.matmul(attention.transpose(-1, -2).type_as(X), X)  # [B, 1, N].[B, N, out_features] => [B, 1, out_features]
        return h_prime 

class InterEdgeAggregate(nn.Module):
    def __init__(self, aggregate_type='mean', node_dim=256, edge_dim=256):
        super(InterEdgeAggregate, self).__init__()
        self.aggregate_type = aggregate_type
        if aggregate_type == 'MLP':
            self.aggregation = nn.Sequential(
                nn.Linear(node_dim*4, edge_dim)
            )
        if self.aggregate_type == 'attention':
            self.attentionLayer = AttentionEdgeLayer(node_dim, node_dim)

    def forward(self, X, A):
        if self.aggregate_type == 'mean':
            #Xe = scatter(X, A, dim=0, reduce='mean')
            Xe = torch.matmul(A.transpose(-1, -2).type_as(X), X) / torch.sum(A, dim=1).unsqueeze(-1)
        if self.aggregate_type == 'MLP':
            index = torch.arange(X.shape[0]).to(X.device)
            # E x 3
            index_per_edge = torch.masked_select(index, A.T == 1)
            index_per_edge = index_per_edge.reshape(A.T.shape[0], A.T.shape[1], -1)
            assert index_per_edge.shape == (A.T.shape[0], A.T.shape[1], 3)
            # E x (3 * d)
            node_cat_edge = X[index_per_edge].reshape(A.T.shape[0], -1)
            Xe = self.aggregation(node_cat_edge)
        if self.aggregate_type == 'attention':
            Xe = self.attentionLayer(X, A)        
        return Xe

class HyperEdgeAggregate(nn.Module):
    def __init__(self, user_agg_type='mean', inter_agg_type='mean', domain_agg_type='mean'):
        super(HyperEdgeAggregate, self).__init__()
        self.interEdgeAggregate = InterEdgeAggregate(inter_agg_type)
        self.domainEdgeAggregate = HyperEdgeLayer(domain_agg_type)
        self.userEdgeAggregate = HyperEdgeLayer(user_agg_type)

    def forward(self, X, A, T):
        
        # b * 1 * m
        eu_index = (T == UT_TYPE).unsqueeze(1).transpose(-1, -2)
        ei_index = (T == IN_TYPE).unsqueeze(1).transpose(-1, -2)
        ed_index = (T == DO_TYPE).unsqueeze(1).transpose(-1, -2)
        # b * m * d
        Eu = self.userEdgeAggregate(X, A)
        Ei = self.interEdgeAggregate(X, A)
        Ed = self.domainEdgeAggregate(X, A)

        edge_states = Eu * eu_index + Ei * ei_index + Ed * ed_index
        return edge_states

class NodeAggregateLayer(nn.Module):
    def __init__(self, aggregate_type='mean', node_dim=256):
        super(NodeAggregateLayer, self).__init__()
        self.aggregate_type = aggregate_type
        if self.aggregate_type == 'attention':
            self.attentionLayer = AttentionEdgeLayer(node_dim, node_dim)

    def forward(self, E, A):
        Ev = E
        if self.aggregate_type == 'mean':
            # degree = torch.sum(A, dim=2).unsqueeze(-1)
            # degree = degree.masked_fill(degree == 0, 1)
            # Ev = torch.matmul(A.type_as(E), E) / degree
            centerA = A[:, :1, :]
            Ev = torch.matmul(centerA.type_as(E), E) / torch.sum(centerA, dim=2)
        if self.aggregate_type == 'attention':
            Ev = self.attentionLayer(E, A.transpose(-1, -2))  
        return Ev

class HyperEdgeLayer(nn.Module):
    def __init__(self, aggregate_type='mean', node_dim=256):
        super(HyperEdgeLayer, self).__init__()
        self.aggregate_type = aggregate_type
        if self.aggregate_type == 'attention':
            self.attentionLayer = AttentionEdgeLayer(node_dim, node_dim)
        if self.aggregate_type == 'proj_mean':
            self.proj = nn.Linear(node_dim, node_dim)

    def forward(self, X, A):
        Xe = X
        if self.aggregate_type == 'mean':
            #Xe = scatter(X, A, dim=0, reduce='mean')
            degree = torch.sum(A, dim=1).unsqueeze(-1)
            degree = degree.masked_fill(degree == 0, 1)
            Xe = torch.matmul(A.transpose(-1, -2).type_as(X), X) / degree
        if self.aggregate_type == 'proj_mean':
            X = F.normalize(self.proj(X), dim=-1)
            degree = torch.sum(A, dim=1).unsqueeze(-1)
            degree = degree.masked_fill(degree == 0, 1)
            Xe = torch.matmul(A.transpose(-1, -2).type_as(X), X) / degree 
        elif self.aggregate_type == 'attention':
            Xe = self.attentionLayer(X, A)
        return Xe


class UserHGNN(nn.Module):
    def __init__(self, agg_type='mean'):
        super(UserHGNN, self).__init__()
        self.personalClickEdgeAggregate = HyperEdgeLayer('mean')
        # self.nodeAggregate = SingleAttentionNodeLayer()
        self.nodeAggregate = NodeAggregateLayer()
    def forward(self, X, A, T):
        # eu_index = (T == PC_TYPE).unsqueeze(1).transpose(-1, -2)
        # # ei_index = (T == IN_TYPE).unsqueeze(1).transpose(-1, -2)
        # # b * m * d
        edge_states = self.personalClickEdgeAggregate(X, A)
        # Ei = self.interEdgeAggregate(X, A)

        # edge_states = Eu * eu_index

        # Xv = self.nodeAggregate(edge_states, A.transpose(-1, -2), X[:, :1, :])
        Xv = self.nodeAggregate(edge_states, A)
        # Xv = X
        return Xv  

class AdverHGNN(nn.Module):
    def __init__(self, agg_type='mean'):
        super(AdverHGNN, self).__init__()
        self.adverClickEdgeAggregate = HyperEdgeLayer('mean')  
        # self.nodeAggregate = SingleAttentionNodeLayer()
        self.nodeAggregate = NodeAggregateLayer()

    def forward(self, X, A, T):
        # eu_index = (T == AC_TYPE).unsqueeze(1).transpose(-1, -2)
        # ei_index = (T == IN_TYPE).unsqueeze(1).transpose(-1, -2)
        # b * m * d
        edge_states = self.adverClickEdgeAggregate(X, A)
        # Ei = self.interEdgeAggregate(X, A)
        # edge_states = Eu * eu_index
        # Xv = self.nodeAggregate(edge_states, A.transpose(-1, -2), X[:, :1, :])
        Xv = self.nodeAggregate(edge_states, A)
        # Xv = X
        return Xv 

class QueryHGNN(nn.Module):
    def __init__(self, user_agg_type='mean'):
        super(QueryHGNN, self).__init__()
        self.sessionEdgeAggregate = HyperEdgeLayer(user_agg_type)
        self.localEdgeAggregate = HyperEdgeLayer(user_agg_type)
        self.nodeAggregate = NodeAggregateLayer(aggregate_type=user_agg_type)

        #self.nodeAggregate = SingleAttentionNodeLayer()
    def forward(self, X, A, T):
        # b * 1 * m
        eu_index = (T == UT_TYPE).unsqueeze(1).transpose(-1, -2)
        # ei_index = (T == IN_TYPE).unsqueeze(1).transpose(-1, -2)
        # b * m * d
        Eu = self.sessionEdgeAggregate(X, A)
        # Ei = self.interEdgeAggregate(X, A)

        edge_states = Eu * eu_index

        #Xv = self.nodeAggregate(edge_states, A.transpose(-1, -2), X[:, :1, :])
        Xv = self.nodeAggregate(edge_states, A)

        return Xv  

class KeywordHGNN(nn.Module):
    def __init__(self, user_agg_type='mean'):
        super(KeywordHGNN, self).__init__()
        self.orderEdgeAggregate = HyperEdgeLayer(user_agg_type)
        self.localEdgeAggregate = HyperEdgeLayer(user_agg_type)
        self.nodeAggregate = NodeAggregateLayer(aggregate_type=user_agg_type)
        # self.nodeAggregate = SingleAttentionNodeLayer()
    def forward(self, X, A, T):
        # b * 1 * m
        eo_index = (T == DO_TYPE).unsqueeze(1).transpose(-1, -2)
        # ei_index = (T == IN_TYPE).unsqueeze(1).transpose(-1, -2)
        # b * m * d
        Eo = self.orderEdgeAggregate(X, A)
        # Ei = self.interEdgeAggregate(X, A)
        edge_states = Eo * eo_index
        # Xv = self.nodeAggregate(edge_states, A.transpose(-1, -2), X[:, :1, :])
        Xv = self.nodeAggregate(edge_states, A)

        return Xv  


class NodeEncoder(nn.Module):
    def __init__(self, text_encoder='bert-base-uncased', encoder_dim=768, node_dim=256, user_size=222000, domain_size=29531, pretrain_user=None, pretrain_domain=None):
        super(NodeEncoder, self).__init__()
        if pretrain_domain:
            domain_emb = np.load(pretrain_domain)
            self.domain_embedding = nn.Embedding.from_pretrained(torch.from_numpy(domain_emb).float())
            self.domain_embedding.weight.requires_grad = True
            logger.info(f"initial domain embedding from {pretrain_domain}")
        else:
            self.domain_embedding = nn.Embedding(domain_size, encoder_dim)
            logger.info("random initial domain embedding")
        if pretrain_user:
            user_emb = np.load(pretrain_user)
            self.user_embedding = nn.Embedding.from_pretrained(torch.from_numpy(user_emb).float())
            self.user_embedding.weight.requires_grad = True
            logger.info(f"initial user embedding from {pretrain_domain}")
        else:
            self.user_embedding = nn.Embedding(user_size, encoder_dim)
            logger.info("random initial user embedding")
        bert_config = BertConfig.from_pretrained(text_encoder)
        bert_config.output_hidden_states = True
        self.q_model = BertModel.from_pretrained(text_encoder, config=bert_config)
        #self.k_model = BertModel.from_pretrained(text_encoder, config=bert_config)
        self.k_model = self.q_model
        self.node_dim = node_dim
        self.q_projection = nn.Linear(encoder_dim, node_dim)
        #self.k_projection = nn.Linear(encoder_dim, node_dim)
        self.k_projection = self.q_projection
        self.d_projection = nn.Linear(encoder_dim, node_dim)
        self.u_projection = nn.Linear(encoder_dim, node_dim)

    def forward(self, nodes, masks, segments, types):

        u_index = (types == U_TYPE).nonzero().squeeze(1)
        q_index = (types == Q_TYPE).nonzero().squeeze(1)
        k_index = (types == K_TYPE).nonzero().squeeze(1)
        d_index = (types == D_TYPE).nonzero().squeeze(1)

        h1 = torch.zeros(nodes.shape[0], self.node_dim).to(nodes.device)
        if u_index.numel():
            ids_u = nodes[u_index][:, 0]

            u_output = self.user_embedding(ids_u)
            # print("u_output")
            # print(u_output[:10, :10])
            u_h = self.u_projection(u_output)
            h1[u_index] = u_h


        if q_index.numel():
            ids_q = nodes[q_index]
            mask_q = masks[q_index]
            token_type_ids_q = segments[q_index]
            # print("ids_q")
            # print(ids_q[:10, :10])
            q_output = self.q_model(ids_q, mask_q, token_type_ids_q)
            q_cls = q_output[0][:, 0]
            # print("q_output")
            # print(q_cls[:10, :10])
            q_h = self.q_projection(q_cls)
            h1[q_index] = q_h

        if k_index.numel():
            ids_k = nodes[k_index]
            # print("ids_k")
            # print(ids_k[:10, :10])
            mask_k = masks[k_index]
            token_type_ids_k = segments[k_index]
            k_output = self.k_model(ids_k, mask_k, token_type_ids_k)
            k_cls = k_output[0][:, 0]
            # print("k_output")
            # print(k_cls[:10, :10])
            k_h = self.k_projection(k_cls)
            h1[k_index] = k_h

        if d_index.numel():
            ids_d = nodes[d_index][:, 0]
            d_output = self.domain_embedding(ids_d)
            # print("d_output")
            # print(d_output[:10, :10])
            d_h = self.d_projection(d_output)
            h1[d_index] = d_h

        return h1 


class Pass(nn.Module):
    def __init__(self, text_encoder='bert-base-uncased', encoder_dim=768, node_dim=256, user_size=222000, domain_size=29531,
    user_agg_type='mean', adver_agg_type='mean', query_agg_type='mean', keyword_agg_type='mean', pretrain_user=None, pretrain_domain=None):
        super(Pass, self).__init__()
        self.nodeEncoder = NodeEncoder(text_encoder=text_encoder, 
                                        encoder_dim=encoder_dim, 
                                        node_dim=node_dim,
                                        user_size=user_size,
                                        domain_size=domain_size,
                                        pretrain_user=pretrain_user,
                                        pretrain_domain=pretrain_domain)
        self.userhgnn = UserHGNN(user_agg_type)
        self.adverhgnn = AdverHGNN(adver_agg_type)
        self.queryhgnn = QueryHGNN(query_agg_type)
        self.keywordhgnn = KeywordHGNN(keyword_agg_type)
        self.uq_projection = nn.Linear(node_dim*2, node_dim)
        self.dk_projection = nn.Linear(node_dim*2, node_dim)

    def forward(self, input_ids, input_masks, input_token_type_ids, node_types, 
                batch_node_id, batch_adj_matrix, batch_edge_type):
        
        node_embeddings = F.normalize(self.nodeEncoder(input_ids, input_masks, input_token_type_ids, node_types), dim=-1)
        batch_ugraph_node_representation = node_embeddings[batch_node_id[0]]
        batch_qgraph_node_representation = node_embeddings[batch_node_id[1]]
        batch_kgraph_node_representation = node_embeddings[batch_node_id[2]]
        batch_dgraph_node_representation = node_embeddings[batch_node_id[3]]

        batch_ugraph_hidden = self.userhgnn(batch_ugraph_node_representation, batch_adj_matrix[0], batch_edge_type[0])
        batch_qgraph_hidden = self.queryhgnn(batch_qgraph_node_representation, batch_adj_matrix[1], batch_edge_type[1])
        batch_kgraph_hidden = self.keywordhgnn(batch_kgraph_node_representation, batch_adj_matrix[2], batch_edge_type[2])
        batch_dgraph_hidden = self.adverhgnn(batch_dgraph_node_representation, batch_adj_matrix[3], batch_edge_type[3])

        batch_u_hidden = (batch_ugraph_hidden[:, 0] + batch_ugraph_node_representation[:, 0]) / 2.0
        batch_q_hidden = (batch_qgraph_hidden[:, 0] + batch_qgraph_node_representation[:, 0]) / 2.0
        batch_k_hidden = (batch_kgraph_hidden[:, 0] + batch_kgraph_node_representation[:, 0]) / 2.0
        batch_d_hidden = (batch_dgraph_hidden[:, 0] + batch_dgraph_node_representation[:, 0]) / 2.0

        batch_uq_embs = torch.cat((batch_u_hidden, batch_q_hidden), -1)
        output1 = F.normalize(self.uq_projection(batch_uq_embs), dim=-1)

        batch_dk_embs = torch.cat((batch_d_hidden, batch_k_hidden), -1)
        output2 = F.normalize(self.dk_projection(batch_dk_embs), dim=-1)
        
        logits = torch.matmul(output1, output2.transpose(-1, -2))
        return logits

    
    def encode_user_query(self, input_ids, input_masks, input_token_type_ids, node_types, 
                batch_node_id, batch_adj_matrix, batch_edge_type):
        
        node_embeddings = F.normalize(self.nodeEncoder(input_ids, input_masks, input_token_type_ids, node_types), dim=-1)
        # b * n * d
        batch_ugraph_node_representation = node_embeddings[batch_node_id[0]]
        batch_qgraph_node_representation = node_embeddings[batch_node_id[1]]
        # b * m * d
        # print("batch_node_representation")
        # print(batch_node_representation[:1, :10, :10])
        batch_ugraph_hidden = self.userhgnn(batch_ugraph_node_representation, batch_adj_matrix[0], batch_edge_type[0])
        batch_qgraph_hidden = self.queryhgnn(batch_qgraph_node_representation, batch_adj_matrix[1], batch_edge_type[1])
        # print("batch_edge_features")
        # print(batch_edge_features[:1, :10, :10])
        batch_u_hidden = batch_ugraph_hidden[:, 0]
        batch_q_hidden = batch_qgraph_hidden[:, 0]
        # b * d
        # print(node_hidden_states[:1, :10, :10])
        batch_uq_embs = torch.cat((batch_u_hidden, batch_q_hidden), -1)
        output = F.normalize(self.uq_projection(batch_uq_embs), dim=-1)
        # print(output[:10, :10])
        return output 

    def encode_domain_keyword(self, input_ids, input_masks, input_token_type_ids, node_types, 
                batch_node_id, batch_adj_matrix, batch_edge_type):
        node_embeddings = F.normalize(self.nodeEncoder(input_ids, input_masks, input_token_type_ids, node_types))
        # b * n * d
        batch_kgraph_node_representation = node_embeddings[batch_node_id[1]]
        batch_dgraph_node_representation = node_embeddings[batch_node_id[0]]
        # b * m * d
        batch_kgraph_hidden = self.keywordhgnn(batch_kgraph_node_representation, 
                                                batch_adj_matrix[1], 
                                                batch_edge_type[1])

        batch_dgraph_hidden = self.adverhgnn(batch_dgraph_node_representation, 
                                            batch_adj_matrix[0], 
                                            batch_edge_type[0])
        # b * n * d
        batch_k_hidden = batch_kgraph_hidden[:, 0]
        batch_d_hidden = batch_dgraph_hidden[:, 0]
        # b * d

        batch_dk_embs = torch.cat((batch_d_hidden, batch_k_hidden), -1)
        output = F.normalize(self.dk_projection(batch_dk_embs), dim=-1)
        return output       

class BPRloss(nn.Module):
    def __init__(self, margin=0.4):
        super(BPRloss, self).__init__()

    def forward(self, logits, labels_index):
        new_logits = logits
        labels_index = labels_index.squeeze(0)
        for i, index in enumerate(labels_index):
            new_logits[i] = torch.cat((logits[i][index:index+1], logits[i][:index], logits[i][index+1:]))
        pos_si = new_logits[:, 0]
        neg_si = new_logits[:, 1:]
        diff = pos_si[:, None] - neg_si
        bpr_loss = - diff.sigmoid().log().mean(1)
        bpr_loss_batch_mean = bpr_loss.mean()
        return bpr_loss_batch_mean