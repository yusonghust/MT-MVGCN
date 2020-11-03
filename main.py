#-*-coding:utf-8-*-
import tensorflow as tf
from Graph import Graph
from Config import Config
from Mtmv import MTMV
from Utils import *
import networkx as nx
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import warnings
warnings.filterwarnings("ignore")
import os


def parse_args():
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter,
                            conflict_handler='resolve')
    parser.add_argument('--views',required=True,help='input view edgelist files')
    parser.add_argument('--nodes',required=True,help='input node file')
    parser.add_argument('--edges',default=None,help='input edge label file for link prediction')
    parser.add_argument('--node_labels',default=None,help='input node label file')
    parser.add_argument('--node_features',default=None,help='input node feature files')
    parser.add_argument('--epochs',default=1000,type=int,help='number of training epochs')
    parser.add_argument('--layers',default=3,type=int,help='number of gcn layers')
    parser.add_argument('--lr',default=0.001,type=float,help='learning rate')
    parser.add_argument('--clf_ratio',default=0.5,type=float,help='training data ratio')
    parser.add_argument('--drop_rate',default=0.5,type=float,help='dropout rate')
    parser.add_argument('--alpha',default=0.5,type=float,help='the importance of task_specific attention')
    parser.add_argument('--beta',default=1.0,type=float,help='the importance of node classification task')
    parser.add_argument('--gamma',default=0.01,type=float,help='the importance of reconstruction error')
    parser.add_argument('--batchsize',default=50000,type=int,help='training batchsize')
    parser.add_argument('--num_heads',default=4,type=int,help='number of heads for multi-head attention')
    parser.add_argument('--biases',action='store_true',help='Using biases in gcn layers')
    parser.add_argument('--size',default='middle',choices=[
        'mini','small','middle','large'],help='The network size will be choosen')
    parser.add_argument('--cuda_devices',default='0,1',type=str,help='which gpus will be used')
    parser.add_argument('--act',default='tanh',choices=[
        'relu',
        'leaky_relu',
        'tanh',
        'sigmoid'
        ],help='The activation function will be used')
    parser.add_argument('--average',default='weighted',choices=[
        'weighted',
        'micro',
        'macro',
        'samples'
        ],help='The average type will be used in auc')
    parser.add_argument('--task',default='link_and_multilabel',choices=[
        'link_and_multilabel', #linkprediction + multilabel classification
        'link_and_multiclass', #linkprediction + multiclass classification
        'multiclass', #multilabel classification
        'multilabel', #multiclass classification
        'linkpred'],help='The task will be evaluated')
    parser.add_argument('--model',default='model',choices=[
        'model',
        'mvgcn'],help='The model will be evaluated')
    args = parser.parse_args()
    return args

class mtmv():
    def __init__(self,cfg):
        self.cfg = cfg

        self.cfg.update_config('gpus',self.cfg.cuda_devices.split(','))

        with open(cfg.edgelists,'r') as f:
            self.views = []
            print('read edgelist......')
            for line in f:
                ls = line.strip().split()
                edgelist = ls[0]
                directed = bool(ls[1])
                weighted = bool(ls[2])
                self.views.append(Graph(edgelist,weighted,directed,cfg.node_labels,cfg.node_features))
        self.cfg.update_config('views',self.views)
        self.cfg.update_config('view_nums',len(self.views))
        f.close()

        self.node_list = []
        self.look_up = dict() #map node to id
        with open(cfg.nodes,'r') as f:
            i = 0
            for line in f:
                ls = line.strip().split()
                self.node_list.append(int(ls[0]))
                self.look_up[int(ls[0])] = i
                i += 1
        self.cfg.update_config('node_list',self.node_list)
        self.cfg.update_config('look_up',self.look_up)
        self.cfg.update_config('node_size',len(self.node_list))
        f.close()

        self.has_features   = False if cfg.node_features is None else True
        self.cfg.update_config('has_features',self.has_features)
        self.has_node_label = False if cfg.node_labels is None else True
        self.cfg.update_config('has_node_label',self.has_node_label)
        self.has_edge_label = False if cfg.edges is None else True
        self.cfg.update_config('has_edge_label',self.has_edge_label)

        self.node_mask = dict()
        if self.has_node_label is True:
            print('preprocess node labels......')
            self.node_label_mat,self.node_mask['train'],self.node_mask['val'],self.node_mask['test'] = preprocess_node_labels(self.cfg)
            self.cfg.update_config('node_mask',self.node_mask)
            self.cfg.update_config('node_label_mat',self.node_label_mat)
        if self.has_edge_label is True:
            print('preprocess edge labels......')
            self.src_node,self.dst_node,self.edge_label_mat= preprocess_edge_labels(self.cfg)
            self.cfg.update_config('src_node',self.src_node)
            self.cfg.update_config('dst_node',self.dst_node)
            self.cfg.update_config('edge_label_mat',self.edge_label_mat)
            # self.cfg.update_config('pos_num',pos_num)
            # self.cfg.update_config('neg_num',neg_num)

        self.adj_mat = dict()
        print('preprocess adjacency matrix......')
        if self.has_features is False:
            for i in range(len(self.views)):
                self.adj_mat['view_%d'%i],self.feature_mat = preprocess_data(self.cfg,i)
        else:
            self.feature_mat = dict()
            for i in range(len(self.views)):
                self.adj_mat['view_%d'%i],self.feature_mat['view_%d'%i] = preprocess_data(self.cfg,i)
        self.cfg.update_config('adj_mat',self.adj_mat)
        self.cfg.update_config('feature_mat',self.feature_mat)

        if cfg.size == 'mini':
            self.num_units = 16
        elif cfg.size == 'small':
            self.num_units = 32
        elif cfg.size == 'middle':
            self.num_units = 64
        elif cfg.size == 'large':
            self.num_units = 128
        # if cfg.model=='mvgcn':
        #     self.cfg.update_config('num_units',int(self.cfg.num_units/self.cfg.view_nums))
        # else:
        self.cfg.update_config('num_units',self.num_units)

        if cfg.act == 'relu':
            self.act = tf.nn.relu
        elif cfg.act == 'leaky_relu':
            self.act = tf.nn.leaky_relu
        elif cfg.act == 'tanh':
            self.act = tf.nn.tanh
        elif cfg.act == 'sigmoid':
            self.act = tf.nn.sigmoid
        self.cfg.update_config('act',self.act)

        self.Network = MTMV(self.cfg)
        self.Network.train_and_evaluate()
        print('current experiment setting is','model = ',self.cfg.model, 'clf_ratio = ',self.cfg.clf_ratio, 'alpha = ',self.cfg.alpha,'beta = ',self.cfg.beta, 'gamma = ',self.cfg.gamma,'layers = ',self.cfg.layers, 'size = ',self.cfg.size, 'average =',self.cfg.average)

def main(args):
    cfg = Config(args)
    M   = mtmv(cfg)

if __name__ == '__main__':
    main(parse_args())









