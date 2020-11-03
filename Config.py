#-*-coding:utf-8-*-
import numpy as np
import tensorflow as tf
import os

class Config:
    def __init__(self,args):
        self._configs = {}
        #parameters
        self._configs['edgelists']          = args.views
        self._configs['nodes']              = args.nodes
        self._configs['edges']              = args.edges
        self._configs['node_labels']        = args.node_labels
        self._configs['node_features']      = args.node_features
        self._configs['size']               = args.size
        self._configs['cuda_devices']       = args.cuda_devices
        self._configs['num_heads']          = args.num_heads
        self._configs['epochs']             = args.epochs
        self._configs['layers']             = args.layers
        self._configs['lr']                 = args.lr
        self._configs['clf_ratio']          = args.clf_ratio
        self._configs['drop_rate']          = args.drop_rate
        self._configs['alpha']              = args.alpha
        self._configs['beta']               = args.beta
        self._configs['gamma']              = args.gamma
        self._configs['act']                = args.act
        self._configs['task']               = args.task
        self._configs['model']              = args.model
        self._configs['biases']             = args.biases
        self._configs['batchsize']          = args.batchsize
        self._configs['average']            = args.average
        self._configs['view_nums']          = 0
        self._configs['node_size']          = 0
        self._configs['num_units']          = 64
        self._configs['pos_num']            = 0
        self._configs['neg_num']            = 0
        self._configs['views']              = []
        self._configs['gpus']               = []
        self._configs['node_list']          = []
        self._configs['has_features']       = False
        self._configs['has_node_label']     = False
        self._configs['has_edge_label']     = False
        self._configs['look_up']            = dict()
        self._configs['adj_mat']            = dict()
        self._configs['node_mask']          = dict()
        self._configs['feature_mat']        = dict()
        self._configs['node_label_mat']     = dict()
        self._configs['src_node']           = dict()
        self._configs['dst_node']           = dict()
        self._configs['edge_label_mat']     = dict()


    @property
    def edgelists(self):
        return self._configs['edgelists']

    @property
    def nodes(self):
        return self._configs['nodes']

    @property
    def edges(self):
        return self._configs['edges']

    @property
    def node_labels(self):
        return self._configs['node_labels']

    @property
    def node_features(self):
        return self._configs['node_features']

    @property
    def size(self):
        return self._configs['size']

    @property
    def epochs(self):
        return self._configs['epochs']

    @property
    def layers(self):
        return self._configs['layers']

    @property
    def lr(self):
        return self._configs['lr']

    @property
    def clf_ratio(self):
        return self._configs['clf_ratio']

    @property
    def drop_rate(self):
        return self._configs['drop_rate']

    @property
    def alpha(self):
        return self._configs['alpha']

    @property
    def beta(self):
        return self._configs['beta']

    @property
    def gamma(self):
        return self._configs['gamma']

    @property
    def act(self):
        return self._configs['act']

    @property
    def task(self):
        return self._configs['task']

    @property
    def model(self):
        return self._configs['model']

    @property
    def cuda_devices(self):
        return self._configs['cuda_devices']

    @property
    def batchsize(self):
        return self._configs['batchsize']

    @property
    def num_heads(self):
        return self._configs['num_heads']

    @property
    def num_units(self):
        return self._configs['num_units']

    @property
    def biases(self):
        return self._configs['biases']

    @property
    def view_nums(self):
        return self._configs['view_nums']

    @property
    def node_size(self):
        return self._configs['node_size']

    @property
    def pos_num(self):
        return self._configs['pos_num']

    @property
    def neg_num(self):
        return self._configs['neg_num']

    @property
    def views(self):
        return self._configs['views']

    @property
    def gpus(self):
        return self._configs['gpus']

    @property
    def average(self):
        return self._configs['average']

    @property
    def has_features(self):
        return self._configs['has_features']

    @property
    def has_node_label(self):
        return self._configs['has_node_label']

    @property
    def has_edge_label(self):
        return self._configs['has_edge_label']

    @property
    def node_list(self):
        return self._configs['node_list']

    @property
    def look_up(self):
        return self._configs['look_up']

    @property
    def adj_mat(self):
        return self._configs['adj_mat']

    @property
    def node_mask(self):
        return self._configs['node_mask']

    @property
    def feature_mat(self):
        return self._configs['feature_mat']

    @property
    def node_label_mat(self):
        return self._configs['node_label_mat']

    @property
    def src_node(self):
        return self._configs['src_node']

    @property
    def dst_node(self):
        return self._configs['dst_node']

    @property
    def edge_label_mat(self):
        return self._configs['edge_label_mat']

    def update_config(self,key,value):
        if key in self._configs.keys():
            self._configs[key] = value
        else:
            raise RuntimeError('Update_Config_Error')