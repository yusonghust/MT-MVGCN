# -*- coding: utf-8 -*-
import tensorflow as tf
import time
from sklearn.metrics import roc_auc_score as auc
from sklearn.metrics import average_precision_score as ap
from Layers import GraphConvLayer
from Graph import Graph
from Utils import *
import os



class MTMV():
    def __init__(self,cfg):

        os.environ["CUDA_VISIBLE_DEVICES"] = cfg.cuda_devices
        self.cfg = cfg
        self.build_placeholders()

    def build_placeholders(self):

        self.placeholders = dict()
        if self.cfg.has_features is False:
            self.placeholders['features']          = tf.sparse_placeholder(tf.float32, shape=tf.constant(self.cfg.feature_mat[2], dtype=tf.int64))
            self.placeholders['is_training']       = tf.placeholder(tf.bool)
            for i in range(self.cfg.view_nums):
                self.placeholders['adj_%d'%i]      = tf.sparse_placeholder(tf.float32,shape=tf.constant(self.cfg.adj_mat['view_%d'%i][2],dtype=tf.int64))
        else:
            self.placeholders['is_training']       = tf.placeholder(tf.bool)
            for i in range(self.cfg.view_nums):
                self.placeholders['adj_%d'%i]      = tf.sparse_placeholder(tf.float32,shape=tf.constant(self.cfg.adj_mat['view_%d'%i][2],dtype=tf.int64))
                self.placeholders['features_%d'%i] = tf.sparse_placeholder(tf.float32,shape=tf.constant(self.cfg.feature_mat['view_%d'%i][2], dtype=tf.int64))

        if self.cfg.has_node_label is True:
            print('build_placeholders for node classification')
            self.placeholders['node_labels']       = tf.placeholder(tf.int32, shape=self.cfg.node_label_mat.shape)#node classification
            self.placeholders['node_labels_mask']  = tf.placeholder(tf.int32) #node classification
        if self.cfg.has_edge_label is True:
            print('build_placeholders for link prediction')
            self.placeholders['src_node']          = tf.placeholder(tf.int32) #link prediction
            self.placeholders['dst_node']          = tf.placeholder(tf.int32) #link prediction
            self.placeholders['edge_labels']       = tf.placeholder(tf.int32, shape=(None,self.cfg.view_nums)) #link prediction

    def construct_feed_dict_linkpred(self,src_node,dst_node,edge_labels,is_training=True):

        feed_dict = dict()
        feed_dict.update({self.placeholders['is_training']:is_training})

        if self.cfg.has_features is False:
            feed_dict.update({self.placeholders['features']:self.cfg.feature_mat})
            for i in range(self.cfg.view_nums):
                feed_dict.update({self.placeholders['adj_%d'%i]:self.cfg.adj_mat['view_%d'%i]})
        else:
            for i in range(self.cfg.view_nums):
                feed_dict.update({self.placeholders['features_%d'%i]:self.cfg.feature_mat['view_%d'%i]})
                feed_dict.update({self.placeholders['adj_%d'%i]:self.cfg.adj_mat['view_%d'%i]})

        feed_dict.update({self.placeholders['src_node']:src_node})
        feed_dict.update({self.placeholders['dst_node']:dst_node})
        feed_dict.update({self.placeholders['edge_labels']:edge_labels})
        return feed_dict

    def construct_feed_dict_nodecls(self,node_mask,is_training=True):

        feed_dict = dict()
        feed_dict.update({self.placeholders['is_training']:is_training})
        feed_dict.update({self.placeholders['node_labels']:self.cfg.node_label_mat})
        feed_dict.update({self.placeholders['node_labels_mask']:node_mask})
        return feed_dict

    def construct_feed_dict_multi(self,src_node,dst_node,edge_labels,node_mask,is_training=True):

        feed_dict = dict()
        feed_dict.update({self.placeholders['is_training']:is_training})

        if self.cfg.has_features is False:
            feed_dict.update({self.placeholders['features']:self.cfg.feature_mat})
            for i in range(self.cfg.view_nums):
                feed_dict.update({self.placeholders['adj_%d'%i]:self.cfg.adj_mat['view_%d'%i]})
        else:
            for i in range(self.cfg.view_nums):
                feed_dict.update({self.placeholders['features_%d'%i]:self.cfg.feature_mat['view_%d'%i]})
                feed_dict.update({self.placeholders['adj_%d'%i]:self.cfg.adj_mat['view_%d'%i]})

        feed_dict.update({self.placeholders['src_node']:src_node})
        feed_dict.update({self.placeholders['dst_node']:dst_node})
        feed_dict.update({self.placeholders['edge_labels']:edge_labels})
        feed_dict.update({self.placeholders['node_labels']:self.cfg.node_label_mat})
        feed_dict.update({self.placeholders['node_labels_mask']:node_mask})
        return feed_dict

    def gcn(self,inputs,indim,view_id):

        L = self.cfg.layers
        for i in range(L):
            if i==0:
                sparse  = True
                y       = inputs
            else:
                sparse  = False
                y       = tf.layers.dropout(y,self.cfg.drop_rate,training = self.placeholders['is_training'])
            y           = GraphConvLayer(self.cfg,
                                        input_dim  = indim,
                                        output_dim = self.cfg.num_units,
                                        name       ='gc%d'%i)(adj_norm = self.placeholders['adj_%d'%view_id], x = y, sparse = sparse)
            indim       = self.cfg.num_units
        return y

    def gcns(self):

        if self.cfg.has_features is False:
            inputs      = self.placeholders['features']
            indim       = self.cfg.feature_mat[2][1]
            gcn_outputs = []
            for i in range(self.cfg.view_nums):
                gcn_outputs.append(self.gcn(inputs,indim,i))

        else:
            gcn_outputs = []
            for i in range(self.cfg.view_nums):
                inputs  = self.placeholders['features_%d'%i]
                indim   = self.cfg.feature_mat['view_%d'%i][2][1]
                gcn_outputs.append(self.gcn(inputs,indim,i))
        return gcn_outputs

    def mvgcn(self):

        gcn_outputs     = self.gcns()
        gcn_concate     = tf.concat(gcn_outputs,axis=1)
        return gcn_concate

    def Dense(self,inputs,output_size,biases=True,scope=None):
        #simple dense layer
        input_size = int(inputs.shape[-1])
        with tf.variable_scope(scope) as scope:
            W = tf.Variable(tf.random_uniform([input_size, output_size], -0.05, 0.05))
            if biases:
                b = tf.Variable(tf.random_uniform([output_size], -0.05, 0.05))
            else:
                b = 0
            outputs = tf.matmul(tf.reshape(inputs,(-1,input_size)),W) + b
            outputs = tf.reshape(outputs, tf.concat([tf.shape(inputs)[:-1], [output_size]], 0))
        return outputs

    def GraphEncoder(self,inputs,output_size,biases=False,scope=None):

        #graph convolutional layer encoder
        AH = []
        for i in range(self.cfg.view_nums):
            ah      = tf.sparse_tensor_dense_matmul(self.placeholders['adj_%d'%i],inputs[i])
            AH.append(ah)
        inputs      = tf.stack(AH,axis=0)
        input_size  = int(inputs.shape[-1])
        with tf.variable_scope(scope) as scope:
            W       = tf.Variable(tf.random_uniform([input_size, output_size], -0.05, 0.05))
            if biases:
                b   = tf.Variable(tf.random_uniform([output_size], -0.05, 0.05))
            else:
                b   = 0
            outputs = tf.matmul(tf.reshape(inputs,(-1,input_size)),W) + b
            outputs = tf.reshape(outputs, tf.concat([tf.shape(inputs)[:-1], [output_size]], 0))
        return outputs

    def GraphDecoder(self,gcn_outputs,O,scope=None):

        with tf.variable_scope(scope) as scope:
            square_loss = 0
            output_size = self.cfg.num_units
            input_size  = O.get_shape().as_list()[-1]
            gcn_hat     = []
            for i in range(self.cfg.view_nums):
                W       = tf.Variable(tf.random_uniform([input_size, output_size], -0.05, 0.05),name='W_%d'%i)
                gcn_hat.append(tf.sparse_tensor_dense_matmul(self.placeholders['adj_%d'%i],tf.matmul(O,W)))
            gcn         = tf.stack(gcn_outputs,axis=0)
            gcn_hat     = tf.stack(gcn_hat,axis=0)
            square_loss = tf.reduce_mean(tf.squared_difference(gcn, gcn_hat))
        return square_loss


    def quary_key_value(self,gcn_outputs):

        gcn_outputs_stack = tf.stack(gcn_outputs,axis=0,name='gcn_outputs_stack') #num_views*num_nodes*input_size
        self.Q_view       = self.Dense(gcn_outputs_stack,self.cfg.num_units,scope='Q_view') #num_views*num_nodes*self.cfg.num_units
        self.K_view       = self.Dense(gcn_outputs_stack,self.cfg.num_units,scope='K_view') #num_views*num_nodes*self.cfg.num_units
        self.V_view       = self.GraphEncoder(gcn_outputs,self.cfg.num_units,scope='V_view') #num_views*num_nodes*self.cfg.num_units

        self.Q_link       = tf.get_variable(name='Q_link',shape=(self.cfg.view_nums,len(self.cfg.look_up),self.cfg.num_units),
                            initializer=tf.contrib.layers.xavier_initializer())
        self.K_link       = tf.get_variable(name='K_link',shape=(self.cfg.view_nums,len(self.cfg.look_up),self.cfg.num_units),
                            initializer=tf.contrib.layers.xavier_initializer())
        self.V_link       = self.GraphEncoder(gcn_outputs,self.cfg.num_units,scope='V_link')

        self.Q_cls        = tf.get_variable(name='Q_cls', shape=(self.cfg.view_nums,len(self.cfg.look_up),self.cfg.num_units),
                            initializer=tf.contrib.layers.xavier_initializer())
        self.K_cls        = tf.get_variable(name='K_cls', shape=(self.cfg.view_nums,len(self.cfg.look_up),self.cfg.num_units),
                            initializer=tf.contrib.layers.xavier_initializer())
        self.V_cls        = self.GraphEncoder(gcn_outputs,self.cfg.num_units,scope='V_cls')

        # self.V = self.GraphEncoder(gcn_outputs,output_size,scope='V')

    def Attention(self):

        self.gcn_outputs = self.gcns()
        self.quary_key_value(self.gcn_outputs)
        nb_head = self.cfg.num_heads
        size_per_head = int(self.cfg.num_units/nb_head)

        #compute attention
        Q_view = tf.reduce_mean(tf.reshape(self.Q_view,(-1,tf.shape(self.Q_view)[1],nb_head,size_per_head)),axis=0,keepdims=True) #1*num_nodes*num_heads*head_size
        Q_view = tf.transpose(Q_view, [0, 2, 1, 3]) # 1*num_heads*num_nodes*head_size
        K_view = tf.reshape(self.K_view,(-1,tf.shape(self.K_view)[1],nb_head,size_per_head)) #num_views*num_nodes*num_heads*head_size
        K_view = tf.transpose(K_view, [0, 2, 1, 3]) #num_views*num_heads*num_nodes*head_size
        V_view = tf.reshape(self.V_view,(-1,tf.shape(self.V_view)[1],nb_head,size_per_head)) #num_views*num_nodes*num_heads*head_size
        V_view = tf.transpose(V_view, [0, 2, 1, 3]) #num_views*num_heads*num_nodes*head_size

        A_view = tf.multiply(Q_view,K_view)/tf.sqrt(float(size_per_head)) #num_views*num_heads*num_nodes*head_size
        A_view = tf.nn.softmax(A_view,axis=0)
        O_view = tf.multiply(A_view,V_view)
        O_view = tf.transpose(O_view,[0,2,1,3])
        O_view = tf.nn.l2_normalize(tf.reduce_sum(tf.reshape(O_view, (-1, tf.shape(O_view)[1], nb_head * size_per_head)),axis=0),axis=1) #num_nodes*dims

        Q_link = tf.reduce_mean(tf.reshape(self.Q_link,(-1,tf.shape(self.Q_link)[1],nb_head,size_per_head)),axis=0,keepdims=True) #1*num_nodes*num_heads*head_size
        Q_link = tf.transpose(Q_link, [0, 2, 1, 3]) # 1*num_heads*num_nodes*head_size
        K_link = tf.reshape(self.K_link,(-1,tf.shape(self.K_link)[1],nb_head,size_per_head)) #num_views*num_nodes*num_heads*head_size
        K_link = tf.transpose(K_link, [0, 2, 1, 3]) #num_views*num_heads*num_nodes*head_size
        V_link = tf.reshape(self.V_link,(-1,tf.shape(self.V_link)[1],nb_head,size_per_head)) #num_views*num_nodes*num_heads*head_size
        V_link = tf.transpose(V_link, [0, 2, 1, 3]) #num_views*num_heads*num_nodes*head_size

        A_link = tf.multiply(Q_link,K_link)/tf.sqrt(float(size_per_head)) #num_views*num_heads*num_nodes*head_size
        A_link = tf.nn.softmax(A_link,axis=0)
        O_link = tf.multiply(A_link,V_link)
        O_link = tf.transpose(O_link,[0,2,1,3])
        O_link = tf.nn.l2_normalize(tf.reduce_sum(tf.reshape(O_link, (-1, tf.shape(O_link)[1], nb_head * size_per_head)),axis=0),axis=1) #num_nodes*dims


        Q_cls  = tf.reduce_mean(tf.reshape(self.Q_cls,(-1,tf.shape(self.Q_cls)[1],nb_head,size_per_head)),axis=0,keepdims=True) #1*num_nodes*num_heads*head_size
        Q_cls  = tf.transpose(Q_cls, [0, 2, 1, 3]) # 1*num_heads*num_nodes*head_size
        K_cls  = tf.reshape(self.K_cls,(-1,tf.shape(self.K_cls)[1],nb_head,size_per_head)) #num_views*num_nodes*num_heads*head_size
        K_cls  = tf.transpose(K_cls, [0, 2, 1, 3]) #num_views*num_heads*num_nodes*head_size
        V_cls  = tf.reshape(self.V_cls,(-1,tf.shape(self.V_cls)[1],nb_head,size_per_head)) #num_views*num_nodes*num_heads*head_size
        V_cls  = tf.transpose(V_cls, [0, 2, 1, 3]) #num_views*num_heads*num_nodes*head_size

        A_cls  = tf.multiply(Q_cls,K_cls)/tf.sqrt(float(size_per_head)) #num_views*num_heads*num_nodes*head_size
        A_cls  = tf.nn.softmax(A_cls,axis=0)
        O_cls  = tf.multiply(A_cls,V_cls)
        O_cls  = tf.transpose(O_cls,[0,2,1,3])
        O_cls  = tf.nn.l2_normalize(tf.reduce_sum(tf.reshape(O_cls, (-1, tf.shape(O_cls)[1], nb_head * size_per_head)),axis=0),axis=1) #num_nodes*dims

        return O_view,O_link,O_cls

    def train_and_evaluate(self):

        if self.cfg.model=='model':
            O_view,O_link,O_cls = self.Attention()
            embs_link = self.cfg.alpha*O_link + (1-self.cfg.alpha)*O_view
            embs_cls  = self.cfg.alpha*O_cls  + (1-self.cfg.alpha)*O_view

            if self.cfg.task=='link_and_multilabel' or self.cfg.task=='link_and_multiclass':
                self.multitask(embs_link,embs_cls)

            elif self.cfg.task=='linkpred':
                self.linkpred(embs_link)

            elif self.cfg.task=='multiclass' or self.cfg.task=='multilabel':
                self.nodecls(embs_cls)

            else:
                print('Error task')

        elif self.cfg.model=='mvgcn':
            gcn_concate = self.mvgcn()

            if self.cfg.task=='link_and_multilabel' or self.cfg.task=='link_and_multiclass':
                self.multitask(gcn_concate,gcn_concate)

            elif self.cfg.task=='linkpred':
                self.linkpred(gcn_concate)

            elif self.cfg.task=='multiclass' or self.cfg.task=='multilabel':
                self.nodecls(gcn_concate)

            else:
                print('Error task')

    def Sess(self):

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        sess = tf.InteractiveSession(config=config)
        init = tf.global_variables_initializer()
        sess.run(init)
        return sess

    def cosine(self,a,b):

        x = tf.reduce_sum(tf.multiply(a,b),axis=1,keepdims=True)
        y_a = tf.norm(a,axis=1)
        y_b = tf.norm(b,axis=1)
        y = tf.expand_dims(tf.multiply(y_a,y_b),1)
        return tf.reshape(tf.div(x,y),(-1,1))

    def group(self,x,kind):
        shape = x.get_shape()
        assert shape.ndims==2
        weights = tf.get_variable('weights',shape=[shape[-1].value,kind])
        biases = tf.get_variable('biases',shape=[kind])
        y = tf.nn.xw_plus_b(x,weights,biases,name='group')
        return y

    def focal_loss(self,logits,labels,alpha=0.5,gamma=2):

        w_pos = self.cfg.neg_num/(self.cfg.pos_num + self.cfg.neg_num)
        w_neg = 1 - w_pos
        pos   = -(1-alpha)*labels*tf.pow(tf.nn.sigmoid(-logits),gamma)*tf.log(tf.nn.sigmoid(logits) + 0.000001)
        neg   = -alpha*(1-labels)*tf.pow(tf.nn.sigmoid(logits),gamma)*tf.log(tf.nn.sigmoid(-logits) + 0.000001)
        loss  = w_pos*pos + w_neg*neg
        # loss = pos + neg
        return loss


    def balanced_focal_loss(self, preds, labels):

        loss = self.focal_loss(logits=preds, labels=labels)
        return tf.reduce_mean(loss)

    def multilabel_accuracy(self, preds, labels):
        preds = tf.nn.sigmoid(preds)
        pred_norm = tf.cast(tf.greater(preds,0.5),tf.int32)
        labels_norm = tf.cast(labels,tf.int32)
        over_norm = tf.bitwise.bitwise_xor(pred_norm,labels_norm)
        over_false = tf.reduce_sum(tf.cast(over_norm,tf.float32),1)
        over_norm = tf.equal(pred_norm,labels_norm)
        over_true = tf.reduce_sum(tf.cast(over_norm,tf.float32),1)
        all_norm = over_true + over_false
        accuracy= tf.reduce_mean(over_true/all_norm)
        return accuracy

    def masked_multilabel_sigmoid_cross_entropy(self, preds, labels, mask):
        """multi-label Sigmoid cross-entropy loss with masking."""
        loss = tf.losses.sigmoid_cross_entropy(multi_class_labels=labels,logits=preds)
        mask = tf.cast(mask, dtype=tf.float32)
        mask /= tf.reduce_mean(mask)
        loss_m = loss*tf.expand_dims(mask,1)
        return tf.reduce_mean(loss_m)

    def masked_multilabel_accuracy(self,preds,labels,mask):

        preds = tf.nn.sigmoid(preds)
        pred_norm = tf.cast(tf.greater(preds,0.5),tf.int32)
        labels_norm = tf.cast(labels,tf.int32)
        mask = tf.cast(mask, dtype=tf.float32)
        mask /= tf.reduce_mean(mask)
        over_norm = tf.bitwise.bitwise_xor(pred_norm,labels_norm)
        over_false = tf.reduce_sum(tf.cast(over_norm,tf.float32),1)
        over_norm = tf.equal(pred_norm,labels_norm)
        over_true = tf.reduce_sum(tf.cast(over_norm,tf.float32),1)
        all_norm = over_true + over_false
        accuracy=tf.reduce_mean((over_true/all_norm)*mask)
        return accuracy

    def masked_multiclass_softmax_cross_entropy(self, preds, labels, mask):

        loss  = tf.nn.softmax_cross_entropy_with_logits(logits=preds, labels=labels)
        mask  = tf.cast(mask, dtype=tf.float32)
        mask /= tf.reduce_mean(mask)
        loss *= mask
        return tf.reduce_mean(loss)

    def masked_multiclass_accuracy(self, preds, labels, mask):

        correct_prediction = tf.equal(tf.argmax(preds, 1), tf.argmax(labels, 1))
        accuracy_all = tf.cast(correct_prediction, tf.float32)
        mask = tf.cast(mask, dtype=tf.float32)
        mask /= tf.reduce_mean(mask)
        accuracy_all *= mask
        return tf.reduce_mean(accuracy_all)



    def cast(self,preds):
        y_0=tf.nn.sigmoid(preds)
        y_pred=0.5*tf.sign(y_0-0.5)+0.5
        y_pre=tf.cast(y_pred,tf.int32)
        return y_pre

    def linkpred(self,embs_link):
        with tf.variable_scope('linkpred') as scope:
            embs_link      = tf.nn.l2_normalize(embs_link,axis=1)
            z_src          = tf.nn.embedding_lookup(embs_link,self.placeholders['src_node'])
            z_dst          = tf.nn.embedding_lookup(embs_link,self.placeholders['dst_node'])

            # w              = tf.Variable(initial_value=tf.random_uniform(shape=(1,1)),name='w_l')
            # b              = tf.Variable(tf.zeros(1),name='b_l',dtype=tf.float32)
            cosine         = self.cosine(z_src,z_dst)
            # print(cosine.get_shape())
            preds          = self.group(cosine,self.cfg.view_nums)
            # preds          = tf.reshape(tf.matmul(preds,w) + b,[-1])

            y_true         = tf.cast(self.placeholders['edge_labels'],tf.float32)
            link_pred_loss = tf.reduce_mean(tf.losses.sigmoid_cross_entropy(multi_class_labels=y_true,logits=preds))
            link_pred_acc  = self.multilabel_accuracy(preds,y_true)
            y_pred         = self.cast(preds)
            preds          = tf.nn.sigmoid(preds)

            if self.cfg.model=='model':
                sq_loss    = self.GraphDecoder(self.gcn_outputs,embs_link,scope='square_loss')
                loss       = link_pred_loss + self.cfg.gamma*sq_loss
            else:
                loss       = link_pred_loss
        opt            = tf.train.AdamOptimizer(self.cfg.lr).minimize(loss)
        sess           = self.Sess()


        feed_dict_val = self.construct_feed_dict_linkpred(self.cfg.src_node['val'],self.cfg.dst_node['val'],self.cfg.edge_label_mat['val'],False)

        batch_size = self.cfg.batchsize
        print('\t'*150)
        start = time.time()
        for i in range(self.cfg.epochs):
            train_index       = np.random.choice(len(self.cfg.src_node['train']),batch_size,replace=False)
            feed_dict_train   = self.construct_feed_dict_linkpred(self.cfg.src_node['train'][train_index],self.cfg.dst_node['train'][train_index],self.cfg.edge_label_mat['train'][train_index],True)
            _,loss_tr,acc_tr,y_t,y_p,pred_t = sess.run([opt,loss,link_pred_acc,y_true,y_pred,preds],feed_dict = feed_dict_train)
            # auc_tr = auc(y_t,y_p)
            # ap_tr  = ap(y_t,y_p)
            if i%50==0:
                loss_v,acc_v,y_t_v,y_p_v,pred_v = sess.run([loss,link_pred_acc,y_true,y_pred,preds],feed_dict = feed_dict_val)
                auc_t = auc(y_t,pred_t,self.cfg.average)
                auc_v = auc(y_t_v,pred_v,self.cfg.average)
                # auc_v = auc(y_t_v,y_p_v)
                # ap_v  = ap(y_t_v,y_p_v)
                end = time.time()
                duration = end - start
                print('training,   step is',i,'link prediction loss is %4f'%(loss_tr),'link prediction accuracy is %4f'%(acc_tr), 'time is %4f'%(duration), 'sec/50_steps')
                print('validation, step is',i,'link prediction loss is %4f'%(loss_v), 'link prediction accuracy is %4f'%(acc_v),  'time is %4f'%(duration), 'sec/50_steps')
                print('training auc is',auc_t,'validation auc is',auc_v)
                print('\t'*150)
                start = time.time()

    def nodecls(self,embs_cls):
        with tf.variable_scope('nodecls') as scope:
            embs_cls        = tf.nn.l2_normalize(embs_cls,axis=1)
            preds           = self.group(embs_cls,self.cfg.node_label_mat.shape[1])
            y_true          = tf.cast(self.placeholders['node_labels'],tf.float32)

            if self.cfg.task=='multilabel':
                node_cls_loss   = self.masked_multilabel_sigmoid_cross_entropy(preds,y_true,self.placeholders['node_labels_mask'])
                node_cls_acc    = self.masked_multilabel_accuracy(preds,self.placeholders['node_labels'],self.placeholders['node_labels_mask'])
            elif self.cfg.task=='multiclass':
                node_cls_loss   = self.masked_multiclass_softmax_cross_entropy(preds,y_true,self.placeholders['node_labels_mask'])
                node_cls_acc    = self.masked_multiclass_accuracy(preds,self.placeholders['node_labels'],self.placeholders['node_labels_mask'])
            else:
                raise RuntimeError('Node_Classification_Task_Error')


            if self.cfg.model=='model':
                square_loss = self.GraphDecoder(self.gcn_outputs,embs_cls,scope='square_loss')
                loss        = node_cls_loss + self.cfg.gamma*square_loss
            else:
                loss        = node_cls_loss
        opt             = tf.train.AdamOptimizer(self.cfg.lr).minimize(loss)
        sess            = self.Sess()


        feed_dict_train = self.construct_feed_dict_nodecls(self.cfg.node_mask['train'],is_training=True)
        feed_dict_val   = self.construct_feed_dict_nodecls(self.cfg.node_mask['val'],is_training=False)
        feed_dict_test  = self.construct_feed_dict_nodecls(self.cfg.node_mask['test'],is_training=False)
        start = time.time()

        print('\t'*150)
        for i in range(self.cfg.epochs):
            _,cls_loss,cls_acc = sess.run([opt,node_cls_loss,node_cls_acc],feed_dict = feed_dict_train)
            if i%50 == 0:
                cls_loss_v,cls_acc_v = sess.run([node_cls_loss,node_cls_acc],feed_dict = feed_dict_val)
                end = time.time()
                duration = end - start
                print('\t'*200)
                print('step {:d} \t node classification train_loss = {:.3f} \t train_accuracy =  {:.4f} \t val_loss = {:.3f} \t val_accuracy = {:.4f} \t ({:.3f} sec/50_steps)'.format(i+1,cls_loss,cls_acc,cls_loss_v,cls_acc_v,duration))
                print('\t'*200)
                start = time.time()

        cls_loss,cls_acc = sess.run([node_cls_loss,node_cls_acc],feed_dict = feed_dict_test)
        print('\t'*150)
        print('after training, node classification test accuracy is %4f'%(cls_acc))

    def multitask(self,embs_link,embs_cls):

        with tf.variable_scope('linkpred') as scope:
            embs_link           = tf.nn.l2_normalize(embs_link,axis=1)
            z_src               = tf.nn.embedding_lookup(embs_link,self.placeholders['src_node'])
            z_dst               = tf.nn.embedding_lookup(embs_link,self.placeholders['dst_node'])
            cosine              = self.cosine(z_src,z_dst)
            preds               = self.group(cosine,self.cfg.view_nums)
            # preds          = tf.reshape(tf.matmul(preds,w) + b,[-1])

            y_true              = tf.cast(self.placeholders['edge_labels'],tf.float32)
            link_pred_loss      = tf.reduce_mean(tf.losses.sigmoid_cross_entropy(multi_class_labels=y_true,logits=preds))
            link_pred_acc       = self.multilabel_accuracy(preds,y_true)
            y_pred              = self.cast(preds)
            preds               = tf.nn.sigmoid(preds)

        with tf.variable_scope('nodecls') as scope:
            embs_cls            = tf.nn.l2_normalize(embs_cls,axis=1)
            preds_cls           = self.group(embs_cls,self.cfg.node_label_mat.shape[1])
            trues_cls           = tf.cast(self.placeholders['node_labels'],tf.float32)
        if self.cfg.task=='link_and_multilabel':
            node_cls_loss   = self.masked_multilabel_sigmoid_cross_entropy(preds_cls,trues_cls,self.placeholders['node_labels_mask'])
            node_cls_acc    = self.masked_multilabel_accuracy(preds_cls,self.placeholders['node_labels'],self.placeholders['node_labels_mask'])
        elif self.cfg.task=='link_and_multiclass':
            node_cls_loss   = self.masked_multiclass_softmax_cross_entropy(preds_cls,trues_cls,self.placeholders['node_labels_mask'])
            node_cls_acc    = self.masked_multiclass_accuracy(preds_cls,self.placeholders['node_labels'],self.placeholders['node_labels_mask'])
        else:
            raise RuntimeError('Node_Classification_Task_Error')

        preds_cls           = self.cast(preds_cls)

        #encoder-decoder loss
        if self.cfg.model == 'model':
            square_loss         = self.GraphDecoder(self.gcn_outputs,embs_link+embs_cls,scope='square_loss')
            loss                = link_pred_loss + self.cfg.beta*node_cls_loss + self.cfg.gamma*square_loss
        else:
            square_loss         = tf.constant(0)
            loss                = link_pred_loss + self.cfg.beta*node_cls_loss

        # return link_pred_loss,node_cls_loss,preds_link,trues_link,node_cls_acc,square_loss
        opt                 = tf.train.AdamOptimizer(self.cfg.lr).minimize(loss)
        sess                = self.Sess()

        feed_dict_val       = self.construct_feed_dict_multi(self.cfg.src_node['val'],self.cfg.dst_node['val'],self.cfg.edge_label_mat['val'],self.cfg.node_mask['val'],False)
        batch_size          = self.cfg.batchsize
        start               = time.time()

        print('\t'*150)
        for i in range(self.cfg.epochs):
            train_index     = np.random.choice(len(self.cfg.src_node['train']),batch_size,replace=False)
            feed_dict_train = self.construct_feed_dict_multi(self.cfg.src_node['train'][train_index],self.cfg.dst_node['train'][train_index],self.cfg.edge_label_mat['train'][train_index],self.cfg.node_mask['train'],True)
            _,pred_loss,cls_loss,sq_loss,y_t_l,y_p_l,y_p_c,y_t_c,acc_l,acc_c = sess.run([opt,link_pred_loss,node_cls_loss,square_loss,y_true,preds,preds_cls,trues_cls,link_pred_acc,node_cls_acc],feed_dict = feed_dict_train)

            if i%50 == 0:
                pred_loss_v,cls_loss_v,sq_loss_v,y_t_lv,y_p_lv,y_p_cv,y_t_cv,acc_lv,acc_cv = sess.run([link_pred_loss,node_cls_loss,square_loss,y_true,preds,preds_cls,trues_cls,link_pred_acc,node_cls_acc],feed_dict = feed_dict_val)
                auc_t = auc(y_t_l,y_p_l,self.cfg.average)
                auc_v = auc(y_t_lv,y_p_lv,self.cfg.average)

                end = time.time()
                duration = end - start
                print('\t'*200)
                print('training square_loss = {:.3f} \t validation square_loss = {:.3f}'.format(sq_loss,sq_loss_v))
                print('step {:d} \t link prediction train_loss = {:.4f} \t train_auc =  {:.4f} \t train_acc = {:.4f} val_loss = {:.4f} \t val_auc = {:.4f} \t  val_acc = {:.4f} \t ({:.4f} sec/50_steps)'.format(i+1,pred_loss,auc_t,acc_l,pred_loss_v,auc_v,acc_lv,duration))
                print('step {:d} \t node classification train_loss = {:.4f} \t train_accuracy =  {:.4f} \t val_loss = {:.4f} \t val_accuracy = {:.4f} \t ({:.4f} sec/50_steps)'.format(i+1,cls_loss,acc_c,cls_loss_v,acc_cv,duration))
                print('\t'*200)
                start = time.time()



