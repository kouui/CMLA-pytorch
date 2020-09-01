# -*- coding: utf-8 -*-
#-----------------------------------------------------------------------------
# TODO :
#    1. dropout layer not yet implemented.
#    2. be aware of the usage of dropout. dropout input? dropout parameters?
#-----------------------------------------------------------------------------

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import copy

_dtype = torch.double
_dtype_np = np.double


def create_x(context_window_, h_input_):
    r"""

    currently only available for batch_size=1,
    padding for minibatch has not yet been completed.

    Parameters:
    ------------
    context_window_ : (batch_size, sequence, context_window)
    h_input_        : (batch_size, nodes, embed_dimension)

    Returns:
    ---------
    x_ : (batch_size, sequence, context_window * embed_dimension)
    """
    x_ = torch.zeros( (context_window_.shape[0],
                       context_window_.shape[1],
                       context_window_.shape[2] * h_input_.shape[2]),
                      dtype=_dtype )

    for i_ in range( context_window_.shape[0] ):
        index_list_ = context_window_[i_].view(-1).tolist()
        x_[i_,:,:] = h_input_[i_][index_list_].view(context_window_.shape[1],
                                     context_window_.shape[2] * h_input_.shape[2])

    return x_


def get_num_directions(rnn_):
    return 2 if rnn_.bidirectional else 1

def get_rnn_h0_ndim1(rnn_):
    return rnn_.num_layers * get_num_directions(rnn_)

def concatenate_h0(h0_, n_):
    #return torch.cat([h0_.reshape(h0_.shape[0],1,h0_.shape[1]),]*n_, 1)
    return h0_.expand(n_, h0_.shape[0], h0_.shape[1]).transpose(0,1)

def embed_to_h_input(emb_, index_embed_, h_input_size_, de_, pad_, punkt_):

    max_h_input_size_ = max( h_input_size_ )
    bs_ = len(h_input_size_)

    h_input_ = torch.zeros( (bs_, max_h_input_size_, de_) ,dtype=_dtype)

    for b_ in range(bs_):
        #for w_ in range(h_input_size_[b_]-2):
        #    h_input_[b_,w_,:] = emb_[ :, index_embed_[b_][w_] ]
        h_input_[b_,:h_input_size_[b_]-2,:] = emb_[ :, index_embed_[b_]].T
        h_input_[:, h_input_size_[b_]-2, :] = pad_[:]
        h_input_[:, h_input_size_[b_]-1, :] = punkt_[:]

    return h_input_



class CMLANet(nn.Module):

    def __init__(self, nh, nc, de, cs, bs, device, nt_a=20, nt_o=20, csv=1, iteration=1):
        r"""
        Parameters:
        -----------
        nh        : dimension of the hidden layer
        nc        : number of classes
        de        : dimension of the word embeddings
        cs        : word window context size
        bs        : batch size
        nt_a      : 20, K for aspect
        nt_o      : 20, K for opinion
        csv       : 1
        iteration : 1, number of memory iterations
        """
        super(CMLANet, self).__init__()

        self.device = device

        self.n_in = n_in = de * cs # embedding-dimension * window-context-size
        self.n_v = n_v = nt_a + nt_o
        self.n_inv = n_inv = n_v * csv
        self.ny = ny = nc
        self.de = de

        self.dropout_dict = {}

        self.gru = nn.GRU(input_size=n_in, hidden_size=nh, num_layers=1, bias=True, batch_first=True)
        self.dropout_dict["gru"] = nn.Dropout(p=0.0, inplace=False)
        ##self.h0 = torch.zeros( (bs,get_rnn_h0_ndim1(self.gru),nh), dtype=_dtype )
        self.h0 = torch.zeros( (get_rnn_h0_ndim1(self.gru),nh), dtype=_dtype ).to(device)

        self.m0_a = torch.as_tensor( 0.2 * np.random.uniform(-1.0, 1.0, (nh,)) ).to(device)
        self.m0_o = torch.as_tensor( 0.2 * np.random.uniform(-1.0, 1.0, (nh,)) ).to(device)


        self.Ua = torch.as_tensor( 0.2 * np.random.uniform(-1.0, 1.0, (nt_a, nh, nh)) ).to(device)
        self.Va = torch.as_tensor( 0.2 * np.random.uniform(-1.0, 1.0, (nt_o, nh, nh)) ).to(device)
        self.Uo = torch.as_tensor( 0.2 * np.random.uniform(-1.0, 1.0, (nt_o, nh, nh)) ).to(device)
        self.Vo = torch.as_tensor( 0.2 * np.random.uniform(-1.0, 1.0, (nt_a, nh, nh)) ).to(device)
        self.dropout_dict["Ua"] = nn.Dropout(p=0.0, inplace=False)
        self.dropout_dict["Va"] = nn.Dropout(p=0.0, inplace=False)
        self.dropout_dict["Uo"] = nn.Dropout(p=0.0, inplace=False)
        self.dropout_dict["Vo"] = nn.Dropout(p=0.0, inplace=False)

        self.gru_a = nn.GRU(input_size=n_v, hidden_size=n_v, num_layers=1, bias=True, batch_first=True)
        self.dropout_dict["gru_a"] = nn.Dropout(p=0.0, inplace=False)
        self.r0_a = torch.zeros( (get_rnn_h0_ndim1(self.gru_a),n_v), dtype=_dtype ).to(device)
        self.gru_o = nn.GRU(input_size=n_v, hidden_size=n_v, num_layers=1, bias=True, batch_first=True)
        self.dropout_dict["gru_o"] = nn.Dropout(p=0.0, inplace=False)
        self.r0_o = torch.zeros( (get_rnn_h0_ndim1(self.gru_o),n_v), dtype=_dtype ).to(device)

        self.va = torch.as_tensor( 0.2 * np.random.uniform(-1.0, 1.0, (n_v,)) ).to(device)
        self.vo = torch.as_tensor( 0.2 * np.random.uniform(-1.0, 1.0, (n_v,)) ).to(device)


        self.Ma = torch.as_tensor( 0.2 * np.random.uniform(-1.0, 1.0, (nh,nh)) ).to(device)
        self.Mo = torch.as_tensor( 0.2 * np.random.uniform(-1.0, 1.0, (nh,nh)) ).to(device)

        self.linear_a = torch.nn.Linear(in_features=n_v, out_features=ny, bias = True)
        self.dropout_dict["linear_a"] = nn.Dropout(p=0.0, inplace=False)
        self.linear_o = torch.nn.Linear(in_features=n_v, out_features=ny, bias = True)
        self.dropout_dict["linear_o"] = nn.Dropout(p=0.0, inplace=False)

        self.padding = torch.as_tensor( np.random.uniform(-0.2, 0.2, (de,)) ).to(device)
        self.punkt   = torch.as_tensor( np.random.uniform(-0.2, 0.2, (de,)) ).to(device)

        self.pars = [self.padding, self.punkt, self.Ma, self.Mo, self.va, self.vo,
                     self.r0_a, self.r0_o, self.Ua, self.Va, self.Uo, self.Vo,
                     self.m0_a, self.m0_o, self.h0]

        for par in self.pars:
            par.requires_grad = True





    def set_dropout_rate(self, p):

        for k, v in self.dropout_dict.items():
            v.p = p

    def set_h_input(self, h_input, h_input_size):

        bs = h_input.shape[0]

        #self.h_input = torch.zeros(h_input.numpy().shape, dtype=_dtype, requires_grad=True)
        #self.h_input[:,:,:] = h_input.numpy()[:,:,:]
        #self.h_input = torch.as_tensor(h_input.numpy())
        #self.h_input = copy.deepcopy( h_input ).to(self.device)
        #self.h_input.requires_grad=True
        self.h_input = h_input.detach().clone().to(self.device)
        #self.h_input.requires_grad = True


        for b in range(bs):
            self.h_input[:, h_input_size[b]-2, :] = self.padding[:] * 1
            self.h_input[:, h_input_size[b]-1, :] = self.punkt[:] * 1

    def set_embed(self, emb, fixEmbed):

        self.emb = torch.as_tensor( emb ).to(self.device)

        if not fixEmbed:
            self.pars.append( self.emb )
            self.emb.requires_grad = True


    # if True : return 1, 1
    def forward(self, context_words, h_input_size, seq_size, index_embed):

        self.seq_size  =  seq_size

        self.h_input = embed_to_h_input(self.emb[:,:], index_embed, h_input_size, self.emb.shape[0], self.padding[:], self.punkt[:])

        bs, n_word, _ = context_words.shape

        #-- x : (batch_size, n_word, n_in)
        x = create_x(context_words, self.h_input).to(self.device)

        bs, n_word, n_in = x.shape
        n_v = self.n_v

        #-- an extra gru for embedding word vectors before hGu tensor product (eq 4)
        #-- h : (batch_size, n_word, n_hidden)
        h, _ = self.gru(x, concatenate_h0(self.h0, x.shape[0]))
        h = self.dropout_dict["gru"](h)

        #-- compute memory state for each iteration
        #-- ma, mo : (batch_size, n_hidden)
        ma, mo = self._memory_iteration(self.m0_a, self.m0_o, h)

        #???
        # ma, mo : (batch_size, 2, n_hidden)
        ##ma = torch.cat([torch.cat([self.m0_a.reshape(1,1,-1),]*bs, 0), ma.reshape(bs,1,-1)], axis=1)
        ##mo = torch.cat([torch.cat([self.m0_o.reshape(1,1,-1),]*bs, 0), mo.reshape(bs,1,-1)], axis=1)
        ma = torch.cat([self.m0_a.reshape(1,1,-1).expand(bs,1,-1), ma.reshape(bs,1,-1)], axis=1)
        mo = torch.cat([self.m0_o.reshape(1,1,-1).expand(bs,1,-1), mo.reshape(bs,1,-1)], axis=1)



        # hidden_a, hidden_o : (bs, 2, n_word, n_v)
        NN = ma.shape[1]
        hidden_a = torch.zeros( (bs, NN, n_word, n_v), dtype=_dtype ).to(self.device)
        hidden_o = torch.zeros( (bs, NN, n_word, n_v), dtype=_dtype ).to(self.device)

        for b_ in range(bs):
            for i_ in range(NN):
                hidden_a[b_,i_,:,:] = self._get_hidden_aspect(torch.unsqueeze(h[b_,:,:],0), ma[b_,i_,:], mo[b_,i_,:])[0,:,:]
                hidden_o[b_,i_,:,:] = self._get_hidden_opinion(torch.unsqueeze(h[b_,:,:],0), ma[b_,i_,:], mo[b_,i_,:])[0,:,:]
        # hidden_a, hidden_o : (bs, n_word, n_v)
        hidden_a = hidden_a.sum(axis=1)
        hidden_o = hidden_o.sum(axis=1)

        # final softmax to get prediction from attention vector r_{i}^{a}
        # ya_pred, yo_pred : (bs, n_word, ny)
        ##ya_pred = torch.zeros( (bs, n_word, self.ny), dtype=_dtype ).to(self.device)
        ##yo_pred = torch.zeros( (bs, n_word, self.ny), dtype=_dtype ).to(self.device)
        ##for b_ in range(bs):
        ##    ya_pred[b_,:,:] = F.softmax( self.dropout_dict["linear_a"](self.linear_a(hidden_a[b_,:,:]) ), dim=1 )
        ##    yo_pred[b_,:,:] = F.softmax( self.dropout_dict["linear_o"](self.linear_o(hidden_o[b_,:,:]) ), dim=1 )
        ya_pred = F.softmax( self.dropout_dict["linear_a"](self.linear_a(hidden_a[:,:,:]) ), dim=2 )
        yo_pred = F.softmax( self.dropout_dict["linear_o"](self.linear_o(hidden_o[:,:,:]) ), dim=2 )

        #-- (bs, n_word), same dimension with ya_label, yo_label
        ##ya_predLabel = ya_pred.argmax(axis=2)
        ##yo_predLabel = yo_pred.argmax(axis=2)


        return ya_pred, yo_pred

    def _memory_iteration(self, ma_t_, mo_t_, h_):
        r"""

        `iteration` parameter not yet implemented.

        Parameters:
        -----------
        ma_t_ : (nh,)
        mo_t_ : (nh,)
        h_    : (bs, n_word, nh)

        Returns:
        --------
        ma_tp1_ : (bs,nh)
        mo_tp1_ : (bs,nh)
        """

        #-- o_{t}^{m}
        #-- ca_tp1_ : (bs, nh)
        #-- ca_tp1_ is the attentioned word vector for aspect
        ca_tp1_, _ = self._attention_pool_aspect(h_, ma_t_, mo_t_)

        #-- co_tp1_ : (bs, nh)
        #-- co_tp1_ is the attentioned word vector for opinion
        co_tp1_, _ = self._attention_pool_opinion(h_, ma_t_, mo_t_)

        ## (eq 6)
        # self.Ma --> V^{m}
        # ma_t    --> u_{t}^{m}
        ma_tp1_ = torch.tanh(torch.matmul(ma_t_, self.Ma)) + ca_tp1_
        mo_tp1_ = torch.tanh(torch.matmul(mo_t_, self.Mo)) + co_tp1_


        return ma_tp1_, mo_tp1_


    def _attention_pool_aspect(self, h_, ma_, mo_):
        r"""
        Parameters:
        -----------
        h_    : (bs, n_word, nh)
        ma_t_ : (nh,)
        mo_t_ : (nh,)

        Returns:
        ---------
        ctx_pool_ : (bs,nh)
        """
        bs, n_word, nh = h_.shape

        #-- hidden_aspect : (bs, n_word, n_v)
        hidden_aspect = self._get_hidden_aspect(h_, ma_, mo_)

        ##e_ = torch.zeros( (bs, n_word), dtype=_dtype ).to(self.device)
        ##ctx_pool_ = torch.zeros( (bs, nh), dtype=_dtype ).to(self.device)
        ##for b_ in range(bs):
        ##    e_[b_,:] = torch.matmul( hidden_aspect[b_,:,:], self.va[:] )
        ##    #-- alpha : (n_word,)
        ##    alpha_ = F.softmax(e_[b_,:], dim=0)
        ##    #print(alpha_.dtype, h_.dtype)
        ##    ctx_pool_[b_,:] = torch.matmul( alpha_[:], h_[b_,:,:] )

        e_ = torch.matmul( hidden_aspect[:,:,:], self.va[:] )
        #-- alpha_ : (bs, n_word)
        alpha_ = F.softmax(e_[:,:], dim=1)

        ctx_pool_ = torch.zeros( (bs, nh), dtype=_dtype ).to(self.device)
        for b_ in range(bs):
            nw_ = self.seq_size[b_]
            ctx_pool_[b_,:] = torch.matmul( alpha_[b_,:nw_], h_[b_,:nw_,:] )

        return ctx_pool_, e_

    def _get_hidden_aspect(self, h_, ma_, mo_):
        r"""

        tensor operation + GRU to get attention vector $r_{i}^{a}$

        Parameters:
        -----------
        h_    : (bs, n_word, nh)
        ma_t_ : (nh,)
        mo_t_ : (nh,)

        Returns:
        ---------
        r_ : (bs, n_word, n_v)
        """
        #-- tensor operation + GRU to get attention vector r_{i}^{a}
        ##a_ = torch.zeros( (h_.shape[0], h_.shape[1], self.n_v), dtype=_dtype ).to(self.device)
        ##for b_ in range(h_.shape[0]):
        ##    for w_ in range(h_.shape[1]):
        ##        #-- (nv_a)
        ##        tensor1_ = self._tensor_product(h_[b_,w_], self.Ua, ma_)
        ##        #-- (nv_o)
        ##        tensor2_ = self._tensor_product(h_[b_,w_], self.Va, mo_)
        ##        a_[b_,w_,:] = torch.cat((tensor1_, tensor2_), 0)

        tensor1_ = self._tensor_product(h_, self.dropout_dict["Ua"](self.Ua), ma_)
        tensor2_ = self._tensor_product(h_, self.dropout_dict["Va"](self.Va), mo_)
        a_ = torch.cat((tensor1_, tensor2_), 2)
        #-- a_ : (bs, n_word, n_v)
        r_, _ = self.gru_a(a_, concatenate_h0(self.r0_a, h_.shape[0]))
        r_ = self.dropout_dict["gru_a"](r_)

        return r_

    def _attention_pool_opinion(self, h_, ma_, mo_):
        r"""
        Parameters:
        -----------
        h_    : (bs, n_word, nh)
        ma_t_ : (nh,)
        mo_t_ : (nh,)

        Returns:
        ---------
        ctx_pool_ : (bs,nh)
        """
        bs, n_word, nh = h_.shape

        #-- hidden_aspect : (bs, n_word, n_v)
        hidden_opinion = self._get_hidden_opinion(h_, ma_, mo_)

        ##e_ = torch.zeros( (bs, n_word), dtype=_dtype ).to(self.device)
        ##ctx_pool_ = torch.zeros( (bs, nh), dtype=_dtype ).to(self.device)
        ##for b_ in range(bs):
        ##    e_[b_,:] = torch.matmul( hidden_opinion[b_,:,:], self.vo[:] )
        ##    #-- alpha : (n_word,)
        ##    alpha_ = F.softmax(e_[b_,:], dim=0)
        ##    #print(alpha_.dtype, h_.dtype)
        ##    ctx_pool_[b_,:] = torch.matmul( alpha_[:], h_[b_,:,:] )

        e_ = torch.matmul( hidden_opinion[:,:,:], self.vo[:] )
        #-- alpha_ : (bs, n_word)
        alpha_ = F.softmax(e_[:,:], dim=1)

        ctx_pool_ = torch.zeros( (bs, nh), dtype=_dtype ).to(self.device)
        for b_ in range(bs):
            nw_ = self.seq_size[b_]
            ctx_pool_[b_,:] = torch.matmul( alpha_[b_,:nw_], h_[b_,:nw_,:] )

        return ctx_pool_, e_

    def _get_hidden_opinion(self, h_, ma_, mo_):
        r"""

        tensor operation + GRU to get attention vector $r_{i}^{a}$

        Parameters:
        -----------
        h_    : (bs, n_word, nh)
        ma_t_ : (nh,)
        mo_t_ : (nh,)

        Returns:
        ---------
        r_ : (bs, n_word, n_v)
        """
        #-- tensor operation + GRU to get attention vector r_{i}^{a}
        ##a_ = torch.zeros( (h_.shape[0], h_.shape[1], self.n_v), dtype=_dtype ).to(self.device)
        ##for b_ in range(h_.shape[0]):
        ##    for w_ in range(h_.shape[1]):
        ##        #-- (nv_a)
        ##        tensor1_ = self._tensor_product(h_[b_,w_], self.Uo, ma_)
        ##        #-- (nv_o)
        ##        tensor2_ = self._tensor_product(h_[b_,w_], self.Vo, mo_)
        ##        a_[b_,w_,:] = torch.cat((tensor1_, tensor2_), 0)

        tensor1_ = self._tensor_product(h_, self.dropout_dict["Uo"](self.Uo), ma_)
        tensor2_ = self._tensor_product(h_, self.dropout_dict["Vo"](self.Vo), mo_)
        a_ = torch.cat((tensor1_, tensor2_), 2)

        #-- a_ : (bs, n_word, n_v)
        r_, _ = self.gru_o(a_, concatenate_h0(self.r0_o, h_.shape[0]))
        r_ = self.dropout_dict["gru_o"](r_)

        return r_


    def _tensor_product(self, h_, Dm_, um_):
        r"""
        Parameters:
        -----------
        h_ :  (bs, nw, nh,)
        Dm_ : (K :: nt_a or nt_o, nh, nh)
        um_ : (nh,)

        Returns:
        ---------
        ti_ : (bs, nw, K :: nv_a or nv_o,)
        """
        #nk_ = Dm_.shape[0]
        #ti_ = torch.zeros( (nk_,), dtype=_dtype ).to(self.device)
        #for k in range(nk_):
        #    ti_[k] = torch.dot( hi_, torch.matmul(Dm_[k], um_) )
        ti_ = torch.matmul( Dm_[:,:,:], um_[:] )
        ti_ = torch.matmul( h_[:,:,:], ti_.T[:,:] )

        return ti_
