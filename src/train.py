# -*- coding: utf-8 -*-
#-----------------------------------------------------------------------------
# TODO :
#    1. pack train and evaluate functions
#    2. model saving
#    3. minibatch with padding
#-----------------------------------------------------------------------------

import pickle
import torch
import time
import datetime
import numpy as np
import os

from seqItem import SeqItem
from util import set_status, save_score_to_text
from model import CMLANet
from dataset import CMLADataset
from score import score_aspect, score_opinion

def create_context_window(index2word_, win_, seq_size_):
    r"""
    Parameters:
    -----------
    index2word_ : torch.Tensor, (batch_size, nNode),
        order of nodes in each sequence
    win_ : int, corresponding to the size of the window
    seq_size_ : int, length of (each) sequence
    """

    assert (win_ % 2) == 1
    assert win_ >= 1

    index2word_list_ = index2word_.tolist()
    out_ = []
    for li_ in index2word_list_:
        padded_ = win_//2 * [seq_size_-2,] + li_ +  win_//2 * [seq_size_-2,]
        out_.append( [ padded_[i_:i_+win_] for i_ in range(len(li_)) ] )

    assert len(out_[0]) == len(index2word_[0])

    return out_


def LossFunc(ya_pred_, yo_pred_, ya_label_, yo_label_):
    r"""
    """

    bs, n_word, ny = ya_pred_.shape
    loss_mean_ = 0
    for b in range(bs):
        loss_ = 0
        for nw in range(n_word):
            loss_ += -torch.log( ya_pred_[b,nw, ya_label_[b,nw] ] )
            loss_ += -torch.log( yo_pred_[b,nw, yo_label_[b,nw] ] )
        loss_ /= n_word

        loss_mean_ += loss_
    loss_mean_ /= bs


    return loss_mean_

if __name__ == "__main__":

#-----------------------------------------------------------------------------
# Configuration Parameters
#-----------------------------------------------------------------------------
    args = {
        "data" : "../data/res15/final_input_res15",
        "embModel" : "../data/res15/word_embeddings200_res15",
        "logStatus" : "terminal",
        "logFile" : "",
        "debugTrain" : False,
        "evaluate" : True,
        "logSequence" : True,
        "version" : "English",
        "text"    : "../txt/outcome.txt",
        "save" : False,
    }
    args["logFile"] = "../log/" + args["version"] + "/" + datetime.datetime.now().strftime("%Y%m%d_%H%M") + ".txt"
#-----------------------------------------------------------------------------
# Learning Control Hyperparameters
#-----------------------------------------------------------------------------
    params = {
        "lr" : 0.05,
        "win" : 3,
        "nHidden" : 50,
        "nEmbedDimension" : 200,
        "nClass" : 3,
        "batchSize" : 1,
        "nEpoch" : 100,
        "dropout" : 0.3,
        "device"  : torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
        "fixEmbed" : False,
    }

#-----------------------------------------------------------------------------
# Prepare Log fileObj
#-----------------------------------------------------------------------------

    if args["logStatus"] == "log":
        idx = args["logFile"].rfind('/')
        folder = args["logFile"][:idx]
        if not os.path.exists(folder) : os.makedirs(folder)
        fileObj = open(args["logFile"], 'w')
    else:
        fileObj = None
#-----------------------------------------------------------------------------
# Data and Embed Model Loading
#-----------------------------------------------------------------------------

    s_ = "loading data"
    set_status(s_=s_, status_=args["logStatus"], fileObj_=fileObj)

    with open(args["data"], 'rb') as handle:
        vocab, train_seq_list, test_seq_list = pickle.load(handle)


    s_ = "loading embed model"
    set_status(s_=s_, status_=args["logStatus"], fileObj_=fileObj)

    with open(args["embModel"], 'rb') as handle:
        emb_model = pickle.load(handle, encoding="bytes")

#-----------------------------------------------------------------------------
# model initialization
#-----------------------------------------------------------------------------

    s_ = "initializing network model"
    set_status(s_=s_, status_=args["logStatus"], fileObj_=fileObj)

    net = CMLANet(nh=params["nHidden"], nc=params["nClass"],
                  de=params["nEmbedDimension"], cs=params["win"],
                  bs=params["batchSize"], device=params["device"]).to(params["device"])
    net.double()

    #-- set dropout in net to params["dropout"] here
    net.set_dropout_rate(params["dropout"])

#-----------------------------------------------------------------------------
# optimizer
#-----------------------------------------------------------------------------

    optimizer = torch.optim.SGD(params=net.parameters(), lr=params["lr"], momentum=0.9, weight_decay=0.0)
    #optimizer = torch.optim.Adam(params=net.parameters(), lr=params["lr"], weight_decay=0, amsgrad=False)

#-----------------------------------------------------------------------------
# dataset and dataloader
#-----------------------------------------------------------------------------

    s_ = "building dataset and dataloader"
    set_status(s_=s_, status_=args["logStatus"], fileObj_=fileObj)

    dataset = {
        "train" : CMLADataset(data_list=train_seq_list, emb_model=emb_model),
        "test"  : CMLADataset(data_list=test_seq_list, emb_model=emb_model)
    }
    dataset["train"].set_n_emb_dim(params["nEmbedDimension"])
    dataset["test"].set_n_emb_dim(params["nEmbedDimension"])

    s_ = f"length of train set : {len(dataset['train'])}"
    set_status(s_=s_, status_=args["logStatus"], fileObj_=fileObj)
    s_ = f"length of test set : {len(dataset['test'])}"
    set_status(s_=s_, status_=args["logStatus"], fileObj_=fileObj)

    #-- batch_size must be 1 because of the variable length of node
    dataloader = {}
    for key, value in dataset.items():
        dataloader[key] = torch.utils.data.DataLoader(value,
                                                      batch_size=params["batchSize"],
                                                      shuffle=True,
                                                      num_workers=0)

#-----------------------------------------------------------------------------
# training epochs
#-----------------------------------------------------------------------------

    s_ = "start training"
    set_status(s_=s_, status_=args["logStatus"], fileObj_=fileObj)

    min_error = float("inf")
    for epoch in range(1, params["nEpoch"]+1):

        net.train()
        epoch_error = 0.
        count = 0
        for i_batch, batch in enumerate(dataloader["train"]):

            h_input, ya_label, yo_label, index2word, index_embed, sent = batch
            h_input = h_input.to(params["device"])
            ya_label = ya_label.to(params["device"])
            yo_label = yo_label.to(params["device"])

            sent = list( zip(*sent) )
            seq_size = h_input.shape[1]

            now = time.time()

#-----------------------------------------------------------------------------
# training sequence(s)
#-----------------------------------------------------------------------------

            context_words = torch.tensor(
                         create_context_window(index2word, params["win"], seq_size),
                         dtype=torch.uint8 ).to(params["device"])


            #-- ya_pred, yo_pred : (bs, n_word, ny)
            ya_pred, yo_pred = net(context_words[:,:,:], h_input[:,:,:])

            error = LossFunc(ya_pred, yo_pred, ya_label.detach(), yo_label.detach())
            net.zero_grad()
            error.backward()
            optimizer.step()

#-----------------------------------------------------------------------------
# modify embedding model
#-----------------------------------------------------------------------------

            if not params["fixEmbed"]:

                h_input_new = net.h_input.detach().cpu().numpy()
                for k in range(h_input_new.shape[0]):
                    for i in index2word[k,:].tolist():
                        try:
                            j = index_embed[k,i].data.item()
                            emb_model[:, j] = h_input_new[k,i,:]
                        except IndexError:
                            continue
#-----------------------------------------------------------------------------

#-----------------------------------------------------------------------------


            epoch_error += error
            count += 1

            if args["logSequence"]:
                s_ = f"epoch: {epoch:02d} i_batch: {i_batch:05d} error: {error:.2f} "
                s_ += f"time collapsed: {time.time()-now:.2f}[sec]"
                set_status(s_=s_, status_=args["logStatus"], fileObj_=fileObj)

            if args["debugTrain"] and i_batch > 50:
                break
        epoch_error /= count

#-----------------------------------------------------------------------------
# evaluate here
#-----------------------------------------------------------------------------
        if args["evaluate"]:
            s_ = "Evaluating...."
            set_status(s_=s_, status_=args["logStatus"], fileObj_=fileObj)
            with torch.no_grad():
                net.eval()

                true_a = []
                true_o = []
                pred_a = []
                pred_o = []

                for i_batch, batch in enumerate(dataloader["test"]):

                    h_input, ya_label, yo_label, index2word, index_embed, sent = batch
                    h_input = h_input.to(params["device"])
                    ya_label = ya_label.to(params["device"])
                    yo_label = yo_label.to(params["device"])

                    sent = list( zip(*sent) )
                    seq_size = h_input.shape[1]

                    context_words = torch.tensor(
                                 create_context_window(index2word, params["win"], seq_size),
                                 dtype=torch.uint8 ).to(params["device"])

                    ya_pred, yo_pred = net(context_words[:,:,:], h_input[:,:,:])

                    ya_predLabel = ya_pred.argmax(axis=2)
                    yo_predLabel = yo_pred.argmax(axis=2)

                    ##true_list.append([str(y) for y in y_label[0,:]])
                    true_a.append([str(y) for y in ya_label[0,:]])
                    true_o.append([str(y) for y in yo_label[0,:]])
                    pred_a.append([str(y) for y in ya_predLabel[0,:]])
                    pred_o.append([str(y) for y in yo_predLabel[0,:]])

                precision_as, recall_as, f1_as = score_aspect(true_a, pred_a)
                precision_op, recall_op, f1_op = score_opinion(true_o, pred_o)

                if not os.path.exists("../txt") : os.mkdir("../txt")
                save_score_to_text(args["text"], epoch, precision_as, recall_as, f1_as, precision_op, recall_op, f1_op)
                print(f"{precision_as:.3f}  {recall_as:.3f}  {f1_as:.3f}  {precision_op:.3f}  {recall_op:.3f}  {f1_op:.3f}")

#-----------------------------------------------------------------------------

#-----------------------------------------------------------------------------
# saving model here
#-----------------------------------------------------------------------------
        #-- save parameters if the current model is better than previous best model
        if epoch_error < min_error:
            min_error = epoch_error

            if not args["debugTrain"] and args["save"]:

                s_ = "saving model"
                set_status(s_=s_, status_=args["logStatus"], fileObj_=fileObj)
                checkpoint = {
                        "model state" : net.state_dict(),
                    }
                folder = f"../checkpoints/{args['version']}"
                if not os.path.exists(folder) : os.makedirs(folder)
                torch.save(checkpoint, os.path.join(folder, f"checkpoint_epoch_{epoch}.pkl") )
#-----------------------------------------------------------------------------
        #-- done with epoch
        s_ = f"done with epoch {epoch:02d}, epoch_error = {epoch_error:.2f}, min_error = {min_error:.2f}"
        set_status(s_=s_, status_=args["logStatus"], fileObj_=fileObj)
