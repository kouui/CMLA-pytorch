# -*- coding: utf-8 -*-
#-----------------------------------------------------------------------------
# Library of pytorch Dataset customization
# steps:
#   1. load data and embedding model
#   2. create node.vec for all nodes in all seq
#   3. create ya_label, yo_label, index2word, sent, h_input
#-----------------------------------------------------------------------------

import torch
import numpy as np

def collate_fn(single_data):
    """Creates mini-batch tensors from the the output of CMLADataset.__getitem__().

    We should build custom collate_fn rather than using default collate_fn,
    because merging caption (including padding) is not supported in default.
    Args:
        data: ...
    Returns:

    """

    # Sort a data list by caption length (descending order).
    single_data.sort(key=lambda pair: pair[-1], reverse=True)

    ya_label_list, yo_label_list, index2word_list, index_embed_list, \
                   sent_list, seq_size_list, h_input_size_list = zip(*single_data)


    max_seq_size = max(seq_size_list)
    max_h_input_size = max( h_input_size_list )
    batch_size = len(seq_size_list)

    ##h_inputs = -1 * torch.ones((batch_size,max_h_input_size,h_input_list[0].shape[1]), dtype=h_input_list[0].dtype)
    ya_labels = -1 * torch.ones((batch_size,ya_label_list[0].shape[0]), dtype = ya_label_list[0].dtype)
    yo_labels = -1 * torch.ones((batch_size,yo_label_list[0].shape[0]), dtype = yo_label_list[0].dtype)
    #index2words = -1 * torch.ones((batch_size,index2word_list[0].shape[0]), index2word_list[0].dtype)
    #index_embeds = -1 * torch.ones((batch_size,index_embed_list[0].shape[0]), index_embed_list[0].dtype)
    #seq_sizes = -1 * torch.ones((batch_size,seq_size_list[0].shape[0]), seq_size_list[0].dtype)
    #ya_labels = ya_label_list
    #yo_labels = yo_label_list
    
    index2words = [i + [j-1,] * (max_seq_size-len(i))
                       for i, j in zip(index2word_list,h_input_size_list)]
    index_embeds = index_embed_list
    sents = sent_list
    seq_sizes = torch.tensor( seq_size_list )
    h_input_sizes = h_input_size_list


    for b in range(batch_size):
        ##h_inputs[b,:h_input_list[b].shape[0],:] = h_input_list[b]
        ya_labels[b,:seq_sizes[b]] = ya_label_list[b][:]
        yo_labels[b,:seq_sizes[b]] = yo_label_list[b][:]

    return ya_labels, yo_labels, index2words, index_embeds, sents, seq_sizes, h_input_sizes


class CMLADataset(torch.utils.data.Dataset):

    def __init__(self, data_list, emb_model):
        r"""

        load data and embedding model

        Parameters:
        -----------
        ...
        """

        self.data_list = data_list
        self.emb_model = emb_model

    def __len__(self):
        r"""

        length of the dataset

        """

        return len(self.data_list)

    def __getitem__(self, index):
        r"""

        return item in dataset based on its index

        """

        seq = self.data_list[index]

        ya_label, yo_label, index2word, index_embed, sent, seq_size, h_input_size = self.preprocess_single_seq(seq)

        ##h_input = torch.as_tensor( h_input )
        ya_label = torch.tensor( ya_label, dtype=torch.int16 )
        yo_label = torch.tensor( yo_label, dtype=torch.int16 )
        #index2word = torch.tensor( index2word, dtype=torch.int16 )
        #index_embed = torch.tensor( index_embed, dtype=torch.int64 )

        return ya_label, yo_label, index2word, index_embed, sent, seq_size, h_input_size

    def set_n_emb_dim(self, n_emb_dim):
        r"""

        set parameter for embedding dimension

        Parameters:
        ------------
        n_emb_dim : int, embedding dimension

        """

        self.n_emb_dim = n_emb_dim

    def filter_seq_has_not_word(self):
        r"""
        pop out seq that has node.is_word==False in self.data_list
        """
        pass
        ##pop_list = []
        ##for i, seq in enumerate( self.data_list ):
        ##    for index, node in enumerate( seq.nodes ):
        ##        if seq.get(index).is_word == 0:
        ##            pop_list.append(seq)
        ##            break
        ##for seq in pop_list:
        ##    self.data_list.pop(seq)


    def preprocess_single_seq(self, seq):
        r"""

        - create node.vec for all nodes in all seq
        - create ya_label, yo_label, index2word, sent, h_input

        Parameters:
        -----------
        seq : seqItem.SeqItem, sequence object

        Returns:
        --------
        h_input : numpy.ndarray, (nNodes+2, n_emb_dim), np.float64,
            array of embedding vectors of nodes
        ya_label : list of int
            labels for aspect, in an order of nodes, 0,1,2
        yo_label : list of int
            labels for opinion, in an order of nodes, 0,1,2
        index2word : list of int
            list of word index, in an order of nodes, nNodes+1 if it is not
            a meaningful word.
        sent : list of str
            a list to store meaningful words

        """

        emb_model = self.emb_model
        n_emb_dim = self.n_emb_dim

        nodes = seq.get_nodes()
        ##for node in nodes:
        ##    # ? why do we need reshape here
        ##    node.vec = emb_model[:, node.ind]#.reshape( (n_emb_dim, 1) )

        nNodes = len( nodes )

        # 2 for aux["padding"] and aux["punkt"]
        ##h_input = np.zeros( (nNodes+2, n_emb_dim), dtype=np.float64 )
        ya_label = []
        yo_label = []
        index2word = []
        index_embed = []
        sent = [] # a list to store aspect and option words
        word_index = 0 # count aspect and option words

        for index, node in enumerate( seq.nodes ):

            if seq.get(index).is_word == 0:
                ya_label.append( 0 )
                yo_label.append( 0 )
                sent.append( "NOT-WORD" )
                index2word.append( nNodes + 1 )
                #index2word.append( -1 )
                seq.hasNotWord = True

            else:
                # class labels :
                #  ya : 1, 2; yo : 0
                #  y0 : 1, 2; ya : 0
                #  ya : 0; yo : 0
                if node.trueLabel == 1 or node.trueLabel == 2:
                    ya_label.append( node.trueLabel )
                    yo_label.append( 0 )
                elif node.trueLabel == 3 or node.trueLabel == 4:
                    ya_label.append( 0 )
                    yo_label.append( node.trueLabel - 2 )
                else:
                    ya_label.append( 0 )
                    yo_label.append( 0 )

                sent.append( node.word )
                index2word.append( word_index )
                index_embed.append( node.ind )

                ##h_input[word_index, :] = node.vec[:]

                word_index += 1

            h_input_size = nNodes + 2

        return ya_label, yo_label, index2word, index_embed, sent, len(index2word), h_input_size


if __name__ == "__main__":

    import pickle
    from seqItem import SeqItem

    with open("../data/res15/final_input_res15", 'rb') as handle:
        vocab, train_seq_list, test_seq_list = pickle.load(handle)
    with open("../data/res15/word_embeddings200_res15", 'rb') as handle:
        emb_model = pickle.load(handle, encoding="bytes")



    dataset = CMLADataset(data_list=train_seq_list, emb_model=emb_model)
    dataset.set_n_emb_dim(200)
    #n_train_seq = len(train_seq_list)

    #-- batch_size must be 1 because of the variable length of node
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=2,
                           shuffle=False, num_workers=0, collate_fn=collate_fn)

    for i_batch, (h_input, ya_label, yo_label, index2word, index_embed, sent, seq_size, h_input_size ) in enumerate(dataloader):

        #print(h_input)
        #print( sent )
        #print( h_input.shape, ya_label.shape, yo_label.shape, index2word.shape )
        print(h_input.shape, ya_label.shape, yo_label.shape)
        for i2w in index2word:
            print(len(i2w))
        print(seq_size, h_input_size)
        break
