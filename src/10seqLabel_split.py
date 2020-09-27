# -*- coding: utf-8 -*-
##############################################################################
# create sequence structures from raw training sentences
# accumulate vocabulary
# ignore lemmatization
# differentiate beginning and inside of aspects and opinions

# execution :
#   $ python /path/to/10seqLabel.py /dir/of/input /dir/of/output
#       ex : $ python src/10seqLabel.py data/res15/ data/res15/
##############################################################################


from seqItem import SeqItem
import sys, pickle, random, os
#import numpy as np
import string
import nltk

from glob import glob

def shuffle_multi(*ls):

    l =list(zip(*ls))
    random.shuffle(l)
    return zip(*l)

def label_generation(sequence, as_label, op_label):

    if sequence.strip():
        sequence = nltk.word_tokenize(sequence)
        # load words into nodes
        nodes = [None for i in range(0, len(sequence))]
        for ind, word in enumerate(sequence):
            if word not in string.punctuation:
                nodes[ind] = word

        seq = SeqItem(nodes)
        aspects = as_label.rstrip()
        ops = op_label.strip()


        if '##' in ops:
            opinions_tuple = ops.split('##')[1].strip()
            opinions_tuple = opinions_tuple.split(',')

            for opinion_tuple in opinions_tuple:
                opinion_tuple = opinion_tuple.strip()
                op_list = opinion_tuple.split()[:-1]
                #op_list = op.split()

                if len(op_list) > 1:
                    for ind, term in enumerate(nodes):
                        if term != None:
                            if term == op_list[0] and ind < len(nodes) - 1 and nodes[ind + 1] != None and nodes[ind + 1] == op_list[1]:
                                seq.get(ind).trueLabel = 3
                                for i in range(len(op_list) - 1):
                                    if nodes[ind + i + 1] != None and nodes[ind + i + 1] == op_list[i + 1]:
                                        seq.get(ind + i + 1).trueLabel = 4

                elif len(op_list) == 1:
                    for ind, term in enumerate(nodes):
                        if term != None:
                            if term == op_list[0] and seq.get(ind).trueLabel == 0:
                                seq.get(ind).trueLabel = 3

        if aspects != 'NULL':
            aspects = aspects.split(',')

            #deal with same word but different labels
            for aspect in aspects:
                aspect = aspect.strip()
                #aspect is a phrase
                if ' ' in aspect:
                    aspect_list = aspect.split()
                    for ind, term in enumerate(nodes):
                        if term == aspect_list[0] and ind < len(nodes) - 1 and nodes[ind + 1] == aspect_list[1]:
                            seq.get(ind).trueLabel = 1

                            for i in range(len(aspect_list) - 1):
                                if ind + i + 1 < len(nodes):
                                    if nodes[ind + i + 1] == aspect_list[i + 1]:
                                        seq.get(ind + i + 1).trueLabel = 2
                            break

                #aspect is a single word
                else:
                    for ind, term in enumerate(nodes):
                        if term == aspect and seq.get(ind).trueLabel == 0:
                            seq.get(ind).trueLabel = 1

    return seq


if __name__ == "__main__":

    indice = 0

    assert len( sys.argv )==4
    dirs = {
        "input" : sys.argv[1],
    }

    _args = {
        "random seed" : int(sys.argv[2]),
        "train ratio" : float(sys.argv[3]),
    }

    # read data and aspect, opinion labels

    _data = open(f'{dirs["input"]}/sentence', 'r').read().splitlines()
    _as_label = open(f'{dirs["input"]}/aspectTerm', 'r').read().splitlines()
    _op_label = open(f'{dirs["input"]}/sentence_op', 'r').read().splitlines()

    random.seed( _args["random seed"] )
    _res = shuffle_multi(_data, _as_label, _op_label)
    _data, _as_label, _op_label = [list(_item) for _item in _res]

    _nTrain = int(len(_data)*_args["train ratio"])
    data_train = _data[:_nTrain]
    data_test = _data[_nTrain:]
    as_label_train = _as_label[:_nTrain]
    as_label_test = _as_label[_nTrain:]
    op_label_train = _op_label[:_nTrain]
    op_label_test = _op_label[_nTrain:]

    train_dict = []
    test_dict = []
    vocab = []

    for item, as_item, op_item in zip(data_train, as_label_train, op_label_train):
        seq_item = label_generation(item, as_item, op_item)
        train_dict.append(seq_item)
        for node in seq_item.get_nodes():
            if node.word.lower() not in vocab:
                vocab.append(node.word.lower())

            node.ind = vocab.index(node.word.lower())

    for item, as_item, op_item in zip(data_test, as_label_test, op_label_test):
        seq_item = label_generation(item, as_item, op_item)
        test_dict.append(seq_item)
        for node in seq_item.get_nodes():
            if node.word.lower() not in vocab:
                vocab.append(node.word.lower())

            node.ind = vocab.index(node.word.lower())

    print( f"length of vocab: {len(vocab)}" )

    with open(f"{dirs['input']}/final_input", "wb") as handle:
        pickle.dump((vocab, train_dict, test_dict), handle)
