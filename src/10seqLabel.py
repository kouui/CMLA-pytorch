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

    assert len( sys.argv )==3
    dirs = {
        "input" : sys.argv[1],
        "output" : sys.argv[2],
    }

    # read data and aspect, opinion labels
    data_train = open(f'{dirs["input"]}/sentence_res15', 'r').read().splitlines()
    data_test = open(f'{dirs["input"]}/sentence_restest15', 'r').read().splitlines()

    train_dict = []
    test_dict = []
    vocab = []

    as_label_train = open(f'{dirs["input"]}/aspectTerm_res15', 'r').read().splitlines()
    as_label_test = open(f'{dirs["input"]}/aspectTerm_restest15', 'r').read().splitlines()

    op_label_train = open(f'{dirs["input"]}/sentence_res15_op', 'r').read().splitlines()
    op_label_test = open(f'{dirs["input"]}/sentence_restest15_op', 'r').read().splitlines()

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

    with open(f"{dirs['output']}/final_input_res15", "wb") as handle:
        pickle.dump((vocab, train_dict, test_dict), handle)
