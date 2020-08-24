# -*- coding: utf-8 -*-
##############################################################################
# word2vec embedding data : https://github.com/RaRe-Technologies/gensim-data
#
# execution :
#   $ python /path/to/20word_embedding.py /path/to/word2vecModel /dir/of/output
##############################################################################

import numpy as np
import pickle

if __name__ == "__main__":

    assert len(sys.argv)==3
    dirs = {
        "word2vec-model" : sys.argv[1],
        "output" : sys.argv[2],
    }

    with open(f"{dirs['word2vec-model']}", "r") as dic_file:
        dic = dic_file.readlines()

    dictionary = {}

    for line in dic:
        word_vector = line.split(",")[:-1]
        vector_list = []
        for element in word_vector[len(word_vector)-200:]:
            vector_list.append(float(element))
        word = ','.join(word_vector[:len(word_vector)-200])

        vector = np.asarray(vector_list)
        dictionary[word] = vector


    with open(f"{dirs['output']}/final_input_res15", "rb") as handle
        final_input = pickle.load(handle)

    vocab = final_input[0]
    word_embedding = np.zeros((200, len(vocab)))

    count = 0

    for ind, word in enumerate(vocab):
        if word in dictionary.keys():
            vec = dictionary[word]
            row = 0
            for num in vec:
                word_embedding[row][ind] = float(num)
                row += 1
            count += 1
        else:
            print word,
            for i in range(200):
                word_embedding[i][ind] = 2 * np.random.rand() - 1

    print( f"length of vocab : {len(vocab)}"" )
    print( f"counting of embedded words : {count}" )

    with open(f"{dirs['output']}/word_embeddings200_res15", "wb") as handle:
        pickle.dump(word_embedding, handle)
