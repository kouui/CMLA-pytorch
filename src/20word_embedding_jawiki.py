# -*- coding: utf-8 -*-
##############################################################################
# word2vec embedding data : https://github.com/RaRe-Technologies/gensim-data
#
# execution :
#   $ python /path/to/20word_embedding.py /path/to/word2vecModel /dir/of/output
##############################################################################

import numpy as np
import pickle

import sys

if __name__ == "__main__":

    assert len(sys.argv)==3
    dirs = {
        "word2vec-model" : sys.argv[1],
        "output" : sys.argv[2],
    }

    with open(f"{dirs['word2vec-model']}", "r") as dic_file:
        dic = dic_file.readlines()

    dictionary = {}

    for i, line in enumerate(dic[1:]):

        words = line.split()
        assert len(words) == 201, f"{words[0]}, l{len(words)}"

        word = words[0].strip()
        vector = np.asarray( [float(v) for v in words[1:]], dtype=np.float32 )

        dictionary[word] = vector

    with open(f"{dirs['output']}/final_input", "rb") as handle:
        final_input = pickle.load(handle)

    vocab = final_input[0]
    word_embedding = np.zeros((200, len(vocab)), dtype=np.float32)

    count = 0
    ng_words = []
    for ind, word in enumerate(vocab):
        if word in dictionary.keys():
            print(f"embeded word : {word}")
            vec = dictionary[word]
            word_embedding[:,ind] = vec[:]
            count += 1
        else:
            ng_words.append( word )
            word_embedding[:,ind] = 2 * np.random.rand(200).astype(np.float32) - 1

    print( f"length of vocab : {len(vocab)}" )
    print( f"counting of embedded words : {count}" )
    print( f"counting of NG words : {len(ng_words)}" )

    path = {
        "word_embeddings200": f"{dirs['output']}/word_embeddings200",
        "ng_words" : f"{dirs['output']}/ng_words.txt"
    }

    with open(path["ng_words"], 'w') as f:
        f.write( '\n'.join(ng_words) )
    with open(f"{dirs['output']}/word_embeddings200", "wb") as handle:
        pickle.dump(word_embedding, handle)

    for key, fname in path.items():
        print(f"{key} has been saved as {fname}")
