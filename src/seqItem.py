
# - an individual node contains the word associated with the index and label
class Node:

    def __init__(self, word):
        if word != None:
            self.word = word
            self.is_word = 1
            self.trueLabel = 0

            # the "ind" variable stores the look-up index of the word in the
            # word embedding matrix We. set this value when the vocabulary is finalized
            self.ind = -1

        else:
            self.is_word = 0

class SeqItem:

    def __init__(self, word_list):
        self.nodes = []
        for word in word_list:
            self.nodes.append(Node(word))

        self.hasNotWord = False

    # return all non-None nodes
    def get_nodes(self):
        return [node for node in self.nodes if node.is_word]


    def get_node_inds(self):
        return [(ind, node) for ind, node in enumerate(self.nodes) if node.is_word]


    # get a node from the raw node list
    def get(self, ind):
        return self.nodes[ind]


    # return the raw text of the sentence
    def get_words(self):
        return ' '.join([node.word for node in self.get_nodes()[1:]])


    def reset_finished(self):
        for node in self.get_nodes():
            node.finished = 0

    def error(self):
        sum = 0.0
        for node in self.get_nodes():
            sum += node.label_error

        return sum
