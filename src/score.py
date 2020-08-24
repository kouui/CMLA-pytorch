import numpy as np

#f1 score
def score_aspect(true_list, predict_list):

    correct = 0
    predicted = 0
    relevant = 0

    i=0
    j=0
    pairs = []
    while i < len(true_list):
        true_seq = true_list[i]
        predict = predict_list[i]

        for num in range(len(true_seq)):
            if true_seq[num] == '1':
                if num < len(true_seq) - 1:
                    #if true_seq[num + 1] == '0' or true_seq[num + 1] == '1':
                    if true_seq[num + 1] != '2':
                        #if predict[num] == '1':
                        if predict[num] == '1' and predict[num + 1] != '2':
                        #if predict[num] == '1' and predict[num + 1] != '1':
                            correct += 1
                            #predicted += 1
                            relevant += 1
                        else:
                            relevant += 1

                    else:
                        if predict[num] == '1':
                            for j in range(num + 1, len(true_seq)):
                                if true_seq[j] == '2':
                                    if predict[j] == '2' and j < len(predict) - 1:
                                    #if predict[j] == '1' and j < len(predict) - 1:
                                        continue
                                    elif predict[j] == '2' and j == len(predict) - 1:
                                    #elif predict[j] == '1' and j == len(predict) - 1:
                                        correct += 1
                                        relevant += 1

                                    else:
                                        relevant += 1
                                        break

                                else:
                                    if predict[j] != '2':
                                    #if predict[j] != '1':
                                        correct += 1
                                        #predicted += 1
                                        relevant += 1
                                        break


                        else:
                            relevant += 1

                else:
                    if predict[num] == '1':
                        correct += 1
                        #predicted += 1
                        relevant += 1
                    else:
                        relevant += 1


        for num in range(len(predict)):
            if predict[num] == '1':
                predicted += 1


        i += 1

    precision = float(correct) / (predicted + 1e-6)
    recall = float(correct) / (relevant + 1e-6)
    f1 = 2 * precision * recall / (precision + recall + 1e-6)

    return precision, recall, f1


def score_opinion(true_list, predict_list):

    correct = 0
    predicted = 0
    relevant = 0

    i=0
    j=0
    pairs = []
    while i < len(true_list):
        true_seq = true_list[i]
        predict = predict_list[i]

        for num in range(len(true_seq)):
            if true_seq[num] == '1':
                if num < len(true_seq) - 1:
                    #if true_seq[num + 1] == '0' or true_seq[num + 1] == '3':
                    if true_seq[num + 1] != '2':
                        #if predict[num] == '3':
                        #if predict[num] == '1' and predict[num + 1] != '1':
                        if predict[num] == '1' and predict[num + 1] != '2':
                            correct += 1
                            #predicted += 1
                            relevant += 1
                        else:
                            relevant += 1

                    else:
                        if predict[num] == '1':
                            for j in range(num + 1, len(true_seq)):
                                if true_seq[j] == '2':
                                    #if predict[j] == '1' and j < len(predict) - 1:
                                    if predict[j] == '2' and j < len(predict) - 1:
                                        continue
                                    #elif predict[j] == '1' and j == len(predict) - 1:
                                    elif predict[j] == '2' and j == len(predict) - 1:
                                        correct += 1
                                        relevant += 1

                                    else:
                                        relevant += 1
                                        break

                                else:
                                    #if predict[j] != '1':
                                    if predict[j] != '2':
                                        correct += 1
                                        #predicted += 1
                                        relevant += 1
                                        break


                        else:
                            relevant += 1

                else:
                    if predict[num] == '1':
                        correct += 1
                        #predicted += 1
                        relevant += 1
                    else:
                        relevant += 1


        for num in range(len(predict)):
            if predict[num] == '1':
                predicted += 1


        i += 1

    precision = float(correct) / (predicted + 1e-6)
    recall = float(correct) / (relevant + 1e-6)
    f1 = 2 * precision * recall / (precision + recall + 1e-6)

    return precision, recall, f1
