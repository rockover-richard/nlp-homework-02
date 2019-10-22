'''
NB.py should take the following parameters:
the training file, the test file,
the file where the parameters of the resulting model will be saved,
and the output file where you will write predictions
made by the classifier on the test data (one example per line).

The last line in the output file should list the overall accuracy
of the classifier on the test data.

The training and test files should have the following format:
one example per line; each line corresponds to an example;
first column is the label, and the other columns are feature values

Train NB classifier on the training partition using BOW features
(use add-one smoothing). Evaluate classifier on the test partition.

Save parameters of your BOW model in a file called movie-review-BOW.NB.
Report the accuracy of your program on the test data with BOW features.
'''
import numpy as np
import math


def create_dict(filepath='aclImdb/imdb.vocab'):
    vocab_dict = {}
    with open(filepath) as vocab:
        for ind, word in enumerate(vocab, 1):
            vocab_dict[word.rstrip()] = ind
    return vocab_dict


def get_filepaths():
    train_file = input('Please input the name of the training file: ')
    test_file = input('Please input the name of the test file: ')
    return train_file, test_file


def build_params(vocab_fp='aclImdb/imdb.vocab',
                 train_fp='vector-file-train.NB',
                 params_file='movie-review-BOW.NB'):
    vocab_dict = create_dict(vocab_fp)
    len_dict = len(vocab_dict)

    neg_count = 0
    neg_params_v = np.zeros(len_dict+1)
    pos_params_v = np.zeros(len_dict+1)

    with open(train_fp, 'r') as train_file:
        for line in train_file:
            vector = np.fromstring(line, dtype=float, sep=' ')
            if vector[0] == 0:
                neg_count += 1
                neg_params_v += vector
            else:
                pos_params_v += vector

    pos_count = pos_params_v[0]
    total_neg = sum(neg_params_v[1:])
    total_pos = sum(pos_params_v[1:])

    # the second entry will the prior probabilities
    neg_params_v[0] = math.log((neg_count)/(neg_count+pos_count), 2)
    pos_params_v[0] = math.log((pos_count)/(neg_count+pos_count), 2)

    for i in range(1, len(neg_params_v)):
        neg_params_v[i] = math.log((neg_params_v[i]+1)/(total_neg+len_dict), 2)
        pos_params_v[i] = math.log((pos_params_v[i]+1)/(total_pos+len_dict), 2)

    # the first entry will be the labels
    neg_params_v = np.insert(neg_params_v, 0, 0)
    pos_params_v = np.insert(pos_params_v, 0, 1)

    with open(params_file, 'w+') as params:
        params.write(' '.join(neg_params_v.astype(str)) + '\n')
        params.write(' '.join(pos_params_v.astype(str)))

    print('results:', neg_params_v, '\n', pos_params_v, '\n')


# incomplete
def pred_NB(test_file='vector-file-test.NB',
            params_file='movie-review-BOW.NB',
            pred_file='BOW-predictions.NB'):

    with open(params_file, 'r') as pf:
        params = pf.readlines()
        neg_vector = np.fromstring(params[0], sep=' ')
        pos_vector = np.fromstring(params[1], sep=' ')

    pred_output = open(pred_file, 'w+')

    total_count = 0
    correct_count = 0
    with open(test_file, 'r') as tf:
        for line in tf:
            neg_prob = neg_vector[1]
            pos_prob = pos_vector[1]
            vector = np.fromstring(line, sep=' ')
            nz_ind = np.nonzero(vector)
            total_count += 1

            for ind in nz_ind[0]:
                if ind == 0:
                    continue
                neg_prob += neg_vector[ind] * vector[ind]
                pos_prob += pos_vector[ind] * vector[ind]

            if neg_prob > pos_prob:
                prediction = '0'
                is_correct = 1 if neg_vector[0] == 0 else 0
                correct_count += is_correct

                cur_line = [prediction, str(neg_prob), str(is_correct)]
                pred_output.write(' '.join(cur_line) + '\n')
            else:
                prediction = '1'
                is_correct = 1 if pos_vector[0] == 1 else 0
                correct_count += is_correct

                cur_line = [prediction, str(pos_prob), str(is_correct)]
                pred_output.write(' '.join(cur_line) + '\n')

        accu_score = correct_count/total_count
        pred_output.write('Overall accuracy: ' + str(accu_score))

    pred_output.close()

# test
# build_params(vocab_fp='small/small.vocab',
#              train_fp='movie-review-small.NB',
#              params_file='small-BOW.NB')

# pred_NB(test_file='movie-review-small-test.NB',
#         params_file='small-BOW.NB',
#         pred_file='small-BOW-pred.NB')

# train_file, test_file = get_filepaths()
build_params()
pred_NB()
