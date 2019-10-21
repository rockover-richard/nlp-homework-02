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


def build_params(train_file, params_file='movie-review-BOW.NB'):
    vocab_dict = create_dict()
    len_dict = len(vocab_dict)

    neg_count = 0
    neg_counts_v = np.zeros(len_dict+1)
    pos_counts_v = np.zeros(len_dict+1)

    for line in train_file:
        vector = np.fromstring(line, dtype=float, sep=' ')
        if vector[0] == 0:
            neg_count += 1
            neg_counts_v += vector
        else:
            pos_counts_v += vector

    pos_count = pos_counts_v[0]
    total_neg = sum(neg_counts_v[1:])
    total_pos = sum(pos_counts_v[1:])

    # the first entry will be the labels
    neg_counts_v = np.insert(neg_counts_v, 0, 0)
    pos_counts_v = np.insert(pos_counts_v, 0, 1)

    # the second entry will the prior probabilities
    neg_counts_v[1] = (neg_count)/(neg_count+pos_count)
    pos_counts_v[1] = (pos_count)/(neg_count+pos_count)

    for i in range(1, len(neg_counts_v)):
        neg_counts_v[i] = (neg_counts_v[i]+1)/(total_neg+len_dict)
        pos_counts_v[i] = (pos_counts_v[i]+1)/(total_pos+len_dict)

    with open(params_file, 'w+') as params:
        params.write(' '.join(neg_counts_v.astype(str)) + '\n')
        params.write(' '.join(pos_counts_v.astype(str)))

    return neg_counts_v, pos_counts_v


# incomplete
def train_NB(train_file, test_file,
             params_file='movie-review-BOW.NB',
             pred_file='BOW-predictions.NB'):

    with open(params_file, 'r') as pf:
        params = pf.readlines()
        neg_vector = params[0]
        pos_vector = params[1]

    pred_output = open(pred_file, 'w+')

    with open(test_file, 'r') as tf:
        for line in tf:
            neg_prob = neg_vector[0]
            pos_prob = pos_vector[0]
            vector = np.fromstring(line, sep=' ')
            nz_ind = np.nonzero(vector)
            for ind in nz_ind[1:]:
                neg_prob *= neg_vector[ind] ** vector[ind]
                pos_prob *= pos_vector[ind] ** vector[ind]
            if neg_prob > pos_prob:
                pred_output.write('neg', neg_prob, 0 if neg_prob)
            else:
                pred_output.write('pos', )
    pred_output.close()

    pass


with open('movie-review-small.NB') as mrs:
    build_params(mrs, params_file='small-BOW.NB')

# train_file, test_file = get_filepaths()
# train_NB = train_NB(train_file, test_file, params_file='small-BOW.NB', pred_file='small-predictions.NB')
