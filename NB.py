'''
Welcome to NB.py, a simple Naive Bayes Classifier
'''

import numpy as np
import math


# Same create_dict from pre-process.py
def create_dict(filepath='aclImdb/imdb.vocab'):
    vocab_dict = {}
    with open(filepath) as vocab:
        for ind, word in enumerate(vocab, 1):
            vocab_dict[word.rstrip()] = ind
    return vocab_dict


'''
Upon starting NB.py, you can input the .NB file for training file and
test file vectors. 

If you just press enter, it will default to 
'vector-file-train.NB' for the training file and
'vector-file-test.NB' for the test file.
'''
def get_filepaths():
    train_file = input('Please input the name of the training file: ')
    if not train_file:
        train_file = 'vector-file-train.NB'
    test_file = input('Please input the name of the test file: ')
    if not test_file:
        test_file = 'vector-file-test.NB'
    return train_file, test_file


'''
This function takes a vector file and creates the log probability parameters
for the two categories in the training vector file.
The categories must be encoded into 0 and 1.

The function takes the filepath to the vocab file, the filepath to the training file,
and the output file name. 

Outputs a file with two vectors on one line each,
one for the negative reviews, one for the positive reviews. 
'''
def build_params(vocab_fp='aclImdb/imdb.vocab',
                 train_fp='vector-file-train.NB',
                 params_file='movie-review-BOW.NB'):
    vocab_dict = create_dict(vocab_fp)
    len_dict = len(vocab_dict)

    # neg_count and pos_count will keep track of the number of neg and pos documents
    neg_count = 0
    pos_count = 0

    # neg_params_v and pos_params_v are the vector files built in the loop below
    neg_params_v = np.zeros(len_dict+1)
    pos_params_v = np.zeros(len_dict+1)

    with open(train_fp, 'r') as train_file:
        for line in train_file:
            vector = np.fromstring(line, dtype=float, sep=' ')

            if vector[0] == 0:
                neg_params_v += vector
                neg_count += 1
            else:
                pos_params_v += vector
                pos_count += 1

    total_count = neg_count + pos_count

    # this is where add-one smoothing happens
    total_neg = sum(neg_params_v[1:])
    total_pos = sum(pos_params_v[1:])
    for i in range(1, len(neg_params_v)):
        neg_params_v[i] = math.log((neg_params_v[i]+1)/(total_neg+len_dict), 2)
        pos_params_v[i] = math.log((pos_params_v[i]+1)/(total_pos+len_dict), 2)

    # the last entry will the prior probabilities for easy reference
    neg_prior = math.log(neg_count/total_count, 2)
    pos_prior = math.log(pos_count/total_count, 2)
    neg_params_v = np.append(neg_params_v, neg_prior)
    pos_params_v = np.append(pos_params_v, pos_prior)

    # the first entry will be the labels for the categories as 0 or 1
    neg_params_v[0] = 0
    pos_params_v[0] = 1

    # write to file
    with open(params_file, 'w+') as params:
        params.write(' '.join(neg_params_v.astype(str)) + '\n')
        params.write(' '.join(pos_params_v.astype(str)))

    print('Total Num of Documents:', total_count)
    print('Total Neg Documents:', neg_count)
    print('Total Pos Documents:', pos_count)
    print('results:', neg_params_v, '\n', pos_params_v, '\n')


'''
This function makes predictions about the category from the Naive Bayes log probabilities
in the parameter vector file. 

The function takes the preprocessed test vector file, the parameter file built by build_params,
and the filename for the output file.

The output file has a prediction per line for each document.
Each line has three columns:
(1) the prediction in 0 or 1
(2) the log probability of that prediction
(3) 0 or 1 to indicate whether the prediction was correct or not
'''
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
            total_count += 1
            neg_prob = neg_vector[-1]
            pos_prob = pos_vector[-1]
            vector = np.fromstring(line, sep=' ')
            nz_ind = np.nonzero(vector)

            for ind in nz_ind[0]:
                if ind == 0:
                    continue
                neg_prob += neg_vector[ind] * vector[ind]
                pos_prob += pos_vector[ind] * vector[ind]

            if neg_prob > pos_prob:
                prediction = 0
                actual = vector[0]
                is_correct = 1 if prediction == actual else 0
                correct_count += is_correct

                cur_line = [str(prediction), str(neg_prob), str(is_correct)]
                pred_output.write(' '.join(cur_line) + '\n')
            else:
                prediction = 1
                actual = vector[0]
                is_correct = 1 if prediction == actual else 0
                correct_count += is_correct

                cur_line = [str(prediction), str(pos_prob), str(is_correct)]
                pred_output.write(' '.join(cur_line) + '\n')

        accu_score = correct_count/total_count
        print('Num of Test Documents:', total_count)
        print('Overall accuracy:', accu_score)
        pred_output.write('Overall accuracy: ' + str(accu_score))

    pred_output.close()


'''
if using TF-IDF-Weight (from http://www.tfidf.com/)
TF(t) = (Number of times term t appears in a document) / (Total number of terms in the document).
IDF(t) = log_2(Total number of documents / Number of documents with term t in it + 1).

However, from Wu et.al. (2017), I used ltf (log_2 of 1 + TF(t)) 
instead because this uses log probabilities,
which works better with my original code. 

The function takes the same inputs as build_params above,
and the output format is the same as well. 
'''
def build_tf_idf(vocab_fp='aclImdb/imdb.vocab',
                 train_fp='vector-file-train.NB',
                 output_file='movie-review-BOW-tf-idf.NB'):
    vocab_dict = create_dict(vocab_fp)
    len_dict = len(vocab_dict)

    doc_count = 0
    pos_count = 0
    neg_count = 0
    neg_params_v = np.zeros(len_dict+1)
    pos_params_v = np.zeros(len_dict+1)
    ltf_vector = np.zeros(len_dict+1)
    idf_vector = np.zeros(len_dict+1)

    # this creates a vector for the idf values of each word
    with open(train_fp, 'r') as train_file:
        for line in train_file:
            vector = np.fromstring(line, dtype=float, sep=' ')
            doc_count += 1
            nz_ind = np.nonzero(vector)
            for ind in nz_ind[0]:
                idf_vector[ind] += 1

        for i, elem in enumerate(idf_vector):
            idf_vector[i] = math.log(doc_count/(elem+1), 2)

    # using ltf
    with open(train_fp, 'r') as train_file:
        for line in train_file:
            vector = np.fromstring(line, dtype=float, sep=' ')

            if vector[0] == 0:
                num_words = sum(vector[1:])
                for i in range(1, len_dict+1):
                    vector[i] *= math.log(1 + (vector[i] / num_words)) + idf_vector[i]
                neg_params_v += vector
                neg_count += 1
            else:
                num_words = sum(vector[1:])
                for i in range(1, len_dict+1):
                    vector[i] *= math.log(1 + (vector[i] / num_words)) + idf_vector[i]
                pos_params_v += vector
                pos_count += 1

    total_count = neg_count + pos_count

    total_neg = sum(neg_params_v[1:])
    total_pos = sum(pos_params_v[1:])
    for i in range(1, len(neg_params_v)):
        neg_params_v[i] = math.log((neg_params_v[i]+1)/(total_neg+len_dict), 2)
        pos_params_v[i] = math.log((pos_params_v[i]+1)/(total_pos+len_dict), 2)

    # the last entry will the prior probabilities
    neg_prior = math.log(neg_count/total_count, 2)
    pos_prior = math.log(pos_count/total_count, 2)
    neg_params_v = np.append(neg_params_v, neg_prior)
    pos_params_v = np.append(pos_params_v, pos_prior)

    # the first entry will be the labels
    neg_params_v[0] = 0
    pos_params_v[0] = 1

    with open(output_file, 'w+') as params:
        params.write(' '.join(neg_params_v.astype(str)) + '\n')
        params.write(' '.join(pos_params_v.astype(str)))


# User can run the file in the console
# May take a moment for each step
ans = input('Build BOW parameters file from training data? (Y/N): ')
if ans.lower() == 'y': 
    print('For default values, just press enter for the following: \n')
    vocab_file = input('Input filepath for vocab: ')
    train_data = input('Input filepath for training data: ')
    output_file = input('Input filename for output file: ')
    if not vocab_file:
        vocab_file = 'aclImdb/imdb.vocab'
    if not train_data:
        train_data = 'vector-file-train.NB'
    if not params_file:
        output_file = 'movie-review-BOW.NB'
    build_params(vocab_fp=vocab_file, train_fp=train_data, params_file=output_file)
else:
    print('Skipping parameters file build...\n') 

ans = input('Build BOW prediction file from test data? (Y/N): ')
if ans.lower() == 'y': 
    print('For default values, just press enter for the following: \n')
    test_file = input('Input filepath for test data: ')
    params = input('Input filepath for parameters file: ')
    output_file = input('Input filename for output file: ')
    if not vocab_file:
        test_file = 'vector-file-test.NB'
    if not params:
        params = 'movie-review-BOW.NB'
    if not output_file:
        output_file = 'BOW-predictions.NB'
    pred_NB(test_file=test_file, params_file=params, pred_file=output_file)
else:
    print('Skipping building BOW predictions...\n') 

# For optional tf_idf feature:
ans = input('Build tf-idf parameters file from training data? (Y/N): ')
if ans.lower() == 'y': 
    print('For default values, just press enter for the following: \n')
    vocab_file = input('Input filepath for vocab: ')
    train_data = input('Input filepath for training data: ')
    output_file = input('Input filename for output file: ')
    if not vocab_file:
        vocab_file = 'aclImdb/imdb.vocab'
    if not train_data:
        train_data = 'vector-file-train.NB'
    if not params_file:
        output_file = 'movie-review-BOW-tf-idf.NB'
    build_tf_idf(vocab_fp=vocab_file, train_fp=train_data, params_file=output_file)
else:
    print('Skipping parameters file build...\n') 

ans = input('Build tf-idf prediction file from test data? (Y/N): ')
if ans.lower() == 'y': 
    print('For default values, just press enter for the following: \n')
    test_file = input('Input filepath for test data: ')
    params = input('Input filepath for parameters file: ')
    output_file = input('Input filename for output file: ')
    if not vocab_file:
        test_file = 'vector-file-test.NB'
    if not params:
        params = 'movie-review-BOW-tf-idf.NB'
    if not output_file:
        output_file = 'BOW-tf-idf-predictions.NB'
    pred_NB(test_file=test_file, params_file=params, pred_file=output_file)
else:
    print('Closing file...\n') 

# test
# build_params(vocab_fp='small/small.vocab',
#              train_fp='movie-review-small.NB',
#              params_file='small-BOW.NB')

# pred_NB(test_file='movie-review-small-test.NB',
#         params_file='small-BOW.NB',
#         pred_file='small-BOW-pred.NB')

# build_tf_idf()
# pred_NB(params_file='movie-review-BOW-tf-idf.NB', pred_file='BOW-tf-idf-predictions.NB')
