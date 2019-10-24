'''
Welcome to pre-process.py
'''

import re
import numpy as np
import os

'''
This functions creates a dictionary out of the vocab file to create a reference hash table
to assign each word a unique index in the vector files. 
'''
def create_dict(filepath='aclImdb/imdb.vocab'):
    vocab_dict = {}
    with open(filepath) as vocab:
        for ind, word in enumerate(vocab, 1):
            vocab_dict[word.rstrip()] = ind
    return vocab_dict


'''
This function performs preprocessing on a file and outputs a list.
'''
def split_text(file):
    # I create a set of emoticons that are in the vocab file to reference.
    emoticons = {':)', ':-)', ';)', ';-)', ');', '=)', '8)', '):', ':o)',
                 ';o)', '=o)', ':(', '(8', ':-(', '(=', '8(', '=('}

    # These are common leading or trailing characters
    sp_char = {':', ';', "'", '(', ')'}

    words = file.read()

    # I first split at the following characters, but leaving ! and ? in.
    temp_list = re.split(r"\s|[,.<>*\"]|'s|([!?])", words.lower())
    word_list = []

    for word in temp_list:
        if not word:
            continue
        # to handle emoticons
        if ')' in word or '(' in word:
            for icon in emoticons:
                if icon in word:
                    word_list.append(icon)
        # to handle leading special characters in words
        # e.g. (hello -> hello
        if word[-1] in sp_char and len(word) > 1:
            word = word[:-1]
        # to handle trailing special characters in words
        # e.g. world: -> world
        if word[0] in sp_char and len(word) > 1:
            word = word[1:]
        word_list.append(word)

    return word_list


'''
This function takes the cat(egory) (i.e. pos or neg), file, dictionary, and the
length of the dictionary to create a vector as a string.
'''
def update_vector(cat, file, vocab_dict, len_dict):
    v = np.zeros(len_dict+1, dtype=int)
    v[0] = cat

    with open(file, 'r', encoding='UTF-8') as ex:
        word_list = split_text(ex)
        for word in word_list:
            if word in vocab_dict:
                v[vocab_dict[word]] += 1

    return v.astype(str)


'''
This is the final function that creates a file with one vector per line
corresponding to one document in the directory.

The function takes the input dir(ectory), which should be organized in folders
by category, the name of the output file, the name of the categories, and
a boolean indicating whether or not it is the training data.

The directory should be organized as follows:
[input_dir]/[train or test]/[category]/

In the output file,
The first column is the category, neg or pos, encoded as a 0 or 1 respectively.
The rest of the columns are a vector with the counts of the words within the document.
'''
def create_vector_file(input_dir='aclImdb/',
                       filename='vector-file.txt',
                       cat1='neg', cat2='pos',
                       train=True):
    vocab_dict = create_dict(filepath=input_dir+'imdb.vocab')
    len_dict = len(vocab_dict)

    vector_file = open(filename, 'w+')

    if train:
        fp = 'train/'
    else:
        fp = 'test/'

    cat1_path = input_dir + fp + cat1 + '/'
    with os.scandir(cat1_path) as cat1_dir:
        for file in cat1_dir:
            vector = update_vector(0, file, vocab_dict, len_dict)
            vector_file.write(' '.join(vector) + '\n')

    cat2_path = input_dir + fp + cat2 + '/'
    with os.scandir(cat2_path) as cat2_dir:
        for file in cat2_dir:
            vector = update_vector(1, file, vocab_dict, len_dict)
            vector_file.write(' '.join(vector) + '\n')

    vector_file.close()


# User can run the file in the console
# May take ~30 minutes per pre-process step
ans = input('Run pre-processing on training corpus? (Y/N): ')
    if ans.lower() = 'y': 
        create_vector_file(filename='vector-file-train.NB')
    else:
        print('Skipping training corpus pre-processing...\n') 

ans = input('Run pre-processing on test corpus? (Y/N): ')
    if ans.lower() = 'y': 
        create_vector_file(filename='vector-file-test.NB', train=False)
    else:
        print('Skipping test corpus pre-processing...\n') 

# for testing:
# with open('testing.txt') as test_file:
#     print(split_text(test_file))

# test of small data set from question 2a
# create_vector_file(input_dir='small/', filename='movie-review-small.NB', cat1='action', cat2='comedy')
# create_vector_file(input_dir='small/', filename='movie-review-small-test.NB', cat1='action', cat2='comedy', train=False)
