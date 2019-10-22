'''
pre-process.py should take the training (or test) directory
containing movie reviews, should perform pre-processing on each file
and output the files in the vector format to be used by NB.py

Pre-processing: prior to building feature vectors, you should separate
punctuation from words and lowercase the words in the reviews.
'''

import re
import numpy as np
import os


def create_dict(filepath='aclImdb/imdb.vocab'):
    vocab_dict = {}
    with open(filepath) as vocab:
        for ind, word in enumerate(vocab, 1):
            vocab_dict[word.rstrip()] = ind
    return vocab_dict

create_dict()

def split_text(file):
    emoticons = {':)', ':-)', ';)', ';-)', ');', '=)', '8)', '):', ':o)',
                 ';o)', '=o)', ':(', '(8', ':-(', '(=', '8(', '=('}
    sp_char = {':', ';', "'", '(', ')'}

    words = file.read()
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

# with open('testing.txt') as test_file:
#     print(split_text(test_file))


def update_vector(cat, file, vocab_dict, len_dict):
    v = np.zeros(len_dict+1, dtype=int)
    v[0] = cat

    with open(file, 'r', encoding='UTF-8') as ex:
        word_list = split_text(ex)
        for word in word_list:
            if word in vocab_dict:
                v[vocab_dict[word]] += 1

    return v.astype(str)


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
            vector_file.write(' '.join(vector) + '\n') # need to check if this shows in NB.py

    cat2_path = input_dir + fp + cat2 + '/'
    with os.scandir(cat2_path) as cat2_dir:
        for file in cat2_dir:
            vector = update_vector(1, file, vocab_dict, len_dict)
            vector_file.write(' '.join(vector) + '\n')

    vector_file.close()


# create_vector_file(input_dir='small/', filename='movie-review-small.NB', cat1='action', cat2='comedy')
# create_vector_file(input_dir='small/', filename='movie-review-small-test.NB', cat1='action', cat2='comedy', train=False)

create_vector_file(filename='vector-file-train.NB')
create_vector_file(filename='vector-file-test.NB', train=False)
