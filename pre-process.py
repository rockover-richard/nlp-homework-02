'''
pre-process.py should take the training (or test) directory
containing movie reviews, should perform pre-processing on each file
and output the files in the vector format to be used by NB.py

Pre-processing: prior to building feature vectors, you should separate
punctuation from words and lowercase the words in the reviews.
'''

def create_dict(filepath='imdb.vocab'):
    vocab_dict = {}
    with open(filepath) as vocab:
        for word in vocab:
            vocab_dict[word] = 0
    return vocab_dict
