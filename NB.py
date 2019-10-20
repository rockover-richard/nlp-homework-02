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


def get_filepaths():
    train_fp = input('Please input the name of the training file: ')
    test_fp = input('Please input the name of the test file: ')
    return train_fp, test_fp


def create_dict(filepath='imdb.vocab'):
    vocab_dict = {}
    with open(filepath) as vocab:
        for word in vocab:
            vocab_dict[word] = 0
    return vocab_dict


def train_NB(data_dir):
    pass
