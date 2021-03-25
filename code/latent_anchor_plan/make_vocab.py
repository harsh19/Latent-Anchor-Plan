"""
Make dictionary.pkl for models given training data.
"""

import argparse
import _pickle as pickle

import data

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train-data', type=str, default='data/penn/train.txt',
                        help='location of the training data corpus')
    parser.add_argument('--valid-data', type=str, default='data/penn/valid.txt',
                        help='location of the valid data corpus')
    parser.add_argument('--test-data', type=str, default='data/penn/test.txt',
                        help='location of the test data corpus')
    parser.add_argument('--output', type=str, default='models/dictionary.pkl',
                        help='location of the file for vocabulary')
    args = parser.parse_args()

    print('Creating dictionary...')
    corpus = data.Corpus(train_path=args.train_data, dev_path=args.valid_data, test_path=args.test_data, output=args.output)




