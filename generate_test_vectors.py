'''
Use to generate nonsense test word vectors of varying size which are easier to work with and debug with than word2vec
trained vectors.  vector_size should be a power of 2 which is smaller than 300 (the size of the vectors in:
https://fasttext.cc/docs/en/english-vectors.html
'''
from __future__ import print_function

import numpy as np
import nltk.data
import re

def main():
    vector_size = 8
    dictionary = get_vocab_dict(vector_size)
    print_out(dictionary)

def get_vocab_dict(size):
    dictionary = {}
    with open('data/test_sentences.txt') as fp:
        nltk.data.load('tokenizers/punkt/english.pickle')
        sentences = nltk.sent_tokenize(fp.read().decode('utf-8'))
    for sentence in sentences:
        for word in re.findall(r"[\w]+|[^\s\w]", sentence):
            if word.lower() not in dictionary:
                dictionary[word.lower()] = np.random.uniform(low=0, high=1, size=(size,))
    return dictionary

def print_out(dictionary):
    for k,v in dictionary.iteritems():
        print(k.encode('utf-8'), end=" ")
        for d in v:
            print(d, end=" ")
        print()

if __name__ == '__main__':
    main()
