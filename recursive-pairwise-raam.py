# Recursiely encodes and decodes pairs of word vectors
from __future__ import division
import tensorflow as tf
import numpy as np
import random
import nltk.data
import re
import math

def main():
    word_vector_size = 300
    padding = word_vector_size // 2
    input_size = 2 * (word_vector_size + padding)
    learning_rate = 0.00001
    num_epochs = 150

    print("Vector size: %d, with padding: %d" % (word_vector_size, padding))
    print("Learning rate: %f" % learning_rate)

    vectors = "data/wiki-news-300d-1M.vec" # File of word vectors
    corpus = "data/austen.txt"

    input1 = tf.placeholder(tf.float32, [None, input_size/2], name="first_half") # first word
    input2 = tf.placeholder(tf.float32, [None, input_size/2], name="second_half") # second word
    inputs = tf.concat([input1, input2], 1, name="full_input")

    # layers
    center, output_layer = generate_layers(inputs, input_size)

    loss = tf.losses.mean_squared_error(labels=inputs, predictions=output_layer)
    train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss)
    sess = tf.InteractiveSession()
    tf.global_variables_initializer().run()

    sentence_dict = generate_samples(vectors, corpus, word_vector_size, padding)
    cut = (4 * len(sentence_dict.values())) // 5
    training_data = sentence_dict.values()[0:cut]
    testing_data = sentence_dict.values()[cut:]
    training_data = length_order(training_data)
    train(sess, train_step, training_data, center, output_layer, loss, input1, input2, input_size, num_epochs)

# Order the training data by sentence length to allow for parallel data training
def length_order(data):
    outs = []
    data.sort(key=lambda x: x.shape[0], reverse=True) # sort the list by length of sublists, longest is first
    last_len = data[0].shape[0]
    outs.append(data[0])
    for arr in data:
        if arr.shape[0] == last_len:
            if arr.ndim == 2 and outs[len(outs) - 1].ndim == 3:
                outs[len(outs) - 1] = np.concatenate((outs[len(outs) - 1],
                    arr.reshape(1, arr.shape[0], arr.shape[1])))
            else:
                outs[len(outs) - 1] = np.asarray([outs[len(outs) - 1], arr])
        else:
            last_len = arr.shape[0]
            outs.append(arr)
    return outs

def generate_layers(inputs, input_size):
    encoded = make_fc(inputs, input_size, "encoder", 3)
    encodedr = make_fc(encoded, input_size, "encoderr", 3)
    encoded2 = make_fc(encodedr, 3*input_size//4, "second_encoder", 3)
    encoded3 = make_fc(encoded2, 3*input_size//4, "third_encoder", 3)
    center = make_fc(encoded3, input_size/2, "center", 3)
    decoded1 = make_fc(center, 3*input_size//2, "decoder1", 3)
    decoded = make_fc(decoded1, 3*input_size//2, "decoder", 3)
    decoded2 = make_fc(decoded, input_size, "second_decoder", 3)
    decoded2r = make_fc(decoded2, input_size, "second_decoderr", 3)
    return center, decoded2r

def make_fc(input_tensor, output_size, name, mode):
    W = tf.get_variable(name + "weights",[input_tensor.get_shape().as_list()[1],output_size],tf.float32,
                                                          tf.random_normal_initializer(stddev=0.1))
    b = tf.Variable(tf.zeros([output_size]))
    if mode == 1: # sigmoid
        x = tf.nn.sigmoid(tf.matmul(input_tensor, W) + b)
    if mode == 2: # relu
        x = tf.nn.relu(tf.matmul(input_tensor, W) + b)
    if mode == 3: # tanh
        x = tf.nn.tanh(tf.matmul(input_tensor, W) + b)
    return x

# Returns a dictionary of sentances and a list of their vector representation
def generate_samples(vectors, corpus, vec_size, pad):
    word_dict = parse_word_vecs(vectors, vec_size, pad)
    sentences = parse_sentences(corpus)
    sentence_dict = {}
    for sentence in sentences:
        res = get_vecs_from_sentence(sentence, word_dict)
        if res is not None:
            sentence_dict[sentence] = res
    return sentence_dict

# Returns an np array of vectors representing the words of the given sentence
def get_vecs_from_sentence(sentence, word_dict):
    arr = []
    for word in re.findall(r"[\w]+|[^\s\w]", sentence): # Each punctuation mark should be its own vector
        cur = word_dict.get(word.lower())
        if cur is None:
            return None
        arr.append(cur)
    return np.array(arr)

# Parses the file containing vector representations of words
def parse_word_vecs(vectors, vec_size, pad):
    print("Parsing word vectors")
    i = 1
    dictionary = {}
    with open(vectors) as fp:
        next(fp) # skip header
        for line in fp:
            parsed = line.lower().split(' ', 1)
            vec = np.fromstring(parsed[1], dtype = float, count = vec_size, sep = " ")
            dictionary[parsed[0]] = np.pad(vec, (0, pad), 'constant') # right pad the vector with 0
            i += 1
            if i % 100000 == 0: # Only use the first 100,000 words
                break
    return dictionary

# Parses the file containing the training and testing sentences
def parse_sentences(corpus):
    print("Parsing input sentences")
    with open(corpus) as fp:
        nltk.data.load('tokenizers/punkt/english.pickle')
        sentences = nltk.sent_tokenize(fp.read().decode('utf-8'))
    return sentences


def train(sess, optimizer, data, encode, decode, loss, input1, input2, size, num_epochs):
    print("Training on %d groups per epoch" % len(data))
    print("Training for %d epochs" % num_epochs)
    for i in range(num_epochs):
        for group in data: # A group is a collection of sentences of the same length
            np.random.shuffle(group)
            if group.ndim == 2: # if there is only one sentene in the group
                group = np.reshape(group, (1, group.shape[0], group.shape[1]))
            while group.shape[1] != 1: # stop when the sentences have been encoded to 1 vector
                train_loss, group = train_inner(sess, optimizer, encode, decode, group, loss, input1, input2, size)
        if i % 5 == 0:
            print("Epoch: " + str(i))
            print("Loss: " + str(train_loss))

# input/output shape : (<number of sentences>, <length of sentences>, <words>)
def train_inner(sess, optimizer, encode, decode, ins, loss, input1, input2, size):
    outs = np.empty((0, size//2), dtype=float)
    while ins.shape[1] > 0: # unconsumed word-vectors
        if ins.shape[1] >= 2: # if there is more than one vector left
            _, train_loss, encoded, _ = sess.run([optimizer, loss, encode, decode],
                    feed_dict={input1:ins[:,0,:], input2:ins[:,1,:]})
            ins = ins[:,2:,:] # pop the top two
            outs = np.concatenate((outs, encoded))
        else: # If there's only one item left, add it to the output to use next round
            outs = np.concatenate((outs, ins[:,0,:]))
            break
    if outs.ndim == 2:
        outs = np.reshape(outs, (1, outs.shape[0], outs.shape[1]))
    return train_loss, outs

# Testing loop
def test(sess, epochs, data, decode, loss, input1, input2):
    num_vectors_total = 0
    total_correct = 0
    print("TESTING")
    for i in range(epochs):
        test_loss, my_decoded, orig = sess.run([loss, decode, inputs], feed_dict={inputs:data[i % len(data)]})

        if i % (test_epochs / 10) == 0:
            print(str((i / test_epochs) * 100) + " percent complete")

        # Reporting
        for original, decode in zip(orig, my_decoded):
            num_vectors_total += 1
            for truth, gen in zip(original, decode):
                if abs(truth-gen) <= 0.1:
                    total_correct += 1

    percent_correct = ((total_correct / num_vectors_total) * 100) / 300
    print(percent_correct)

if __name__ == "__main__":
    main()
