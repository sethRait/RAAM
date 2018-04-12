# Recursiely encodes and decodes pairs of word vectors
from __future__ import division
import tensorflow as tf
import numpy as np
import random
import nltk.data
import re
import math
from scipy import spatial

def main(learning_rate):
	word_vector_size = 300
	padding = word_vector_size // 2
	input_size = 2 * (word_vector_size + padding)
	num_epochs = 500
	sen_len = 32

	print("Vector size: %d, with padding: %d" % (word_vector_size, padding))
	print("Learning rate: %f" % learning_rate)

	vectors = "data/wiki-news-300d-1M.vec" # File of word vectors
	corpus = "data/austen.txt"

	original_sentence = tf.placeholder(tf.float32, [None, sen_len, word_vector_size + padding])
	ingest = original_sentence

	# ingest
	depth_ingest = int(math.ceil(math.log(sen_len, 2)))
	new_sen_len = sen_len
        with tf.name_scope('encoder'):
            for i in range(depth_ingest):
                with tf.name_scope(str(i)):
                    R_array = []
                    for j in range(0, new_sen_len, 2):
                        if j == new_sen_len-1:
                            R_array.append(ingest[:,j])
                        else:
                            temp = tf.concat([ingest[:,j], ingest[:,j+1]], axis=1)
                            R = build_encoder(temp)
                            R_array.append(R)
                    ingest = tf.stack(R_array, axis=1)
                    new_sen_len //= 2

	# egest
	egest = ingest
        new_sen_len = 1
        with tf.name_scope('decoder'):
            for i in range(depth_ingest):
                with tf.name_scope(str(i)):
                    R_array = []
                    for j in range(new_sen_len):
                        R = build_decoder(egest[:,j])
                        R_array.extend([R[:,:input_size//2], R[:,input_size//2:]])
                    egest = tf.stack(R_array, axis=1)
                    new_sen_len *=2
            egest = egest[:,0:sen_len,:]

	loss = tf.losses.mean_squared_error(labels=original_sentence, predictions=egest)
	
	train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss)
	sess = tf.InteractiveSession()
	tf.global_variables_initializer().run()
	writer = tf.summary.FileWriter("checkpoints/", sess.graph)
	print '*'*80
        for i in tf.trainable_variables():
            print(i)
	print '*'*80

	sentence_dict = generate_samples(vectors, corpus, word_vector_size, padding)

	# use 4/5 of the sentences to train, and 1/5 to validate
	cut = (4 * len(sentence_dict.values())) // 5
	training_data = sentence_dict.values()[0:cut]
	testing_data = sentence_dict.values()[cut:]

	# Where the magic happens
	train(sess, train_step, np.array(training_data), loss, num_epochs, ingest, egest, original_sentence)
	test(sess, np.array(testing_data), loss, ingest, egest, original_sentence)


def build_encoder(inputs):
	size = inputs.shape[1].value
	with tf.name_scope('encoder') as scope:
            encoded = make_fc(inputs, size, "E_first")
            encoded2 = make_fc(encoded, 3*size//4, "E_second")
	with tf.name_scope('center') as scope:
            center = make_fc(encoded2, size/2, "center")
	return center

def build_decoder(inputs):
	size = inputs.shape[1].value
	with tf.name_scope('decoder') as scope:
            decoded = make_fc(inputs, 3*size//2, "D_first")
            decoded2 = make_fc(decoded, 2*size, "D_second")
	return decoded2

def make_fc(input_tensor, output_size, name):
	input_size = input_tensor.get_shape().as_list()[1]
        with tf.variable_scope('FC', reuse=tf.AUTO_REUSE):
            W = tf.get_variable(name+"weights",[input_size, output_size],tf.float32,
                                                    tf.random_normal_initializer(stddev=0.1))
            b = tf.get_variable(name+'bias',[output_size],tf.float32,tf.zeros_initializer())
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
			# Now we need the sentence to be length 30 (sentence.shape[0] == 30)
			if res.shape[0] < 32:
				padding = 32 - res.shape[0]
				res = np.pad(res, [(0, padding), (0, 0)], mode='constant')
			elif res.shape[0] > 32:
				res = res[0:32]
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
	with open(corpus) as fp:
		nltk.data.load('tokenizers/punkt/english.pickle')
		sentences = nltk.sent_tokenize(fp.read().decode('utf-8'))
	return sentences


def train(sess, optimizer, data, loss, num_epochs, ingest, egest, orig):
	print("Shape is: ")
	print(data.shape)
	for i in range(num_epochs):
		_, train_loss, encoded, decoded = sess.run([optimizer, loss, ingest, egest], feed_dict={orig: data})
		if i % 25 == 0:
			print("Epoch: " + str(i))
			print("Loss: " + str(train_loss))

# Testing loop
def test(sess, data, loss, ingest, egest, orig):
	test_loss, _encoded, decoded = sess.run([loss, ingest, egest], feed_dict={orig: data})
	check_data = data[0]
	check_output = decoded[0]
	zipped = zip(check_data, check_output)
	result = 1 - spatial.distance.cosine(check_data[0], check_output[0])
	print("cosine: " + str(result))
	print("Validation loss: " + str(test_loss))

if __name__ == "__main__":
	learning_rate = .001
	for i in range(10):
		main(learning_rate)
		learning_rate *= 0.5	

