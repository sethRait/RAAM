# Encodes and decodes pairs of word vectors as well as pairs of encoded pairs of word vectors.

from __future__ import division
import tensorflow as tf
import numpy as np
import random
import nltk.data
import re
import math

def main():
	word_vector_size = 8
	padding = word_vector_size // 2
	input_size = 2 * (word_vector_size + padding)
	test_epochs = 1500
	learning_rate = 0.00002
	vectors = "data/test_vectors.vec" # File of word vectors
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

	train(sess, train_step, sentence_dict.values(), center, output_layer, loss, input1, input2, input_size)
	test(sess, test_epochs, sentence_dict.values(), output_layer, loss)

def generate_layers(inputs, input_size):
	encoded = make_fc(inputs, input_size, "encoder", 1)
	encodedr = make_fc(encoded, input_size, "encoderr", 2)
	encoded2 = make_fc(encodedr, 3*input_size//4, "second_encoder", 2)
	center = make_fc(encoded2, input_size/2, "center", 2)
	decoded = make_fc(center, 3*input_size//2, "decoder", 2)
	decoded2 = make_fc(decoded, input_size, "second_decoder", 2)
	decoded2r = make_fc(decoded2, input_size, "second_decoderr", 1)
	return center, decoded2r

def make_fc(input_tensor, output_size, name, mode):
	W = tf.get_variable(name + "weights",[input_tensor.get_shape().as_list()[1],output_size],tf.float32,
			                                      tf.random_normal_initializer(stddev=0.1))
	b = tf.Variable(tf.zeros([output_size]))
	if mode == 1: # sigmoid
		x = tf.nn.sigmoid(tf.matmul(input_tensor, W) + b)
	else: # relu
		x = tf.nn.relu(tf.matmul(input_tensor, W) + b)
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
			if i % 10000 == 0: # Only use the first 10,000 words
				break
	return dictionary

# Parses the file containing the training and testing sentences
def parse_sentences(corpus):
	print("Parsing input sentences")
	with open(corpus) as fp:
		nltk.data.load('tokenizers/punkt/english.pickle')
		sentences = nltk.sent_tokenize(fp.read().decode('utf-8'))
	return sentences


def train(sess, optimizer, data, encode, decode, loss, input1, input2, size):
	print("Training")
	for i in range(200):
		for sentence in data:
			train_loss = 0.0
			while len(sentence) != 1:
				train_loss, sentence = train_inner(sess, optimizer, encode, decode, sentence, loss, input1, input2, size)
		print("Loss: " + str(train_loss))

def train_inner(sess, optimizer, encode, decode, ins, loss, input1, input2, size):
	outs = []
	while ins.shape[0] > 0:
		if ins.shape[0] >= 2:
			print(ins[0])
			_, train_loss, encoded, _ = sess.run([optimizer, loss, encode, decode],
											  feed_dict={input1:ins[0].reshape(1,size//2),
														 input2:ins[1].reshape(1,size//2)})
			ins = ins[2:] # pop the top two
			outs.append(encoded)
		else: # If there's only one item left, add it to the output to use next round
			outs.append((ins[0]).reshape(1, size//2))
			break
	outs = np.array(outs)
	outs = outs.reshape(outs.shape[0], size//2)
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
