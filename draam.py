# Recursiely encodes and decodes pairs of word vectors
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
	learning_rate = 0.0001
	num_epochs = 150
	sen_len = 30

	print("Vector size: %d, with padding: %d" % (word_vector_size, padding))
	print("Learning rate: %f" % learning_rate)

	vectors = "data/test_vectors.vec" # File of word vectors
	corpus = "data/test_sentences.txt"

	original_sentence = tf.placeholder(tf.float32, [None, sen_len, word_vector_size + padding])
	# ingest = tf.expand_dims(original_sentence, axis=1)
	ingest = original_sentence

	# ingest
	depth_ingest = int(math.ceil(math.log(sen_len, 2)))
	new_sen_len = sen_len
        with tf.name_scope('encoder'):
            for i in range(depth_ingest):
                with tf.name_scope(str(i)):
                    R_array = []
                    # DONE
                    # if ingest.get_shape()[2] == 1:
                    #         break
                    print("shape of ingest is:")
                    print(ingest.get_shape())
                    
                    for j in range(0, new_sen_len, 2):
                        if j == new_sen_len-1:
                            R_array.append(ingest[:,j])
                        else:
                            temp = tf.concat([ingest[:,j], ingest[:,j+1]], axis=1)
                            print("shape of temp is:")
                            print(temp.get_shape())
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
                    # egest = R_array
                    egest = tf.stack(R_array, axis=1)
                    new_sen_len *=2
            egest = egest[:,0:sen_len,:]
	# original_sentence = tf.expand_dims(original_sentence, axis=1)
	loss = tf.losses.mean_squared_error(labels=original_sentence, predictions=egest)

	
	train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss)
	sess = tf.InteractiveSession()
	tf.global_variables_initializer().run()
	writer = tf.summary.FileWriter("checkpoints/", sess.graph)
	# import sys
	# sys.exit(1)
	print '*'*80
        for i in tf.trainable_variables():
            print(i)
	print '*'*80

	sentence_dict = generate_samples(vectors, corpus, word_vector_size, padding)
	cut = (4 * len(sentence_dict.values())) // 5
	training_data = sentence_dict.values()[0:cut]
	testing_data = sentence_dict.values()[cut:]
	# quit()
	train(sess, train_step, training_data, loss, input_size, num_epochs, ingest, egest)


#input.get_shape() == (1, vector_size + padding)
def build_encoder(inputs):
	size = inputs.shape[1].value
	#inputs = tf.expand_dims(inputs, axis=0)
	with tf.name_scope('encoder') as scope:
            encoded = make_fc(inputs, size, "E_first")
            encoded2 = make_fc(encoded, 3*size//4, "E_second")
	with tf.name_scope('center') as scope:
            center = make_fc(encoded2, size/2, "center")
	print("shape of center is:")
	print(center.get_shape())
	return center

def build_decoder(inputs):
	size = inputs.shape[1].value
	with tf.name_scope('decoder') as scope:
            decoded = make_fc(inputs, 3*size//2, "D_first")
            decoded2 = make_fc(decoded, 2*size, "D_second")
	return decoded2

def make_fc(input_tensor, output_size, name):
	input_size = input_tensor.get_shape().as_list()[1]
	# with tf.name_scope('FC') as scope:
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


#train(sess, train_step, training_data, loss, input_size, num_epochs, injest, egest)
def train(sess, optimizer, data, loss, size, num_epochs, injest, egest):
	print("Training on %d groups per epoch" % len(data))
	print("Training for %d epochs" % num_epochs)
	print("Shape is: ")
	# print(data.shape)
	quit()
	for i in range(num_epochs):
		for group in data: # A group is a collection of sentences of the same length
			np.random.shuffle(group)
			if group.ndim == 2: # if there is only one sentene in the group
				group = np.reshape(group, (1, group.shape[0], group.shape[1]))
		if i % 5 == 0:
			print("Epoch: " + str(i))
			print("Loss: " + str(train_loss))

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


