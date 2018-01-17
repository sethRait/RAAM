# Encodes and decodes pairs of word vectors as well as pairs of encoded pairs of word vectors.
# I.E this is a RAAM with a recursion depth of 1.

from __future__ import division
import tensorflow as tf
import numpy as np
import random
import nltk.data
import re

input_size = 300 
test_epochs = 2000
learning_rate = 0.0005
vector_filepath = "wiki-news-300d-1M.vec" # File of word vectors
sentence_filepath = "austen.txt"

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
def generate_samples():
	word_dict = parse_word_vecs()
	sentences = parse_sentences()
	sentence_dict = {}
	for sentence in sentences:
		res = get_vecs_from_sentence(sentence, word_dict)
		if res is not None:
			sentence_dict[sentence] = res
	return sentence_dict	

# Returns an np array of vectors representing the words of the given sentence
def get_vecs_from_sentence(sentence, word_dict):
	arr = [] 
	#for word in re.split('[.,;?:]', sentence): #sentence.split():
	for word in re.findall(r"[\w]+|[^\s\w]", sentence):
		cur = word_dict.get(word.lower())
		if cur is None:
			return None
		arr.append(cur)
	return np.array(arr)	
	
# Parses the file containing vector representations of words
def parse_word_vecs():
	print("Parsing word vectors")
	i = 1
	dictionary = {}
	with open(vector_filepath) as fp:
		next(fp) # skip header
		for line in fp:
			parsed = line.lower().split(' ', 1)
			dictionary[parsed[0]] = np.fromstring(parsed[1], dtype = float, count = input_size, sep = " ")
			i += 1
			if i % 100000 == 0:
				print(str((i/1000000)*100) + " percent complete")
	return dictionary

# Parses the file containing the training and testing sentences
def parse_sentences():
	print("Parsing input sentences")
	sentences = []
	with open(sentence_filepath) as fp:
		tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
		sentences = nltk.sent_tokenize(fp.read().decode('utf-8'))
	return sentences

inputs = tf.placeholder(tf.float32, [None, input_size]) # word vector

# layers
encoded = make_fc(inputs, input_size, "encoder", 1)
center = make_fc(encoded, input_size/2, "center", 1)
decoded = make_fc(center, input_size, "decoder", 1)

loss = tf.losses.mean_squared_error(labels=inputs, predictions=decoded)
train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss)
sess = tf.InteractiveSession()
tf.global_variables_initializer().run()

sentence_dict = generate_samples()
# Train on terminals
print("TRAINING TERMINALS")
for i in range(25000):
	_, train_loss, _, = sess.run([train_step, loss, decoded], feed_dict={inputs:sentence_dict.values()[i%len(sentence_dict)]})
	if i % 2500 == 0:
		print("epoch: " + str(i))
		print("loss: " + str(train_loss))

# Alternate training on terminals and intermediary values
#print("\nTRAINING ENCODED PAIRS")
#for i in range(25000):
#	np.random.shuffle(x)
#	np.random.shuffle(y)
#	encoded_pairs_A = sess.run(center, feed_dict={input1:x, input2:y})
#	np.random.shuffle(x)
#	np.random.shuffle(y)
#	encoded_pairs_B = sess.run(center, feed_dict={input1:x, input2:y})
#
#	_, intermediary_loss, _, = sess.run([train_step, loss, decoded], feed_dict={input1:encoded_pairs_A, input2:encoded_pairs_B})
#	_, terminal_loss, _, = sess.run([train_step, loss, decoded], feed_dict={input1:x, input2:y})
#	if i % 750 == 0:
#		print("epoch: " + str(i))
#		print("intermediary loss: " + str(intermediary_loss))
#		print("terminal loss: " + str(terminal_loss))
#		print("")

# Testing loop
divisor = 0
print("TESTING")
total_correct = 0
for i in range(test_epochs//100):
	test_loss, my_decoded, orig = sess.run([loss, decoded, inputs], feed_dict={inputs:sentence_dict.values()[i % len(sentence_dict)]})
	print(my_decoded)
	print(orig)
	if i % (test_epochs / 10) == 0:
		print(str((i / test_epochs) * 100) + " percent complete")

	# Reporting
        for original, decode in zip(orig, my_decoded):
            divisor += 1
            if np.allclose(original, decode, atol=0.2):
                total_correct += 1

percent_correct = (total_correct / divisor) * 100
print("total correct: " + str(total_correct))
print(str(percent_correct) + " percent correct")

