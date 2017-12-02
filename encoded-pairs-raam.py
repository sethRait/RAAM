# Encodes and decodes pairs of one-hot vectors as well as pairs of encoded pairs of one-hot vectors.
# I.E this is a RAAM with a recursion depth of 1.

from __future__ import division
import tensorflow as tf
import numpy as np
import random

input_size = 16 # 2 letters, input_size bits each, 1-hot
test_epochs = 2000
learning_rate = 0.002

def make_fc(input_tensor, output_size, name, mode):
	W = tf.get_variable(name + "weights",[input_tensor.get_shape().as_list()[1],output_size],tf.float32,
			                                      tf.random_normal_initializer(stddev=0.1))
	b = tf.Variable(tf.zeros([output_size]))
	if mode == 1:
		x = tf.nn.sigmoid(tf.matmul(input_tensor, W) + b)
	else:
		x = tf.nn.relu(tf.matmul(input_tensor, W) + b)
	return x

# Creates a list of all pairwise combinations of 'size' distinct one-hot vectors
def generate_samples(n):
	a = generate_one_hots(n)
	idx = np.array([np.argmax(i) for i in a])
	putval = (idx[:,None] == np.arange(n)).astype(int)
	out = np.zeros((n,n,2,n),dtype=int)
	out[:,:,0,:] = putval[:,None,:]
	out[:,:,1,:] = putval
	out.shape = (n**2,2,-1)
	return out	

# Creates a list of 'size' one-hot vectors
def generate_one_hots(size):
	out = (np.random.choice(size, size, replace=0)[:,None] == range(size)).astype(int)
	return list(map(list, out))

def chunks(l, n):
	# split l into n-sized chunks
	for i in range(0, len(l), n):
		yield l[i:i + n]


input1 = tf.placeholder(tf.float32, [None, input_size/2]) # first letter
input2 = tf.placeholder(tf.float32, [None, input_size/2]) # second letter
input_full = tf.concat([input1, input2], 1) # not 2None x 6

# layers
encoded = make_fc(input_full, input_size, "encoder", 1)
encoded2 = make_fc(encoded, 3*input_size/4, "second_hidden", 2)
encoded3 = make_fc(encoded2, input_size/2, "third_hidden", 2)
decoded1 = make_fc(encoded3, 3*input_size/4, "decoder", 2)
decoded2 = make_fc(decoded1, input_size, "second_decoder", 1)

loss = tf.losses.mean_squared_error(labels=input_full, predictions=decoded2)
train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss)
sess = tf.InteractiveSession()
tf.global_variables_initializer().run()

training_set = generate_samples(input_size//2)

x = np.array([training_set[j][0] for j in range(training_set.shape[0])])
y = np.array([training_set[j][1] for j in range(training_set.shape[0])])

# Train on terminals
print("TRAINING TERMINALS")
for i in range(10000):
	np.random.shuffle(x)
	np.random.shuffle(y)

	_, train_loss, _, = sess.run([train_step, loss, decoded2], feed_dict={input1:x, input2:y})
	if i % 500 == 0:
		print("epoch: " + str(i))
		print("loss: " + str(train_loss))

# Alternate training on terminals and intermediary values
print("\nTRAINING ENCODED PAIRS")
for i in range(12000):
	np.random.shuffle(x)
	np.random.shuffle(y)
	encoded_pairs_A = sess.run(encoded3, feed_dict={input1:x, input2:y})
	np.random.shuffle(x)
	np.random.shuffle(y)
	encoded_pairs_B = sess.run(encoded3, feed_dict={input1:x, input2:y})

	_, intermediary_loss, _, = sess.run([train_step, loss, decoded2], feed_dict={input1:encoded_pairs_A, input2:encoded_pairs_B})
#	_, terminal_loss, _, = sess.run([train_step, loss, decoded2], feed_dict={input1:x, input2:y})
	if i % 500 == 0:
		print("epoch: " + str(i))
		print("intermediary loss: " + str(intermediary_loss))
#		print("terminal loss: " + str(terminal_loss))
		print("")

# Testing loop
divisor = 0
print("TESTING")
total_correct = 0
for i in range(test_epochs):
	x = np.array([training_set[j][0] for j in range (training_set.shape[0])])
	y = np.array([training_set[j][1] for j in range (training_set.shape[0])])
	np.random.shuffle(x)
	np.random.shuffle(y)
	
	test_loss, my_decoded, orig = sess.run([loss, decoded2, input_full], feed_dict={input1:x, input2:y})

	if i % (test_epochs / 10) == 0:
		print(str((i / test_epochs) * 100) + " percent complete")

	# Reporting
	decode_iter = np.nditer(my_decoded)
	orig_iter = np.nditer(orig)
	iter_count = 0
	while not orig_iter.finished:
		divisor += 1
		if np.allclose(orig_iter[0], decode_iter[0], atol=0.2):
			total_correct += 1
		orig_iter.iternext()
		decode_iter.iternext()
percent_correct = (total_correct / divisor) * 100
print("total correct: " + str(total_correct))
print(str(percent_correct) + " percent correct")
