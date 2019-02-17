import pickle
import gzip

from PIL import Image
import os
import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

def neural_network():
	# Import MNIST data
	from tensorflow.examples.tutorials.mnist import input_data
	mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)



	USPSMat  = []
	USPSTar  = []
	curPath  = 'USPSdata/Numerals'

	for j in range(0,10):
	    curFolderPath = curPath + '/' + str(j)
	    imgs =  os.listdir(curFolderPath)
	    for img in imgs:
	        curImg = curFolderPath + '/' + img
	        if curImg[-3:] == 'png':
	            img = Image.open(curImg,'r')
	            img = img.resize((28, 28))
	            imgdata = list(img.getdata())
	            
	            USPSMat.append(imgdata)
	            USPSTar.append(j)
	#Storing image and labels in arrays to be used for training   
	USPS_img_array = np.array(USPSMat)
	USPS_img_array = np.subtract(255, USPS_img_array)
	USPS_label_array = np.array(USPSTar)
	print(USPS_label_array.shape)
	nb_classes = 10
	targets = np.array(USPS_label_array).reshape(-1)
	aa = np.eye(nb_classes)[targets]
	USPS_label_array = np.array(aa, dtype=np.int32)
	#print(USPS_label_array)
	USPS_img_array = np.float_(np.array(USPS_img_array))
	for z in range(len(USPS_img_array)):
	    USPS_img_array[z] /= 255.0 
	
	# Parameters
	learning_rate = 0.001
	training_epochs = 15
	batch_size = 100
	display_step = 1

	# Network Parameters
	n_hidden_1 = 256 # 1st layer number of neurons
	#n_hidden_2 = 256 #2nd layer number of neurons
	n_input = 784 # MNIST data input (img shape: 28*28)
	n_classes = 10 # MNIST total classes (0-9 digits)

	# tf Graph input
	X = tf.placeholder("float", [None, n_input])
	Y = tf.placeholder("float", [None, n_classes])

	# Store layers weight & bias
	weights = {
	    'h1': tf.Variable(tf.random_normal([n_input, n_hidden_1])),
	    #'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
	    'out': tf.Variable(tf.random_normal([n_hidden_1, n_classes]))
	    
	}
	biases = {
	    'b1': tf.Variable(tf.random_normal([n_hidden_1])),
	    #'b2': tf.Variable(tf.random_normal([n_hidden_2])),
	    'out': tf.Variable(tf.random_normal([n_classes]))
	}									


	# Create model
	def multilayer_perceptron(x):
	    # Hidden fully connected layer with 256 neurons
	    layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
	    #layer_2 = tf.add(tf.matmul(x, weights['h2']), biases['b2'])
	    out_layer = tf.matmul(layer_1, weights['out']) + biases['out']
	    return out_layer

	# Construct model
	logits = multilayer_perceptron(X)

	# Define loss and optimizer
	loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=Y))
	optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
	train_op = optimizer.minimize(loss_op)

	# Initializing the variables
	init = tf.global_variables_initializer()

	with tf.Session() as sess:
	    sess.run(init)
	# Training cycle
	    for epoch in range(training_epochs):
	        avg_cost = 0.
	        #total_batch = int(len(training_data)/batch_size)
	        total_batch = int(mnist.train.num_examples/batch_size)
	        # Loop over all batches
	        for i in range(total_batch):
	            #batch_x, batch_y = training_data.next_batch(batch_size)
	            batch_x, batch_y = mnist.train.next_batch(batch_size)
	            # Run optimization op (backprop) and cost op (to get loss value)
	            _, c = sess.run([train_op, loss_op], feed_dict={X: batch_x,
	                                                            Y: batch_y})
	            # Compute average loss
	            avg_cost += c / total_batch

	        # Display logs per epoch step
	        if epoch % display_step == 0:
	            print("Epoch:", '%04d' % (epoch+1), "cost={:.9f}".format(avg_cost))
	    print("Optimization Finished!")

	# Test model
	    pred = tf.nn.softmax(logits)  # Apply softmax to logits
	    correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(Y, 1))
	# Calculate accuracy
	    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
	    print("Accuracy of MNIST Train Data:", accuracy.eval({X: mnist.train.images, Y: mnist.train.labels}))
	    print("Accuracy of MNIST Validation Data:", accuracy.eval({X: mnist.validation.images, Y: mnist.validation.labels}))
	    print("Accuracy of MNIST Test Data:", accuracy.eval({X: mnist.test.images, Y: mnist.test.labels}))
	    prediction = tf.argmax(pred,1)
	    pred_matrix = prediction.eval(feed_dict={X: mnist.test.images}, session=sess)
	    print(pred_matrix)
	    test_labels = np.argmax(mnist.test.labels, axis=1)
	    print(test_labels)
	    results = confusion_matrix(test_labels,pred_matrix)
	    print ('Confusion matrix :')
	    print (results)
	    print ('Report : ')
	    print (classification_report(test_labels,pred_matrix))
	    print("Accuracy for USPS Numerals:", accuracy.eval({X: USPS_img_array, Y: USPS_label_array}))
	    prediction = tf.argmax(pred,1)
	    pred_usps = prediction.eval(feed_dict={X: USPS_img_array}, session=sess)
	    print(pred_matrix)
	return(pred_matrix)
if __name__=='__main__':
	pred_mlp = neural_network()
