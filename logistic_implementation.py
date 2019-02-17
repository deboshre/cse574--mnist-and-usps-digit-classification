#Reference: https://www.tensorflow.org/tutorials/
#Reference: https://www.kdnuggets.com/2016/07/softmax-regression-related-logistic-regression.html
import tensorflow as tf
import numpy as np

import pickle
import gzip
import os
from PIL import Image
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

from tensorflow.examples.tutorials.mnist import input_data
def logistic_regression():
	mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

	#Extract feature values from MNIST dataset
	mnist_train_labels =  np.array(mnist.train.labels)
	mnist_train_images =  np.array(mnist.train.images)
	mnist_valid_images =  np.array(mnist.validation.images)
	mnist_valid_labels =  np.array(mnist.validation.labels)
	mnist_test_labels =  np.array(mnist.test.labels)
	mnist_test_images =  np.array(mnist.test.images)

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
	#print(USPS_label_array.shape)
	nb_classes = 10
	targets = np.array(USPS_label_array).reshape(-1)
	aa = np.eye(nb_classes)[targets]
	USPS_label_array = np.array(aa, dtype=np.int32)
	#print(USPS_label_array)


	USPS_img_array = np.float_(np.array(USPS_img_array))
	for z in range(len(USPS_img_array)):
	    USPS_img_array[z] /= 255.0 

	# Parameters
	learning_rate = 0.01
	training_epochs = 25
	batch_size = 100
	display_step = 1

	# tf Graph Input
	x = tf.placeholder(tf.float32, [None, 784]) # mnist data image of shape 28*28=784
	y = tf.placeholder(tf.float32, [None, 10]) # 0-9 digits recognition => 10 classes labels

	# Set model weights
	W = tf.Variable(tf.zeros([784, 10]))
	b = tf.Variable(tf.zeros([10]))

	# Construct model
	pred = tf.nn.softmax(tf.matmul(x, W) + b) # Softmax predicted values
	#print (pred.shape)
	# Minimize error using cross entropy
	cost = tf.reduce_mean(-tf.reduce_sum(y*tf.log(pred), reduction_indices=1))
	# Gradient Descent
	optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

	# Initialize the variables (i.e. assign their default value)
	init = tf.global_variables_initializer()

	# Start training
	with tf.Session() as sess:

	    # Run the initializer
	    sess.run(init)

	    # Training cycle
	    for epoch in range(training_epochs):
	        avg_cost = 0.
	        total_batch = int(mnist.train.num_examples/batch_size)
	        # Loop over all batches
	        for i in range(total_batch):
	            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
	            # Run optimization op (backprop) and cost op (to get loss value)
	            _, c = sess.run([optimizer, cost], feed_dict={x: batch_xs,
	                                                          y: batch_ys})
	            # Compute average loss
	            avg_cost += c / total_batch
	        # Display logs per epoch step
	        if (epoch+1) % display_step == 0:
	            print("Epoch:", '%04d' % (epoch+1), "cost=", "{:.9f}".format(avg_cost))

	    print("Optimization Finished!")
	    # Test model
	    correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
	    
	    # Calculate accuracy
	    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
	    print("Accuracy of MNIST Training Set:", accuracy.eval({x: mnist_train_images, y: mnist_train_labels}))
	    print("Accuracy of MNIST Validation Set:", accuracy.eval({x: mnist_valid_images, y: mnist_valid_labels}))
	    print("Accuracy of MNIST Testing Set:", accuracy.eval({x: mnist_test_images, y: mnist_test_labels}))
	    prediction = tf.argmax(pred,1)
	    pred_matrix = prediction.eval(feed_dict={x: mnist.test.images}, session=sess)
	    print(pred_matrix)
	    test_labels = np.argmax(mnist_test_labels, axis=1)
	    print(test_labels)
	    results = confusion_matrix(test_labels,pred_matrix)
	    print ('Confusion matrix :')
	    print (results)
	    print ('Report : ')
	    print (classification_report(test_labels,pred_matrix))
	    print("Accuracy of USPS Numeral Set:", accuracy.eval({x: USPS_img_array , y: np.float_(USPS_label_array)}))
	    prediction = tf.argmax(pred,1)
	    pred_usps = prediction.eval(feed_dict={x: USPS_img_array}, session=sess)
	    print(pred_usps)
	return (pred_usps)

if __name__=='__main__':
	pred_logistic = logistic_regression()