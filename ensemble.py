import numpy as np
import gzip
import pickle
from functools import reduce
import os
from PIL import Image

from logistic_implementation import logistic_regression
from neural_mnist import neural_network
from rsv_implementation import random_forest
from svm_implementation import svm_implement

filename = 'mnist.pkl.gz'
f = gzip.open(filename, 'rb')
training_data, validation_data, test_data = pickle.load(f, encoding='latin1')
f.close()

mnist_train_images = np.array(training_data[0])
mnist_train_labels = np.array(training_data[1])
mnist_validation_images = np.array(validation_data[0])
mnist_validation_labels = np.array(validation_data[1])
mnist_test_images = np.array(test_data[0])
mnist_test_labels = np.array(test_data[1])

USPSMat  = []
USPSTar  = []
curPath  = 'USPSdata/Numerals'
savedImg = []

for j in range(0,10):
    curFolderPath = curPath + '/' + str(j)
    imgs =  os.listdir(curFolderPath)
    for img in imgs:
        curImg = curFolderPath + '/' + img
        if curImg[-3:] == 'png':
            img = Image.open(curImg,'r')
            img = img.resize((28, 28))
            savedImg = img
            imgdata = (255-np.array(img.getdata()))/255
            USPSMat.append(imgdata)
            USPSTar.append(j)

clf1 = logistic_regression()
print(clf1)
clf2 = neural_network()
print(clf2.shape)
clf3 = random_forest()
print(clf3.shape)
clf4 = svm_implement()
result = []

match = 0

for i in range(0,10000):
	majority = {}
	if clf1[i] in majority:
		majority[clf1[i]] += 1
	else:
		majority[clf1[i]] = 1

	if clf2[i] in majority:
		majority[clf2[i]] += 1
	else:
		majority[clf2[i]] = 1

	if clf3[i] in majority:
		majority[clf3[i]] += 1
	else:
		majority[clf3[i]] = 1
		
	if clf4[i] in majority:
		majority[clf4[i]] += 1
	else:
		majority[clf4[i]] = 1


	max_occurence  = reduce((lambda x,y: x if majority[x] > majority[y] else y), majority.keys())
	result.append(max_occurence)
	if max_occurence == mnist_test_labels[i]:
		match += 1

print("------------------Accuracy for ensembled model is:---------------------------")
print((match / 10000) *100)
