import pickle
import gzip
from PIL import Image
import os
import numpy as np
from sklearn import cross_validation
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import pandas as pd
import math

def random_forest():
	filename = 'mnist.pkl.gz'
	f = gzip.open(filename, 'rb')
	training_data, validation_data, test_data = pickle.load(f, encoding='latin1')
	f.close()

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

	#for mnist dataset
	rf_class = RandomForestClassifier()
	rf_class.fit(training_data[0], training_data[1])
	y_pred_rf = rf_class.predict(test_data[0])
	acc_rf = accuracy_score(test_data[1], y_pred_rf)

	results = confusion_matrix(test_data[1], y_pred_rf)
	print ('Confusion matrix for MNIST dataset :')
	print(results)
	print("random forest accuracy for MNIST dataset: " + str(acc_rf))
	print ('Report : ')
	print (classification_report(test_data[1], y_pred_rf))

	#for usps dataset
	y_pred_rf_1 = rf_class.predict(USPSMat)
	acc_rf = accuracy_score(USPSTar, y_pred_rf_1)
	results = confusion_matrix(USPSTar, y_pred_rf_1)
	print ('Confusion matrix for USPS dataset:')
	print(results)
	print("random forest accuracy for USPS dataset: " + str(acc_rf))
	print ('Report : ')
	print (classification_report(USPSTar, y_pred_rf_1))
	return y_pred_rf
if __name__=='__main__':
	pred_rfc = random_forest()


