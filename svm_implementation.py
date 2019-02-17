from PIL import Image
import os
import numpy as np
import pickle
import gzip
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

def svm_implement():
    filename = 'mnist.pkl.gz'
    f = gzip.open(filename, 'rb')
    training_data, validation_data, test_data = pickle.load(f, encoding='latin1')
    f.close()
    mnist_train_images = np.array(training_data[0])
    #mnist_train_images = mnist_train_images[0:10000]
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
    print("Execution in progress!!!")

    #train the model on MNIST training data
    clf = svm.SVC(kernel='rbf', C= 10.0, gamma= 0.1)
    mnist_train_images = np.asmatrix(mnist_train_images[:(50000*784)]).reshape(50000, 784)

    clf.fit(mnist_train_images, mnist_train_labels[:50000])
    #testing the data
    mnist_test_images = np.asmatrix(mnist_test_images).reshape(10000, 784)
    clf_predictions = clf.predict(mnist_test_images)
    result = clf.score(mnist_test_images, mnist_test_labels)
    print("Accuracy for MNIST Dataset: ",result)
    results = confusion_matrix(mnist_test_labels, clf_predictions)
    print ('Confusion matrix for MNIST dataset :')
    print(results)
    print ('Report : ')
    print (classification_report(mnist_test_labels, clf_predictions))

    USPSMat = np.array(USPSMat)
    clf_predictions = clf.predict(USPSMat)
    result = clf.score(USPSMat, USPSTar)
    print("Accuracy for USPS Dataset: ",result)
    return clf_predictions

if __name__=='__main__':
    pred_svm = svm_implement()