from sklearn import svm
import numpy as np
from six.moves import cPickle as pickle
import cv2

train_labels = []
valid_labels = []
test_labels = []
train_data = []
valid_data=[]
test_data=[]

try:
	with open('meterData.pickle','rb') as f:
		save = pickle.load(f)
		train_data = save['train_dataset']
		train_labels = save['train_labels']
		test_data = save['test_dataset']
		test_labels = save['test_labels']
		del save
except Exception as e:
	print(e)


num_train = len(train_data)
train_data = np.reshape(train_data,(num_train,48*48))
num_test = len(test_data)
test_data = np.reshape(test_data,(num_test,48 * 48))
print(train_labels)
#num_test = len(test_data)
'''print(num_train)
print(train_data)
clf = svm.SVC()
clf.fit(train_data, train_labels)
predicition = clf.predict(test_data)
print(predicition)'''
model = cv2.ml.SVM_create()
model.setGamma(0.5)
model.setC(1)
model.setKernel(cv2.ml.SVM_RBF)
model.setType(cv2.ml.SVM_C_SVC)
model.train(train_data, cv2.ml.ROW_SAMPLE, train_labels)
print(model.predict(test_data)[1].ravel())
print(test_labels)