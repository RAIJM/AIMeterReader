import tflearn
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.estimator import regression
from tflearn.metrics import Accuracy


acc = Accuracy()
network = input_data(shape=[None,28,28,1])


network = conv_2d(network, 64, 3, strides=1, activation='relu', name= 'conv1_3_3_1')
network = max_pool_2d(network,2 ,strides=2)
network = conv_2d(network, 64, 3, strides=1, activation='relu', name= 'conv1_3_3_2')
network = conv_2d(network, 64, 3, strides=1, activation='relu', name='conv1_3_3_3')
network = max_pool_2d(network, 2, strides=2)

#Fully connected layer
network = fully_connected(network, 1024, activation='tanh')

network = dropout(network, 0.5)

network = fully_connected(network, 10, activation='softmax')

network = regression(network, optimizer='momentum',
					 loss='categorical_crossentropy',
					 learning_rate=0.001, metric=acc)	

model = tflearn.DNN(network, checkpoint_path='model-',best_checkpoint_path='best-model')


