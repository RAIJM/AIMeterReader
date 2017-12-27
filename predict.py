import deepneuralnet as net
import random
import tflearn.datasets.mnist as mnist
from skimage import io


model = net.model
path_to_model = 'final-model.tflearn'

_,_,testX, _ = mnist.load_data(one_hot=True)
model.load(path_to_model)

rand_index = random.randint(0,len(testX) -1)

x = testX[rand_index].reshape((28,28,1))


result = model.predict([x])[0]

prediction = result.tolist().index(max(result))

print("Predicition",prediction)

io.imsave('testimage.jpg', x.reshape(28,28))