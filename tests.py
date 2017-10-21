import unittest
from prepare_dataset import *
import numpy as np
from train import *


class TestCaseClass(unittest.TestCase):

	def test_vector_to_one_hot(self):
   		self.assertTrue(np.array_equal(vector_to_one_hot([0,1,1,0]),[[1,0],[0,1],[0,1],[1,0]]))

   	def test_randomize(self):
   		dataset = np.ndarray((5,2,2),dtype=np.int32)
   		labels = np.ndarray(5, dtype=np.int32)
   		dataset[0] = [[1,0],[1,1]]
   		dataset[1] = [[2,0], [5,1]]
   		dataset[2] = [[1,2] , [5,6]]
   		dataset[3] = [[7,8] , [1,4]]
   		dataset[4] = [[0,2] , [8,9]]
   		labels[0:5] = [5,6,4,2,1]
   		new_data,new_labels = randomize(dataset,labels)
   		self.assertTrue(not(np.array_equal(labels,new_labels)) and not(np.array_equal(dataset,new_data)))

   	def test_make_arrays(self):
   		dataset,labels = make_arrays(10,50)
   		self.assertTrue(dataset != None and labels != None)


    

   	


if __name__ == '__main__':
    unittest.main()

