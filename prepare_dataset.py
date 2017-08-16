from __future__ import print_function
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import tarfile
import IPython.display
from six.moves.urllib.request import urlretrieve
from six.moves import cPickle as pickle
from six.moves import range
#from IPython.display import display, Image
from PIL import Image
from scipy import ndimage
import gzip
import tensorflow as tf


url = 'https://drive.google.com/uc?export=download&id=0B82rWclWut72cENaN0U1am05S1k'
last_percent_reported = None
data_root = '.'
def download_progress_hook(count, blockSize, totalSize):

	global last_percent_reported
	percent = int(count * blockSize * 100 / totalSize)

	if last_percent_reported != percent:
		if percent % 5 == 0:
			sys.stdout.write("%s%%" % percent)
			sys.stdout.flush()
		else:
			sys.stdout.write(".")
			sys.stdout.flush()

		last_percent_reported = percent

def maybe_download(filename,force=False):
	dest_filename = os.path.join(data_root, filename)
	if force or not os.path.exists(dest_filename):
		print('Attempting to download:',filename)
		filename = urlretrieve(url, dest_filename, reporthook=download_progress_hook)
		print('\nDownload Complete')
	statinfo = os.stat(dest_filename)
	if statinfo.st_size >=0:
		print("File downloaded")
	else:
		raise Exception('Failed to download')
	return dest_filename


data_filename = maybe_download('meter_dataset.tar.gz')

num_classes = 2

def mabye_extract(filename, force=False):
	root = os.path.splitext(os.path.splitext(filename)[0])[0]
	if os.path.isdir(root) and not force:
		print("%s already present - Skipping extraction of %s. " % (root,filename))
	else:
		print("Extracting data for %s This may take a while. Please wait. " %root)
		tar = tarfile.open(filename)
		sys.stdout.flush()
		tar.extractall(data_root)
		tar.close()
	data_folders = [
		os.path.join(root, d) for d in sorted(os.listdir(root))
		if os.path.isdir(os.path.join(root, d))]
	if len(data_folders) != num_classes:
		raise Exception(
			'Expected %d folders, one per class. Found %d instead.' %(num_classes, len(data_folders)))
	print(data_folders)
	return data_folders


data_folders = mabye_extract(data_filename)


image_size = 96

def load_class(folder):
	image_files = os.listdir(folder)
	dataset = np.ndarray(shape=(len(image_files),image_size,image_size),dtype=np.float32)

	print(folder)
	num_images = 0
	for image in image_files:
		image_file = os.path.join(folder,image)
		try:
			image_data = Image.open(image_file) #opens image
			image_data = image_data.convert('L') #converts to grayscale
			image_data = image_data.resize((image_size,image_size)) #resize image to 96x96
			imarray = np.array(image_data,dtype=np.float32)
			dataset[num_images,:,:] = imarray
			num_images += 1
		except IOError as e:
			print('Could not read:',image_file,':',e)

	dataset = dataset[0:num_images,:,:]
	print('Shape',dataset.shape)
	return dataset

def mabye_pickle(folders,force=False):
	dataset_names = []
	for folder in folders:
		filename = folder + '.pickle'
		dataset_names.append(filename)
		if os.path.exists(filename) and not force:
			print('%s already present. Skipping pickling' %filename)
		else:
			print('Pickling %s' %filename)
			dataset = load_class(folder)
			try:
				with open(filename,'wb') as f:
					pickle.dump(dataset, f, pickle.HIGHEST_PROTOCOL)
			except:
				print('Unable to pickle file')
	return dataset_names





#dataset = load_class(data_folders[0])
#plt.imshow(dataset[0])
#plt.show()
dataset_names = mabye_pickle(data_folders)

train_size = 10
valid_size = 5
test_size = 5

def make_arrays(num_rows,image_size):
	if num_rows:
		dataset = np.ndarray((num_rows,image_size,image_size),dtype=np.float32)
		labels = np.ndarray(num_rows, dtype=np.int32)
	else:
		dataset,labels = None,None
	return dataset,labels

def merge_datasets(pickle_files,train_size,valid_size,test_size):
	num_classes = len(pickle_files)

	train_dataset,train_labels = make_arrays(num_classes * train_size,image_size)
	valid_dataset,valid_labels = make_arrays(num_classes * valid_size,image_size)
	test_dataset,test_labels = make_arrays(num_classes * test_size,image_size)

	train_pos,valid_pos,test_pos = 0,0,0
	for label, pickle_file in enumerate(pickle_files):
		try:
			with open(pickle_file, 'rb') as f:
				class_set = pickle.load(f)
				np.random.shuffle(class_set)
				train_dataset[train_pos:train_pos+train_size,:,:] = class_set[:train_size,:,:]
				valid_dataset[valid_pos:valid_pos+valid_size,:,:] = class_set[train_size:train_size+valid_size,:,:]
				test_dataset[test_pos:test_pos+test_size,:,:] = class_set[train_size+valid_size:train_size+valid_size+test_size,:,:]
				train_labels[train_pos:train_pos+train_size] = label
				valid_labels[valid_pos:valid_pos+valid_size] = label
				test_labels[test_pos:test_pos+test_size] = label

				train_pos += train_size
				valid_pos += valid_size
				test_pos += test_size
		except Exception as e:
			print('Unable to process data:',e)
	return train_dataset,train_labels,valid_dataset,valid_labels,test_dataset,test_labels


train_dataset,train_labels,valid_dataset,valid_labels,test_dataset,test_labels = merge_datasets(dataset_names,train_size,valid_size,test_size)

def randomize(dataset,labels):
	permutation = np.random.permutation(labels.shape[0])
	shuffled_dataset = dataset[permutation,:,:]
	shuffled_labels = labels[permutation]
	return shuffled_dataset, shuffled_labels

train_dataset,train_labels = randomize(train_dataset,train_labels)
valid_dataset,valid_size = randomize(valid_dataset,valid_labels)
test_dataset,test_size = randomize(test_dataset,test_labels)

print('Training:', train_dataset.shape,train_labels.shape)
print('Validation', valid_dataset.shape,valid_labels.shape)
print('Testing', test_dataset.shape,test_labels.shape)
print('Example',train_labels)
#res = tf.one_hot(indices=train_labels,depth=2)
res = (np.arange(2) == train_labels[:,None]).astype(np.int32)
print('One Hot',res)
pickle_file = 'meterData.pickle'
try:
	f = open(pickle_file, 'wb')
	save = {
	     'train_dataset': train_dataset,
	     'train_labels': train_labels,
	     'valid_dataset': valid_dataset,
	     'valid_labels': valid_labels,
	     'test_dataset': test_dataset,
	     'test_labels': test_labels,
	}
	pickle.dump(save, f, pickle.HIGHEST_PROTOCOL)
	f.close()
except Exception as e:
	print('Unable to save data to',pickle_file,':',e)

statinfo = os.stat(pickle_file)
print('Compressed pcikle size:', statinfo.st_size)







