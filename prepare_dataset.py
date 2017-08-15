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
from IPython.display import display, Image
from scipy import ndimage



url = 'https://drive.google.com/uc?export=download&id=0B82rWclWut72X0tCS3J3Uy1FQ3M'
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


data_filename = maybe_download('meter_dataset.zip')

