import glob
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os
from pynput import keyboard






'''sorted_names = []
if(os.path.exists('image_names.txt')):
	with open('image_names.txt','r') as f:
		lines = f.read().split("\n")
		#print(lines)
		sorted_names = [(l.strip().split()[0]).split("/")[-1] for l in lines if len(l)>0]
		#print(sorted_names)

	f.close()

print(len(sorted_names))
def get_filenames(folder):
	files = []
	print(len(os.listdir(folder)))
	for filename in os.listdir(folder):
		if (filename.endswith('.jpg') or filename.endswith('.png')) and filename.strip() not in sorted_names:
			files.append(os.path.join(folder,filename))
	

	#print(files)
	return files


folder = '/home/stone/Documents/MeterImages'
files = get_filenames(folder)
print(len(files))

with open('image_names.txt','a+') as f:
	for file in files:
		img = mpimg.imread(file)
		imgplot = plt.imshow(img)
		plt.draw()
		plt.pause(0.0005)
		i = input('Enter folder number(1=analog,2=digital,3=other,4=exit)\n')
		if(i==1):
			folder_name = os.path.join(folder,'analogue')
		elif(i==2):
			folder_name = os.path.join(folder,'digital')
		elif(i==4):
			break
		else:
			folder_name = os.path.join(folder,'other')
		file_name = file.split('/')[-1]
		f.write(file+" "+ str(i)+"\n")

	f.close()
		#os.rename(file,os.path.join(folder_name,file_name))


        #return False # remove this if want more keys'''

def move_files_to_folder():
	if(os.path.exists('image_names.txt')):
		with open('image_names.txt','r') as f:
			lines = f.read().split("\n")
			for line in lines:
				line_sep = line.split()
				if(len(line_sep) > 0):
					folder_number = int(line_sep[1])
					print(folder_number)
					file = line_sep[0]
					if(os.path.exists(file)):
						dir_sep = file.split("/")
						file_name = dir_sep[-1]
						folder = "/".join(dir_sep[:-1])
						folder_name=""
						if(folder_number == 1):
							folder_name = os.path.join(folder,'analogue')
						elif(folder_number==2):
							folder_name = os.path.join(folder,'digital')
						elif(folder_number==3):
							folder_name = os.path.join(folder,'other')
						print(file)
						print(folder_name)
						print(file_name)

						os.rename(file,os.path.join(folder_name,file_name))
			f.close()

if __name__ == "__main__":
    move_files_to_folder()




