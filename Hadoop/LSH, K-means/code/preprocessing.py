import sys
import os
import numpy as np

# Reference: StackOverflow by Punnerud
def preprocess(img_f, label_f, train = True):
	image_size = 28
	if train == True:
		img_num = 60000
		file_name = 'train'
	else: 
		img_num = 10000
		file_name = 'test'
	img_f = open(img_f, 'rb')
	label_f = open(label_f, 'rb')

	# bypass the header info
	img_f.read(16)
	label_f.read(8)
	# load the data
	img_buf = img_f.read(image_size * image_size * img_num)
	label_buf = label_f.read(img_num)
	# reformat and reshape data
	img_data = np.frombuffer(img_buf, dtype=np.uint8).astype(np.int)
	label_data = np.frombuffer(label_buf, dtype=np.uint8).astype(np.int)
	img_data = img_data.reshape(img_num, image_size*image_size)
	label_data = label_data.reshape(img_num,1)
	# integrate the result
	output_dir = './' + file_name + '.txt'
	with open(output_dir, mode = 'w', encoding = 'utf-8') as pre:
		for i in range(img_num):
			img = ' '.join(str(d) for d in img_data[i])
			label = str(label_data[i][0])
			pre.write(img+'\t'+label+'\n')
	return 'Finished Converting '+ file_name + ' files.'



preprocess('train-images.idx3-ubyte', 'train-labels.idx1-ubyte', True)
preprocess('t10k-images.idx3-ubyte', 't10k-labels.idx1-ubyte', False)
