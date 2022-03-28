#!/usr/bin/env python3

import sys
import os
import numpy as np

cur_s_index = None
count = 1
centroid = np.zeros(28*28)

# save as centroid format file
# centroid: 1 * 784, count: 1*1
def format_centroid(cur_s_index, centroid, count):
	assert (len(centroid) == 784), "Wrong dimension: "+str(len(centroid))+ " It should be (784,)"
	new_centroid = [str(centroid[i] / count) for i in range(28*28)]
	new_centroid = ' '.join(new_centroid)
	# save the count number
	count = str(count)
	rst = "Centroid " + str(cur_s_index) + ":" + new_centroid + "," + count
	return rst

for line in sys.stdin:
	line = line.strip()
	s_index, img = line.split('\t')
	img = img.split(' ')
	s_index = int(s_index)
	# deal with only one index at a time

	if(cur_s_index == s_index):
		count = count + 1
		centroid = [centroid[i] + int(img[i]) for i in range(28*28)]
	else:
		# cal the average, get new centroids and print
		if cur_s_index is not None:
			print(format_centroid(cur_s_index, centroid, count))
		# renew a centroid
		centroid = np.zeros(28*28)
		count = 1
	# renew the index
	cur_s_index = s_index

# last line: end of iteration
if count > 0:
	# if the last few lines has the same index with the above line
	print(format_centroid(cur_s_index, centroid, count))
else:
	# if the last line has a different index with the above lines
	img = [int(img[i]) for i in range(28*28)]
	print(format_centroid(cur_s_index, img, count))