#!/usr/bin/env python3

import sys
import os
import random
import numpy as np

# set up the initial centroid random seed
random_seed = sys.argv[1]
random.seed(random_seed)
centroid_file = 'centroid_'+str(random_seed)+'_new.txt'
dim = 20
# for question C
# part = sys.argv[1]
# random_seed = 324
# random.seed(random_seed)
# centroid_file = 'centroid_p'+str(part)+'_new.txt'



# calculate the distance between two points (represented with vector)
# a: centroid 1*784, b: sample 1*784, index: which centroid
def distance(centroid, sample):
	return np.linalg.norm(centroid - sample)

# initialize the centroids and counts
centroids = np.zeros((10, dim))
count = np.zeros((10, 1))
# read the centorids file
with open(centroid_file) as cents:
	for line in cents.readlines():
		line = line.strip()
		line, count = line.split(',')
		header, cent = line.split(':')
		header, index = header.split(' ')
		cent = cent.split(' ')
		centroids[int(index)] = cent
assert (os.path.getsize(centroid_file) > 0), "Empty initial centroid file!"



# map: only need to find out the closest centroid for each point

for line in sys.stdin:
	line = line.strip()
	img, label = line.split('\t')
	img = [float(img_item) for img_item in img.split(' ')]
	# label = int(label)

	# shortest distance, start with index 0
	# d: distance, s_d: shortest distance, s_index: shortest centroid index
	s_d = distance(centroids[0], img)
	s_index = 0
	for i in range(1, 10):
		d = distance(centroids[i], img)
		if d < s_d:
			s_d = d
			s_index = i
		else: continue

	# format intermediate result as string
	img = ' '.join(str(k) for k in img)
	print('%s\t%s' % (str(s_index), img))




