#!/usr/bin/env python3

import sys
import os
import random
import numpy as np

# set up the initial centroid random seed
# random_seed = 324
# random.seed(random_seed)
# centroid_file = 'centroid_'+str(random_seed)+'.txt'

# for question C
part = sys.argv[1]
random_seed = 324
random.seed(random_seed)
centroid_file = 'centroid_p'+str(part)+'.txt'

# calculate the distance between two points (represented with vector)
# a: centroid 1*784, b: sample 1*784, index: which centroid
def distance(centroid, sample):
	return np.linalg.norm(centroid - sample)

# save as centroid format file
# centroid_file: save to this file, centroids: 10 * 784, counts: 10 * 1
def save_centroid(centroid_file, centroids, counts):
	with open(centroid_file, mode = 'w', encoding = 'utf-8') as cent:
		for i in range(0, 10):
			cent_i = 'Centroid ' + str(i) + ':'
			img = [str(centroids[i][j]) for j in range(28*28)]
			img = ' '.join(img)
			cent_i = cent_i + img + ',' + str(counts[i]) + '\n'
			cent.write(cent_i)
	cent.close()


# # initialize centroid and create file if it doesn't exist at the first iteration
# if not os.path.exists(centroid_file):
# 	# randomly generated centroid
# 	random_img = [[random.randint(0, 225) for j in range(28*28)] for i in range(10)]
# 	# intialize as 0
# 	counts = [0 for i in range(10)]
# 	save_centroid(centroid_file, random_img, counts)

assert (os.path.getsize(centroid_file) > 0), "Empty initial centroid file!"

# initialize the centroids and counts
centroids = np.zeros((10, 28 * 28))
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

# map: only need to find out the closest centroid for each point

for line in sys.stdin:
	line = line.strip()
	img, label = line.split('\t')
	img = [int(img_item) for img_item in img.split(' ')]
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


