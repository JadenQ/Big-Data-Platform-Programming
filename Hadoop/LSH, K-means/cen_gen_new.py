import numpy as np
import random
import os
import sys
# set up the initial centroid random seed

part = sys.argv[1]
random_seed = 324
# random_seed = sys.argv[1]
random.seed(random_seed)
centroid_file = 'centroid_p'+str(part)+'.txt'
# centroid_file = 'centroid_'+str(random_seed)+'.txt'
file = "./data/train_part_"+str(part)+".txt"
# file = "./data/train.txt"

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

dice = [random.randint(1, 56000) for i in range(10)] 
sample = []
with open(file) as samples:
	line_count = 0
	for line in samples.readlines():
		line_count += 1
		if line_count in dice:
			line.strip()
			img, label = line.split('\t')
			img = [int(img_item) for img_item in img.split(' ')]
			sample.append(img)
		else: continue


# initialize centroid and create file if it doesn't exist at the first iteration
if not os.path.exists(centroid_file):
	# randomly selected centroid from data points
	random_img = sample
	# centroid from randomly generated values 
	# random_img = [[random.randint(0, 225) for j in range(28*28)] for i in range(10)]
	# intialize as 0
	counts = [0 for i in range(10)]
	save_centroid(centroid_file, random_img, counts)

assert (os.path.getsize(centroid_file) > 0), "Empty initial centroid file!"
