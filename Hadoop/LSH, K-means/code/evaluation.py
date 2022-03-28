# TODOs
# 1. count the number of images belong to this cluster
# 2. find the major label for this cluster
# 3. count the number of correctly clustered images
# 4. cal the calssification accuracy
# Files needed: a) K-means result, b)train.txt

import sys
import os
import numpy as np
import gc
import pandas as pd

random_seed = 1229
train_file = './train.txt'
cluster_file = './cluster_'+str(random_seed)+'.txt'

img_train = []
label_train = []
with open(train_file) as samples:
	for line in samples.readlines():
		line = line.strip()
		img, label = line.split('\t')
		label = int(label)
		# img = [int(img_item) for img_item in img.split(' ')]
		img_train.append(img)
		label_train.append(label)
# ground truth table

GT = pd.DataFrame({'label':label_train, 'img':img_train})

del img_train, label_train
gc.collect()

img_cluster = []
index_cluster = []
with open(cluster_file) as data:
	for line in data.readlines():
		line = line.strip()
		index, img = line.split('\t')
		index = int(index)
		# img = [int(img_item) for img_item in img.split(' ')]
		img_cluster.append(img)
		index_cluster.append(index)

cluster_result = pd.DataFrame({'index':index_cluster, 'img':img_cluster})

del img_cluster, index_cluster
gc.collect()

GT_cluster = GT.merge(cluster_result, how = 'left', on='img')

del GT, cluster_result
gc.collect()

print(GT_cluster)




