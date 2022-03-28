#!/usr/bin/env python


import sys

# input comes from STDIN
dataset = []
for line in sys.stdin:
	line = line.strip()
	follower, followee_list = line.split('\t')
	followee_list = followee_list[1:-1].split(',')
	dataset.append([follower, followee_list])

for i in range(len(dataset)):
	for j in range(i+1, len(dataset)):
		if len(dataset[i][1])==0 or len(dataset[j][1])==0:
			continue
		else:
			common_followee = list(set(dataset[i][1]) & set(dataset[j][1]))
			num_common_followee = len(common_followee)
			total_followee = len(dataset[i][1]) + len(dataset[j][1]) - num_common_followee
			print("%s\t%s\t%s" % (dataset[i][0], num_common_followee, dataset[j][0])

