#!/usr/bin/env python

import sys

# Aggregate the output of the Job2 mapper
current_triplet = None # a itemset checker used to aggregate the same candidate itemset from different chuncks
overall_count = 0 # the frequency of each candidate itemsets across the overall input file
threshold = 0.0025 # support threshold

basket_num = 4340061


s = threshold * basket_num
# get the triplet - count
for line in sys.stdin:
	line = line.strip()
	triplet, count = line.split('\t')
	count = int(count)
	if triplet == current_triplet:
		 overall_count += count # the same itemset, agg_sum
	else:
		if current_triplet:
			if overall_count >= s: # if the iteration is not over AND s is more than the threshold			 
				print("%s\t%s" % (current_triplet, overall_count))
		overall_count = count
		current_triplet = triplet # update the itemset checker
