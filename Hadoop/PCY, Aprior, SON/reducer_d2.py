#!/usr/bin/env python

import sys

# Aggregate the output of the Job2 mapper
current_pair = None # a itemset checker used to aggregate the same candidate itemset from different chuncks
overall_count = 0 # the frequency of each candidate itemsets across the overall input file
threshold = 0.005 # support threshold
basket_num = 0 # num of baskets

basket_num = 4340061

s = threshold * basket_num
# get the pair - count
for line in sys.stdin:
	line = line.strip()
	pair, count = line.split('\t')
	count = int(count)
	if pair == current_pair:
		 overall_count += count # the same itemset, agg_sum
	else:
		if current_pair:
			if overall_count >= s: # if the iteration is not over AND s is more than the threshold			 
				print("%s\t%s" % (current_pair, overall_count))
		overall_count = count
		current_pair = pair # update the itemset checker


