#!/usr/bin/env python
import sys

# Mapper for Job2
# for all the candidate itemsets produced by Job1, count the frequency in local chunk

# stdin: the output result of last MR work
fname = 'freqPair_c.txt'
freq_pairs = []
freq_triplets = {} # key: item triplet, value: frequency
threshold = 0.0025 # support threshold
basket_num = 0

# unify the triplets
def orderTriplet(a, b, c):
	ordered_list = [a, b, c]
	ordered_list.sort()
	return ' '.join(ordered_list)

# unify the triplets
# align with the process of Task B
def orderPair(a,b):
	if (a <= b):
		pair = a + " " + b
	else: pair = b + " " + a
	return pair

with open(fname) as f:
	for line in f.readlines():
		pair, count = line.strip().split('\t')
		freq_pairs.append(pair)


for line in sys.stdin:
	line = line.strip()
	words = line.split(' ')
	words = list(set(words)) # remove the duplicates in one basket
	# the basket number of each chunk
	basket_num += 1
	for i in range(0, len(words) - 2):
		for j in range(i+1, len(words) - 1):
			for k in range(j+1, len(words)):
				# find the frequent pairs
				pair1 = orderPair(words[i], words[j])
				pair2 = orderPair(words[j], words[k])
				pair3 = orderPair(words[i], words[k])
				# 1. pairs of frequent triplets should all be frequent
				# 2. there should be no order within item triplet
				if (pair1 in freq_pairs) and (pair2 in freq_pairs) and (pair3 in freq_pairs):
					triplet = orderTriplet(words[i], words[j], words[k])
					if triplet in freq_triplets:
						freq_triplets[triplet] += 1
					else: freq_triplets[triplet] = 1

# save intermediate result
s = threshold * basket_num
# only triplets more frequent than s can be saved
for triplet in freq_triplets:
	if (freq_triplets[triplet] >= s):
		print("%s\t%s" % (triplet, freq_triplets[triplet]))
