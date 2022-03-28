#!/usr/bin/env python

import sys
import gc

# Mapper for Job1
# Run A-Priori Algorithm in each chunk
basket_num = 0 # num of baskets
word_count = {} # key: word (individual item), value: frequency
freq_item = {} # key:word (frequent individual item), value: frequency
threshold = 0.0025 # support threshold
freq_pairs = {} # key: item pair, value: frequency

baskets = [] # the list to store baskets for counting pass
# s = threshold * basket_number * p , calculate for each chunk

# income from standard input
for line in sys.stdin:
	line = line.strip()
	words = line.split(' ')
	words = list(set(words))
	baskets.append(words)
	for word in words:
		if word not in word_count:
			word_count[word] = 1
		else: word_count[word] += 1

# find frequent itemset in subsets

basket_num = len(baskets)
s = threshold * basket_num

for word in word_count:
	if word_count[word] >= s:
		freq_item[word] = word_count[word]

# garbage collect
del word_count
gc.collect()

for basket in baskets:
	for i in range(0, len(basket) - 1):
		for j in range(i+1, len(basket)):
			if (basket[i] in freq_item) and (basket[j] in freq_item): # both elements are frequent
				# there should be no order within item pair
				if (basket[i] <= basket[j]):
					pair = basket[i] + " " + basket[j]
				else: pair = basket[j] + " " + basket[i]
				if pair in freq_pairs:
					freq_pairs[pair] += 1
				else: freq_pairs[pair] = 1

# save intermediate result
# only pairs more frequent than s can be saved
for pair in freq_pairs:
	if (freq_pairs[pair] >= s):
		print("%s\t%s" % (pair, freq_pairs[pair]))

# a b	count 