#!/usr/bin/env python

import sys
import gc

# Mapper for Job1
# Run A-Priori Algorithm in each chunk
basket_num = 0 # num of baskets
word_count = {} # key: word (individual item), value: frequency
freq_item = {} # key:word (frequent individual item), value: frequency
threshold = 0.005 # support threshold
freq_pairs = {} # key: item pair, value: frequency

baskets = [] # the list to store baskets for counting pass

def orderPair(a,b):
	if (a <= b):
		pair = a + " " + b
	else: pair = b + " " + a
	return pair

# income from standard input
hashTable = [0 for i in range(100000)]

for line in sys.stdin:
	line = line.strip()
	words = line.split(' ')
	words = list(set(words))
	baskets.append(words)
	for word in words:
		if word not in word_count:
			word_count[word] = 1
		else: word_count[word] += 1
	for i in range(0, len(words) - 1):
		for j in range(i+1, len(words)):
		# define a hashtable
		# hash all the pairs into 100000 bucktes
			hashIndex = hash(orderPair(words[i], words[j])) % 100000 # index of the hash table
			hashTable[hashIndex] += 1 # count into the hash table

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
				pair = orderPair(basket[i], basket[j])
				# verify freq pairs using hashtable
				hashIndex = hash(pair) % 100000
				if hashTable[hashIndex] >= s:
					if pair in freq_pairs:
						freq_pairs[pair] += 1
					else: freq_pairs[pair] = 1

# save intermediate result
# only pairs more frequent than s can be saved
for pair in freq_pairs:
	if (freq_pairs[pair] >= s):
		print("%s\t%s" % (pair, freq_pairs[pair]))

# a b	count 