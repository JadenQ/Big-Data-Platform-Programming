#!/usr/bin/env python
import sys

# Mapper for Job2
# for all the candidate itemsets produced by Job1, count the frequency in local chunk

# stdin: the output result of last MR work
fname = 'candPair_b.txt'
freq_pairs = {} # key: item pair, value: frequency



with open(fname) as f:
	for line in f.readlines():
		pair = line.strip()
		freq_pairs[pair] = 0 # reset to 0

# frequent pairs counts in each chunk
for line in sys.stdin:
	line = line.strip()
	words = line.split(' ')
	words = list(set(words)) # remove the duplicates in one basket
	for i in range(0, len(words) - 1):
		for j in range(i+1, len(words)):
				# there should be no order within item pair
				if (words[i] <= words[j]):
					pair = words[i] + " " + words[j]
				else: pair = words[j] + " " + words[i]
				# only count the frequent pairs
				if pair in freq_pairs:
					freq_pairs[pair] += 1


for pair in freq_pairs:
	print("%s\t%s" % (pair, freq_pairs[pair]))