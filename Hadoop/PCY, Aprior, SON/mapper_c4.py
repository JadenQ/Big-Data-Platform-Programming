#!/usr/bin/env python
import sys

# Mapper for Job2
# for all the candidate itemsets produced by Job1, count the frequency in local chunk

# stdin: the output result of last MR work
fname = 'candTriplet_c.txt'
freq_triplets = {} # key: item triplet, value: frequency


# unify the triplets
def orderTriplet(a, b, c):
	ordered_list = [a, b, c]
	ordered_list.sort()
	return ' '.join(ordered_list)

with open(fname) as f:
	for line in f.readlines():
		triplet = line.strip()
		freq_triplets[triplet] = 0 # reset to 0

# frequent triplets counts in each chunk
for line in sys.stdin:
	line = line.strip()
	words = line.split(' ')
	words = list(set(words)) # remove the duplicates in one basket
	for i in range(0, len(words) - 2):
		for j in range(i+1, len(words) - 1):
			for k in range(j+1, len(words)):
				# there should be no order within item triplet
				triplet = orderTriplet(words[i], words[j], words[k])
				# only count the frequent triplets
				if triplet in freq_triplets:
					freq_triplets[triplet] += 1


for triplet in freq_triplets:
	print("%s\t%s" % (triplet, freq_triplets[triplet]))