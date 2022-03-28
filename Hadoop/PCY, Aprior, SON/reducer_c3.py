#!/usr/bin/env python
import sys

# aggregate the candidate triplet
current_triplet = None
for line in sys.stdin:
	line = line.strip()
	triplet, count = line.split('\t')
	if triplet == current_triplet:
		continue
	else:
		if current_triplet:
			print(current_triplet)
		current_triplet = triplet
	# discard c and output all candidate itemsets F's
if current_triplet == triplet:
	print(current_triplet)