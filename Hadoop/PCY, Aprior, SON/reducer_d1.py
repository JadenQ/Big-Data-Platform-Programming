#!/usr/bin/env python
import sys

current_pair = None
for line in sys.stdin:
	line = line.strip()
	pair, count = line.split('\t')
	if pair == current_pair:
		continue
	else:
		if current_pair:
			print(current_pair)
		current_pair = pair
	# discard c and output all candidate itemsets F's
if current_pair == pair:
	print(current_pair)