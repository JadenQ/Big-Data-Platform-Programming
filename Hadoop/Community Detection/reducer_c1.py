#!/usr/bin/env python
import sys
import re

last_community_label = None
count = 0
# input file is output3_all
for line in sys.stdin:
	line = line.strip()
	community_label, actLabel = line.split('\t')
	# count all the 1 label in one coummnity
	if community_label == last_community_label:
		actLabel = int(actLabel)
		count = count + actLabel
	else:
		if last_community_label:
			print("%s\t%s" % (last_community_label, count))
		last_community_label = community_label
		count = 0
# last line
if last_community_label == community_label:
	print("Community %s:%s" % (last_community_label, count))

