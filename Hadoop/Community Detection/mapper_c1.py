#!/usr/bin/env python
import sys
import re

# input file is output3_all
for line in sys.stdin:
	line = line.strip()
	community_label, actLabel, blogID = line.split('\t')
	print("%s\t%s" % (community_label, actLabel))
