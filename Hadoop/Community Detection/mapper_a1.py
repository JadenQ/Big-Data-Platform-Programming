#!/usr/bin/env python

import sys

# input comes from STDIN
for line in sys.stdin:
	line = line.strip()
	followee, follower = line.split(' ')
	# switch the key and value, we need to sort according to follower and count followee
	print("%s\t%s" %(follower, followee))
		


