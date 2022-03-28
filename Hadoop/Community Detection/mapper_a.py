#!/usr/bin/env python

import sys

# input comes from STDIN
# switch key and value in the map process
for line in sys.stdin:
	line = line.strip()
	users = line.split()
	for followee, follower in users:
		print '%s\t%s' % (follower, followee)


