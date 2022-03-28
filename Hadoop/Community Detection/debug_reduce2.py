#!/usr/bin/env python

from operator import itemgetter
import sys


last_follower = None
# input comes from STDIN
max_followees = 0
most_common_user = None
# followee_list = []

for line in sys.stdin:
	line = line.strip()
	cur_follower, followee_num, followee = line.split('\t')
	if cur_follower == last_follower:
		if int(followee_num) > int(max_followees):
			max_followees = followee_num
			most_common_user = followee
			# followee_list.append(most_common_user)
		else: continue
	else:
		if last_follower is not None and cur_follower != most_common_user:
			print("%s\t%s\t%s" % (cur_follower, most_common_user, max_followees))
		last_follower = cur_follower
		max_followees = followee_num
		most_common_user = followee

if cur_follower != most_common_user:
	print("%s\t%s\t%s" % (cur_follower, most_common_user, max_followees))