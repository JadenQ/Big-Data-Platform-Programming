#!/usr/bin/env python

from operator import itemgetter
import sys


last_follower = None
# input comes from STDIN
max_followees = 0
most_common_user = None
# followee_list = []

# last 4 numbers of my studentID are 1,0,4,8
def studentIdFlag(blogID):
	blogID = str(blogID)
	if '1' in blogID and '0' in blogID and '4' in blogID and '8' in blogID:
		return True
	else: return False

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
		if last_follower is not None and cur_follower != most_common_user and studentIdFlag(cur_follower):
			print("%s\t%s\t%s" % (cur_follower, most_common_user, max_followees))
		last_follower = cur_follower
		max_followees = followee_num
		most_common_user = followee

if cur_follower != most_common_user and studentIdFlag(cur_follower):
	print("%s\t%s\t%s" % (cur_follower, most_common_user, max_followees))