#!/usr/bin/env python

from operator import itemgetter
import sys
import re

# input comes from STDIN

# last 4 numbers of my studentID are 1,0,4,8
def studentIdFlag(blogID):
	blogID = str(blogID)
	if '1' in blogID and '0' in blogID and '4' in blogID and '8' in blogID:
		return True
	else: return False

def trimId(list_):
	new_list = []
	if len(list_) == 1 and list_[0] == '':
		return new_list
	else:
		for item in list_:
			if len(item)>0:
				num = re.findall(r"\d+\.?\d*",item)
				if len(num)>0:
					new_list.append(num[0])
			else: continue
		return new_list

for line in sys.stdin:
	line = line.strip()
	blog1, blog2, num_common_followee, num_total_followee, followee_list = line.split('\t')
	if float(num_total_followee) == 0:
		similarity = 0
	else:
		similarity = float(num_common_followee) / float(num_total_followee)
	followee_list = trimId(followee_list[1:-1].split(','))
	if studentIdFlag(blog1) and similarity > 0:
		print("%s\t%s\t{%s}\t%s" % (blog1, blog2,followee_list, similarity))
