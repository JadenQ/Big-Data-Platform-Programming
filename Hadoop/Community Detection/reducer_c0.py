#!/usr/bin/env python

from operator import itemgetter
import sys
import re

# input comes from STDIN



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


# one key to 2 values
blog1_flw = {} # save blog1_followeeList key-value pair
blog1_com = {} # save blog1_community key-value pair
last_blog1 = None
incre_followee_list = []

for line in sys.stdin:
	line = line.strip()
	blog1, blog2, followee_list, community = line.split('\t')
	followee_list = trimId(followee_list[1:-1].split(','))
	blog1_com[blog1] = community
	# aggregate the followee list according to blog1 ID
	if blog1 == last_blog1:
		incre_followee_list = incre_followee_list + followee_list 
	else:
		last_blog1 = blog1
		blog1_flw[blog1] = incre_followee_list
		incre_followee_list = []

blog1_list = list(blog1_com.keys())


for blog1_i in blog1_list:
	count = 0
	for blog1_j in blog1_list:
		if blog1_i in blog1_flw[blog1_j] and blog1_com[blog1_i] != blog1_com[blog1_j]:
			count = 1
			break
		else: continue
	print("%s\t%s\t%s" % (blog1_com[blog1_i], count, blog1_i))



