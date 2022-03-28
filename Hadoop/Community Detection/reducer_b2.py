#!/usr/bin/env python
import sys
import re

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

last_blog1 = None
count = 0
for line in sys.stdin:
	line = line.strip()
	cur_blog1, blog2, flwee_list, sim = line.split('\t')
	# flwee_list = trimId(flwee_list[2:-2].strip(','))
	if cur_blog1 == last_blog1:
		count = count + 1
	else:
		last_blog1 = cur_blog1
		count = 0
	if count < 3:
		print("%s:%s, %s, %s" % (cur_blog1, blog2, flwee_list, sim))
	
