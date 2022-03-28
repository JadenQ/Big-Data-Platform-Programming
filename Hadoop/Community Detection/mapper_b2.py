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


for line in sys.stdin:
	line = line.strip()
	blog1, blog2, flwee_list, sim = line.split('\t')
	flwee_list = trimId(flwee_list[2:-2].strip(','))
	print("%s\t%s\t%s\t%s" % (blog1, blog2, flwee_list, sim))
