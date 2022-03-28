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

# input file is small, medium and large label
community_list = {} # to save community info
with open('medium_label') as f:
	for line in f.readlines():
		community = line.strip()
		blog_id, community = community.split(' ')
		community_list.update({blog_id: community})


# input comes from STDIN
dataset = []
for line in sys.stdin:
	line = line.strip()
	follower, followee_list = line.split('\t')
	followee_list = trimId(followee_list[1:-1].split(','))
	dataset.append([follower, followee_list])

for i in range(len(dataset)):
	for j in range(len(dataset)):
		common_followee = list(set(dataset[i][1]) & set(dataset[j][1]))
		community = community_list[dataset[i][0]]
		if len(common_followee) > 0:
			print("%s\t%s\t%s\t%s" % (dataset[i][0], dataset[j][0], common_followee, community))