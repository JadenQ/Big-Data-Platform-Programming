# #!/usr/bin/env python

# import sys

# test_list = []
# with open('test.txt') as f:
# 	for line in f.readlines():
# 		test_word = line.strip()
# 		users = line.split(' ')
# 		test_list.append([int(users[0]), int(users[1])])

# print(test_list)

# for i in range(len(test_list)):
	
# # input comes from STDIN
# # switch key and value in the map process
# # for line in sys.stdin:
# # 	line = line.strip()
# # 	users = line.split(' ')




hadoop jar /usr/lib/hadoop-mapreduce/hadoop-streaming.jar \
-file mapper_a1.py -mapper mapper_a1.py \
-file reducer_a1.py -reducer reducer_a1.py \
-input ./hw1_a/small_relation \
-output ./a1/output1


-D mapred.map.tasks = 2 \
-D mapred.reduce.tasks = 3 \

		
#!/usr/bin/env python


import sys

# input comes from STDIN
dataset = []
for line in sys.stdin:
	line = line.strip()
	follower, followee_list = line.split('\t')
	dataset.append([follower, followee_list])

for i in range(len(dataset)):
	for j in range(i+1, len(dataset)):
		common_followee = list(set(dataset[i][1]) & set(dataset[j][1]))
		num_common_followee = len(common_followee)
		total_followee = len(dataset[i][1]) + len(dataset[j][1]) - num_common_followee

		print("%s\t%s" % (dataset[i][0], [dataset[j][0], common_followee, num_common_followee, total_followee]))


