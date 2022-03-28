### Homework1 Report

#### Table of Content

[TOC]

#### 0.Data Preparation

##### a. Download dataset from website

```shell
wget --http-user=ierg4300 --http-passwd=fall2021ierg http://mobitec.ie.cuhk.edu.hk/ierg4300Fall2021/homework/small.zip
unzip small.zip
rm small.zip
wget --http-user=ierg4300 --http-passwd=fall2021ierg http://mobitec.ie.cuhk.edu.hk/ierg4300Fall2021/homework/medium.zip
unzip medium.zip
rm medium.zip
wget --http-user=ierg4300 --http-passwd=fall2021ierg http://mobitec.ie.cuhk.edu.hk/ierg4300Fall2021/homework/large.zip
unzip large.zip
rm large.zip
```

##### b. Put dataset on hdfs file system

```shell
 hdfs dfs -mkdir hw1_a
 hdfs dfs -copyFromLocal hw1/data/small_relation ./hw1_a
 hdfs dfs -copyFromLocal hw1/data/medium_relation ./hw1_a
```

#### 1.Homework1_a

##### a. Step1 - Get the list of followees of each user

###### Write python scripts

Mapper_a1.py

```python
#!/usr/bin/env python
import sys
# input comes from STDIN
for line in sys.stdin:
	line = line.strip()
	followee, follower = line.split(' ')
	# switch the key and value, we need to sort according to follower and count followee
	print("%s\t%s" %(follower, followee))
```

Reducer_a1.py

```python
#!/usr/bin/env python
from operator import itemgetter
import sys

last_follower = None
followee_list = []

for line in sys.stdin:
	line = line.strip()
	cur_follower, followee = line.split('\t')

	if cur_follower == last_follower:
		followee_list.append(followee)
	else:
		if last_follower is not None:
			print("%s\t%s" % (last_follower, followee_list))
		last_follower = cur_follower
		followee_list = []

print("%s\t%s" % (last_follower, followee_list))
```

###### Authorize the python file

```shell
chmod +x hw1/hw1_a/a1/mapper_a1.py
chmod +x hw1/hw1_a/a1/reducer_a1.py
```

###### Debug the python script

```shell
cat ./data/small_relation | python3 ./hw1_a/a1/mapper_a1.py
cat ./data/small_relation | python3 ./hw1_a/a1/mapper_a1.py | sort -k1 | python3 ./hw1_a/a1/reducer_a1.py
```

###### Run the job

```shell
# hdfs dfs -rm -r /user/s1155161048/a1/output1  
hadoop jar /usr/lib/hadoop-mapreduce/hadoop-streaming.jar \
-D mapred.output.key.comparator.class=org.apache.hadoop.mapred.lib.KeyFieldBasedComparator \
-D mapred.text.key.comparator.options=-n \
-file mapper_a1.py -mapper mapper_a1.py \
-file reducer_a1.py -reducer reducer_a1.py \
-input ./hw1_a/medium_relation \
-output ./a1/output1
```

###### Output1

```shell
hdfs dfs -cat a1/output1/part-00000
```

![1633523400047](pic\1.png)

##### b. Step2 - Count the common followees and find the maximum

###### Python script

In this step, I use follower id as key and the number of common followees, id of the other user as value.

mapper_a2.py `Error`

```python
#!/usr/bin/env python
import sys
# input comes from STDIN
dataset = []

# This part is incorrect:
for line in sys.stdin:
	line = line.strip()
	follower, followee_list = line.split('\t')
	followee_list = followee_list[1:-1].split(',')
	dataset.append([follower, followee_list])

# This part should also be implemented in mappers, instead of single machine
for i in range(len(dataset)):
	for j in range(i+1, len(dataset)):
		if len(dataset[i][1])==0 or len(dataset[j][1])==0:
			continue
		else:
			common_followee = list(set(dataset[i][1]) & set(dataset[j][1]))
			num_common_followee = len(common_followee)
			total_followee = len(dataset[i][1]) + len(dataset[j][1]) - num_common_followee
			print("%s\t%s\t%s" % (dataset[i][0], num_common_followee, dataset[j][0])
# id of each user | number of common followees | id of the other user
```

reducer_a2.py

```python
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
```

###### Use small relation to debug

```shell
hdfs dfs -get a1/small_output1/part-00000 ../a2/small_part-00000
cat small_part-00000 | python3 mapper_a2.py

cat ./small_part-00000 | sort -nk1 | python3 ./mapper_a2.py | sort -nk1 | python3 ./reducer_a2.py
```

###### Run on Hadoop

```shell
# hdfs dfs -rm -r /user/s1155161048/a2/output2
hadoop jar /usr/lib/hadoop-mapreduce/hadoop-streaming.jar \
-D mapred.output.key.comparator.class=org.apache.hadoop.mapred.lib.KeyFieldBasedComparator \
-D mapred.text.key.comparator.options=-n \
-D mapred.job.name='Job_a2' \
-D mapred.map.tasks=100 \
-D mapred.reduce.tasks=10 \
-file mapper_a2.py -mapper mapper_a2.py \
-file reducer_a2.py -reducer reducer_a2.py \
-input ./a1/output1 \
-output ./a2/output2
```

###### Output2

The output of question (a) is in this format, the blogs without any followee are ruled out.

```
blogID recommendedBlogID #commonBlogs
```

```shell
hdfs dfs -cat a2/output2/part-00000
```

![1633572629850](pic\2.png)

#### 2.Homework1_b

In task b, I used mapreduce jobs and python script to calculate similarity and sort/format the result respectively.

##### a. Map task - Calculate number of total followees and number of common followees for every blog pairs

In this part, I used the result of the first mapreduce job, i.e. the result of mapper_a1.py and reducer_a1.py

mapper_b1.py

```python
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
# input comes from STDIN
dataset = []
for line in sys.stdin:
	line = line.strip()
	follower, followee_list = line.split('\t')
	followee_list = trimId(followee_list[1:-1].split(','))
	dataset.append([follower, followee_list])

for i in range(len(dataset)):
	for j in range(i+1, len(dataset)):
		common_followee = list(set(dataset[i][1]) & set(dataset[j][1]))
		num_common_followee = len(common_followee)
		total_followee = len(dataset[i][1]) + len(dataset[j][1]) - num_common_followee

		print("%s\t%s\t%s\t%s\t%s" % (dataset[i][0], dataset[j][0], num_common_followee, total_followee, common_followee))
```

So, the intermediate result should look like this.

```
blog1ID blog2ID #commonFollowee #totalFollowee commonFollowees
```

##### b. Reduce task - Calculate similarity of blog pairs

```python
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
# trim the result of common followee list to only numbers of blog ID
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
    # similarity = #commonFollowees / #total_followees
	similarity = float(num_common_followee) / float(num_total_followee)
	followee_list = trimId(followee_list[1:-1].split(','))
	if studentIdFlag(blog1):
		print("%s\t%s\t{%s}\t%s" % (blog1, blog2,followee_list, similarity))
```

##### c. Debug and run

###### Debug

```
cat ../hw1_a/a2/small_part-00000 | sort -nk1 | python3 ./mapper_b1.py
cat ../hw1_a/a2/small_part-00000 | sort -nk1 | python3 ./mapper_b1.py | sort -nk1 | python3 ./reducer_b1.py
```

###### Run on Hadoop

```shell
# hdfs dfs -rm -r /user/s1155161048/b1/output3
hadoop jar /usr/lib/hadoop-mapreduce/hadoop-streaming.jar \
-D mapred.output.key.comparator.class=org.apache.hadoop.mapred.lib.KeyFieldBasedComparator \
-D mapred.text.key.comparator.options=-n \
-D mapred.job.name='Job_b1' \
-D mapred.map.tasks=100 \
-D mapred.reduce.tasks=10 \
-file mapper_b1.py -mapper mapper_b1.py \
-file reducer_b1.py -reducer reducer_b1.py \
-input ./a1/output1 \
-output ./b1/output3
```

###### Rule out blog pairs with zero similarity

```python
# modification of reducer_b1.py
if studentIdFlag(blog1) and similarity > 0:
		print("%s:%s,{%s},%s" % (blog1, blog2,followee_list, similarity))
```

###### Output3

```shell
hdfs dfs -ls b1/output3
hdfs dfs -cat b1/output3/part-00001
```

![1633588833256](pic\3.png)

##### d. Sort and find the Top-K result

###### Use sort command to sort

```shell
hdfs dfs -get b1/output3 ./
# concate the output results
cat ./output3/* > ./output3/output3_all
# use the blog1 and similarity to sort
cat ./output3/output3_all |sort -n -k1 -k4r > ./output3/output3_all_sort
hdfs dfs -copyFromLocal hw1/hw1_b/output3/output3_all_sort ./b1
```

![1633589354901](pic\4.png)

###### Use the following python script to limit the Top-K (K = 3)

Since we have done sorting and calculating, there is no need for this step to do mapreduce.

```shell
 cat ./output3/output3_all_sort | python3 formatTopK.py > result_b
```

formatTopK.py

```python
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
	flwee_list = trimId(flwee_list[1:-1].strip(','))
	if cur_blog1 == last_blog1:
		count += 1
	else:
		count = 0
	if count < 3:
		print("%s:%s, {%s}, %s" % (cur_blog1, blog2, flwee_list, sim))
```

###### Output_b

![1633592535977](pic\5.png)

#### 3.Homework1_c

I re-run the last mapreduce job to get all the blog pairs and it's common list, i.e. the mapper_b1.py and reducer_b1.py are modified as mapper_c0.py and reducer_c0.py (eliminate student ID and other useless calculation, list all the blog id at the first column).

In the first step, I will join the *medium_label* table and add a line to label the community; in the second step, I will calculate the result.

##### a. Step1 - add community information to blog pairs and label

In this step, the *key* is blog id, *value* is the followees list to label the blog which act as a common followee of other community. *actLabel* = 1 means it acts as other community's common followee, vice versa.

###### mapper_c0.py

```python
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
```

The result of mapper_c0.py is shown as follow:

![1633656050219](pic\6.png)

###### reducer_c0.py

```python
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
	actLabel = 0
	for blog1_j in blog1_list:
		if blog1_i in blog1_flw[blog1_j] and blog1_com[blog1_i] != blog1_com[blog1_j]:
			actLabel = 1
			break
		else: continue
	print("%s\t%s\t%s" % (blog1_com[blog1_i], actLabel, blog1_i))
```

The output formats as community label, actLabel, blog_ID. I put community label first because I am going to use it as key in the second step.

###### Debug and Output4

```shell
cat ../hw1_a/a2/small_part-00000 | sort -nk1 | python3 ./mapper_c0.py | sort -nk1 | python3 ./reducer_c0.py
```

```shell
# hdfs dfs -rm -r /user/s1155161048/c0/output4
hadoop jar /usr/lib/hadoop-mapreduce/hadoop-streaming.jar \
-D mapred.output.key.comparator.class=org.apache.hadoop.mapred.lib.KeyFieldBasedComparator \
-D mapred.text.key.comparator.options=-n \
-D mapred.job.name='Job_c1' \
-D mapred.map.tasks=100 \
-D mapred.reduce.tasks=10 \
-file medium_label \
-file mapper_c0.py -mapper mapper_c0.py \
-file reducer_c0.py -reducer reducer_c0.py \
-input ./a1/output1 \
-output ./c0/output4
```

![1633662094692](pic\7.png)

##### b. Step2 - count the number

In step2, I use community label as key, and the actLabel as value.

###### mapper_c1.py

```python
#!/usr/bin/env python
import sys
import re

# input file is output3_all
for line in sys.stdin:
	line = line.strip()
	community_label, actLabel, blogID = line.split('\t')
	print("%s\t%s" % (community_label, actLabel))
```

###### reducer_c1.py

```python
#!/usr/bin/env python
import sys
import re

last_community_label = None
count = 0
# input file is output3_all
for line in sys.stdin:
	line = line.strip()
	community_label, actLabel = line.split('\t')
	# count all the 1 label in one coummnity
	if community_label == last_community_label:
		actLabel = int(actLabel)
		count = count + actLabel
	else:
		if last_community_label:
			print("%s\t%s" % (last_community_label, count))
		last_community_label = community_label
		count = 0
# last line
if last_community_label == community_label:
	print("Community %s:%s" % (last_community_label, count))
```

###### Debug and output

```shell
cat ./output4/* | sort -nk1 | python3 ./mapper_c1.py | sort -nk1 | python3 ./reducer_c1.py
```

```shell
# hdfs dfs -rm -r /user/s1155161048/c1/output5
hadoop jar /usr/lib/hadoop-mapreduce/hadoop-streaming.jar \
-D mapred.output.key.comparator.class=org.apache.hadoop.mapred.lib.KeyFieldBasedComparator \
-D mapred.text.key.comparator.options=-n \
-D mapred.job.name='Job_c2' \
-D mapred.map.tasks=5 \
-D mapred.reduce.tasks=2 \
-file mapper_c1.py -mapper mapper_c1.py \
-file reducer_c1.py -reducer reducer_c1.py \
-input ./c0/output4 \
-output ./c1/output5
```

```shell
hdfs dfs -cat ./c1/output5/*
```

![1633662854545](C:\Users\Jaden\AppData\Roaming\Typora\typora-user-images\1633662854545.png)

#### 4.Homework1_d

I use the follow command to run the mapreduce experiments, and set $m$ and $n$ to different combinations.

```shell
# hdfs dfs -rm -r /user/s1155161048/a2/output2
hadoop jar /usr/lib/hadoop-mapreduce/hadoop-streaming.jar \
-D mapred.output.key.comparator.class=org.apache.hadoop.mapred.lib.KeyFieldBasedComparator \
-D mapred.text.key.comparator.options=-n \
-D mapred.job.name='Job_m_r' \
-D mapred.map.tasks=10 \
-D mapred.reduce.tasks=10 \
-file mapper_a2.py -mapper mapper_a2.py \
-file reducer_a2.py -reducer reducer_a2.py \
-input ./a1/output1 \
-output ./a2/output2
```

| Mapper num | Reducer num | Max mapper time | Min mapper time | Avg mapper time | Max reducer time | Min reducer time | Avg reducer time | Total job |         Note          |
| :--------: | :---------: | :-------------: | :-------------: | :-------------: | :--------------: | :--------------: | :--------------: | :-------: | :-------------------: |
|     10     |      1      |       9s        |       8s        |       8s        |       32s        |       32s        |       32s        |    60s    |    Shuffle time:1s    |
|     10     |      5      |       9s        |       8s        |       9s        |        6s        |        6s        |        6s        |    41s    |           -           |
|     10     |     10      |       10s       |       8s        |       9s        |        3s        |        3s        |        3s        |    54s    |           -           |
|    100     |     10      |       2s        |       1s        |       1s        |        0s        |        0s        |        0s        |    95s    |   Shuffle time: 57s   |
|    100     |     100     |       2s        |       1s        |       1s        |        0s        |        0s        |        0s        |   185s    |   Shuffle time: 56s   |
|     50     |     100     |       3s        |       1s        |       2s        |        0s        |        0s        |        0s        |   144s    |   Shuffle time: 27s   |
|    300     |     10      |       2s        |       1s        |       1s        |        0s        |        0s        |        0s        |   263s    | Avg shuffle time: 19s |

##### Explanation

###### 1.The number of reducers

In the first 4 experiments, I increase the number of reducers.

At the beginning, suitable number of reducer can increase the processing speed because there will be more processors involved to compute, but when I set reducer to 100, it turns out the reduce task takes more time, especially shuffle time.

The reduce need to start up and be created in the nodes, most of the time are wasted in starting up redundant reducers.

###### 2.The number of mappers

When *increasing the number of mappers*, the computing load are distributed into multiple different mapper tasks, so the average/min/max time of mapper task decrease. But the minimal time of mapper is 1s to 2s according to all the experiments, because as the number mappers increase to over 100, *the average mapper time does not decrease*, it means there are *redundant mappers* and reached the shortest time for each mapper to process.

###### 3.Conclusion

In practice, to increase the speed of our processing, we should set up number of mappers and reducers appropriately by doing experiments or theoretical analysis.

The *number of mapper* should be set up according to the number of cores and data block size, suppose there are 100 data nodes, each data node has 8 cores to run 5 mappers per node, then we can run 500 mappers in a cluster, too many mappers will be useless.

The *number of reducer* should be set up in a range, according to the number_of_nodes*maximum_container_per_node.

#### 5.Homework1_e - Bonus 

##### a. Try to run with the current Python script

###### Mapreduce Job_e1

Details are explained in Part1 - Homework1_a.

```shell
# hdfs dfs -rm -r /user/s1155161048/e1/output6  
hadoop jar /usr/lib/hadoop-mapreduce/hadoop-streaming.jar \
-D mapred.output.key.comparator.class=org.apache.hadoop.mapred.lib.KeyFieldBasedComparator \
-D mapred.text.key.comparator.options=-n \
-D mapred.job.name='Job_e1' \
-D mapred.map.tasks=10 \
-D mapred.reduce.tasks=10 \
-file mapper_a1.py -mapper mapper_a1.py \
-file reducer_a1.py -reducer reducer_a1.py \
-input ./hw1_e/large_relation \
-output ./e1/output6
```

![1633667447803](pic\8.png)

![1633667485964](D:\document\CUHK\Web-scale\homework1\pic\9.png)

###### Mapreduce Job_e2

Details are explained in Part2 - Homework1_b.

```shell
# hdfs dfs -rm -r /user/s1155161048/e2/output7
hadoop jar /usr/lib/hadoop-mapreduce/hadoop-streaming.jar \
-D mapred.output.key.comparator.class=org.apache.hadoop.mapred.lib.KeyFieldBasedComparator \
-D mapred.text.key.comparator.options=-n \
-D mapred.job.name='Job_e2' \
-D mapred.map.tasks=40 \
-D mapred.reduce.tasks=10 \
-file mapper_b1.py -mapper mapper_b1.py \
-file reducer_b1.py -reducer reducer_b1.py \
-input ./e1/output6 \
-output ./e2/output7
```

![1633672112566](pic\10.png)

![1633672164157](pic\11.png)

###### Final Result

```shell
hdfs dfs -get e2/output7 ./
# concate the output results
cat ./output7/* > ./output7/output7_all
# use the blog1 and similarity to sort
cat ./output7/output7_all |sort -n -k1 -k4r > ./output7/output7_all_sort
cat ./output7/output7_all_sort | python3 formatTopK.py > ./result_e
```

![1633672377669](pic\12.png)

#### References

1. Use sort function in Linux: https://www.runoob.com/linux/linux-comm-sort.html

2. The function of reducer and shuffle: https://stackoverflow.com/questions/39541718/why-increasing-the-number-of-reducers-increases-the-time-for-running-reduce-phas

3. Linux sort function:https://segmentfault.com/a/1190000005713784
4. Implement Join in mapreduce:https://blog.matthewrathbone.com/2013/02/09/real-world-hadoop-implementing-a-left-outer-join-in-hadoop-map-reduce.html
5. Number of mappers and reducer:https://data-flair.training/forums/topic/how-one-can-decide-for-a-job-how-many-mapper-reducers-are-required/

#### Further Investigation

1.How to reuse the result of map task to different reduce job?

2.How to prevent $O(n^2)$ algorithm and make it faster.

3.How to implement join in mapreduce job?

4.How to implement composite key in python.