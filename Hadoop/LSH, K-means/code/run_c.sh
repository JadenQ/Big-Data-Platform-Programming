# give permission
chmod +x mapper_a1.py reducer_a1.py
# remove old centroid file
# rm centroid_p*

# generate the initial centroid file
python3 cen_gen_new.py 1
# iteration loop
for i in `seq 1 15`;
do
	# remove old output
	hdfs dfs -rm -r ./hw3/output_a1
	# run mapreduce
	hadoop jar /usr/lib/hadoop-mapreduce/hadoop-streaming.jar \
	-D mapred.output.key.comparator.class=org.apache.hadoop.mapred.lib.KeyFieldBasedComparator \
	-D mapred.job.name='Job_c1' \
	-D mapred.map.tasks=20 \
	-D mapred.reduce.tasks=10 \
	-D mapred.text.key.comparator.options=-k1n \
	-file mapper_a1.py -mapper 'mapper_a1.py 1' \
	-file reducer_a1.py -reducer reducer_a1.py \
	-file centroid_p1.txt \
	-input ./hw3/train_part_1.txt \
	-output ./hw3/output_a1
	# save output
	hdfs dfs -cat ./hw3/output_a1/* > centroid_p1_new.txt
	# save the final
	cat centroid_p1_new.txt > centroid_p1.txt
	# save all iterations as log
	cat centroid_p1_new.txt >> centroid_p1_all.txt
	# delete the output
	hdfs dfs -rm -r ./hw3/output_a1
done

python3 cen_gen_new.py 2
# iteration loop
for i in `seq 1 15`;
do
	# remove old output
	hdfs dfs -rm -r ./hw3/output_a1
	# run mapreduce
	hadoop jar /usr/lib/hadoop-mapreduce/hadoop-streaming.jar \
	-D mapred.output.key.comparator.class=org.apache.hadoop.mapred.lib.KeyFieldBasedComparator \
	-D mapred.job.name='Job_c2' \
	-D mapred.map.tasks=20 \
	-D mapred.reduce.tasks=10 \
	-D mapred.text.key.comparator.options=-k1n \
	-file mapper_a1.py -mapper 'mapper_a1.py 2' \
	-file reducer_a1.py -reducer reducer_a1.py \
	-file centroid_p2.txt \
	-input ./hw3/train_part_2.txt \
	-output ./hw3/output_a1
	# save output
	hdfs dfs -cat ./hw3/output_a1/* > centroid_p2_new.txt
	# save the final
	cat centroid_p2_new.txt > centroid_p2.txt
	# save all iterations as log
	cat centroid_p2_new.txt >> centroid_p2_all.txt
	# delete the output
	hdfs dfs -rm -r ./hw3/output_a1
done

python3 cen_gen_new.py 3
# iteration loop
for i in `seq 1 15`;
do
	# remove old output
	hdfs dfs -rm -r ./hw3/output_a1
	# run mapreduce
	hadoop jar /usr/lib/hadoop-mapreduce/hadoop-streaming.jar \
	-D mapred.output.key.comparator.class=org.apache.hadoop.mapred.lib.KeyFieldBasedComparator \
	-D mapred.job.name='Job_c3' \
	-D mapred.map.tasks=20 \
	-D mapred.reduce.tasks=5 \
	-D mapred.text.key.comparator.options=-k1n \
	-file mapper_a1.py -mapper 'mapper_a1.py 3' \
	-file reducer_a1.py -reducer reducer_a1.py \
	-file centroid_p3.txt \
	-input ./hw3/train_part_3.txt \
	-output ./hw3/output_a1
	# save output
	hdfs dfs -cat ./hw3/output_a1/* > centroid_p3_new.txt
	# save the final
	cat centroid_p3_new.txt > centroid_p3.txt
	# save all iterations as log
	cat centroid_p3_new.txt >> centroid_p3_all.txt
	# delete the output
	hdfs dfs -rm -r ./hw3/output_a1
done

python3 cen_gen_new.py 4
# iteration loop
for i in `seq 1 15`;
do
	# remove old output
	hdfs dfs -rm -r ./hw3/output_a1
	# run mapreduce
	hadoop jar /usr/lib/hadoop-mapreduce/hadoop-streaming.jar \
	-D mapred.output.key.comparator.class=org.apache.hadoop.mapred.lib.KeyFieldBasedComparator \
	-D mapred.job.name='Job_c4' \
	-D mapred.map.tasks=20 \
	-D mapred.reduce.tasks=10 \
	-D mapred.text.key.comparator.options=-k1n \
	-file mapper_a1.py -mapper 'mapper_a1.py 4' \
	-file reducer_a1.py -reducer reducer_a1.py \
	-file centroid_p4.txt \
	-input ./hw3/train_part_4.txt \
	-output ./hw3/output_a1
	# save output
	hdfs dfs -cat ./hw3/output_a1/* > centroid_p4_new.txt
	# save the final
	cat centroid_p4_new.txt > centroid_p4.txt
	# save all iterations as log
	cat centroid_p4_new.txt >> centroid_p4_all.txt
	# delete the output
	hdfs dfs -rm -r ./hw3/output_a1
done

python3 cen_gen_new.py 5
# iteration loop
for i in `seq 1 15`;
do
	# remove old output
	hdfs dfs -rm -r ./hw3/output_a1
	# run mapreduce
	hadoop jar /usr/lib/hadoop-mapreduce/hadoop-streaming.jar \
	-D mapred.output.key.comparator.class=org.apache.hadoop.mapred.lib.KeyFieldBasedComparator \
	-D mapred.job.name='Job_c5' \
	-D mapred.map.tasks=20 \
	-D mapred.reduce.tasks=5 \
	-D mapred.text.key.comparator.options=-k1n \
	-file mapper_a1.py -mapper 'mapper_a1.py 5' \
	-file reducer_a1.py -reducer reducer_a1.py \
	-file centroid_p5.txt \
	-input ./hw3/train_part_5.txt \
	-output ./hw3/output_a1
	# save output
	hdfs dfs -cat ./hw3/output_a1/* > centroid_p5_new.txt
	# save the final
	cat centroid_p5_new.txt > centroid_p5.txt
	# save all iterations as log
	cat centroid_p5_new.txt >> centroid_p5_all.txt
	# delete the output
	hdfs dfs -rm -r ./hw3/output_a1
done


# give permission
chmod +x mapper_b2.py reducer_b2.py

# remove old output
hdfs dfs -rm -r ./hw3/output_b2
# run mapreduce
hadoop jar /usr/lib/hadoop-mapreduce/hadoop-streaming.jar \
-D mapred.output.key.comparator.class=org.apache.hadoop.mapred.lib.KeyFieldBasedComparator \
-D mapred.job.name='Job_c_eva1' \
-D mapred.map.tasks=20 \
-D mapred.reduce.tasks=5 \
-D mapred.text.key.comparator.options=-k1n \
-file mapper_b2.py -mapper 'mapper_b2.py 1' \
-file reducer_b2.py -reducer reducer_b2.py \
-file centroid_p1_new.txt \
-input ./hw3/xaa \
-output ./hw3/output_b2

# save output
hdfs dfs -cat ./hw3/output_b2/* > cluster_p1.txt
hdfs dfs -rm -r ./hw3/output_b2


# run mapreduce
hadoop jar /usr/lib/hadoop-mapreduce/hadoop-streaming.jar \
-D mapred.output.key.comparator.class=org.apache.hadoop.mapred.lib.KeyFieldBasedComparator \
-D mapred.job.name='Job_c_eva2' \
-D mapred.map.tasks=20 \
-D mapred.reduce.tasks=5 \
-D mapred.text.key.comparator.options=-k1n \
-file mapper_b2.py -mapper 'mapper_b2.py 2' \
-file reducer_b2.py -reducer reducer_b2.py \
-file centroid_p2_new.txt \
-input ./hw3/xab \
-output ./hw3/output_b2

# save output
hdfs dfs -cat ./hw3/output_b2/* > cluster_p2.txt
hdfs dfs -rm -r ./hw3/output_b2


# run mapreduce
hadoop jar /usr/lib/hadoop-mapreduce/hadoop-streaming.jar \
-D mapred.output.key.comparator.class=org.apache.hadoop.mapred.lib.KeyFieldBasedComparator \
-D mapred.job.name='Job_c_eva3' \
-D mapred.map.tasks=20 \
-D mapred.reduce.tasks=5 \
-D mapred.text.key.comparator.options=-k1n \
-file mapper_b2.py -mapper 'mapper_b2.py 3' \
-file reducer_b2.py -reducer reducer_b2.py \
-file centroid_p3_new.txt \
-input ./hw3/xac \
-output ./hw3/output_b2

# save output
hdfs dfs -cat ./hw3/output_b2/* > cluster_p3.txt
hdfs dfs -rm -r ./hw3/output_b2

# run mapreduce
hadoop jar /usr/lib/hadoop-mapreduce/hadoop-streaming.jar \
-D mapred.output.key.comparator.class=org.apache.hadoop.mapred.lib.KeyFieldBasedComparator \
-D mapred.job.name='Job_c_eva4' \
-D mapred.map.tasks=20 \
-D mapred.reduce.tasks=5 \
-D mapred.text.key.comparator.options=-k1n \
-file mapper_b2.py -mapper 'mapper_b2.py 4' \
-file reducer_b2.py -reducer reducer_b2.py \
-file centroid_p4_new.txt \
-input ./hw3/xad \
-output ./hw3/output_b2

# save output
hdfs dfs -cat ./hw3/output_b2/* > cluster_p4.txt
hdfs dfs -rm -r ./hw3/output_b2

# run mapreduce
hadoop jar /usr/lib/hadoop-mapreduce/hadoop-streaming.jar \
-D mapred.output.key.comparator.class=org.apache.hadoop.mapred.lib.KeyFieldBasedComparator \
-D mapred.job.name='Job_c_eva5' \
-D mapred.map.tasks=20 \
-D mapred.reduce.tasks=5 \
-D mapred.text.key.comparator.options=-k1n \
-file mapper_b2.py -mapper 'mapper_b2.py 5' \
-file reducer_b2.py -reducer reducer_b2.py \
-file centroid_p5_new.txt \
-input ./hw3/xae \
-output ./hw3/output_b2

# save output
hdfs dfs -cat ./hw3/output_b2/* > cluster_p5.txt
hdfs dfs -rm -r ./hw3/output_b2


