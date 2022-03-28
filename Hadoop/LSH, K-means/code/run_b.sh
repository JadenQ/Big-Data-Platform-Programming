# give permission
chmod +x mapper_b2.py reducer_b2.py

# remove old output
hdfs dfs -rm -r ./hw3/output_b2
# run mapreduce
hadoop jar /usr/lib/hadoop-mapreduce/hadoop-streaming.jar \
-D mapred.output.key.comparator.class=org.apache.hadoop.mapred.lib.KeyFieldBasedComparator \
-D mapred.job.name='Job_b2' \
-D mapred.map.tasks=20 \
-D mapred.reduce.tasks=5 \
-D mapred.text.key.comparator.options=-k1n \
-file mapper_b2.py -mapper 'mapper_b2.py 1229' \
-file reducer_b2.py -reducer reducer_b2.py \
-file centroid_1229_new.txt \
-input ./hw3/train.txt \
-output ./hw3/output_b2

# save output
hdfs dfs -cat ./hw3/output_b2/* > cluster_1229.txt
hdfs dfs -rm -r ./hw3/output_b2

hadoop jar /usr/lib/hadoop-mapreduce/hadoop-streaming.jar \
-D mapred.output.key.comparator.class=org.apache.hadoop.mapred.lib.KeyFieldBasedComparator \
-D mapred.job.name='Job_b2' \
-D mapred.map.tasks=20 \
-D mapred.reduce.tasks=5 \
-D mapred.text.key.comparator.options=-k1n \
-file mapper_b2.py -mapper 'mapper_b2.py 324' \
-file reducer_b2.py -reducer reducer_b2.py \
-file centroid_324_new.txt \
-input ./hw3/train.txt \
-output ./hw3/output_b2
# save output
hdfs dfs -cat ./hw3/output_b2/* > cluster_324.txt
hdfs dfs -rm -r ./hw3/output_b2

hadoop jar /usr/lib/hadoop-mapreduce/hadoop-streaming.jar \
-D mapred.output.key.comparator.class=org.apache.hadoop.mapred.lib.KeyFieldBasedComparator \
-D mapred.job.name='Job_b2' \
-D mapred.map.tasks=20 \
-D mapred.reduce.tasks=5 \
-D mapred.text.key.comparator.options=-k1n \
-file mapper_b2.py -mapper 'mapper_b2.py 1001' \
-file reducer_b2.py -reducer reducer_b2.py \
-file centroid_1001_new.txt \
-input ./hw3/train.txt \
-output ./hw3/output_b2
# save output
hdfs dfs -cat ./hw3/output_b2/* > cluster_1001.txt
hdfs dfs -rm -r ./hw3/output_b2