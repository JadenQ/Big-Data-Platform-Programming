# give permission
chmod +x mapper_a1.py reducer_a1.py
# remove old centroid file
# rm centroid_*

# generate the initial centroid file
python3 cen_gen_new.py 324
# iteration loop
for i in `seq 1 15`;
do
	# remove old output
	hdfs dfs -rm -r ./hw4/output_a1
	# run mapreduce
	hadoop jar /usr/lib/hadoop-mapreduce/hadoop-streaming.jar \
	-D mapred.output.key.comparator.class=org.apache.hadoop.mapred.lib.KeyFieldBasedComparator \
	-D mapred.job.name='Job_a1' \
	-D mapred.map.tasks=20 \
	-D mapred.reduce.tasks=5 \
	-D mapred.text.key.comparator.options=-k1n \
	-file mapper_a1.py -mapper mapper_a1.py \
	-file reducer_a1.py -reducer reducer_a1.py \
	-file centroid_324.txt \
	-input ./hw4/train_my_20_pca.txt \
	-output ./hw4/output_a1
	# save output
	hdfs dfs -cat ./hw4/output_a1/* > centroid_324_new.txt
	# save the final
	cat centroid_324_new.txt > centroid_324.txt
	# save all iterations as log
	cat centroid_324_new.txt >> centroid_324_all.txt
	# delete the output
	hdfs dfs -rm -r ./hw4/output_a1
done

# generate the initial centroid file
python3 cen_gen_new.py 1229
# iteration loop
for i in `seq 1 15`;
do
	# remove old output
	hdfs dfs -rm -r ./hw4/output_a1
	# run mapreduce
	hadoop jar /usr/lib/hadoop-mapreduce/hadoop-streaming.jar \
	-D mapred.output.key.comparator.class=org.apache.hadoop.mapred.lib.KeyFieldBasedComparator \
	-D mapred.job.name='Job_a1' \
	-D mapred.map.tasks=20 \
	-D mapred.reduce.tasks=5 \
	-D mapred.text.key.comparator.options=-k1n \
	-file mapper_a1.py -mapper mapper_a1.py \
	-file reducer_a1.py -reducer reducer_a1.py \
	-file centroid_1229.txt \
	-input ./hw4/train_my_20_pca.txt \
	-output ./hw4/output_a1
	# save output
	hdfs dfs -cat ./hw4/output_a1/* > centroid_1229_new.txt
	# save the final
	cat centroid_1229_new.txt > centroid_1229.txt
	# save all iterations as log
	cat centroid_1229_new.txt >> centroid_1229_all.txt
	# delete the output
	hdfs dfs -rm -r ./hw4/output_a1
done


# generate the initial centroid file
python3 cen_gen_new.py 1001
# iteration loop
for i in `seq 1 15`;
do
	# remove old output
	hdfs dfs -rm -r ./hw4/output_a1
	# run mapreduce
	hadoop jar /usr/lib/hadoop-mapreduce/hadoop-streaming.jar \
	-D mapred.output.key.comparator.class=org.apache.hadoop.mapred.lib.KeyFieldBasedComparator \
	-D mapred.job.name='Job_a1' \
	-D mapred.map.tasks=20 \
	-D mapred.reduce.tasks=5 \
	-D mapred.text.key.comparator.options=-k1n \
	-file mapper_a1.py -mapper mapper_a1.py \
	-file reducer_a1.py -reducer reducer_a1.py \
	-file centroid_1001.txt \
	-input ./hw4/train_my_20_pca.txt \
	-output ./hw4/output_a1
	# save output
	hdfs dfs -cat ./hw4/output_a1/* > centroid_1001_new.txt
	# save the final
	cat centroid_1001_new.txt > centroid_1001.txt
	# save all iterations as log
	cat centroid_1001_new.txt >> centroid_1001_all.txt
	# delete the output
	hdfs dfs -rm -r ./hw4/output_a1
done