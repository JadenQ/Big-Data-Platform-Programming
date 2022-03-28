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
	hdfs dfs -rm -r ./hw3/output_a1
	# run mapreduce
	hadoop jar /usr/lib/hadoop-mapreduce/hadoop-streaming.jar \
	-D mapred.output.key.comparator.class=org.apache.hadoop.mapred.lib.KeyFieldBasedComparator \
	-D mapred.job.name='Job_b1' \
	-D mapred.map.tasks=20 \
	-D mapred.reduce.tasks=10 \
	-D mapred.text.key.comparator.options=-k1n \
	-file mapper_a1.py -mapper mapper_a1.py \
	-file reducer_a1.py -reducer reducer_a1.py \
	-file centroid_p1.txt \
	-input ./hw3/train.txt \
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
