### Homework2 Report

[TOC]

#### Question 1

##### a) Bonus: Install Pig

We assume the Hadoop cluster (Java, Hadoop environment) is established according to IEMS5709(T1)/IERG4300 Homework0.

###### Download and unzip

```shell
wget https://dlcdn.apache.org/pig/pig-0.16.0/pig-0.17.0.tar.gz
wget https://dlcdn.apache.org/pig/pig-0.16.0/pig-0.17.0-src.tar.gz
tar zxvf pig-0.17.0-src.tar.gz 
tar zxvf pig-0.17.0.tar.gz 
# move pig resource files under hadoop-pig
mv pig-0.17.0-src.tar.gz/* /home/Hadoop/Pig/
```

###### Setup Environment

```shell
export PIG_HOME = /home/Hadoop/Pig
export PATH  = $PATH:/home/Hadoop/pig/bin
export PIG_CLASSPATH = $HADOOP_HOME/conf
```

###### Configuration

```shell
# setup the logging,performance tuning and other details
pig -h properties 
```

##### b) Upload and join

###### Download dataset to local machine

```shell
wget http://storage.googleapis.com/books/ngrams/books/googlebooks-eng-all-1gram-20120701-a.gz
wget http://storage.googleapis.com/books/ngrams/books/googlebooks-eng-all-1gram-20120701-b.gz
gzip -d *.gz
```

###### Upload dataset to HDFS

```shell
hdfs dfs -mkdir ./hw2
hdfs dfs -copyFromLocal ./* ./hw2
hdfs dfs -ls ./hw2
```

![1646030579944](D:\document\CUHK\Big Data\homework\hw2\pics\1646030579944.png)

###### Join the two tables

1. Start the Pig Grunt Shell

   ```shell
   pig –x tez
   cd hdfs://dicvmd2.ie.cuhk.edu.hk:8020/user/s1155161048/hw2
   ```

2. Load the data

   ```sql
   tableA = LOAD './hw2/googlebooks-eng-all-1gram-20120701-a' USING PigStorage('\t') as (bigram: chararray, year: int, match_count: int, volume_count: int);
   tableB = LOAD './hw2/googlebooks-eng-all-1gram-20120701-b' USING PigStorage('\t') as (bigram: chararray, year: int, match_count: int, volume_count: int);
   ```

3. Join (Union) the two tables

   ```SQL
   allGram = UNION tableA, tableB;
   ```

4. Verify the relation (for me to check the correctness)

   ```
   dump allGram
   ```

![1646032610185](D:\document\CUHK\Big Data\homework\hw2\pics\1646032610185.png)

![1646032540458](D:\document\CUHK\Big Data\homework\hw2\pics\1646032540458.png)

5. Save to HDFS

   ```SQL
   STORE allGram INTO './hw2/allGram' USING PigStorage (',');
   ```

![1646033823164](D:\document\CUHK\Big Data\homework\hw2\pics\1646033823164.png)

##### c) Average occurrences

Expected fields of output:

bigram avgCount

I use tmux to manage screen sessions, 'tmux new -s pig', 'tmux a -t pig' and 'tmux detach'.

###### Create a pig script file

```sql
/*Since the bigram-year pair is unique, we can directly group by bigram*/

allGram = LOAD './hw2/allGram/*' USING PigStorage(',') as (bigram: chararray, year: int, match_count: int, volume_count: int);

gramGroup = group allGram by bigram;

gramAvg1 = foreach gramGroup generate group as bigram, AVG(allGram.match_count) as avgCount;

gramAvg2 = ORDER gramAvg1 BY avgCount DESC;

STORE gramAvg2 INTO './hw2/gramAvgCount2' USING PigStorage (',');
```

###### Execute pig script

```shell
hdfs dfs -copyFromLocal ./c1_avgCount.pig ./hw2
pig -x tez hdfs://dicvmd2.ie.cuhk.edu.hk:8020/user/s1155161048/hw2/c1_avgCount.pig
```

Time consumed is 9 min 3s and 657ms.

![1646055862246](D:\document\CUHK\Big Data\homework\hw2\pics\1646055862246.png)

##### d) Top 20

```shell
hdfs dfs -get ./hw2/gramAvgCount2
# I list top 30 records
head -n 30 ./gramAvgCount2/part-v005-o000-r-00000
```

![1646630285167](D:\document\CUHK\Big Data\homework\hw2\pics\1646630285167.png)

#### Question 2

##### a) Bonus: Install Hive

We assume the Hadoop cluster (Java, Hadoop environment) is established according to IEMS5709(T1)/IERG4300 Homework0. I list the critical commands.

The hive-2.3.8 is no longer on apache, hive-2.3.9 is the only option.

<img src="D:\document\CUHK\Big Data\homework\hw2\pics\1646711928910.png" alt="1646711928910" style="zoom:50%;" />

###### Download and unzip

```shell
wget https://dlcdn.apache.org/hive/hive-2.3.9/
tar zxvf apache-hive-0.14.0-bin.tar.gz
# move hive source file to corresponding directory
mv apache-hive-0.14.0-bin /usr/local/hive
```

###### Setup environment

```shell
export HIVE_HOME=/usr/local/hive
export PATH=$PATH:$HIVE_HOME/bin
export CLASSPATH=$CLASSPATH:/usr/local/Hadoop/lib/*:.
export CLASSPATH=$CLASSPATH:/usr/local/hive/lib/*:.
```

Execute bash file.

```shell
source ~/.bashrc
```

###### Configure Hive

```shell
cd $HIVE_HOME/conf
cp hive-env.sh.template hive-env.sh
```

Edit hive-env.sh by adding:

```shell
export HADOOP_HOME=/usr/local/hadoop
```

And other configurations:

![1646712778358](C:\Users\Jaden\AppData\Roaming\Typora\typora-user-images\1646712778358.png)

Create the /tmp folder and a separate Hive folder in HDFS. Here, I use the /user/hive/warehouse folder. Hive will store its tables on HDFS and those locations needs to be bootstrapped:

```
chmod g+w
$HADOOP_HOME/bin/hadoop dfs -mkdir /tmp 
$HADOOP_HOME/bin/hadoop dfs -mkdir /user/hive/warehouse
$HADOOP_HOME/bin/hadoop dfs -chmod g+w /tmp 
$HADOOP_HOME/bin/hadoop dfs -chmod g+w /user/hive/warehouse
```

##### b1) Upload and union

1. Load dataset

```sql
CREATE DATABASE hw2;
USE hw2;
CREATE TABLE tableA (bigram string, year int, match_count int, volume_count int) ROW FORMAT DELIMITED FIELDS TERMINATED BY '\t';
CREATE TABLE tableB (bigram string, year int, match_count int, volume_count int) ROW FORMAT DELIMITED FIELDS TERMINATED BY '\t';

LOAD DATA LOCAL INPATH './googlebooks-eng-all-1gram-20120701-a' INTO TABLE tableA;
LOAD DATA LOCAL INPATH './googlebooks-eng-all-1gram-20120701-b' INTO TABLE tableB;
```

2. Union the two tables

```SQL
CREATE TABLE tableAll AS SELECT * FROM tableA UNION SELECT * FROM tableB;
```

![1646396408002](D:\document\CUHK\Big Data\homework\hw2\pics\1646396408002.png)

##### b2) Average occurrences

```sql
CREATE TABLE tableAVG AS SELECT bigram, AVG(match_count) AS match_count_avg FROM tableAll GROUP BY bigram;
```

![1646446335982](D:\document\CUHK\Big Data\homework\hw2\pics\1646446335982.png)

##### b3) Top20  

```SQL
CREATE TABLE tableAVGTop AS SELECT * FROM tableAVG ORDER BY match_count_avg DESC;
SELECT * FROM tableAVGTop LIMIT 20;
```

![1646447174485](D:\document\CUHK\Big Data\homework\hw2\pics\1646447174485.png)

##### Time Comparison

| Time |  Union   | Average  |
| :--: | :------: | :------: |
| Pig  |   349s   | 543.657s |
| Hive | 504.589s | 285.597s |

#### Question 3

##### a) Output the number of movies they both watched

###### Number of movie they both watched

`Q3_a_3.pig / Q3_a_4.pig`

```sql
tableLarge = LOAD './hw2/movielens_large_updated.csv' USING PigStorage(',') as (userID: int, movieID: int);
tableSmall = LOAD './hw2/movielens_small.csv' USING PigStorage(',') as (userID: int, movieID: int);
userMovieAll_0 = UNION tableLarge, tableSmall;
userMovieAll = DISTINCT userMovieAll_0;

-- get the number of movies they both watched S(A & B)
movielens_grpd = GROUP userMovieAll BY movieID;
movielens_grpd_dbl = FOREACH movielens_grpd GENERATE group, userMovieAll.userID AS userID, userMovieAll.userID AS userID2;
cowatch = FOREACH movielens_grpd_dbl GENERATE FLATTEN(userID) as userID, FLATTEN(userID2) as userID2;
cowatch_filtered = FILTER cowatch BY userID < userID2;

D = FOREACH (GROUP cowatch_filtered BY (userID, userID2)) 
GENERATE FLATTEN(group) AS (userID, userID2), COUNT(cowatch_filtered.userID) AS both_watch;

result = ORDER D BY both_watch DESC;

STORE result INTO './hw2/both_watch' USING PigStorage (',');
```

```powershell
pig -x tez hdfs://dicvmd2.ie.cuhk.edu.hk:8020/user/s1155161048/hw2/Q3_a_3.pig
pig -x tez hdfs://dicvmd2.ie.cuhk.edu.hk:8020/user/s1155161048/hw2/Q3_a_4.pig
```

###### Find top 10 user pairs with most both watched movies

```shell
hdfs dfs -get ./hw2/both_watch # all dataset union
hdfs dfs -get ./hw2/both_watch_3 # dataset 3
hdfs dfs -get ./hw2/both_watch_4 # dataset 4

head -n 10 ./both_watch/part-v007-o000-r-00000 # all dataset top 10
head -n 10 ./both_watch_3/part-v006-o000-r-00000 # dataset 3 top 10
head -n 10 ./both_watch_4/part-v006-o000-r-00000 # dataset 4 top 10
```

![1646660364607](D:\document\CUHK\Big Data\homework\hw2\pics\1646660364607.png)

![1646660218252](D:\document\CUHK\Big Data\homework\hw2\pics\1646660218252.png)

![1646660820281](D:\document\CUHK\Big Data\homework\hw2\pics\1646660820281.png)

##### b) Top-K most similar users

###### Calculate the similarity

`similarity_3.pig / similarity_4.pig`

```sql
tableLarge = LOAD './hw2/movielens_large_updated.csv' USING PigStorage(',') as (userID: int, movieID: int);
-- tableSmall = LOAD './hw2/movielens_small.csv' USING PigStorage(',') as (userID: int, movieID: int);
-- userMovieAll_0 = UNION tableLarge, tableSmall;
userMovieAll = DISTINCT tableLarge;
-- userMovieAll = DISTINCT tableSmall;
-- STEP 1. sum the number of movie 2 users watched. S(A) + S(B)

userMovieList = FOREACH (GROUP userMovieAll BY userID)
GENERATE group AS userID, COUNT(userMovieAll.movieID) as movieList;

userMovieList_2 = FOREACH userMovieList GENERATE userID AS userID2, movieList AS movieList2;

pairs = CROSS userMovieList, userMovieList_2;

B = FILTER pairs BY userID < userID2;

C = FOREACH B GENERATE userID AS userID, userID2 AS userID2,
(movieList + movieList2) AS addup;

-- STEP 2. -- get the number of movies they both watched S(A & B)

-- directly load from question3-a
-- D = LOAD './hw2/both_watch_3/part*' USING PigStorage(',') as (userID: int, userID2: int, both_watch: long);

D = LOAD './hw2/both_watch_4/part*' USING PigStorage(',') as (userID: int, userID2: int, both_watch: long);

-- join the table
E = JOIN C BY (userID, userID2) LEFT OUTER, D BY (userID, userID2);

-- Fill null with 0
F = FOREACH E GENERATE $0 AS userID, $1 AS userID2, $2 AS addup, 
($5 is null ? 0 : $5) AS both_watch;

-- STEP 3. S(A || B) = S(A) + S(B) - S(A & B), similarity = S(A & B) / S(A || B)

simi_data = FOREACH F GENERATE $0 AS userID, $1 AS userID2, ((float)$3 / (float)($2 - $3)) AS similarity;

-- STORE result INTO './hw2/both_watched1' USING PigStorage (',');
-- id1, id2, 
STORE simi_data INTO './hw2/similarity_table_4' USING PigStorage (',');
```

###### Execute

Calculate similarity between user pairs on the dataset [3] and [4] with `similarity.pig` above, and get `similarity_table_3` and `similarity_table_4`.

The scripts for running this part is:

```shell
pig -x tez hdfs://dicvmd2.ie.cuhk.edu.hk:8020/user/s1155161048/hw2/similarity_3.pig
pig -x tez hdfs://dicvmd2.ie.cuhk.edu.hk:8020/user/s1155161048/hw2/similarity_4.pig
```

###### i) Small dataset

`simi_ID_3.pig`

```sql
A_0 = LOAD './hw2/similarity_table_3/*' USING PigStorage(',') as (userID: int, userID2: int, similarity: float);

A = FOREACH A_0 GENERATE (chararray)$0 AS userID, (chararray)$1 AS userID2, $2 AS similarity;
A_ver = FOREACH A_0 GENERATE (chararray)$1 AS userID, (chararray)$0 AS userID2, $2 AS similarity;

-- match the user shares the same last 2 digits with my student ID: 1155161048.
B = FILTER (UNION A, A_ver) BY (userID MATCHES '.*48');

C = FOREACH (GROUP B BY userID) {
	top = TOP(3, 2, B);
	GENERATE group AS userID, FLATTEN(BagToTuple(top.userID2));
}

STORE C INTO './hw2/result_b_i' USING PigStorage(',');
```

###### ii) Large dataset

`simi_ID_4.pig`

```sql
A_1 = LOAD './hw2/similarity_table_4/*' USING PigStorage(',') as (userID: int, userID2: int, similarity: float);

A = FOREACH A_1 GENERATE (chararray)$0 AS userID, (chararray)$1 AS userID2, $2 AS similarity;
A_ver = FOREACH A_1 GENERATE (chararray)$1 AS userID, (chararray)$0 AS userID2, $2 AS similarity;

-- match the user shares the same last 4 digits with my student ID: 1155161048.
B = FILTER (UNION A, A_ver) BY (userID MATCHES '.*1048');

C = FOREACH (GROUP B BY userID) {
	top = TOP(3, 2, B);
	GENERATE group AS userID, FLATTEN(BagToTuple(top.userID2));
}

STORE C INTO './hw2/result_b_ii' USING PigStorage(',');
```

The overall script is:

```shell
pig -x tez hdfs://dicvmd2.ie.cuhk.edu.hk:8020/user/s1155161048/hw2/similarity_3.pig
pig -x tez hdfs://dicvmd2.ie.cuhk.edu.hk:8020/user/s1155161048/hw2/similarity_4.pig
pig -x tez hdfs://dicvmd2.ie.cuhk.edu.hk:8020/user/s1155161048/hw2/simi_ID_3.pig
pig -x tez hdfs://dicvmd2.ie.cuhk.edu.hk:8020/user/s1155161048/hw2/simi_ID_4.pig
```

###### Results

i)

![1646707554085](D:\document\CUHK\Big Data\homework\hw2\pics\1646707554085.png)

ii)

![1646707662262](D:\document\CUHK\Big Data\homework\hw2\pics\1646707662262.png)

##### c) Bonus: Use Hive to do b)

Start from the resource from question 3a, which is the both_watched table from `both_watched_3` and `both_watched_4`, format as: userID, userID2, number_of_both_watched_movies. Since the hive is not working on cluster, the scripts is as following.

###### Calculate similarity

```sql
CREATE DATABASE hw2;
USE hw2;

CREATE TABLE movielens (userID string, movieID int) ROW FORMAT DELIMITED FIELDS TERMINATED BY ',';
CREATE TABLE both_watch (userID string, userID2 string, both_watch int) ROW FORMAT DELIMITED FIELDS TERMINATED BY ',';

LOAD DATA INPATH './hw2/movielens_small.csv' INTO TABLE movielens;
LOAD DATA INPATH './hw2/both_watch_3/*' INTO TABLE both_watch;

-- LOAD DATA INPATH './hw2/movielens_large_updated.csv' INTO TABLE movielens;
-- LOAD DATA INPATH './hw2/both_watch_4/*' INTO TABLE both_watch;

-- CALCULATE THE NUMBER OF MOVIE WATCHED BY EACH USER
CREATE TABLE common_watch1 AS
SELECT userID, COUNT(DISTINCT movieID) AS mv_watch
FROM movielens
GROUP BY userID;

-- DUPLICATE THE TABLE FOR CROSS
CREATE TABLE common_watch2 AS 
SELECT * FROM common_watch1;

-- GET NUMBER OF MOVIES USERS WATCHED
CREATE TABLE common_watch AS
SELECT common_watch1.userID AS userID, common_watch2.userID AS userID2, 
(common_watch1.mv_watch +common_watch_2.mv_watch) AS cm_watch
FROM common_watch1 CROSS JOIN common_watch2;

-- GET THE SIMILARITY TABLE
CREATE TABLE similarity_table AS
SELECT c.userID AS userID1, c.userID2 AS userID2,
(CAST(b.both_watch AS FLOAT) / CAST((c.cm_watch - b.both_watch) AS FLOAT)) AS similarity
FROM both_watch b LEFT JOIN common_watch c 
ON c.userID = b.userID AND
c.userID2 = b.userID2
```

###### Find the top3 similar user

Use the similarity table to find the top3 similar users.

```sql
-- GET THE RANKING OF EACH PAIR'S SIMILARITY SCORE BY EACH USER
CREATE TABLE rank_simi AS
SELECT userID1, userID2, RANK() OVER (PARTITION BY userID1 ORDER BY similarity DESC) AS rank
FROM similarity_table
WHERE userID1 LIKE '%48'
-- WHERE userID1 LIKE '%1048'

-- PIVOT SOME ROWS INTO COLUMN
CREATE TABLE top3 AS
SELECT userID1,
CASE WHEN rank=1 THEN userID2 END AS similarID1,
CASE WHEN rank=2 THEN userID2 END AS similarID2,
CASE WHEN rank=3 THEN userID2 END AS similarID3
FROM rank_simi 
WHERE rank<4;
```

#### References

1. Pig tutorial https://www.tutorialspoint.com/apache_pig/apache_pig_reading_data.htm
2. Create table in Hive. https://phoenixnap.com/kb/hive-create-table
3. Hive tutorial https://www.tutorialspoint.com/hive/index.htm