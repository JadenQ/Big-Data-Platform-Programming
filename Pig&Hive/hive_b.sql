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

-- GET THE RANKING OF EACH PAIR'S SIMILARITY SCORE BY EACH USER
CREATE TABLE rank_simi AS
SELECT userID1, userID2, RANK() OVER (PARTITION BY userID1 ORDER BY similarity DESC) AS rank
FROM similarity_table
WHERE userID1 LIKE '%48'

-- PIVOT SOME ROWS INTO COLUMN
CREATE TABLE top3 AS
SELECT userID1,
CASE WHEN rank=1 THEN userID2 END AS similarID1,
CASE WHEN rank=2 THEN userID2 END AS similarID2,
CASE WHEN rank=3 THEN userID2 END AS similarID3
FROM rank_simi 
WHERE rank<4;


