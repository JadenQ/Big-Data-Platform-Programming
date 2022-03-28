import org.apache.spark.{SparkContext, SparkConf, HashPartitioner}

object adv_pagerank{

    def main(args: Array[String]){
        val iters = args(0).toInt
        val hash = args(1).toInt
        val input = args(2)
        val output = args(3)

        val conf = new SparkConf().setAppName("adv_pagerank").setMaster("yarn")
        val sc = new SparkContext(conf)
        val raw = sc.textFile(input)
        // filter the headers
        val links = raw.map{ 
        s => val parts = s.split("\\s+")
        (parts(0), parts(1))
        }.distinct().groupByKey().partitionBy(new HashPartitioner(hash))
        var ranks = links.mapValues(v => 1.0).distinct()
        // collect garbage
        raw.unpersist()
        for (i <- 1 to iters) {
            val contribs = links.join(ranks).flatMap{
            case(url, (links, rank)) =>
            links.map(dest => (dest, rank/(links.size).toDouble))
        }
        ranks = contribs.reduceByKey(_+_)
        .mapValues(0.15 + 0.85 * _)
        }
        ranks.saveAsTextFile(output)
        sc.stop()
    }
}