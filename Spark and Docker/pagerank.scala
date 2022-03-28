import org.apache.spark.{SparkContext, SparkConf}

object pagerank{

    def main(): Unit = {
        val ITERATIONS = 10
        val conf = new SparkConf().setAppName("pagerank").setMaster("local")
        val sc = new SparkContext(conf)
        val raw = sc.textFile("./hw3/web-Google.txt")
        // filter the headers
        val links = raw.map{ 
        s => val parts = s.split("\\s+")
        (parts(0), parts(1))
        }.distinct().groupByKey()

        var ranks = links.mapValues(v => 1.0).distinct()
        // collect garbage
        raw.unpersist()

        for (i <- 1 to ITERATIONS) {
            val contribs = links.join(ranks).flatMap{
            case(url, (links, rank)) =>
            links.map(dest => (dest, rank/(links.size).toDouble))
        }
        ranks = contribs.reduceByKey(_+_)
        .mapValues(0.15 + 0.85 * _)
        }
        ranks.saveAsTextFile("q2_a")
        sc.stop()
    }   
}

pagerank.main()

