import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions.countDistinct
import org.apache.spark.sql.functions._

object sparkSQL_b{
    def main(args: Array[String]){
        val input = "./hw3/Crime_Incidents_in_2013.csv"
        val spark = SparkSession
        .builder()
        .appName("crimeInc2")
        .config("spark.submit.deployMode","cluster")
        .master("yarn-client")
        .getOrCreate()
        import spark.implicits._
        val df = spark.read.option("header","true").csv(input)

        df.createOrReplaceTempView("crime")
        var crimeDF = spark.sql(
            "SELECT CCN, REPORT_DAT, OFFENSE, METHOD, END_DATE, DISTRICT FROM crime")
        crimeDF = crimeDF.na.drop("any")
        // the number of each offenses
        val offenceCount = crimeDF.groupBy("OFFENSE").count().orderBy(desc("count"))
        val timeCount = crimeDF.groupBy("REPORT_DAT").count().orderBy(desc("count"))
        
        offenceCount.write.csv("./hw3/q3_b_offenceCount.csv")
        timeCount.write.csv("./hw3/q3_b_timeCount.csv")
        spark.stop()
    }
}
