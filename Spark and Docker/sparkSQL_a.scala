import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions._

object sparkSQL_a{
    def main(args: Array[String]){
        val input = "./hw3/Crime_Incidents_in_2013.csv"
        val spark = SparkSession
        .builder()
        .appName("crimeInc")
        .config("spark.submit.deployMode","cluster")
        .master("yarn-client")
        .getOrCreate()
        import spark.implicits._
        val df = spark.read.option("header","true").csv(input)
        df.createOrReplaceTempView("crime")
        var crimeDF = spark.sql(
            "SELECT CCN, REPORT_DAT, OFFENSE, METHOD, END_DATE, DISTRICT FROM crime")
        crimeDF = crimeDF.na.drop("any")
        crimeDF.write.csv("./hw3/q3_a")
        spark.stop()
    }
}
