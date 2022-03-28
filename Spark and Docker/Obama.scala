import org.apache.spark.sql.{SparkSession,Row}
import spark.implicits._
import org.apache.spark.sql.types.{StructType,StructField,StringType}
import org.apache.spark.sql.functions.countDistinct

object sparkSQL_c{
    def main(args: Array[String]){
        val spark = SparkSession
        .builder()
        .appName("crimeInc3")
        .getOrCreate()
        val schema = StructType(
            StructField("REPORT_DAT", StringType, true) ::
            StructField("OFFENSE", StringType, true) :: Nil)
        var crimeDF = spark.createDataFrame(spark.sparkContext.emptyRDD[Row], schema)

        for(i <- 2010 to 2018){
            var input = s"./hw3/crimeIncidents/Crime_Incidents_in_$i.csv"
            val df = spark.read.option("header","true").csv(input).select("REPORT_DAT","OFFENSE")
            crimeDF = crimeDF.union(df)
        }
        crimeDF = crimeDF.na.drop("any")
        crimeDF = crimeDF.withColumn("YEAR", substring(col("REPORT_DAT"),1,4)).select("YEAR","OFFENSE")
        val crimeStats = crimeDF.groupBy("OFFENSE","YEAR").count()
        var crimeYear = crimeDF.groupBy("YEAR").count()
        crimeYear = crimeYear.withColumnRenamed("count","yearCount").withColumnRenamed("YEAR","YEAR_1")
        var crimeStats2 = crimeStats.join(crimeYear, crimeStats("YEAR") === crimeYear("YEAR_1"),"left")
        crimeStats2 = crimeStats2.select("OFFENSE","YEAR","count","yearCount")
                                 .withColumn("crimeRate",col("count")/col("yearCount"))
                                 .where(crimeStats2("OFFENSE")==="ASSAULT W/DANGEROUS WEAPON")
                                 .select("OFFENSE","YEAR","crimeRate")
                                 .orderBy(desc("YEAR"))
        
        crimeStats2.write.csv("./hw3/q3_c")
        spark.stop()
    }
}
