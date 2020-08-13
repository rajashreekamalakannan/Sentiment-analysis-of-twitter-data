import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.sql.{SaveMode, SparkSession}
object PageRank{

  def main(args: Array[String]): Unit = {

    // if argument parameter missing print the proper argument
    if (args.length != 3) {
      println("Usage: PageRank APSD -> Inputfile NoOfIterations OutputDirectory")
    }


    val sparkConf = new SparkConf().setAppName("Page Rank")
    val sparkContext = new SparkContext(sparkConf)
    val sqlContext = new SparkSession.Builder()
      .config(sparkConf)
      .getOrCreate()

    //Read the command line argument
    val inputFile = args(0)
    val noOfIterations = args(1).toInt
    val outputDirectory = args(2)


    import sqlContext.implicits._

    // read the input file and create Data frame
    val dataSource = sqlContext.read.format("csv").option("header","true").option("inferSchema","true").option("delimiter", ",").load(inputFile).select("ORIGIN", "DEST")
    // Group Destinations by airport code
    val airportGroupByOrigin = sparkContext.parallelize(dataSource.map(row => (row.getString(0), row.getString(1))).collect()).groupByKey()
    // Initial Page Rank 10 for all origin
    val noOfNode = airportGroupByOrigin.count()
    var pageRankMap = airportGroupByOrigin.mapValues(v => 10.0)
    //  no of  iterations
    for (i <- 1 to noOfIterations) {
      val distributeMap = airportGroupByOrigin.join(pageRankMap).values.flatMap{ case (airports, pageRank) =>
        airports.map(airport => (airport, pageRank/airports.size))
      }
      pageRankMap = distributeMap.reduceByKey(_ + _).mapValues((0.15/noOfNode) + 0.85 * _)
    }

    //sort and save the result to the output directory
    val result = pageRankMap.sortBy(_._2, false).coalesce(1).saveAsTextFile(outputDirectory)
  }

}