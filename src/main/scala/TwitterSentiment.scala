import org.apache.spark.SparkConf
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.feature.{HashingTF, StopWordsRemover, StringIndexer, Tokenizer}
import org.apache.spark.ml.tuning.{CrossValidator, ParamGridBuilder}
import org.apache.spark.mllib.evaluation.MulticlassMetrics
import org.apache.spark.sql.{Row, SaveMode, SparkSession}

object TwitterSentiment {

  def main(args: Array[String]): Unit = {
    // if argument parameter missing print the proper argument
    if (args.length != 2) {
      println("Usage: Twitter Sentiment Analysis -> Inputfile OutputFile")
    }


    val sparkConf = new SparkConf().setAppName("Twitter Sentiment Analysis")
    val sqlContext = new SparkSession.Builder()
      .config(sparkConf)
      .getOrCreate()

    //Read the command line argument
    val inputFile = args(0)
    val outputFile = args(1)
    import sqlContext.implicits._
    // read the input file and create Data frame
    val dataSource = sqlContext.read.option("header","true").option("inferSchema","true").csv(inputFile)

    val dataSource_filtered = dataSource.filter(dataSource.col("text").isNotNull).drop("tweet_coord", "airline_sentiment_gold", "negativereason_gold")

    //split data source into training and test
    val Array(training, test) = dataSource_filtered.randomSplit(Array(0.8, 0.2))

    //create tokenizer pipeline
    val tokenizer = new Tokenizer().setInputCol("text")
      .setOutputCol("tokenized_words")

    //create stopword remover
    val remover = new StopWordsRemover()
      .setInputCol(tokenizer.getOutputCol)
      .setOutputCol("stop_filtered")


    val hashingTF = new HashingTF().setNumFeatures(1000)
      .setInputCol(remover.getOutputCol)
      .setOutputCol("features")

    //indexer
    val indexer = new StringIndexer().setInputCol("airline_sentiment")
      .setOutputCol("label")

    //logistic regression
    val logRig = new LogisticRegression().setMaxIter(15)
      .setRegParam(0.1)
   //add all tasks to pipeline
    val pipeline = new Pipeline().setStages(Array(tokenizer,remover, hashingTF, indexer, logRig))

    //param grid builder
    val paramGridBuilder = new ParamGridBuilder()
      .addGrid(hashingTF.numFeatures, Array(10, 100, 1000))
      .addGrid(logRig.regParam, Array(0.1, 0.01))
      .build()

    //create cross validator with estimator,evaluator,param
    val crossValidator = new CrossValidator()
      .setEstimator(pipeline)
      .setEvaluator(new MulticlassClassificationEvaluator)
      .setEstimatorParamMaps(paramGridBuilder)
      .setNumFolds(5)

    //filt the model
    val crossValidationModel = crossValidator.fit(training)

    //get prediction
    val prediction = crossValidationModel.bestModel.transform(test).select("prediction", "label").map {
      case Row(label: Double, prediction: Double) => (prediction, label)
    }
    val multiClassMetrics = new MulticlassMetrics(prediction.rdd)

    //get all the metrix for the model
    val resultDF = Seq(("Weighted Precision", multiClassMetrics.weightedPrecision),
      (("Weighted Recall", multiClassMetrics.weightedRecall)),
      ("Weighted F Measure", multiClassMetrics.weightedFMeasure),
      ("Weighted True Positive Rate", multiClassMetrics.weightedTruePositiveRate),
      ("Weighted False Positive Rate", multiClassMetrics.weightedFalsePositiveRate)).toDF("Metric","value")

    //write to csv file
    resultDF.coalesce(1)
      .write
      .mode(SaveMode.Overwrite)
      .format("csv")
      .option("header", "true")
      .save(outputFile)
  }
}