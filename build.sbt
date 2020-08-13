name := "PageRankTwitterAna"

version := "0.1"

scalaVersion := "2.11.8"

val sparkVersion = "2.4.2"


libraryDependencies ++= Seq(
  "org.apache.spark" %% "spark-core" % sparkVersion,
  "org.apache.spark" %% "spark-sql" % sparkVersion,
  "org.apache.spark" %% "spark-mllib" % sparkVersion,
  "org.apache.spark" %% "spark-streaming" % sparkVersion,
  "org.apache.spark" %% "spark-hive" % sparkVersion,
  "org.apache.spark" % "spark-graphx_2.11" % sparkVersion % "provided"
)