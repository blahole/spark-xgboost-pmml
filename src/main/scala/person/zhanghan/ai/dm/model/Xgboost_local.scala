package person.zhanghan.ai.dm.model

import java.io.FileOutputStream

import ml.dmlc.xgboost4j.scala.spark.XGBoostClassifier
import org.apache.log4j.{Level, Logger}
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.types.DoubleType
import org.jpmml.model.PMMLUtil
import org.jpmml.sparkml.PMMLBuilder

object Xgboost_local {

  def main(args: Array[String]): Unit = {
    Logger.getLogger("org.apache.org.apache.spark").setLevel(Level.ERROR)
    val spark = SparkSession
      .builder()
      .master("local[*]")
      .config("org.apache.spark.serializer", "org.apache.org.apache.spark.serializer.KryoSerializer")
      .config("org.apache.spark.rdd.compress", "true")
      .getOrCreate()

    //
    import org.apache.spark.sql.functions._
    val newInput = spark.read.option("header", "true").csv("data/iris3.csv")
      .select(
        col("1").cast(DoubleType),
        col("2").cast(DoubleType),
        col("3").cast(DoubleType),
        col("4").cast(DoubleType),
        col("class").cast(DoubleType)
      )
    newInput.show()
    newInput.select("class").distinct().show()

    //
    val vectorAssembler = new VectorAssembler()
      .setInputCols(Array("1", "2", "3", "4"))
      .setOutputCol("features")

    //
    val xgbParam = Map(
      "eta" -> 0.1f,
      "max_depth" -> 3,
      "objective" -> "multi:softprob",
      "num_round" -> 3,
      "num_class" -> 3,
      "num_workers" -> 2
    )
    val xgbClassifier = new XGBoostClassifier(xgbParam).setFeaturesCol("features").setLabelCol("class").setProbabilityCol("probabilities")
    val pipeline = new Pipeline().setStages(Array(vectorAssembler, xgbClassifier))
    val model = pipeline.fit(newInput)

    //
    val pmml = new PMMLBuilder(newInput.schema, model).build
    val targetFile = "person/zhanghan/temp/xgboost.pmml"
    PMMLUtil.marshal(pmml, new FileOutputStream(targetFile))

  }
}
