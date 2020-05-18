package person.zhanghan.ai.dm.model

import javax.xml.transform.stream.StreamResult
import ml.dmlc.xgboost4j.scala.spark.{XGBoostClassificationModel, XGBoostClassifier}
import org.apache.hadoop.conf.Configuration
import org.apache.hadoop.fs.permission.{FsAction, FsPermission}
import org.apache.hadoop.fs.{FileSystem, Path}
import org.apache.log4j.{Level, Logger}
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.types.{DoubleType, IntegerType}
import org.jpmml.model.JAXBUtil
import org.jpmml.sparkml.PMMLBuilder

object Xgboost2 {

  def main(args: Array[String]): Unit = {
    Logger.getLogger("org.apache.spark").setLevel(Level.ERROR)
    val spark = SparkSession
      .builder()
      .master("local[*]")
      .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
      .config("spark.rdd.compress", "true")
      .getOrCreate()

    //
    import org.apache.spark.sql.functions._
    val newInput = spark.read.option("header", "true").csv("data/iris.csv")
      .select(
        col("1").cast(DoubleType),
        col("2").cast(DoubleType),
        col("3").cast(DoubleType),
        col("4").cast(DoubleType),
        col("class").cast(IntegerType)
      )
    newInput.show()

    //
    val vectorAssembler = new VectorAssembler()
      .setInputCols(Array("1", "2", "3", "4"))
      .setOutputCol("features")

    //
    val xgbParam = Map(
      "eta" -> 0.1f,
      "max_depth" -> 3,
      "objective" -> "binary:logistic",
      "num_round" -> 3,
      "num_workers" -> 2
    )
    val xgbClassifier = new XGBoostClassifier(xgbParam).setFeaturesCol("features").setLabelCol("class").setProbabilityCol("probabilities")
    val pipeline = new Pipeline().setStages(Array(vectorAssembler, xgbClassifier))
    val model = pipeline.fit(newInput)

    //
    val pmml = new PMMLBuilder(newInput.schema, model).build
    val targetFile = "person/zhanghan/temp/pmml"
    val conf = new Configuration();//加载配置文件
    conf.set("fs.hdfs.impl.disable.cache","true")
    val fs = FileSystem.get(conf);//初始化文件系统
    val permission = new FsPermission(FsAction.ALL, FsAction.ALL, FsAction.ALL);
    val fs2 = FileSystem.create(fs, new Path(targetFile), permission)
    JAXBUtil.marshalPMML(pmml, new StreamResult(fs2))

    //
    val predictResult = model.transform(newInput)
    predictResult.show(false)
    val xgBoostClassificationModel = model.stages(1).asInstanceOf[XGBoostClassificationModel]
    println(xgBoostClassificationModel.extractParamMap)
    val evaluator = new BinaryClassificationEvaluator().setLabelCol("class").setRawPredictionCol("probabilities")
    val aucArea = evaluator.evaluate(predictResult)
    System.out.println("auc is :" + aucArea)
    model.write.overwrite.save("person/zhanghan/temp/bin")
  }
}
