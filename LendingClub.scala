package com.huawei.LendingClubScala

import ml.dmlc.xgboost4j.scala.spark.{ XGBoost,XGBoostModel }
import org.apache.hadoop.conf.Configuration
import org.apache.hadoop.hbase.HBaseConfiguration
import org.apache.hadoop.hbase.client._
import org.apache.hadoop.hbase.mapreduce.TableInputFormat
import org.apache.hadoop.hbase.protobuf.ProtobufUtil
import org.apache.hadoop.hbase.util.{ Base64, Bytes }
import org.apache.spark.ml.classification.GBTClassificationModel
import org.apache.spark.ml.feature.LabeledPoint
import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.ml.regression.GBTRegressionModel
import org.apache.spark.{SparkConf, SparkContext }
import org.apache.spark.mllib.classification._
import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics
import org.apache.spark.mllib.evaluation.MulticlassMetrics
import org.apache.spark.mllib.feature.{ PCA, PCAModel}
import org.apache.spark.mllib.optimization.LogisticGradient
import org.apache.spark.mllib.optimization.SimpleUpdater
import org.apache.spark.mllib.tree.RandomForest
import org.apache.spark.mllib.tree.model.RandomForestModel
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.hive.HiveContext
import org.slf4j.LoggerFactory
import org.apache.spark.mllib.optimization.SquaredL2Updater
import org.apache.spark.sql.SQLContext
import org.apache.spark.sql.types.StringType
import org.apache.spark.sql.types.IntegerType
import org.apache.spark.sql.types.StructField
import org.apache.spark.sql.types.StructType
import org.apache.spark.sql.Row

import scala.tools.scalap.Main
import org.apache.spark.ml.feature.VectorSlicer
import org.apache.spark.mllib.feature.StandardScaler
object LendingClub {
  val log = LoggerFactory.getLogger(this.getClass)
  /**
   * hive查询数据
   */
  
  def queryHive(sc: SparkContext, sql: String): RDD[(LabeledPoint)]={
    val hiveContext = new HiveContext(sc)
    val hiveRDD = hiveContext.sql(sql).rdd
    val resvalue = hiveRDD.map { t =>
      var hivestr = t.toString.substring(1,t.toString.length - 1) //全量查询
      val cols = hivestr.split(",")
      val value = cols(cols.length -1)
      
      var label = 0
      if (value.equals("1") || value.equals("l")) {
        label = 1;
      } else {
        label = 0;
      }
      val parts = new Array[String](cols.length - 1) //全量查询
      for (i <- 0 until cols.length -1) { //全量查询
        parts(i) = cols(i) //全量查询
      }
      val features = Vectors.dense(parts.slice(0,parts.length).map(_.trim.toDouble))
      LabeledPoint(label,features)
    }
    return resvalue;
  }
  
  /**
   *  将Scan转换为String作为设置参数输入
   *  
   *  @param scan
   *  @return
   */
  
  //def convertScanToString(scan: Scan) = {
  //  val proto = ProtobufUtil.toScan(scan)
  //  Base64.encodeBytes(proto.toByteArray)
  //}
  
  //数据标准化后再进行模型训练
  def getLRLBFGSModelNormal(input: RDD[org.apache.spark.mllib.regression.LabeledPoint],numClasses: Int): LogisticRegressionModel = {
    val model = new LogisticRegressionWithLBFGS() //采用
    GBTClassificationModel
    GBTRegressionModel
    val modelinput = input.map {
      point =>
        val featarrs = point.features.toArray
        val label = point.label
        val features = Vectors.dense(featarrs.slice(1,featarrs.length))
        val featuresMLlib = org.apache.spark.mllib.linalg.Vectors.fromML(features)
        org.apache.spark.mllib.regression.LabeledPoint(label,featuresMLlib)
    }
    val scaler = new StandardScaler(true,true).fit(modelinput.map(f => f.features))
    
    val modelinput1 = input.map {
      point =>
        val featarrs = point.features.toArray
        val label = point.label
        val features = Vectors.dense(featarrs.slice(1,featarrs.length))
        val featuresMLlib = org.apache.spark.mllib.linalg.Vectors.fromML(features)
        org.apache.spark.mllib.regression.LabeledPoint(label,featuresMLlib)
    }
    model.setNumClasses(numClasses)
    val trained = model.run(modelinput1)
    return trained;
  }
  
  //数据分析
  def getLRLBFGSModel(input:RDD[org.apache.spark.mllib.regression.LabeledPoint], numClasses:Int):LogisticRegressionModel = {
    val model = new LogisticRegressionWithLBFGS()//采用
    GBTClassificationModel
    GBTRegressionModel
  
    val modelinput = input.map {
      point =>
        val featarrs = point.features.toArray
        val label = point.label
        val features = Vectors.dense(featarrs.slice(1,featarrs.length))
        val featuresMLlib = org.apache.spark.mllib.linalg.Vectors.fromML(features)
        org.apache.spark.mllib.regression.LabeledPoint(label,featuresMLlib)
    }
    model.setNumClasses(numClasses)
    val trained = model.run(modelinput)
    return trained;
  }
  
  def getLRSGD(input: RDD[org.apache.spark.mllib.regression.LabeledPoint],numClasses: Int): LogisticRegressionModel = {
    val model = new LogisticRegressionWithSGD()
    
    val modelinput = input.map{
      point =>
        val featarrs = point.features.toArray
        val label = point.label
        val features = Vectors.dense(featarrs.slice(1,featarrs.length))
        val featuresMLlib = org.apache.spark.mllib.linalg.Vectors.fromML(features)
        org.apache.spark.mllib.regression.LabeledPoint(label,featuresMLlib)
    }
    model.optimizer.setNumIterations(500).setUpdater(new SquaredL2Updater()).setStepSize(0.0001)
    val trained = model.run(modelinput)
    return trained;
  }  
  //训练随机森林算法
  def getRandomForest(input:RDD[org.apache.spark.mllib.regression.LabeledPoint], numClasses: Int):RandomForestModel = {
    val categoricalFeaturesInfo = Map[Int,Int]()//用map储存类别（离散）特征及每个类特征对应值（类别）的数量
    val impurity = "gini"//纯度计算方法，用于信息增益的计算
    val number = 100//构建树的数量
    val maxDepth = 8//树的最大高度
    val maxBins = 81//用于分裂特征的最大划分数量
    val modelinput = input.map{
      point =>
        val featarrs = point.features.toArray
        val label = point.label
        val features = Vectors.dense(featarrs.slice(1,featarrs.length))
        val featuresMLlib = org.apache.spark.mllib.linalg.Vectors.fromML(features)
        org.apache.spark.mllib.regression.LabeledPoint(label,featuresMLlib)
    }
    val model = RandomForest.trainClassifier(modelinput, numClasses, categoricalFeaturesInfo, number,"auto",impurity,maxDepth,maxBins)
    
    return model
  }
  
  /**
  
  
   * 在训练构建RandomForestModel
   * @param model
   * @param data
   * @return
   */
  def getMetrics(model: RandomForestModel, data: RDD[org.apache.spark.mllib.regression.LabeledPoint]): MulticlassMetrics = {
    val predictionsAndLabels = data.map(example =>
      (model.predict(example.features), example.label))
    predictionsAndLabels.saveAsTextFile("/tmp/LR/Output"+
        System.currentTimeMillis())
    new MulticlassMetrics(predictionsAndLabels)
  }
  
  def getPCAData(input: RDD[org.apache.spark.mllib.regression.LabeledPoint]): Array[RDD[org.apache.spark.mllib.regression.LabeledPoint]]={
    val pca = new PCA(30).fit(input.map(_.features))
    val hiveRDDPCA = input.map(p => p.copy(features = pca.transform(p.features)))
    val hiveRDDSplit = hiveRDDPCA.randomSplit(Array(0.7,0.3))
    return hiveRDDSplit
  }
 

  def main(args: Array[String]): Unit = {
    val conf = new SparkConf().setMaster("local[*]").setAppName("LendingClub")
    val sc = new SparkContext(conf)
    //val hivesql_train = "select id,member_id,loan_amnt,int_rate,grade,dti,open_acc,out_prncp,total_pymnt,total_rec_late_fee,recoveries,last_pymnt_amnt from default.training"
    //val hivesql_test = "select id,member_id,loan_amnt,int_rate,grade,dti,open_acc,out_prncp,total_pymnt,total_rec_late_fee,recoveries,last_pymnt_amnt from default.testing"
    
    //手动选择属性
    val hivesql_train = "select id,loan_amnt,int_rate,out_prncp,total_pymnt,total_rec_late_fee,recoveries,collection_recovery_fee,last_pymnt_amnt,home_ownership_ANY,home_ownership_NONE,home_ownership_OTHER,purpose_car,purpose_educational,purpose_wedding,application_type_INDIVIDUAL,initial_list_status_f,pymnt_plan_n,loan_status from default.training"
    val hivesql_test = "select id,loan_amnt,int_rate,out_prncp,total_pymnt,total_rec_late_fee,recoveries,collection_recovery_fee,last_pymnt_amnt,home_ownership_ANY,home_ownership_NONE,home_ownership_OTHER,purpose_car,purpose_educational,purpose_wedding,application_type_INDIVIDUAL,initial_list_status_f,pymnt_plan_n,loan_status from default.testing"
    val hiveRDD_train: RDD[LabeledPoint] = queryHive(sc, hivesql_train)

    val hiveRDD_test: RDD[LabeledPoint] = queryHive(sc, hivesql_test)
    val hiveRDDtrain = hiveRDD_train.map { key => 
      {
      val features = key.features
      val featuresMLlib = org.apache.spark.mllib.linalg.Vectors.fromML(features)
      val labels = key.label
      org.apache.spark.mllib.regression.LabeledPoint(labels, featuresMLlib)
      }
    }

    val hiveRDDtest = hiveRDD_test.map { key => 
      {
      val features = key.features
      val featuresMLlib = org.apache.spark.mllib.linalg.Vectors.fromML(features)
      val labels = key.label
      org.apache.spark.mllib.regression.LabeledPoint(labels, featuresMLlib)
      }
    }
    
    
    //PCA选择属性
//    val hivesql_train = "select * from default.training"
//    val hiveRDD_train: RDD[LabeledPoint] = queryHive(sc, hivesql_train)
//    val hiveRDDtrain = hiveRDD_train.map { key => 
//      {
//      val features = key.features
//      val featuresMLlib = org.apache.spark.mllib.linalg.Vectors.fromML(features)
//      val labels = key.label
//      org.apache.spark.mllib.regression.LabeledPoint(labels, featuresMLlib)
//      }
//    }
//    val PCAsplit = getPCAData(hiveRDDtrain)
//    val train=PCAsplit(0)
//    val test=PCAsplit(1)
    

/*    val trainmodel = getRandomForest(hiveRDDtrain,2);
    
    val labelAndPreds = hiveRDDtest.map { point =>
      val featarr = point.features.toArray
      val featurestest = org.apache.spark.mllib.linalg.Vectors.dense(featarr.slice(1, featarr.length))
      val id = featarr.apply(0).toInt
      val prediction = trainmodel.predict(featurestest)

      (point.label.toInt, prediction.toInt, featurestest, id)
    }*/
    
    //原数据没标准化的处理
    val trainmodel = getLRLBFGSModelNormal(hiveRDDtrain,2);
    
    val hivemodelRDDnorm = hiveRDDtest.map { point =>
      val featarr = point.features.toArray
      val featurestest = org.apache.spark.mllib.linalg.Vectors.dense(featarr.slice(1, featarr.length))
      featurestest
    }
    val testscaler = new StandardScaler(true,true).fit(hivemodelRDDnorm)
    
    val labelAndPreds = hiveRDDtest.map { point =>
      val featarr = point.features.toArray
      val featurestest = org.apache.spark.mllib.linalg.Vectors.dense(featarr.slice(1, featarr.length))
      val featurestestnorm = testscaler.transform(featurestest)
      val id = featarr.apply(0).toInt
      val prediction = trainmodel.predict(featurestestnorm)

      (point.label.toInt, prediction.toInt, featurestest, id)
    }

    
        
    val trainprecision = labelAndPreds.filter(r => r._1 == r._2).count.toDouble / labelAndPreds.count
    val predfeatures = hiveRDDtest.map { point =>
      val featarr = point.features.toArray
      val featurestest = org.apache.spark.mllib.linalg.Vectors.dense(featarr.slice(1, featarr.length))
      featurestest
    }
    val prediction = trainmodel.predict(predfeatures)
    val predictionAndLabels = prediction.zip(hiveRDDtest.map(_.label))
    val metrics = new BinaryClassificationMetrics(predictionAndLabels)
    val auROC = metrics.areaUnderROC()
    val areaUnderPR = metrics.areaUnderPR()
    println("Area under ROC = " + auROC)
    println("Area under Precision = " + trainprecision)
    println("Area under precision-recall = " + areaUnderPR)
    
    //输出csv文件
    val sqlContext = new SQLContext(sc)
    val schemaString = "id,label,predlabel,loan_amnt,int_rate,out_prncp,total_pymnt,total_rec_late_fee,recoveries,collection_recovery_fee,last_pymnt_amnt,home_ownership_ANY,home_ownership_NONE,home_ownership_OTHER,purpose_car,purpose_educational,purpose_wedding,application_type_INDIVIDUAL,initial_list_status_f,pymnt_plan_nm"
    val schema = StructType(schemaString.split (",").map(fieldName => StructField(fieldName,StringType,true)))

    val rowRDD = labelAndPreds.map(p => Row(p._4.toString(), p._1.toString(), p._2.toString(), p._3.apply(0).toString(), p._3.apply(1).toString(), p._3.apply(2).toString(), p._3.apply(3).toString(), p._3.apply(4).toString(), p._3.apply(5).toString(), p._3.apply(6).toString(), p._3.apply(7).toString(), p._3.apply(8).toString(), p._3.apply(9).toString(), p._3.apply(10).toString(), p._3.apply(11).toString(), p._3.apply(12).toString(), p._3.apply(13).toString(), p._3.apply(14).toString(), p._3.apply(15).toString(), p._3.apply(16).toString()))
    val labelAndPredsSchemadf = sqlContext.createDataFrame(rowRDD, schema)
    labelAndPredsSchemadf.write.option("header", "true").csv("/tmp/LR/Output/result" + System.currentTimeMillis())
    //输出结束
    
    
    println("done!")
    sc.stop()
  }
}