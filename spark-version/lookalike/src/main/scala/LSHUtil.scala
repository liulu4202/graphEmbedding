import java.io.Serializable

import org.apache.spark.SparkContext
import org.apache.spark.ml.feature.MinHashLSH
import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.functions.col
import org.apache.spark.sql.{SparkSession, functions}

import scala.collection.Map

/**
 * @Description: BucketedRandomProjectionLSH局部敏感哈希
 * 为欧几里得距离度量实现局部敏感哈希函数
 *
 * 输入是密集或稀疏向量，每个向量代表欧几里得距离空间中的一个点。 输出将是可配置尺寸的向量。 相同维中的哈希值由相同的哈希函数计算。
 **/
object LSHUtil extends Serializable {

  var spark: SparkSession = null
  var context: SparkContext = null
  var numPartition = 2000
  var numHashTables = 3
  var bucketLength = 2.0
  var threshold = 0.1
  var date = "0"

  def setup(spark: SparkSession, context: SparkContext, param: Map[String, String]): this.type = {
    this.spark = spark
    this.context = context
    this.numPartition = param("NUM_PARTITIONS").toInt
    this.numHashTables = param("NUM_HASHBUCKET").toInt
    this.bucketLength = param("NUM_BUCKETLENGTH").toInt
    this.threshold = param("LSH_THRESHOLD").toDouble
    this.date = param("DATE")

    this
  }

  def getScore(distance: Double): Double = {
    1.0 - distance
  }

  def MD5Handle(line: String): String = {
    Md5Util.evaluate(line)
  }

  def tpId(): String = {
    "0"
  }

  def eMode(): String = {
    "4"
  }

  def registerUDF() = {
    spark.udf.register("getScore", getScore _)
    spark.udf.register("md5Handle", MD5Handle _)
    spark.udf.register("tpId", tpId _)
    spark.udf.register("eMode", eMode _)

    this
  }

  def createSimilarity(item2vec: RDD[(String, Array[Double])]) = {

    val df = spark.createDataFrame(item2vec.map{ case (id, vector) =>
      (id, Vectors.dense(vector))
    }).toDF("id", "vec")

    val brp = new MinHashLSH().setInputCol("vec").setOutputCol("hashes").setSeed(123)
      //增大参数降低假阴性率，但以增加计算复杂性为代价
      .setNumHashTables(numHashTables)
      //每个哈希存储桶的长度（较大的存储桶可降低假阴性）
      //.setBucketLength(bucketLength)

    val model = brp.fit(df)

//    val keyline = "-0.17444848,0.13710116,0.036809597,-0.013806498,0.026735978,0.029838856,-0.020916983,0.0014806592,0.11776896,-0.10643057,-0.0895567,0.027163643,0.023766866,-0.03849636,-0.13616785,-0.029958485,-0.032577883,0.13655794,-0.04397826,-0.08548063,0.11460877,-0.01666804,0.008186754,0.021788638,-0.022014046,0.10938472,0.017683098,0.07670139,-0.12029679,-0.028684918,0.05113399,-0.113058016,0.052198954,0.14172015,0.04587456,-0.1036608,-0.06369583,0.01555655,-0.106039695,-0.0022123074,-0.14272456,0.031948663,-0.07656527,0.014764928,0.028740572,-0.020207867,0.012064475,-0.06299816,-0.07875888,0.046803568,0.045074813,-0.034257557,0.13523105,-0.09164691,-0.03331726,0.11805607,-0.014179055,-0.14654447,0.00920304,-0.007138808,-0.0027223688,0.02266644,0.056159787,0.0998705,-0.116545565,-0.16214822,-0.015079361,-0.041908678,0.021727538,-0.0066337953,-0.014094111,-0.0003349924,0.06409463,0.10041777,0.08123548,0.10157579,-0.13761082,-0.112167284,0.005857735,-0.12677114,0.114418454,-0.02858712,-0.03699925,0.063630216,-0.011167165,0.043917067,0.009071561,0.024705062,0.0058992277,0.008956694,-0.08780625,-0.023080673,0.04874955,-0.051052783,0.025091192,-0.10834372,-0.055088677,-0.04328814,-0.00071972614,0.103448965,0.09651871,0.013013168,0.03645502,-0.16584149,0.09575695,-0.07993959,0.024883622,0.021971965,0.14378943,-0.10537865,-0.09493358,0.094411224,0.031722646,-0.10165089,0.03251076,-0.07510443,-0.00429169,-0.029993985,0.040604614,0.06463966,-0.022239022,-0.12926033,0.075517945,-0.10969244,-0.017435469,-0.04517842,0.0658391,-0.15180442,-0.028503388,0.015611426,-0.06440611,0.065610245,-0.012086352,-0.006206612,-0.051527128,-0.09286374,0.11125911,-0.20562118,-0.07552552,-0.007750443,-0.061116524,-0.03924982,-0.006784758,0.0630935,0.066750385,0.04459811,0.070741124,-0.13401248,0.012925395,0.13811992,0.036385667,-0.07361795,0.012512187,-0.20348479,-0.026341457,0.06348557,0.18100889,-0.11873324,0.072731264,0.05758772"
//    val key = Vectors.dense(keyline.split(",").map(_.toDouble))
//    val k = 10
//    model.approxNearestNeighbors(df, key, k).rdd.saveAsTextFile("/user/predict/liulu/lookalike/graph/lsh")

    // Feature Transformation
    //println("The hashed dataset where hashed values are stored in the column 'hashes':")
    //model.transform(df)

    // Compute the locality sensitive hashes for the input rows, then perform approximate similarity join.
    // We could avoid computing hashes by passing in the already-transformed dataset,
    // e.g. `model.approxSimilarityJoin(transformedA, transformedB, 2.5)`
    //println("Approximately joining dfA and dfB on Euclidean distance smaller than 2.5:")
    registerUDF()

    model.approxSimilarityJoin(df, df, threshold, "EuclideanDistance")
      .select(col("datasetA.id").alias("user1"),
        col("datasetB.id").alias("user2"),
        col("EuclideanDistance").alias("distance"))
      .filter("user1 != user2")
      .withColumn("score", functions.callUDF("getScore", functions.col("distance")))
      .withColumn("user1_md5", functions.callUDF("md5Handle", functions.col("user1")))
      .withColumn("user2_md5", functions.callUDF("md5Handle", functions.col("user2")))
      .withColumn("user2_tp_id", functions.callUDF("tpId"))
      .withColumn("emode", functions.callUDF("eMode"))
      .select(col("user1"), col("user1_md5"),
        col("user2"), col("user2_tp_id"), col("user2_md5"),
        col("score"), col("emode"))
      .createOrReplaceTempView("Tmp_lookalike_table")

    spark.sql(
      s"""
         |insert into table aps.lookalike_user_similar_pairs
         |partition(dt=${date})
         |select * from Tmp_lookalike_table
         |""".stripMargin)
  }

  def loadRes() {
    registerUDF()
    val assist = new Assist
    var res = context.textFile("/user/predict/liulu/lookalike/graph/rec_0")
      .map(_.split("\t")).filter(_.length == 3).map(x=>(x(0), x(1), x(2).toDouble))
    for(iter <- 1 until 8) {
      if (assist.check_path(s"/user/predict/liulu/lookalike/graph/rec_${iter}")){
        res = res.union(context.textFile(s"/user/predict/liulu/lookalike/graph/rec_${iter}")
          .map(_.split("\t")).filter(_.length == 3).map(x=>(x(0), x(1), x(2).toDouble)))
      }
    }
    spark.createDataFrame(res).toDF("user1", "user2", "score")
      .filter("user1 != user2")
      .withColumn("user1_md5", functions.callUDF("md5Handle", functions.col("user1")))
      .withColumn("user2_tp_id", functions.callUDF("tpId"))
      .withColumn("user2_md5", functions.callUDF("md5Handle", functions.col("user2")))
      .withColumn("emode", functions.callUDF("eMode"))
      .select(col("user1"), col("user1_md5"),
        col("user2"), col("user2_tp_id"), col("user2_md5"),
        col("score"), col("emode"))
      .createOrReplaceTempView("Tmp_lookalike_table")

    spark.sql(
      s"""
         |insert overwrite table aps.lookalike_user_similar_pairs
         |partition(dt=${date})
         |select * from Tmp_lookalike_table
         |""".stripMargin)
  }

}
