import org.apache.log4j.{Level, Logger}
import org.apache.spark.SparkConf
import org.apache.spark.sql.SparkSession
import org.apache.spark.storage.StorageLevel

import scala.collection.Map

object randomWalks {
object randomWalks {

  def parseArguments(args: Array[String]): Map[String, String] = {
    Map(
      "TRAIN_PATH"          -> "/user/aps/recom/cf-rating/train-data/",
      "DATE"                -> args(0),
      "RATING_THRE"         -> "0.00005",
      "HIS_LENGTH"          -> "100",
      "HIS_LENGTH_SELECTED" -> "10",
      "NUM_OF_RANDOM_WALKS" -> "10",
      "RANDOM_WALK_LENGTH"  -> "10",
      "VECTOR_DIM"          -> "128",
      "NUM_PARTITIONS"      -> "2000",
      "NUM_ITERATIONS"      -> "5",
      "WINDOW_SIZE"         -> "5",
      "OUTPUT_DIR"          -> "/user/predict/liulu/lookalike/graph/",
      "NODE2IDPATH"         -> "/user/predict/liulu/lookalike/graph/node2id",
      "NUM_HASHBUCKET"      -> "10",
      "NUM_BUCKETLENGTH"    -> "8",
      "LSH_THRESHOLD"       -> "0.01"
    )
  }

  def main(args: Array[String]) {

    // setup spark session
    val conf = new SparkConf()
    conf.set("spark.sql.shuffle.partitions", "2000")
    conf.set("spark.default.parallelism", "2000")
    conf.set("spark.dynamicAllocation.maxExecutors","2000")
    conf.set("spark.kryoserializer.buffer.max", "1024m")
    conf.set("spark.driver.maxResultSize", "16G")
    conf.set("spark.dynamicAllocation.enabled", "true")
    conf.set("spark.shuffle.service.enabled", "true")
    conf.set("spark.rdd.compress", "true")
    conf.set("spark.yarn.maxAppAttempts", "1")
    conf.set("spark.network.timeout", "10000000")
    conf.set("spark.executor.memoryOverhead", "4096m")

    val spark = SparkSession
      .builder()
      .config(conf)
      .appName("sequence_maker_19046349")
      .enableHiveSupport()
      .getOrCreate()

    val context = spark.sparkContext
    try {
      Logger.getRootLogger().setLevel(Level.ERROR)

      val config = parseArguments(args)
      val Assist = new _root_.Assist
      OriginDataProcess.process(args, spark, context, config)

      val age = Array("60q", "60h", "70h", "80h", "90h", "95h", "00h", "-")

      for (iter <- 0 until age.length){ //age.length
        val G = graphConstruct2.setup(spark, context, config).export_edgelist(iter)
          .persist(StorageLevel.MEMORY_AND_DISK_SER)

        var nodesWeightSum = context.broadcast(G.mapPartitions(x => x.map(l => (l._1, l._3)))
          .reduceByKey(_ + _).collectAsMap())

        val GProcessed = G.mapPartitions{ x =>
          var res : List[((Int, Long), List[(Long, Double)])] = List()
          while (x.hasNext) {
            val (memb1, memb2, weight) = x.next()
            val weightSum = nodesWeightSum.value(memb1)
            res .::= ((scala.util.Random.nextInt(100), memb1), List((memb2, weight/weightSum)))
          }
          res.toIterator
        }.reduceByKey(_ ++ _)
          .mapPartitions(x => x.map(l => (l._1._2, l._2.sortWith(_._2 > _._2).take(100))))
          .reduceByKey(_ ++ _)
          .mapPartitions { x=>
            var res: List[(Long, Long, Double)] = List()
            while (x.hasNext) {
              var (memb1, list) = x.next()
              list = list.sortWith(_._2 > _._2).take(100)
              res ++= list.map(l => (memb1, l._1, l._2))
            }
            res.toIterator
          }

        nodesWeightSum = context.broadcast(G.mapPartitions(x => x.map(l => (l._1, l._3)))
          .reduceByKey(_ + _).collectAsMap())

        val subGraphWalks = HyperGraph.loadEdges(GProcessed, config("NUM_PARTITIONS").toInt)
          .getRandomWalks(config("RANDOM_WALK_LENGTH").toInt,
            config("NUM_OF_RANDOM_WALKS").toInt)

        Assist.overwrite_path(s"${config("OUTPUT_DIR")}/seq_${iter}")
        subGraphWalks.mapPartitions(x => x.map(l => l.mkString(" "))).repartition(10)
          .saveAsTextFile(s"${config("OUTPUT_DIR")}/seq_${iter}")
      }

      user2vec.main(spark, context, config)
    }
    finally {
      spark.stop()
    }
  }
}
