import java.io.Serializable

import org.apache.spark.SparkContext
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.SparkSession
import org.apache.spark.storage.StorageLevel

import scala.collection.Map
import scala.collection.mutable.ArrayBuffer

object graphConstruct2 extends Serializable {

  case class member2idx(member_id:String, index: String)
  case class member_interests(member_id:String, gds: String, score: String)

  var spark: SparkSession = null
  var context: SparkContext = null
  var train_path = ""
  var rating_thres = 0.00005
  var user_action_length = 1000
  var date = ""
  var node2id_path = ""

  def setup(spark: SparkSession, context: SparkContext,
            param: Map[String, String]): this.type = {
    this.spark = spark
    this.context = context
    this.train_path = param("TRAIN_PATH") + param("DATE")
    this.rating_thres = param("RATING_THRE").toDouble
    this.user_action_length = param("HIS_LENGTH").toInt
    this.date = param("DATE")
    this.node2id_path = param("NODE2IDPATH")

    this
  }

  def export_edgelist(gidSelected: Int): RDD[(Long, Long, Double)] = {

    val train_RDD = context.textFile("/user/aps/liulu/lookalike/graph/selected_records")
      .map(_.split(",")).filter(_.length == 3)
      .mapPartitions(x => x.map(l=> (l(0), l(1), l(2).toDouble)))

    val user_processed = context.textFile(node2id_path+"_gid").map(x=>x.split("\t"))
      .filter(x=>x.length == 3).map(x=>(x(0), (x(1).toLong, x(2).toInt))).filter(x => x._2._2 == gidSelected)

    val nodeRDD = train_RDD.map(x=>(x._1, (x._2, x._3))).join(user_processed)
      .mapPartitions{ x =>
        var res: List[(String, (List[(Long, Double)], Int))] = List()
        while (x.hasNext)
        {
          val (memb, ((gds, score), (index, gid))) = x.next()
          res .::= (s"${gds}", (List((index, score)), 1))
        }
        res.iterator
      }.reduceByKey((x, y) => (x._1 ++ y._1, x._2 + y._2))
      .persist(StorageLevel.MEMORY_AND_DISK_SER)

    train_RDD.unpersist(blocking = false)

    var graphEdges = nodeRDD.filter(x=>x._2._2 < 1000).mapPartitions{ x=>
      var res: List[((Int, Long, Long), Double)] = List()
      while (x.hasNext)
      {
        val (gds_gid, (list, len)) = x.next()
        val t_list = ArrayBuffer[((Long, Long), Double)]()
        list.combinations(2).foreach { i =>
          if(i(0)._1 < i(1)._1) {
            t_list.append(((i(0)._1, i(1)._1), i(0)._2*i(1)._2))
          }
          else {t_list.append(((i(1)._1, i(0)._1), i(0)._2*i(1)._2))}
        }
        val r = scala.util.Random.nextInt(10)
        res ++= t_list.toArray.map(l => ((r, l._1._1, l._1._2), l._2)).toList
      }
      res.toIterator
    }

    graphEdges = graphEdges.union(nodeRDD.filter(x=>x._2._2 > 1000).mapPartitions{ x=>
      var res: List[((Int, Long, Long), Double)] = List()
      while (x.hasNext)
      {
        val (gds_gid, (list, len)) = x.next()
        val t_list = ArrayBuffer[((Long, Long), Double)]()
        list.combinations(2).foreach { i =>
          if(i(0)._1 < i(1)._1) {
            t_list.append(((i(0)._1, i(1)._1), i(0)._2*i(1)._2))
          }
          else {t_list.append(((i(1)._1, i(0)._1), i(0)._2*i(1)._2))}
        }
        val r = scala.util.Random.nextInt(10)
        res ++= t_list.toArray.map(l => ((r, l._1._1, l._1._2), l._2)).toList
      }
      res.toIterator
    })

    val edgeOutput = graphEdges.reduceByKey(_+_)
      .mapPartitions{ x =>
        var res: List[((Long, Long), Double)] = List()
        while (x.hasNext)
        {
          val ((r, memb1, memb2), weight) = x.next()
          res .::= ((memb1, memb2), weight)
        }
        res.iterator
      }.reduceByKey(_+_)
      .mapPartitions{ x =>
        var res: List[(Long, Long, Double)] = List()
        while (x.hasNext)
        {
          val ((memb1, memb2), weight) = x.next()
          res .::= (memb1, memb2, weight)
          res .::= (memb2, memb1, weight)
        }
        res.iterator
      }
    nodeRDD.unpersist(false)

    edgeOutput
  }
}
