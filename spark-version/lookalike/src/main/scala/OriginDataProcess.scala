import java.io.Serializable

import org.apache.spark.SparkContext
import org.apache.spark.sql.SparkSession

import scala.collection.Map

object OriginDataProcess extends Serializable {

  def process(args: Array[String],
           spark: SparkSession,
           context: SparkContext,
           config: Map[String, String]) {

    val train_path = config("TRAIN_PATH")
    val rating_thres = config("RATING_THRE").toDouble
    val date = config("DATE")
    val numSelected = config("HIS_LENGTH_SELECTED").toInt
    val node2id_path = config("NODE2IDPATH")
    val assist = new Assist

    val train_RDD = context.textFile(s"${train_path}/${date}").map(assist.parse_line)
      .filter(x=>(x._2._3 == 1) & (x._2._1.length > 0))//(member_id, (gds,score))
      .map(x=>((x._1, x._2._1), List(x._2._2))).reduceByKey(_++_)
      .map(x=>(x._1._1, x._1._2, x._2.max))
      .filter(x => x._3 >= rating_thres)

    val invalidGds = context.broadcast(train_RDD.mapPartitions(x => x.map(l => (l._2, 1)))
      .reduceByKey(_+_).filter(_._2 > 10000).mapPartitions(x => x.map(l => l._1)).collect())

    val selected = train_RDD.filter(x => !invalidGds.value.contains(x._2))
      .mapPartitions(x => x.map(l => ((scala.util.Random.nextInt(10), l._1), List((Math.pow(l._3, 2), l._2)))))
      .reduceByKey((x, y) => x ++ y)
      .mapPartitions(x => x.map(l => (l._1._2, l._2.sortWith(_._1 > _._1).take(numSelected))))
      .reduceByKey((x, y) => x ++ y)
      .filter(x=>x._2.length <= 100)
      .mapPartitions{x =>
        var res: List[(String, String, String)] = List()
        while (x.hasNext){
          val (memb, list) = x.next()
          val selectedList = list.sortWith(_._1 > _._1).take(numSelected)
          val ws = selectedList.map(_._1).sum
          res ++= selectedList.map(l => (memb, l._2, (l._1 / ws).formatted("%.5f")))
        }
        res.toIterator
      }
    assist.overwrite_path("/user/predict/liulu/lookalike/graph/selected_records")
    selected.map(x=>s"${x._1},${x._2},${x._3}").repartition(100)
      .saveAsTextFile("/user/predict/liulu/lookalike/graph/selected_records")

    val user_valid = selected.mapPartitions(x => x.map(l => (l._1, 1)))
      .reduceByKey((x, y) => x + y)

    spark.createDataFrame(user_valid).toDF("memb_id", "his_length")
      .createOrReplaceTempView("Tmp_table")

    val age = Array("60q","60h","70h","80h","90h","95h","00h","-")
    val groupId = context.broadcast(age.zipWithIndex.toMap)

    val user_processed = spark.sql(
      s"""
         |select t1.memb_id,
         |case when t2.age in ("60q","60h","70h","80h","90h","95h","00h") then t2.age
         |when t2.age="05h" then "00h"
         |else '-' end as age,
         |case when t2.gender in ("124000000020","124000000010") then t2.gender
         |else '-' end as gender from
         |(select memb_id from Tmp_table)t1
         |left join
         |(select memb_id,gender_cd as gender,age_group as age
         |from predict.t_user_base_feature_info
         |where statis_date=${date})t2
         |on t1.memb_id=t2.memb_id
         |""".stripMargin).rdd
      .map(x=>(x.getString(0),
        groupId.value(x.getString(1))))

    assist.overwrite_path(node2id_path+"_gid")
    user_processed.filter(_._2 != groupId.value("-"))
      .mapPartitions(x => x.map(l => (l._2, List(l._1))))
      .reduceByKey(_++_).mapPartitions{x =>
      var res: List[String] = List()
      while (x.hasNext) {
        val (gid, membList) = x.next()
        res ++= membList.zipWithIndex.map(l => l._1+"\t"+l._2+"\t"+gid)
      }
      res.toIterator
    }.union(user_processed.filter(_._2 == groupId.value("-")).zipWithIndex()
      .mapPartitions{x =>
        var res: List[String] = List()
        while (x.hasNext) {
          val ((memb, gid), index) = x.next()
          res .::= (memb+"\t"+index+"\t"+gid)
        }
        res.toIterator
      }).saveAsTextFile(node2id_path+"_gid")

  }
}
