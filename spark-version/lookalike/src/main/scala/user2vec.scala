import java.io.Serializable

import org.apache.log4j.{Level, Logger}
import org.apache.spark.SparkContext
import org.apache.spark.sql.SparkSession

import scala.collection.Map

object user2vec extends Serializable {

  def main(spark: SparkSession, context: SparkContext, config: Map[String, String]) {

    try {
      Logger.getRootLogger().setLevel(Level.ERROR)
      val Assist = new _root_.Assist
      val age = Array("60q", "60h", "70h", "80h", "90h", "95h", "00h", "-")
      for (iter <- 0 until age.length){
        val subGraphWalks = context.textFile(s"${config("OUTPUT_DIR")}/seq_${iter}")
          .map(x=>x.split(" ").map(_.toLong))

        val id2Node = context.textFile(config("NODE2IDPATH")+"_gid")
          .map(x=>x.split("\t")).filter(x=>x.length == 3)
          .filter(x => x(2) == iter.toString).map(x=>(x(1).toLong, x(0)))

        val word2vec = Word2vec.setup(context, config)
          .fit(subGraphWalks.map(_.map(_.toString).toIterable))
          .getVectors.toList

        val user2vecSub = context.parallelize(word2vec)
          .map { case (nodeId, vector) =>
            val l2_norm = scala.math.sqrt(vector.map(scala.math.pow(_, 2)).sum)
            (nodeId.toLong, vector.map(_/l2_norm))
          }.join(id2Node).map { case (nodeId, (vector, name)) => (name, vector) }

        Assist.overwrite_path(config("OUTPUT_DIR") + s"/emb${iter}")
        user2vecSub.map(x => x._1 + "\t" + x._2.mkString(",")).repartition(10)
          .saveAsTextFile(config("OUTPUT_DIR") + s"/emb${iter}")

        }
      }
      finally {
        spark.stop()
      }
    }
  }
