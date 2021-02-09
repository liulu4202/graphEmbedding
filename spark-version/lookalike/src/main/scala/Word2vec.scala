import java.io.Serializable

import org.apache.spark.SparkContext
import org.apache.spark.mllib.feature.{Word2Vec, Word2VecModel}
import org.apache.spark.rdd.RDD

import scala.collection.Map

object Word2vec extends Serializable {
  var context: SparkContext = null
  var word2vec = new Word2Vec()
  var model: Word2VecModel = null
  var vectorSize = 128
  var learningRate = 0.025
  var numPartitions = 4000
  var numIterations = 2
  var minCount = 1
  var maxSentenceLength = 11

  def setup(context: SparkContext, param: Map[String, String]): this.type = {
    this.context = context
    /**
     * model = sg
     * update = hs
     */
    word2vec.setLearningRate(learningRate)
      .setNumIterations(numIterations)
      .setNumPartitions(param("NUM_PARTITIONS").toInt)
      .setMinCount(minCount)
      .setVectorSize(param("VECTOR_DIM").toInt)

    val word2vecWindowField = word2vec.getClass.getDeclaredField("org$apache$spark$mllib$feature$Word2Vec$$window")
    word2vecWindowField.setAccessible(true)
    word2vecWindowField.setInt(word2vec, param("WINDOW_SIZE").toInt)

    this
  }

//  def read(path: String): RDD[Iterable[String]] = {
//    context.parallelize(context.textFile(path).take(100000)).repartition(200).map(_.split("\\s").toSeq)
//  }

  def fit(input: RDD[Iterable[String]]): this.type = {
    model = word2vec.fit(input)

    this
  }

  def save(outputPath: String): this.type = {
    model.save(context, s"$outputPath.bin")
    this
  }

  def load(path: String): this.type = {
    model = Word2VecModel.load(context, path)

    this
  }

  def getVectors = this.model.getVectors

}
