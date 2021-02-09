import java.io.Serializable

import org.apache.spark.HashPartitioner
import org.apache.spark.rdd.RDD
import org.apache.spark.storage.StorageLevel

import scala.util.Random

class HyperGraph(edges: RDD[(Long, Long, Double)],
                 numPartition: Int) extends Serializable {

  val adjacencyList = edges.mapPartitions{ x =>
    var res: List[(Long, List[(Long, Double)])] = List()
    while (x.hasNext)
    {
      val (memb1, memb2, weight) = x.next
      res .::= (memb1, List((memb2, weight)))
    }
    res.iterator
  }.reduceByKey(_++_, numPartition)
    .partitionBy(new HashPartitioner(numPartition))
    .persist(StorageLevel.MEMORY_AND_DISK_SER)

  def getNextNode(neighbors: List[(Long, Double)]): Long = {
    val x = scala.util.Random.nextDouble()
    var cumulative_prob = 0.0
    var nextNode: Long = -1L
    neighbors.sortWith(_._2<_._2).foreach{ case(neighborNode, prob) =>
      cumulative_prob += prob
      if (x < cumulative_prob ) nextNode = neighborNode
    }
    nextNode
  }

  def getSingleRandomWalks(walkLength: Int): RDD[Array[Long]] = {

    // Bootstrap the random walk from every vertex
    var keyedRandomWalks = adjacencyList.keys.map(id => {
      val walk = new Array[Long](walkLength)
      walk(0) = id
      (id, walk)
    })

    // Grow the walk choosing a random neighbour uniformly at random
    for (iter <- 1 until walkLength) {
      val grownRandomWalks =
        adjacencyList.join(keyedRandomWalks)
          .map {
            case (node_id, (neighbours, walk)) => {
              val r = new Random()
              val randomNeighbour = getNextNode(neighbours)
              walk(iter) = randomNeighbour
              (randomNeighbour, walk)
            }
          }

      keyedRandomWalks.unpersist()
      keyedRandomWalks = grownRandomWalks

    }

    keyedRandomWalks.values
  }


  def getRandomWalks(walkLength: Int,
                      walksPerVertex: Int): RDD[Array[Long]] = {

    // Bootstrap the random walk from every vertex
    var keyedRandomWalks = adjacencyList.keys.flatMap(id => {
      for (iter <- 1 to walksPerVertex)
        yield {
          val walk = new Array[Long](walkLength + 1)
          walk(0) = id
          (id, walk)
        }
    })

    // Grow the walk choosing a random neighbour uniformly at random
    for (iter <- 1 to walkLength) {
      val grownRandomWalks =
        adjacencyList.join(keyedRandomWalks)
          .map {
            case (node_id, (neighbours, walk)) => {
              val randomNeighbour = getNextNode(neighbours)
              walk(iter) = randomNeighbour
              (randomNeighbour, walk)
            }
          }

      keyedRandomWalks.unpersist(false)
      keyedRandomWalks = grownRandomWalks

    }

    keyedRandomWalks.values
  }

  def unpersist_resource()  = {
    adjacencyList.unpersist()

    this
  }
}

object HyperGraph extends Serializable {

//  def edgeListFile (spark: SparkSession,
//                    path : String,
//                    numPartition: Int): HyperGraph = {
//
//    val edges = spark.sparkContext
//      .textFile(path, 8)
//      .flatMap { line => {
//        val fields = line.split("\\s+")
//        val a = fields(0).toLong
//        val b = fields(1).toLong
//        val w = fields(2).toDouble
//        Array((a,b,w), (b,a,w))
//      } }
//
//    new HyperGraph(edges, numPartition)
//  }

  def loadEdges (edges: RDD[(Long, Long, Double)],
                 numPartition: Int): HyperGraph = {
    new HyperGraph(edges, numPartition)
  }

}
