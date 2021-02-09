import java.io.{File, PrintWriter, Serializable}
import java.text.SimpleDateFormat
import java.util.{Calendar, Date}

import org.apache.hadoop.fs.Path
import org.apache.spark.rdd.RDD

import scala.collection.mutable.ArrayBuffer

class Assist extends Serializable {

  def getToday: String = {
    val now: Date = new Date()
    val dateFormat: SimpleDateFormat = new SimpleDateFormat("yyyyMMdd")
    dateFormat.format(now)
  }

  def getDateDiff(start_time:String, end_Time:String): Int ={
    val df:SimpleDateFormat = new SimpleDateFormat("yyyyMMdd")
    val begin:Date = df.parse(start_time)
    val end:Date = df.parse(end_Time)
    val between:Long = (end.getTime() - begin.getTime())/ 1000 / 3600 / 24//转化成秒

    between.toInt
  }

  def getDeltaDay(beginTime: String, delta: Int): String = {
    val dateFormat: SimpleDateFormat = new SimpleDateFormat("yyyyMMdd")
    val dt = dateFormat.parse(beginTime)

    val cal: Calendar = Calendar.getInstance()
    cal.setTime(dt)

    cal.add(Calendar.DATE, delta)
    dateFormat.format(cal.getTime)
  }

  def check_path(output_path: String): Boolean = {
    val path = new Path(output_path)
    val hdfs = org.apache.hadoop.fs.FileSystem.get(
      new java.net.URI(output_path), new org.apache.hadoop.conf.Configuration())

    hdfs.exists(path)
  }

  def overwrite_path(output_path: String): Unit= {
    val path = new Path(output_path)
    val hdfs = org.apache.hadoop.fs.FileSystem.get(
      new java.net.URI(output_path), new org.apache.hadoop.conf.Configuration())
    if (hdfs.exists(path))
      hdfs.delete(path, true)
  }

  def parse_line(line: String): (String, (String, Double, Int)) = {
    val cols = line.replace("\"", "").split("\\s+")
    if (cols.length != 4 && cols.length != 5 && cols.length != 6) {
      ("000", ("000", 0.0, -1))
    } else {
      val user = cols(0)
      val item = cols(1)
      val rating = cols(2).toDouble
      val vip = cols(3)
      if (vip == "1") {
        (user, (item, rating, 1))
      } else if (vip == "0"){
        (user, (item, rating, 0))
      } else {
        ("000", ("000", 0.0, -1))
      }
    }
  }

  def make_item_pairs(pair: (String, List[(String, Double)])):
  Array[((String, String), Double)]={
    val t_list = ArrayBuffer[((String, String), Double)]()
    pair._2.combinations(2).foreach { i =>
      t_list.append(((i(0)._1, i(1)._1), i(0)._2*i(1)._2))
      t_list.append(((i(1)._1, i(0)._1), i(0)._2*i(1)._2))
    }
    t_list.toArray
  }

  def make_item_pairs2(pair: (String, List[(Long, Double)])):
  Array[((Long, Long), Double)]={
    val t_list = ArrayBuffer[((Long, Long), Double)]()
    pair._2.combinations(2).foreach { i =>
      t_list.append(((i(0)._1, i(1)._1), i(0)._2*i(1)._2))
      t_list.append(((i(1)._1, i(0)._1), i(0)._2*i(1)._2))
    }
    t_list.toArray
  }

  def getNowWeekEnd():String={
    var period:String=""
    var cal:Calendar =Calendar.getInstance()
    var df:SimpleDateFormat = new SimpleDateFormat("yyyy-MM-dd")
    cal.set(Calendar.DAY_OF_WEEK, Calendar.SUNDAY)
    cal.add(Calendar.WEEK_OF_YEAR, 1)
    period=df.format(cal.getTime())
    period
  }

  def writeCSVFile[T](
                       path: String,
                       schema: Array[String],
                       data: Array[Array[T]]) = {
    val writer = new PrintWriter(new File(path))
    writer.write(schema.mkString(",")+"\n")
    data.foreach {(x: Array[T]) => writer.write(x.mkString(",")+"\n") }
    writer.close()
  }

  def saveVectors(
                   path: String,
                   header: Array[Int],
                   data: Map[String, Array[Float]]) = {

    val writer = new PrintWriter(new File(path))
    writer.write(header.mkString(" ")+"\n")
    data.foreach {
      case (node:String, vec: Array[Float]) =>
        writer.write(node+" "+ vec.mkString(" ") +"\n")
    }
    writer.close()
  }

  def vertexVisitCounts(walks: RDD[List[Int]]): Array[Array[Long]] = {
    walks.flatMap((walk: List[Int]) => walk)
      .countByValue
      .values
      .groupBy(identity)
      .mapValues(_.size)
      .map(kv => Array(kv._1, kv._2))
      .toArray

  }

}
